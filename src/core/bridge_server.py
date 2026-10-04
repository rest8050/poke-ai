"""마인크래프트(Cobblemon + RCT) 모드 ↔ 모델 브리지 서버.

모드(mod/pokeai)가 BattleAI.choose()가 불릴 때마다 Cobblemon이 쌓은 Showdown 원본 메시지 덩어리(update / sideupdate)를 보내면,
1) 우리 쪽 정보만 남기고(상대의 sideupdate·비밀 줄 제거) 표준 Showdown 형식으로 정규화하고 (Cobblemon은 포켓몬 이름 자리에 UUID를 씀)
2) poke-env Battle에 반영 → 학습 때와 같은 인코더 → 신경망(all8a) → 행동 하나를 돌려준다.
행동은 {"kind": "move", "id", "tera"} / {"kind": "switch", "uuid"} / {"kind": "default"}.

실행: python -m src.core.bridge_server            (기본 127.0.0.1:8765, 체크포인트는 환경변수 POKEAI_CKPT)
"""
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from poke_env.battle import Battle

from src.core.model import DeepPokemonBattleTransformerNet, load_compatible
from src.core.rct_player import eff_move
from src.core.search import decode_action
from src.core.tensor_encoder import BattleTensorEncoder

CKPT = os.environ.get("POKEAI_CKPT", "checkpoints/supervised_v2_fp_all8a.pt")
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
DETAILS_RE = re.compile(r"([^|,\"]+), (" + UUID + r")")           # "Monferno, <uuid>, L69, M" → 종 이름 ↔ uuid
IDENT_RE = re.compile(r"(p[12][abc]?): (" + UUID + r")")            # "p1a: <uuid>" → "p1a: Monferno"
DROP_PREFIXES = ("|pp_update|", "|error|", "|debug|", "|inactive|", "|inactiveoff|")

log = logging.getLogger("bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
torch.set_num_threads(1)

model = DeepPokemonBattleTransformerNet()
load_compatible(model, torch.load(CKPT, map_location="cpu"))
model.eval()
encoder = BattleTensorEncoder()


def warmup():
    """서버 시작 때 모델을 한 번 돌려 둠 (첫 호출이 0.5초 걸리는 초기화를 배틀 중에 겪지 않도록)."""
    from src.core.tensor_encoder import FIELD_DIM, NUM_DIM
    g = torch.Generator().manual_seed(0)
    my_num = torch.rand(1, 6, NUM_DIM, generator=g)
    my_num[:, :, 0] = 0
    my_num[:, 0, 0] = 1
    opp_num = torch.rand(1, 6, NUM_DIM, generator=g)
    opp_num[:, :, 0] = 0
    opp_num[:, 1, 0] = 1
    obs = (torch.randint(1, 8, (1, 6, 11), generator=g), my_num, torch.rand(1, 6, 4, 46, generator=g),
           torch.randint(1, 8, (1, 6, 11), generator=g), opp_num, torch.rand(1, 6, 4, 46, generator=g),
           torch.rand(1, FIELD_DIM, generator=g))
    t0 = time.perf_counter()
    model.get_action(*obs, None, torch.ones(1, 22, dtype=torch.bool))
    log.info("모델 예열 완료 (%.0fms)", (time.perf_counter() - t0) * 1000)


warmup()
app = FastAPI(title="PokeAI bridge")
sessions: "OrderedDict[str, Session]" = OrderedDict()
MAX_SESSIONS = 200


class Session:
    def __init__(self, side: str):
        self.side = side
        self.battle: Optional[Battle] = None
        self.history = None
        self.species_of_uuid = {}   # uuid → 종 이름 (프로토콜의 UUID 이름을 poke-env가 아는 이름으로 바꾸는 표)
        self.uuid_of_ident = {}     # "Charizard" → uuid (교체 응답에 uuid를 돌려주기 위한 표)
        self.done = False
        self.n_calls = 0
        self.preview_done = False
        self.last = time.time()

    def learn_names(self, text: str):
        for sp, u in DETAILS_RE.findall(text):
            self.species_of_uuid[u] = sp.strip()

    def normalize(self, text: str) -> str:
        self.learn_names(text)
        text = re.sub(r", " + UUID, "", text)                                   # details에서 uuid 제거
        return IDENT_RE.sub(lambda m: f"{m.group(1)}: {self.species_of_uuid.get(m.group(2), m.group(2))}", text)


def filter_lines(lines: List[str], side: str) -> List[str]:
    """|split|pX 뒤 두 줄(비밀, 공개) 중 우리 쪽은 비밀 줄(정확한 체력), 상대 쪽은 공개 줄만 남김. 비표준/불필요한 줄 제거."""
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|split|"):
            who = ln.split("|")[2]
            secret, public = (lines[i + 1] if i + 1 < len(lines) else None), (lines[i + 2] if i + 2 < len(lines) else None)
            keep = secret if who == side else public
            if keep:
                out.append(keep)
            i += 3
            continue
        if ln and not ln.startswith(DROP_PREFIXES):
            out.append(ln)
        i += 1
    return out


def split_chunk(chunk: str):
    """Cobblemon이 쌓은 덩어리 → (종류, 대상 side 또는 None, 줄 목록). 종류: update / sideupdate / 그 외"""
    lines = chunk.replace("\r", "").split("\n")
    kind = lines[0].strip() if lines else ""
    if kind == "sideupdate":
        return kind, (lines[1].strip() if len(lines) > 1 else None), lines[2:]
    return kind, None, lines[1:]


class ChooseRequest(BaseModel):
    battle_id: str
    side: str
    chunks: List[str] = []
    opp_preview: List[dict] = []    # 모드가 알려주는 상대 파티 [{"species": showdown id, "level": n}] — Showdown 팀 프리뷰와 같은 수준의 정보(종과 레벨)


def ensure_battle(sess: Session, req: ChooseRequest) -> Battle:
    if sess.battle is None:
        name = None
        for ch in req.chunks:                       # 우리 쪽 요청의 side.name(=UUID)이 poke-env가 아는 "내 이름"
            kind, who, lines = split_chunk(ch)
            if kind == "sideupdate" and who == req.side:
                for ln in lines:
                    if ln.startswith("|request|") and ln[9:].strip():
                        try:
                            name = json.loads(ln[9:])["side"]["name"]
                        except Exception:
                            pass
        sess.battle = Battle(f"battle-{req.battle_id}", name or req.side, logging.getLogger("poke_env"), gen=9)
        sess.battle.player_role = req.side
    return sess.battle


def ingest(sess: Session, req: ChooseRequest):
    battle = ensure_battle(sess, req)
    if req.opp_preview and not sess.preview_done:
        # 학습 때 팀 프리뷰로 알던 정보와 같은 형태로 등록: 상대 파티의 종/레벨 (기술/도구/능력치는 여전히 등장·사용해야 보임)
        opp_side = "p1" if sess.side == "p2" else "p2"
        for m in req.opp_preview:
            sp, lv = str(m["species"]), int(m.get("level", 100))
            for cand in (sp, re.sub(r"(female|male)$", "", sp)):   # meowsticmale 같은 성별 폼은 기본 종으로
                try:
                    battle._register_teampreview_pokemon(opp_side, f"{cand}, L{lv}")
                    break
                except Exception:
                    continue
            else:
                log.warning("프리뷰 등록 실패 (무시): %r", m)
        sess.preview_done = True
    for ch in req.chunks:
        kind, who, lines = split_chunk(ch)
        if kind == "sideupdate":
            if who != sess.side:                    # 상대 쪽 요청/비공개 정보는 버림
                continue
            for ln in lines:
                if ln.startswith("|request|") and ln[9:].strip():
                    raw = ln[9:]
                    # 정규화 전 원본에서 "ident → uuid" 표를 만듦 (교체 응답에 uuid를 돌려주기 위함)
                    for ident, u in re.findall(r'"ident":"p[12]: ([^"]+)","details":"[^"]*?(' + UUID + r')', raw):
                        sess.uuid_of_ident[ident] = u
                    battle.parse_request(json.loads(sess.normalize(raw)))
            continue
        if kind != "update":
            continue
        for ln in filter_lines(lines, sess.side):
            ln = sess.normalize(ln)
            if ln.startswith("|win|") or ln.startswith("|tie"):
                sess.done = True
            try:
                battle.parse_message(ln.split("|"))
            except NotImplementedError:
                pass
            except Exception as e:                  # 메시지 하나가 배틀 전체를 죽이지 않도록
                log.warning("메시지 처리 실패 (무시): %r | %s", ln[:120], e)


def _next_legal_alt(idx, probs, mask, decode_fn, is_excluded):
    """확률 내림차순으로 다음 합법 행동 중 is_excluded가 아닌 첫 번째를 찾음. 없으면 None."""
    for alt_idx in torch.argsort(probs[0], descending=True).tolist():
        if alt_idx == idx or not mask[0, alt_idx]:
            continue
        alt = decode_fn(alt_idx)
        if alt is not None and not is_excluded(alt):
            return alt_idx, alt
    return None


def _is_trickroom_move(decoded) -> bool:
    return bool(decoded) and decoded[0] == "move" and decoded[1].id == "trickroom"


def _fields_have_trickroom(battle) -> bool:
    fields = getattr(battle, "fields", None) or {}
    return any("trickroom" in re.sub(r"[^a-z0-9]", "", str(k).lower()) for k in fields)


def avoid_redundant_trickroom(idx, probs, mask, decoded, decode_fn, trickroom_active):
    """필드에 트릭룸이 이미 켜져 있는데 모델이 또 트릭룸을 고르면(쓰면 스스로 꺼짐 — 교사도 가끔 틀림)
    다음으로 확률 높은 합법 행동으로 대체. 그 외엔 그대로 반환."""
    if not (trickroom_active and _is_trickroom_move(decoded)):
        return idx, decoded
    alt = _next_legal_alt(idx, probs, mask, decode_fn, _is_trickroom_move)
    return alt if alt is not None else (idx, decoded)  # 대체할 합법 행동이 없으면(극단적 예외) 원래 선택 유지


_WEATHER_BALL_TYPE = {
    "sunnyday": "fire", "desolateland": "fire",
    "raindance": "water", "primordialsea": "water",
    "sandstorm": "rock",
    "hail": "ice", "snow": "ice", "snowscape": "ice",
}


def _is_weatherball_move(decoded) -> bool:
    return bool(decoded) and decoded[0] == "move" and decoded[1].id == "weatherball"


def _weatherball_is_bad(battle, target) -> bool:
    """날씨 없으면(위력 반토막) 항상 나쁨. 날씨 있으면 그 날씨가 만드는 타입이 상대한테 반감(0.5배) 이하면 나쁨."""
    if target is None:
        return False  # 대상을 모르면 판단 불가 → 손대지 않음
    w = next((re.sub(r"[^a-z0-9]", "", str(k).lower()) for k in (getattr(battle, "weather", None) or {})), "")
    if not w:
        return True
    return eff_move(battle, _WEATHER_BALL_TYPE.get(w, "normal"), target) <= 0.5


def avoid_bad_weatherball(idx, probs, mask, decoded, decode_fn, battle):
    """웨더볼이 지금 날씨 기준으로 타입이 나빠지면(교사도 가끔 놓침) 다음 합법 행동으로 대체."""
    if not _is_weatherball_move(decoded):
        return idx, decoded
    if not _weatherball_is_bad(battle, battle.opponent_active_pokemon):
        return idx, decoded
    alt = _next_legal_alt(idx, probs, mask, decode_fn, _is_weatherball_move)
    return alt if alt is not None else (idx, decoded)


def pick(sess: Session):
    battle = sess.battle
    if battle is None or not (battle.available_moves or battle.available_switches):
        return {"kind": "default", "why": "선택 가능한 행동 없음"}
    tensors = encoder.encode_battle(battle)
    mask = tensors[7]
    idx, probs, _, value, sess.history = model.get_action(*tensors[:7], sess.history, mask)
    decoded = decode_action(battle, idx)
    decode_fn = lambda i: decode_action(battle, i)  # noqa: E731
    idx, decoded = avoid_redundant_trickroom(idx, probs, mask, decoded, decode_fn, _fields_have_trickroom(battle))
    # 근본 원인(인코더 고정 타입 버그)은 tensor_encoder.py에서 고쳤지만, 지금 체크포인트는 그 수정 이전 학습이라
    # 재학습 전까진 안전장치로 유지. 재학습 끝나면 순수 성능 확인차 다시 꺼볼 것.
    idx, decoded = avoid_bad_weatherball(idx, probs, mask, decoded, decode_fn, battle)
    top = torch.topk(probs[0], 3)
    info = {"preview": len(battle.teampreview_opponent_team), "seen": len(battle.opponent_team), "idx": int(idx), "value": round(float(value), 3),
            "top": [(int(i), round(float(p), 3)) for p, i in zip(top.values, top.indices)]}
    if decoded is None:
        return {"kind": "default", "why": f"행동 {idx} 실행 불가", **info}
    if decoded[0] == "switch":
        mon = decoded[1]
        key = next((k for k, v in battle.team.items() if v is mon), None)
        ident = key.split(": ", 1)[1] if key else None
        return {"kind": "switch", "uuid": sess.uuid_of_ident.get(ident), "ident": ident, **info}
    _, move, z, tera = decoded
    return {"kind": "move", "id": move.id, "tera": bool(tera), **info}


@app.get("/health")
def health():
    return {"status": "ok", "ckpt": CKPT, "sessions": len(sessions)}


@app.post("/choose")
def choose(req: ChooseRequest):
    t0 = time.perf_counter()
    sess = sessions.get(req.battle_id)
    if sess is None:
        sess = sessions[req.battle_id] = Session(req.side)
        while len(sessions) > MAX_SESSIONS:
            sessions.popitem(last=False)
    sess.n_calls += 1
    sess.last = time.time()
    for k in [k for k, v in sessions.items() if time.time() - v.last > 1800]:   # 30분 넘게 조용한 배틀은 끝난 것으로 보고 정리
        sessions.pop(k, None)
    try:
        ingest(sess, req)
        out = pick(sess)
    except Exception as e:
        log.exception("choose 실패: %s", e)
        out = {"kind": "default", "why": f"서버 예외: {e}"}
    out["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    log.info("[%s] call#%d turn=%s -> %s", req.battle_id[:8], sess.n_calls, getattr(sess.battle, "turn", "?"), json.dumps(out, ensure_ascii=False))
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("POKEAI_PORT", "8765")))
