"""마인크래프트(Cobblemon + RCT) 모드 ↔ 모델 브리지 서버.

모드(mod/pokeai)가 BattleAI.choose()가 불릴 때마다 Cobblemon이 쌓은 Showdown 원본 메시지 덩어리(update / sideupdate)를 보내면,
1) 우리 쪽 정보만 남기고(상대의 sideupdate·비밀 줄 제거) 표준 Showdown 형식으로 정규화하고 (Cobblemon은 포켓몬 이름 자리에 UUID를 씀)
2) poke-env Battle에 반영 → 학습 때와 같은 인코더 → 신경망(all8a) → 행동 하나를 돌려준다.
행동은 {"kind": "move", "id", "tera"} / {"kind": "switch", "uuid"} / {"kind": "default"}.

실행: python -m src.core.bridge_server            (기본 127.0.0.1:8765)

모델 교체: 서비스 재시작 없이 포인터 파일(기본 checkpoints/CURRENT, 환경변수 POKEAI_POINTER)에 체크포인트 파일 이름 한 줄을 쓰면 된다.
  - 새 배틀(세션)이 시작될 때 포인터가 바뀐 걸 발견하면(최대 5초 지연) 새 모델을 읽어 예열한 뒤 바꿔 끼움. 진행 중인 배틀은 시작한 모델로 끝까지 둠
    (히스토리 상태가 모델마다 달라서). 읽기에 실패하면 이전 모델을 그대로 쓰고 경고만 남김. 포인터가 없으면 환경변수 POKEAI_CKPT.
  - 구조는 체크포인트 옆 .cfg.json으로 복원(model_from_ckpt) — 새 구조는 코드(src/core)가 서버에 있어야 함 (deploy/push_model.sh --code)
"""
import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from poke_env.battle import Battle

from src.core.model import model_from_ckpt
from src.core.rct_player import eff_move
from src.core.search import decode_action
from src.core.tensor_encoder import BattleTensorEncoder

CKPT = os.environ.get("POKEAI_CKPT", "checkpoints/supervised_v2_fp_all8a.pt")      # 포인터 파일이 없을 때 쓰는 기본값
POINTER = os.environ.get("POKEAI_POINTER", "checkpoints/CURRENT")                   # 한 줄: 체크포인트 파일 이름(포인터와 같은 폴더) 또는 경로
RECHECK_SEC = 5.0
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
DETAILS_RE = re.compile(r"([^|,\"]+), (" + UUID + r")")           # "Monferno, <uuid>, L69, M" → 종 이름 ↔ uuid
IDENT_RE = re.compile(r"(p[12][abc]?): (" + UUID + r")")            # "p1a: <uuid>" → "p1a: Monferno"
DROP_PREFIXES = ("|pp_update|", "|error|", "|debug|", "|inactive|", "|inactiveoff|")

log = logging.getLogger("bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
torch.set_num_threads(1)

encoder = BattleTensorEncoder()


def warmup(model):
    """모델을 한 번 돌려 둠 (첫 호출이 0.5초 걸리는 초기화를 배틀 중에 겪지 않도록)."""
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


class ModelHolder:
    """현재 모델 보관 + 포인터 파일 감시. 새 세션이 시작될 때만 확인해서(최대 RECHECK_SEC마다) 진행 중인 배틀의 모델은 안 바뀜."""

    def __init__(self):
        self.lock = threading.Lock()
        self.model, self.name, self.sig, self.loaded_at, self.checked = None, None, None, 0.0, 0.0
        self.refresh(force=True)

    @staticmethod
    def target():
        if os.path.exists(POINTER):
            name = open(POINTER, encoding="utf-8").read().strip().splitlines()[0].strip()
            path = name if os.path.isabs(name) or os.path.dirname(name) else os.path.join(os.path.dirname(POINTER) or ".", name)
        else:
            path = CKPT
        return path

    def refresh(self, force=False):
        now = time.time()
        if not force and now - self.checked < RECHECK_SEC:
            return self.model
        with self.lock:
            self.checked = now
            try:
                path = self.target()
                cfg = path + ".cfg.json"
                sig = (path, os.path.getmtime(path), os.path.getmtime(cfg) if os.path.exists(cfg) else 0)
                if sig == self.sig:
                    return self.model
                model = model_from_ckpt(path)
                model.eval()
                warmup(model)
                log.info("모델 교체: %s -> %s", self.name, os.path.basename(path))
                self.model, self.name, self.sig, self.loaded_at = model, os.path.basename(path), sig, now
            except Exception as e:                      # 실패하면 이전 모델 유지 (처음 시작이면 서버가 뜨지 못하게 그대로 예외)
                if self.model is None:
                    raise
                log.error("모델 교체 실패, 이전 모델(%s) 유지: %s", self.name, e)
        return self.model


holder = ModelHolder()
app = FastAPI(title="PokeAI bridge")
sessions: "OrderedDict[str, Session]" = OrderedDict()
MAX_SESSIONS = 200


class Session:
    def __init__(self, side: str, model, model_name: str):
        self.side = side
        self.model = model            # 배틀 시작 때의 모델로 끝까지 (히스토리 상태가 모델마다 다름)
        self.model_name = model_name
        self.battle_id = ""
        self.stats_checked = False
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
    idx, probs, _, value, sess.history = sess.model.get_action(*tensors[:7], sess.history, mask)
    decoded = decode_action(battle, idx)
    decode_fn = lambda i: decode_action(battle, i)  # noqa: E731
    idx, decoded = avoid_redundant_trickroom(idx, probs, mask, decoded, decode_fn, _fields_have_trickroom(battle))
    # 근본 원인(인코더 고정 타입 버그)은 tensor_encoder.py에서 고쳤지만, 지금 체크포인트는 그 수정 이전 학습이라
    # 재학습 전까진 안전장치로 유지. 재학습 끝나면 순수 성능 확인차 다시 꺼볼 것.
    idx, decoded = avoid_bad_weatherball(idx, probs, mask, decoded, decode_fn, battle)
    top = torch.topk(probs[0], 3)
    if not sess.stats_checked and battle.team:             # 내 팀 실능력치가 요청에서 들어오는지 한 번 확인 (매치업 특징은 실능력치를 전제로 함)
        sess.stats_checked = True
        miss = [k for k, m in battle.team.items() if not all((getattr(m, "stats", None) or {}).get(x) for x in ("hp", "atk", "def", "spa", "spd", "spe"))]
        log.info("[%s] 내 팀 실능력치 %s", sess.battle_id[:8], "모두 있음" if not miss else f"누락: {miss}")
    info = {"model": sess.model_name, "preview": len(battle.teampreview_opponent_team), "seen": len(battle.opponent_team), "idx": int(idx), "value": round(float(value), 3),
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
    return {"status": "ok", "ckpt": holder.name, "loaded_age_s": round(time.time() - holder.loaded_at), "pointer": POINTER, "sessions": len(sessions)}


@app.post("/reload")
def reload_model():
    """포인터를 지금 바로 다시 확인 (보통은 새 배틀이 시작될 때 자동)"""
    holder.refresh(force=True)
    return health()


@app.post("/choose")
def choose(req: ChooseRequest):
    t0 = time.perf_counter()
    sess = sessions.get(req.battle_id)
    if sess is None:
        holder.refresh()
        sess = sessions[req.battle_id] = Session(req.side, holder.model, holder.name)
        sess.battle_id = req.battle_id
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
