"""관전자 시점 사람 대전 로그(Metamon) → Foul Play가 재생할 수 있는 플레이어 시점 기록 한 판(rec).
foul-play/fp/replay_label.py는 서버가 플레이어에게 보낸 프레임(내 쪽 정확한 능력치/기술/PP가 든 |request| JSON, 절대 HP)을 요구하는데,
관전자 로그에는 그게 없다. 상대 쪽 정보는 플레이어가 보는 것과 같으므로 내 쪽만 복원한다:
  - 세트: 판 전체에서 드러난 기술(안 드러난 칸은 PAD 기술로 채움 = FP의 선택지가 학생의 선택지와 같아짐) + 도구/특성/테라/노력치/성격은 드러난 값, 없으면 Smogon 사용률 최빈값
  - HP: 퍼센트 → 추정 최대 HP 기준 절대값 (|switch|/|-damage|/|-heal| 줄도 다시 씀), 상태/기절/도구 소모는 자체 추적
  - |request|: 팀 프리뷰 1번, 매 |turn|N 줄 뒤, 내 포켓몬이 쓰러진 뒤 교체/유턴류로 나가기 직전
사용: convert(log_text, side, user, tag) -> rec dict (tag, user, export, frames, meta) 또는 실패 사유 문자열
알려진 한계(근사): 고정 기술(구애 계열은 도구가 드러난 때만, 충전/재충전만 처리), 갇힘, 상대 압박(Pressure)·내 특성 변화, 사용률 최빈 노력치로 인한 스피드 오차"""
import json
import os
import re

from poke_env.battle import Move
from poke_env.data import GenData

from src.training.replay_dataset import _hindsight, _norm

PAD = "splash"
USAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "foul-play", "fp", "data", "smogon_stats_cache", "gen9ou-0.json")
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
NATURE = {"lonely": ("atk", "def"), "brave": ("atk", "spe"), "adamant": ("atk", "spa"), "naughty": ("atk", "spd"),
          "bold": ("def", "atk"), "relaxed": ("def", "spe"), "impish": ("def", "spa"), "lax": ("def", "spd"),
          "timid": ("spe", "atk"), "hasty": ("spe", "def"), "jolly": ("spe", "spa"), "naive": ("spe", "spd"),
          "modest": ("spa", "atk"), "mild": ("spa", "def"), "quiet": ("spa", "spe"), "rash": ("spa", "spd"),
          "calm": ("spd", "atk"), "gentle": ("spd", "def"), "sassy": ("spd", "spe"), "careful": ("spd", "spa")}
CHOICE = {"choicescarf", "choiceband", "choicespecs"}
_USAGE = None
_DEX = None


def _usage():
    global _USAGE
    if _USAGE is None:
        _USAGE = {_norm(k): v for k, v in json.load(open(USAGE_PATH, encoding="utf-8")).items()}
    return _USAGE


def _dex():
    global _DEX
    if _DEX is None:
        _DEX = GenData.from_gen(9).pokedex
    return _DEX


def _top(d, skip=("other",)):
    d = {k: v for k, v in (d or {}).items() if k not in skip}
    return max(d, key=d.get) if d else None


def calc_stats(base, nature, evs, level):
    up, down = NATURE.get(nature, (None, None))
    out = {}
    for st, ev in zip(STATS, evs):
        core = (2 * base[st] + 31 + ev // 4) * level // 100
        if st == "hp":
            out[st] = 1 if base[st] == 1 else core + level + 10
        else:
            out[st] = int((core + 5) * (1.1 if st == up else 0.9 if st == down else 1.0))
    return out


def estimate(sp, details, hind):
    """드러난 정보 + 사용률 최빈값 → 세트 dict (tera는 id, item은 id, moves는 PAD로 4칸)"""
    dex = _dex()[sp]
    u = _usage().get(sp, {})
    spread = _top(u.get("Spreads"))
    nature, evs = (spread.split(":")[0].lower(), [int(x) for x in spread.split(":")[1].split("/")]) if spread else ("serious", [84] * 6)
    ability = hind.get("ability") or _norm(_top(u.get("Abilities")) or dex["abilities"]["0"])
    item = hind["item"] if hind.get("item") else (_top(u.get("Items")) or "leftovers")
    tera = hind.get("tera") or _top(u.get("Tera Types")) or _norm(dex["types"][0])
    level = int(m.group(1)) if (m := re.search(r"L(\d+)", details)) else 100
    stats = calc_stats(dex["baseStats"], nature, evs, level)
    moves = list(hind.get("moves") or [])[:4]
    return {"species": sp, "details": details, "hind_item": hind.get("item"), "nature": nature, "evs": evs, "ability": ability, "item": "" if item == "nothing" else item,
            "tera": tera, "level": level, "stats": stats, "maxhp": stats.pop("hp"), "nrev": len(moves), "moves": moves + [PAD] * (4 - len(moves))}


class Mon:
    def __init__(self, est):
        self.e, self.hp, self.status, self.item, self.ability = est, est["maxhp"], None, est["item"], est["ability"]
        self.details, self.tera, self.terad, self.last, self.lock, self.recharge = est["details"], est["tera"], False, None, None, False
        self.used = {}

    def pct_to_hp(self, pct):
        return self.e["maxhp"] if pct >= 100 else max(1, round(pct * self.e["maxhp"] / 100)) if pct > 0 else 0

    def set_cond(self, cond):
        """'67/100 par' | '0 fnt' → hp(절대), status"""
        if "fnt" in cond:
            self.hp, self.status = 0, None
            return
        hp, _, rest = cond.partition(" ")
        self.hp = self.pct_to_hp(int(hp.split("/")[0]))
        self.status = rest or None

    def condition(self):
        return "0 fnt" if self.hp <= 0 else f"{self.hp}/{self.e['maxhp']}" + (f" {self.status}" if self.status else "")


def _move_json(mid, used, locked=False):
    try:
        mv = Move(mid, gen=9)
        name, maxpp, target = mv.entry["name"], int(mv.max_pp), mv.entry.get("target", "normal")
    except Exception:
        name, maxpp, target = mid, 8, "normal"
    return {"move": name, "id": mid} if locked else {"move": name, "id": mid, "pp": max(0, maxpp - used), "maxpp": maxpp, "target": target, "disabled": False}


class Team:
    def __init__(self, side, user, previews, hind):
        self.side, self.user, self.rqid, self.used_tera = side, user, 0, False
        self.mons = {}                                  # 미리보기 종 id → Mon
        for details in previews:
            sp = _norm(details.split(",")[0])
            self.mons[sp] = Mon(estimate(sp, details, hind.get(sp, {})))
        self.order = list(self.mons.values())           # PS의 side.pokemon 순서 (맨 앞이 활성)
        self.nick = {}                                  # 'p2: Nick' → Mon
        self.active = None

    def find(self, details):
        sp = _norm(details.split(",")[0])
        if sp in self.mons:
            return self.mons[sp]
        return next((m for k, m in self.mons.items() if k.startswith(sp) or sp.startswith(k)), None)

    def mon_of(self, ident):
        return self.nick.get(ident[:2] + ident[3:])

    def hp_text(self, ident, cond):
        """관전자 HP(퍼센트) → 추정 절대 HP 문자열 (상태 접미사 유지)"""
        mon = self.mon_of(ident)
        if mon is None or "fnt" in cond:
            return cond
        hp, _, rest = cond.partition(" ")
        return f"{mon.pct_to_hp(int(hp.split('/')[0]))}/{mon.e['maxhp']}" + (f" {rest}" if rest else "")

    def update(self, s):
        """내 쪽 줄 하나를 상태에 반영 (s = '|'로 나눈 줄)"""
        ev = s[1]
        if len(s) < 3 or not s[2].startswith(self.side) or len(s[2]) < 3 or s[2][2] not in "abc":
            return
        key = s[2][:2] + s[2][3:]
        if ev in ("switch", "drag"):
            mon = self.find(s[3])
            if mon is None:
                raise KeyError(s[3])
            self.nick[key] = mon
            mon.details, mon.last, mon.lock, mon.recharge = s[3], None, None, False
            mon.set_cond(s[4])
            if self.active is None:                     # 선두: 팀 프리뷰에서 고른 포켓몬이 맨 앞, 나머지는 원래 순서
                self.order.remove(mon)
                self.order.insert(0, mon)
            elif self.active is not mon:                # PS는 활성 자리와 들어오는 자리를 맞바꿈
                ia, ib = self.order.index(self.active), self.order.index(mon)
                self.order[ia], self.order[ib] = self.order[ib], self.order[ia]
            self.active = mon
            return
        mon = self.nick.get(key)
        if mon is None:
            return
        if ev in ("-damage", "-heal"):
            mon.set_cond(s[3])
        elif ev == "-status":
            mon.status = s[3]
        elif ev == "-curestatus":
            mon.status = None
        elif ev == "-cureteam":
            for m in self.mons.values():
                m.status = None
        elif ev == "faint":
            mon.hp, mon.status = 0, None
        elif ev == "-enditem":
            mon.item = ""
        elif ev == "-item":
            mon.item = _norm(s[3])
        elif ev == "-terastallize":
            mon.terad, mon.tera, self.used_tera = True, _norm(s[3]), True
        elif ev == "detailschange":
            mon.details = s[3]
        elif ev == "-prepare":
            mon.lock = _norm(s[3])
        elif ev == "-mustrecharge":
            mon.recharge = True
        elif ev == "move" and not any(x.startswith("[from]") for x in s[4:]):
            mid = _norm(s[3])
            mon.used[mid], mon.last = mon.used.get(mid, 0) + 1, mid
            if mon.lock == mid:
                mon.lock = None
            mon.recharge = False

    def side_json(self):
        mons = []
        for i, m in enumerate(self.order):
            e = m.e
            mons.append({"ident": f"{self.side}: {self._nick(m)}", "details": m.details, "condition": m.condition(), "active": i == 0,
                         "stats": dict(e["stats"]), "moves": list(e["moves"]), "baseAbility": e["ability"], "item": m.item,
                         "pokeball": "pokeball", "ability": m.ability, "commanding": False, "reviving": False,
                         "teraType": e["tera"].capitalize() if not m.terad else m.tera.capitalize(), "terastallized": m.tera.capitalize() if m.terad else ""})
        return {"name": self.user, "id": self.side, "pokemon": mons}

    def _nick(self, m):
        return next((k[len(self.side) + 2:] for k, v in self.nick.items() if v is m), m.e["details"].split(",")[0])

    def request(self, kind):
        self.rqid += 1
        side = self.side_json()
        if kind == "tp":
            return {"teamPreview": True, "side": side, "rqid": self.rqid}
        if kind == "forced":
            return {"forceSwitch": [True], "side": side, "noCancel": True, "rqid": self.rqid}
        m = self.active
        if m.recharge:
            act = {"moves": [_move_json("recharge", 0, True)], "trapped": True}
        elif m.lock:
            act = {"moves": [_move_json(m.lock, 0, True)], "trapped": True}
        else:
            moves = [_move_json(mid, m.used.get(mid, 0)) for mid in m.e["moves"]]
            if m.item in CHOICE and m.item == m.e["hind_item"] and m.last:      # 구애 계열은 도구가 드러났을 때만 고정으로 취급
                for mv in moves:
                    mv["disabled"] = mv["id"] != m.last
            act = {"moves": moves}
            if not self.used_tera and not m.terad:
                act["canTerastallize"] = m.e["tera"].capitalize()
        return {"active": [act], "side": side, "rqid": self.rqid}


def export_text(team):
    out = []
    for k in team.mons:
        e = team.mons[k].e
        evs = " / ".join(f"{v} {n.capitalize()}" for v, n in zip(e["evs"], ("HP", "Atk", "Def", "SpA", "SpD", "Spe")))
        out.append("\n".join([f"{e['details'].split(',')[0]} @ {e['item']}" if e["item"] else e["details"].split(",")[0], f"Ability: {e['ability']}",
                              f"Tera Type: {e['tera'].capitalize()}", f"EVs: {evs}", f"{e['nature'].capitalize()} Nature"] + [f"- {m}" for m in e["moves"]]))
    return "\n\n".join(out)


def _no_choice(lines, side):
    """한 번 등장해서 서로 다른 기술을 둘 이상 쓴 포켓몬 = 구애 계열이 아니라는 증거"""
    cur, used, out = None, set(), set()
    for s in lines:
        if len(s) > 3 and s[2].startswith(side) and len(s[2]) > 2 and s[2][2] in "abc":
            if s[1] in ("switch", "drag"):
                cur, used = _norm(s[3].split(",")[0]), set()
            elif s[1] == "move" and cur and not any(x.startswith("[from]") for x in s[4:]):
                used.add(_norm(s[3]))
                if len(used) >= 2:
                    out.add(cur)
    return out


def convert(log_text, side, user, tag, oracle=None, world=None):
    """관전자 로그 텍스트 → rec (실패하면 사유 문자열). oracle = {종 id: 세트 필드 덮어쓰기}: 충실도 측정에서 추정 오차의 기여를 가를 때만 씀.
    world = (종 id, 드러난 정보, 추정 세트) -> 덮어쓰기 dict: 세트 사후분포에서 뽑은 월드 하나 (tools/set_posterior.make_world)"""
    lines = [s for s in (l.split("|") for l in log_text.split("\n") if l.startswith("|") and l != "|") if len(s) > 1]
    names = {s[2]: s[3] for s in lines if s[1] == "player" and len(s) > 3}
    if side not in names or "p1" not in names or "p2" not in names:
        return "no_player"
    previews = [s[3] for s in lines if s[1] == "poke" and len(s) > 3 and s[2] == side]
    if not previews:
        return "no_preview"
    hind = _hindsight(lines, side)
    for sp in _no_choice(lines, side):
        hind[sp]["no_choice"] = True
    try:
        team = Team(side, user, previews, hind)
    except KeyError as ex:
        return f"unknown_species:{ex}"
    revealed = sorted({m for k in team.mons for m in team.mons[k].e["moves"][:team.mons[k].e["nrev"]]})
    if world is not None:
        oracle = {k: world(k, hind.get(k, {}), team.mons[k].e) for k in team.mons}
    for k, o in (oracle or {}).items():
        if k in team.mons:
            m = team.mons[k]
            sampled = o.pop("sampled", False)
            m.e.update(o)
            m.hp, m.item, m.ability = m.e["maxhp"], m.e["item"], m.e["ability"]
            if "item" in o and not sampled:             # 뽑은 도구는 가정이라 구애 고정에는 쓰지 않음 (드러난 도구만 고정으로 취급)
                m.e["hind_item"] = o["item"]
    t_i = next((i for i, s in enumerate(lines) if s[1] == "teampreview"), None)
    if t_i is None:
        return "no_teampreview"
    head = lines[:t_i + 1]
    j = lambda parts: "|".join(parts)
    pl = [s for s in head if s[1] == "player"]
    rest = [s for s in head if s[1] != "player" and s[1] != "gametype"]
    frames = [f">{tag}\n|init|battle\n|title|{names['p1']} vs. {names['p2']}\n", f">{tag}\n|t:|0\n|gametype|singles", f">{tag}\n{j(pl[0])}",
              f">{tag}\n" + "\n".join([j(pl[1])] + [j(s) for s in rest])]
    own = lambda s: len(s) > 2 and s[2].startswith(side) and len(s[2]) > 2 and s[2][2] in "abc"
    pivot = lambda s: s[1] == "switch" and any(x.startswith("[from]") for x in s[5:])
    frames.append(f">{tag}\n|request|{json.dumps(team.request('tp'), separators=(',', ':'))}")
    chunk, faint = [], False                            # faint: 내 포켓몬이 쓰러져 교체 요청을 기다리는 중 (이젝트 버튼/팩이 터진 뒤에도 같은 처리)
    body = lines[t_i + 1:]

    def flush():
        if chunk:
            frames.append(f">{tag}\n" + "\n".join(chunk))
            chunk.clear()

    def ask(kind):
        flush()
        frames.append(f">{tag}\n|request|{json.dumps(team.request(kind), separators=(',', ':'))}")

    def more_switch(i):
        for t in body[i + 1:]:
            if t[1] in ("turn", "win", "tie"):
                return False
            if t[1] == "switch" and own(t):
                return True
        return False

    for i, s in enumerate(body):
        if s[1] in ("win", "tie"):
            break
        if own(s) and s[1] == "switch" and team.active is not None and (pivot(s) or faint):
            ask("forced")
            faint = False
        out = list(s)
        if own(s):
            try:
                if s[1] in ("switch", "drag"):
                    team.update(s)
                    out[4] = team.hp_text(s[2], s[4])
                elif s[1] in ("-damage", "-heal") and len(s) > 3:
                    out[3] = team.hp_text(s[2], s[3])
                    team.update(s)
                else:
                    team.update(s)
            except (KeyError, ValueError) as ex:
                return f"track_fail:{ex!r}"
            if s[1] == "faint" or (s[1] == "-enditem" and _norm(s[3]) in ("ejectbutton", "ejectpack")):
                faint = True
        chunk.append(j(out))
        if s[1] == "upkeep" and faint and more_switch(i):
            ask("forced")
            faint = False
        elif s[1] == "turn":
            ask("turn")
    flush()
    meta = {"revealed": revealed, "pad": PAD, "est": {k: {"nrev": m.e["nrev"], "moves": m.e["moves"], "item": m.e["item"], "ability": m.e["ability"], "tera": m.e["tera"],
                                                           "nature": m.e["nature"], "evs": m.e["evs"], "stats": m.e["stats"], "maxhp": m.e["maxhp"]} for k, m in team.mons.items()}}
    return {"tag": tag, "user": user, "export": export_text(team), "frames": frames, "meta": meta}
