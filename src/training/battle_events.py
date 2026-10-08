"""Foul Play 대전 프로토콜 → 행동 사건 목록 (결과 예측기 학습과 손계산 오차 진단 tools/damage_residual_report.py의 입력).
표준 라이브러리만 씀 — poke-env/torch 없이 대량 처리.

사건 (dict, JSON으로 그대로 저장)
  move   기술 한 번. 직전 상태 u(사용자, 쪽 us) / t(대상, 쪽 ts) / field / sc(양쪽 장) / fnt(양쪽 기절 수)와 그 기술의 결과 out:
         dmg(대상 HP 비율 감소 합), hp_after, ko, hits, crit, eff("super"/"resisted"), immune, miss, fail, protect, sub(대타가 받음),
         user_dmg / user_heal(반동, 생명의구슬, 울퉁불퉁멧, 흡수 등), boosts{u,t}(랭크 변화), status{u,t}, side_start[[쪽, 장]],
         weather / terrain / fieldstart, reflected(매직미러 등으로 되돌아감), abilities / items([[쪽, 이름]] 발동), items_end, starts / ends
         via: 다른 효과로 나간 기술(예: "ability:magicbounce"로 되돌린 기술). 자기 선택이 아니므로 진단에서 뺌. ord: 이번 턴 몇 번째 기술인가
  switch 등장 한 번(교체/강제 교체). 직전 상태 mon(들어온 포켓몬, 장판 피해 전) / opp(상대 활성) / field / sc / fnt와 결과 out:
         hazard(장판 피해 비율), status(독압정 등), opp_boosts(상대 랭크 변화: 위협 등), self_boosts, weather / terrain, abilities, items
상태 표기: sp(종, 정규화), hp(비율), b(0이 아닌 랭크), st(상태이상), tera(테라 중이면 타입), item, ab, sub(대타 유지 중)

규칙
  - 결과는 현재 열린 사건(cur)에 붙임. 빈 줄 "|"(턴 시작/잔여 효과 구간), |turn|, |upkeep|, |cant|에서 닫힘 → 먹다남은음식/독 같은 잔여 효과는 안 붙음
  - 등장 효과는 줄 순서가 아니라 주체로 찾아 붙임: 장판 피해는 그 포켓몬의 이번 턴 등장 사건으로, 등장 특성(위협/가뭄 등)의 "-ability"와
    "[from] ability: .. [of] .."가 붙은 날씨/필드는 그 포켓몬의 등장 사건으로 되돌아가 이어지는 줄(상대 랭크 하락 등)까지 받음
    (동시 교체에서 특성이 두 번째 등장 줄 뒤에 나오는 경우)
  - "|split|"은 다음 두 줄 중 첫 줄(정확한 HP)만 씀
  - 세트(노력치/성격/도구/특성)는 start 레코드의 팀에서. 모르는 쪽(외부 상대)은 공개된 도구/특성만 채움
"""
import re

HAZARDS = {"stealthrock", "spikes"}
PROTECT = {"protect", "detect", "kingsshield", "spikyshield", "banefulbunker", "silktrap", "burningbulwark", "obstruct", "maxguard"}
# 등장할 때 발동하는 특성: 이 특성의 "-ability" 줄은 그 포켓몬의 이번 턴 등장 사건으로 붙임
ENTRY_ABILITIES = {
    "intimidate", "drought", "drizzle", "sandstream", "snowwarning", "electricsurge", "psychicsurge", "grassysurge", "mistysurge",
    "orichalcumpulse", "hadronengine", "download", "trace", "frisk", "anticipation", "forewarn", "pressure", "moldbreaker",
    "teravolt", "turboblaze", "unnerve", "airlock", "cloudnine", "neutralizinggas", "intrepidsword", "dauntlessshield",
    "supremeoverlord", "costar", "zerotohero", "protosynthesis", "quarkdrive", "screencleaner", "curiousmedicine", "hospitality",
    "asoneglastrier", "asonespectrier", "embodyaspectteal", "embodyaspecthearthflame", "embodyaspectwellspring",
    "embodyaspectcornerstone", "terashift", "terashell", "teraformzero",
}
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x or "").lower())


def _ident(s):
    """"p1a: 이름" → ("p1", "이름"). 쪽 표기("p1: 사용자")도 받음"""
    m = re.match(r"\s*(p[1-4])[a-z]?\s*:\s*(.*)$", s or "")
    return (m.group(1), m.group(2).strip()) if m else (None, None)


def _side_of(s):
    m = re.match(r"\s*(p[1-4])", s or "")
    return m.group(1) if m else None


def _other(side):
    return "p2" if side == "p1" else "p1"


def _hp(s):
    """"254/357 par" → (0.711, "par"), "0 fnt" → (0.0, "fnt")"""
    parts = (s or "").split()
    if not parts:
        return None, None
    frac = None
    if "/" in parts[0]:
        a, b = parts[0].split("/", 1)
        try:
            frac = float(a) / float(b) if float(b) > 0 else 0.0
        except ValueError:
            frac = None
    elif parts[0] == "0":
        frac = 0.0
    return frac, (parts[1] if len(parts) > 1 else None)


def _tag(args, key="[from]"):
    """"[from] item: Life Orb" → ("item", "lifeorb"), "[from] Stealth Rock" → ("", "stealthrock"), 없으면 None"""
    for a in args:
        if a.startswith(key):
            v = a[len(key):].strip()
            if ":" in v:
                kind, name = v.split(":", 1)
                return norm(kind), norm(name)
            return "", norm(v)
    return None


def _effname(s):
    """"move: Stealth Rock" / "ability: Sturdy" / "Substitute" → 정규화한 이름"""
    s = s or ""
    if ":" in s and s.split(":", 1)[0].strip().lower() in ("move", "ability", "item"):
        s = s.split(":", 1)[1]
    return norm(s)


def norm_set(m):
    """start 레코드의 팀원 dict(Foul Play 형식) → 정규화한 세트"""
    def six(v, default):
        if isinstance(v, dict):
            return [int(v.get(k, default) or default) for k in STAT_KEYS]
        if isinstance(v, (list, tuple)) and len(v) == 6:
            return [int(x) for x in v]
        return [default] * 6
    return {"sp": norm(m.get("species")), "item": norm(m.get("item")) or None, "ab": norm(m.get("ability")) or None,
            "tera": norm(m.get("tera_type")) or None, "nature": norm(m.get("nature")) or None,
            "evs": six(m.get("evs"), 0), "ivs": six(m.get("ivs"), 31), "level": int(m.get("level") or 100),
            "moves": [norm(x) for x in m.get("moves", [])]}


def find_set(sets, sp):
    """세트 목록에서 종으로 찾기 (폼 차이는 접두사로: 'palafinhero' ↔ 'palafin')"""
    for s in sets or []:
        if s["sp"] == sp:
            return s
    return next((s for s in sets or [] if s["sp"] and (sp.startswith(s["sp"]) or s["sp"].startswith(sp))), None)


class BattleParser:
    def __init__(self, teams_by_user=None):
        self.teams_by_user = teams_by_user or {}
        self.users, self.sets = {}, {}                      # 쪽 -> 사용자 이름 / 정규화한 세트 목록
        self.mons = {"p1": {}, "p2": {}}                     # 쪽 -> 종 -> 상태
        self.names = {"p1": {}, "p2": {}}                    # 쪽 -> 표시 이름 -> 종
        self.active = {"p1": None, "p2": None}
        self.boosts = {"p1": {}, "p2": {}}
        self.vol = {"p1": set(), "p2": set()}
        self.sc = {"p1": {}, "p2": {}}
        self.fnt = {"p1": 0, "p2": 0}
        self.weather = self.terrain = None
        self.trickroom = False
        self.turn = self.order = 0
        self.events, self.cur = [], None
        self.entry = {"p1": None, "p2": None}                # 쪽 -> 이번 턴 등장 사건
        self._split = 0

    # ---- 상태 ----
    def _mon(self, side, key):
        mons = self.mons[side]
        if key not in mons:
            s = find_set(self.sets.get(side), key) or {}
            mons[key] = {"sp": key, "hp": 1.0, "st": None, "item": s.get("item"), "ab": s.get("ab"), "tera": None}
        return mons[key]

    def _key(self, side, name):
        return self.names[side].get(name) or self.active[side] or norm(name)

    def _snap(self, side):
        key = self.active.get(side) if side else None
        if key is None:
            return None
        m = self.mons[side][key]
        return {"sp": key, "hp": round(m["hp"], 4), "b": {k: v for k, v in self.boosts[side].items() if v}, "st": m["st"],
                "tera": m["tera"], "item": m["item"], "ab": m["ab"], "sub": "substitute" in self.vol[side]}

    def _ctx(self):
        return {"field": {"weather": self.weather, "terrain": self.terrain, "trickroom": self.trickroom},
                "sc": {s: dict(c) for s, c in self.sc.items()}, "fnt": dict(self.fnt)}

    @staticmethod
    def _role(ev, side, key):
        if ev["u"] and side == ev["us"] and key == ev["u"]["sp"]:
            return "u"
        if ev["t"] and side == ev["ts"] and key == ev["t"]["sp"]:
            return "t"
        return None

    @staticmethod
    def _note(ev, field, val):
        if ev is not None:
            lst = ev["out"].setdefault(field, [])
            if val not in lst:
                lst.append(val)

    def _add(self, ev, field, val):
        ev["out"][field] = round(ev["out"].get(field, 0.0) + val, 4)

    def _merge_boosts(self, dst, delta):
        for k, v in delta.items():
            dst[k] = dst.get(k, 0) + v

    def _entry_of(self, side, key):
        ev = self.entry.get(side) if side else None
        return ev if ev is not None and ev["mon"]["sp"] == key else None

    # ---- 줄 처리 ----
    def feed(self, line):
        if self._split:
            self._split -= 1
            if self._split == 0:        # split 뒤 두 번째 줄 = 공개용 중복
                return
        if not line.startswith("|"):
            return
        parts = line.rstrip("\r\n").split("|")[1:]
        cmd, args = parts[0], parts[1:]
        if cmd == "split":
            self._split = 2
            return
        handler = self._HANDLERS.get(cmd)
        if handler is not None:
            handler(self, cmd, args)
        self._reveal(args)

    def _blank(self, cmd, args):
        self.cur = None

    def _player(self, cmd, args):
        if len(args) >= 2 and args[1]:
            self.users[args[0]] = args[1]
            team = self.teams_by_user.get(args[1])
            if team:
                self.sets[args[0]] = [norm_set(m) for m in team]

    def _turn(self, cmd, args):
        self.turn = int(args[0]) if args and args[0].isdigit() else self.turn + 1
        self.order, self.cur = 0, None
        self.entry = {"p1": None, "p2": None}

    def _switch(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None:
            return
        key = norm(args[1].split(",")[0]) if len(args) > 1 else norm(name)
        self.names[side][name] = key
        self.active[side] = key
        m = self._mon(side, key)
        hp, st = _hp(args[2]) if len(args) > 2 else (None, None)
        if hp is not None:
            m["hp"] = hp
            m["st"] = st
        if cmd == "replace":            # 일루전 정체 공개: 이름→종만 고침 (랭크/대타는 그대로)
            return
        self.boosts[side], self.vol[side] = {}, set()
        ev = {"type": "switch", "turn": self.turn, "side": side, "mon": self._snap(side), "opp": self._snap(_other(side)),
              **self._ctx(), "out": {}}
        self.events.append(ev)
        self.cur = self.entry[side] = ev

    def _forme(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None or len(args) < 2:
            return
        old_key, key = self._key(side, name), norm(args[1].split(",")[0])
        if key == old_key:
            return
        self.mons[side][key] = dict(self._mon(side, old_key), sp=key)      # HP/상태/도구/특성/테라는 그대로
        self.names[side][name] = key
        if self.active[side] == old_key:
            self.active[side] = key

    def _move(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None or len(args) < 2:
            return
        via = _tag(args[2:])
        tgt = args[2] if len(args) > 2 and args[2] and not args[2].startswith("[") else None
        tside = None if "[notarget]" in args else (_side_of(tgt) if tgt else _other(side))
        if via and self.cur is not None and self.cur["type"] == "move" and via[1] in ("magicbounce", "magiccoat"):
            self.cur["out"]["reflected"] = True
        ev = {"type": "move", "turn": self.turn, "ord": self.order, "move": norm(args[1]),
              "via": ":".join(x for x in via if x) if via else None,
              "us": side, "u": self._snap(side), "ts": tside, "t": self._snap(tside) if tside else None, **self._ctx(), "out": {}}
        if not via:
            self.order += 1
        self.events.append(ev)
        self.cur = ev

    def _hp_change(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None or len(args) < 2:
            return
        key = self._key(side, name)
        m = self._mon(side, key)
        hp, st = _hp(args[1])
        if hp is None:
            return
        prev, m["hp"] = m["hp"], hp
        if st:
            m["st"] = st
        if cmd == "-sethp":
            return
        src = _tag(args[2:])
        delta = prev - hp
        if cmd == "-damage" and src and src[1] in HAZARDS:
            ev = self._entry_of(side, key)
            if ev is not None:
                self._add(ev, "hazard", delta)
            return
        ev = self.cur
        if ev is None or ev["type"] != "move":
            return
        role = self._role(ev, side, key)
        if role == "t" and cmd == "-damage" and src is None:
            self._add(ev, "dmg", delta)
            ev["out"]["hp_after"] = round(hp, 4)
        elif role == "u" and delta:     # 반동/생명의구슬/울퉁불퉁멧/흡수 회복 등 사용자 HP 변화
            self._add(ev, "user_dmg" if delta > 0 else "user_heal", abs(delta))

    def _boost(self, cmd, args):
        if cmd == "-clearallboost":
            for side in ("p1", "p2"):
                if self.active[side]:
                    self._apply_boosts(side, self.active[side], lambda b: {})
            return
        side, name = _ident(args[0] if args else "")
        if side is None:
            return
        key = self._key(side, name)
        if cmd in ("-boost", "-unboost") and len(args) >= 3:
            n = int(args[2]) if args[2].lstrip("-").isdigit() else 0
            n = n if cmd == "-boost" else -n
            self._apply_boosts(side, key, lambda b: {**b, args[1]: max(-6, min(6, b.get(args[1], 0) + n))})
        elif cmd == "-setboost" and len(args) >= 3 and args[2].lstrip("-").isdigit():
            self._apply_boosts(side, key, lambda b: {**b, args[1]: int(args[2])})
        elif cmd == "-clearboost":
            self._apply_boosts(side, key, lambda b: {})
        elif cmd == "-clearnegativeboost":
            self._apply_boosts(side, key, lambda b: {k: v for k, v in b.items() if v > 0})
        elif cmd == "-clearpositiveboost":
            self._apply_boosts(side, key, lambda b: {k: v for k, v in b.items() if v < 0})
        elif cmd == "-invertboost":
            self._apply_boosts(side, key, lambda b: {k: -v for k, v in b.items()})
        elif cmd == "-copyboost" and len(args) >= 2:      # SOURCE가 TARGET의 랭크를 복사
            tside, _ = _ident(args[1])
            if tside:
                src = dict(self.boosts[tside])
                self._apply_boosts(side, key, lambda b: src)

    def _apply_boosts(self, side, key, fn):
        old = self.boosts[side]
        new = {k: v for k, v in fn(dict(old)).items() if v}
        self.boosts[side] = new
        delta = {k: new.get(k, 0) - old.get(k, 0) for k in set(old) | set(new) if new.get(k, 0) != old.get(k, 0)}
        ev = self.cur
        if not delta or ev is None:
            return
        if ev["type"] == "move":
            role = self._role(ev, side, key)
            if role:
                self._merge_boosts(ev["out"].setdefault("boosts", {}).setdefault(role, {}), delta)
        else:
            self._merge_boosts(ev["out"].setdefault("self_boosts" if side == ev["side"] else "opp_boosts", {}), delta)

    def _status(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None:
            return
        key = self._key(side, name)
        m = self._mon(side, key)
        if cmd == "-curestatus":
            m["st"] = None
            return
        st = norm(args[1]) if len(args) > 1 else None
        m["st"] = st
        ev = self.cur
        if ev is None:
            return
        if ev["type"] == "move":
            role = self._role(ev, side, key)
            if role:
                ev["out"].setdefault("status", {})[role] = st
        elif side == ev["side"] and key == ev["mon"]["sp"]:
            ev["out"]["status"] = st

    def _faint(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None:
            return
        key = self._key(side, name)
        m = self._mon(side, key)
        m["hp"], m["st"] = 0.0, "fnt"
        self.fnt[side] += 1
        ev = self.cur
        if ev is not None and ev["type"] == "move":
            role = self._role(ev, side, key)
            if role == "t":
                ev["out"]["ko"] = True
            elif role == "u":
                ev["out"]["user_fainted"] = True

    def _effect(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        eff = _effname(args[1]) if len(args) > 1 else ""
        if side and cmd == "-start" and eff == "substitute":
            self.vol[side].add("substitute")
        elif side and cmd == "-end" and eff == "substitute":
            self.vol[side].discard("substitute")
        ev = self.cur
        if ev is None or ev["type"] != "move":
            return
        out = ev["out"]
        if cmd == "-crit":
            out["crit"] = True
        elif cmd in ("-supereffective", "-resisted"):
            out["eff"] = "super" if cmd == "-supereffective" else "resisted"
        elif cmd == "-immune":
            out["immune"] = True
        elif cmd == "-miss":
            out["miss"] = True
        elif cmd == "-fail":
            out["fail"] = True
        elif cmd == "-hitcount" and len(args) > 1 and args[1].isdigit():
            out["hits"] = int(args[1])
        elif cmd == "-activate":
            if eff in PROTECT:
                out["protect"] = True
            elif eff == "substitute" and "[damage]" in args:
                out["sub"] = True
            elif eff:
                self._note(ev, "activates", eff)
        elif cmd in ("-start", "-end") and side:
            self._note(ev, "starts" if cmd == "-start" else "ends", [self._role(ev, side, self._key(side, name)), eff])

    def _ability(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None or len(args) < 2:
            return
        key, ab = self._key(side, name), norm(args[1])
        m = self._mon(side, key)
        m["ab"] = m["ab"] or ab
        if ab in ENTRY_ABILITIES and self._entry_of(side, key) is not None:
            self.cur = self.entry[side]
        self._note(self.cur, "abilities", [side, ab])

    def _field(self, cmd, args):
        src, owner = _tag(args), next((a[4:].strip() for a in args if a.startswith("[of]")), None)
        if src and src[0] == "ability" and owner and src[1] in ENTRY_ABILITIES:
            oside, oname = _ident(owner)
            if oside and self._entry_of(oside, self._key(oside, oname)) is not None:
                self.cur = self.entry[oside]
        if cmd == "-weather":
            w = norm(args[0]) if args else ""
            self.weather = None if w in ("", "none") else w
            if "[upkeep]" not in args and self.cur is not None:
                self.cur["out"]["weather"] = w
        elif cmd == "-fieldstart":
            c = _effname(args[0]) if args else ""
            if c == "trickroom":
                self.trickroom = True
            elif c.endswith("terrain"):
                self.terrain = c
            if self.cur is not None:
                self.cur["out"]["terrain" if c.endswith("terrain") else "fieldstart"] = c
        elif cmd == "-fieldend":
            c = _effname(args[0]) if args else ""
            if c == "trickroom":
                self.trickroom = False
            elif c == self.terrain:
                self.terrain = None

    def _side(self, cmd, args):
        if cmd == "-swapsideconditions":
            self.sc["p1"], self.sc["p2"] = self.sc["p2"], self.sc["p1"]
            return
        side = _side_of(args[0] if args else "")
        if side is None or len(args) < 2:
            return
        c = _effname(args[1])
        if cmd == "-sidestart":
            self.sc[side][c] = self.sc[side].get(c, 0) + 1
            self._note(self.cur, "side_start", [side, c])
        else:
            self.sc[side].pop(c, None)

    def _item(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side is None or len(args) < 2:
            return
        m, item = self._mon(side, self._key(side, name)), norm(args[1])
        src = _tag(args[2:])
        if cmd == "-enditem":
            m["item"] = None
            self._note(self.cur, "items_end", [side, item])
        elif "[identify]" in args or (src and src[1] == "frisk"):     # 공개만 (도구가 바뀐 게 아님)
            m["item"] = m["item"] or item
        else:                                                         # 트릭/바꿔치기로 바뀌었거나 풍선 공개
            m["item"] = item
            self._note(self.cur, "items", [side, item])

    def _tera(self, cmd, args):
        side, name = _ident(args[0] if args else "")
        if side and len(args) > 1:
            self._mon(side, self._key(side, name))["tera"] = norm(args[1])

    def _reveal(self, args):
        """[from] ability/item 태그: 주인([of], 없으면 이 줄의 대상)의 특성/도구를 공개하고 열린 사건에 발동으로 기록"""
        src = _tag(args)
        if not src or src[0] not in ("ability", "item"):
            return
        owner = next((a[4:].strip() for a in args if a.startswith("[of]")), None) or (args[0] if args else None)
        side, name = _ident(owner)
        if side is None:
            return
        m = self._mon(side, self._key(side, name))
        if src[0] == "ability":
            m["ab"] = m["ab"] or src[1]
        else:
            m["item"] = m["item"] or src[1]
        self._note(self.cur, "abilities" if src[0] == "ability" else "items", [side, src[1]])

    _HANDLERS = {
        "": _blank, "upkeep": _blank, "cant": _blank, "win": _blank, "tie": _blank,
        "player": _player, "turn": _turn,
        "switch": _switch, "drag": _switch, "replace": _switch, "detailschange": _forme, "-formechange": _forme,
        "move": _move, "-damage": _hp_change, "-heal": _hp_change, "-sethp": _hp_change,
        "-boost": _boost, "-unboost": _boost, "-setboost": _boost, "-clearboost": _boost, "-clearallboost": _boost,
        "-clearnegativeboost": _boost, "-clearpositiveboost": _boost, "-invertboost": _boost, "-copyboost": _boost,
        "-status": _status, "-curestatus": _status, "faint": _faint,
        "-crit": _effect, "-supereffective": _effect, "-resisted": _effect, "-immune": _effect, "-miss": _effect, "-fail": _effect,
        "-hitcount": _effect, "-activate": _effect, "-start": _effect, "-end": _effect,
        "-ability": _ability, "-weather": _field, "-fieldstart": _field, "-fieldend": _field,
        "-sidestart": _side, "-sideend": _side, "-swapsideconditions": _side,
        "-item": _item, "-enditem": _item, "-terastallize": _tera,
    }


def parse_battle(lines, teams_by_user=None):
    """프로토콜 줄 목록 + {사용자 이름: start 레코드의 팀} → (배틀 머리 정보, 사건 목록)"""
    p = BattleParser(teams_by_user)
    for line in lines:
        p.feed(line)
    header = {"users": p.users, "sets": p.sets, "known": {s: bool(p.sets.get(s)) for s in p.users}}
    return header, p.events
