"""도구/특성 공개와 추론 + 행동 기록(마지막 사용 기술, 등장 후 쓴 기술, 소원, 미래예지) 추적.
Foul Play가 결정 시점에 아는 정보(fp/battle/protocol.py의 공개 규칙, fp/battle/inference.py의 추론 규칙)를 poke-env 경로로 옮긴 것이라
학습 데이터 재생, 평가, 실전이 같은 경로(events.py가 parse_message를 감싸서 호출)로 같은 값을 만든다. 규칙마다 Foul Play와 같은 조건을 쓰고, 어긋나면 tools/state_audit.py가 잡는다.

공개: |-ability|, |-item|, |-enditem|, "[from] ability: X" / "ability: X" / "[from] item: X" / "item: X" 가 붙은 모든 줄 ([of] 가 있으면 그쪽이 주인, 없으면 줄의 첫 대상)
추론 (상대만): 돌격 부츠 (장판이 있는데 교체 후 피해/상태/끈끈이를 안 받음), 구애 스카프 (같은 우선도에서 최대 속도 가정으로도 못 앞서는데 먼저 움직임),
  날씨 바위 (날씨가 5번째 upkeep 이후에도 안 끝남), 구애 도구는 서로 다른 기술을 쓰면 취소 (can_choice)
행동 기록: 양쪽의 마지막 행동(기술/교체), 등장 후 쓴 기술 집합과 등장한 턴, 소원 / 미래예지 남은 턴
알려지지 않은 것은 None. 종 이름(_n(species))을 키로 씀.
ponytail: 구애 스카프 추론은 두 번 움직인 턴만 (한 번만 움직인 턴은 Foul Play가 내가 고른 기술을 알아야 해서 생략), 기술/특성이 복사/교환되는 경우(트레이스 등)는 마지막 공개가 덮어씀,
          데미지 계산 기반의 세트 거르기(update_dataset_possibilities)는 도구/특성을 직접 정하지 않아 옮기지 않음"""
import re

from poke_env.battle import Field, Move, MoveCategory, SideCondition, Status, Weather

ROCK = {"sun": "heatrock", "rain": "damprock", "sand": "smoothrock", "hail": "icyrock", "snow": "icyrock"}
CHOICE = {"choiceband", "choicespecs", "choicescarf"}
_ABIL = re.compile(r"^(?:\[from\]\s*)?ability:\s*(.+)$")
_ITEM = re.compile(r"^(?:\[from\]\s*)?item:\s*(.+)$")
_IDENT = re.compile(r"^p[1-4][a-d]?:")
_BOOTS_END = {"move", "upkeep", "turn", "", "cant", "switch", "drag"}
_POISON_IMMUNE = {"immunity", "pastelveil"}


def _n(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _boost(b):
    return (2 + b) / 2 if b >= 0 else 2 / (2 - b)


def _abilities(mon):
    """종이 가질 수 있는 특성 이름 집합 (정규화)"""
    pa = getattr(mon, "possible_abilities", None) or {}
    return {_n(a) for a in (pa.values() if isinstance(pa, dict) else pa)}


class RevealTracker:
    def __init__(self):
        self.item = {"me": {}, "opp": {}}               # 종 -> 지금 들고 있다고 아는 도구 (없다고 확인되면 None)
        self.inferred = {"me": {}, "opp": {}}           # 종 -> 추론으로 알아낸 것인가
        self.ability = {"me": {}, "opp": {}}
        self.impossible_boots = set()                    # 종 (상대): 부츠가 아님이 확인됨
        self.can_choice = {}                             # 종 (상대) -> 구애 도구를 가질 수 있는가
        self.last_action = {"me": None, "opp": None}     # ("move", 기술) | ("switch", 종)
        self.last_move_sp = {"me": None, "opp": None}    # 마지막으로 기술을 쓴 종과 기술
        self.used = {"me": {}, "opp": {}}                # 종 -> 등장 후 쓴 기술 집합
        self.entry = {"me": {}, "opp": {}}               # 종 -> 등장한 턴
        self.wish = {"me": 0, "opp": 0}
        self.future = {"me": 0, "opp": 0}
        self.weather = None                              # {"name", "upkeeps", "side", "sp"}
        self.last_move_ident = None
        self.pending_boots = None
        self.chunk = []                                  # 직전 |turn| 이후 줄들
        self.turn = 0
        self.hp_seen = {"me": {}, "opp": {}}             # 종 -> 마지막으로 본 HP 비율 (교체 후 회복으로 재생력 추론)
        self.synth = {"me": set(), "opp": set()}         # 활성 포켓몬의 합성 휘발성 효과 (Foul Play가 직접 붙이는 lockedmove, unburden, truant, 충전 준비 등)
        self.prep = {"me": None, "opp": None}            # 충전 중인 기술 이름 (-prepare)
        self.age = {"me": {}, "opp": {}}                 # 종 -> {"status", "age"} 수면/맹독이 시작된 뒤 지난 턴

    # ---- 조회
    def known_item(self, side, sp):
        return self.item[side].get(sp)

    def known_ability(self, side, sp):
        return self.ability[side].get(sp)

    def status_age(self, side, sp, status):
        a = self.age[side].get(sp)
        return a["age"] if a and a["status"] == status else None

    # ---- 보조
    @staticmethod
    def _side(battle, ident):
        if not ident or not _IDENT.match(ident):
            return None
        return "me" if ident[:2] == battle.player_role else "opp"

    @staticmethod
    def _mon(battle, side):
        return battle.active_pokemon if side == "me" else battle.opponent_active_pokemon

    def _owner(self, battle, ident):
        side = self._side(battle, ident)
        mon = self._mon(battle, side) if side else None
        return (side, _n(mon.species)) if mon is not None else (None, None)

    def _set_ability(self, battle, ident, name):
        side, sp = self._owner(battle, ident)
        if side:
            self.ability[side][sp] = _n(name)

    def _set_item(self, battle, ident, name, inferred=False):
        side, sp = self._owner(battle, ident)
        if side:
            self.item[side][sp] = _n(name) if name else None
            self.inferred[side][sp] = inferred

    # ---- 메시지 처리 (poke-env의 parse_message가 끝난 뒤 호출)
    def on_message(self, battle, parts):
        kind = parts[1] if len(parts) > 1 else ""
        if self.pending_boots is not None:
            if kind in _BOOTS_END:
                self._resolve_boots(battle)
            else:
                self.pending_boots["lines"].append(parts)
        if kind == "turn":
            self._check_scarf(battle, self.chunk)
            self.chunk = []
            self.turn = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else self.turn + 1
            for s in ("me", "opp"):
                mon = self._mon(battle, s)
                a = self.age[s].get(_n(mon.species)) if mon is not None else None
                if a and a["status"] in ("slp", "tox"):
                    a["age"] += 1
        else:
            self.chunk.append("|".join(parts))
        self._reveal(battle, parts, kind)
        if kind in ("switch", "drag"):
            self._switch(battle, parts)
        elif kind == "move":
            self._move(battle, parts)
        elif kind == "-weather":
            self._weather(battle, parts)
        elif kind == "upkeep":
            for s in ("me", "opp"):
                self.wish[s] = max(0, self.wish[s] - 1)
                self.future[s] = max(0, self.future[s] - 1)
        elif kind == "-start" and len(parts) > 3 and "Future Sight" in parts[3]:
            s = self._side(battle, parts[2])
            if s:
                self.future[s] = 3
        elif kind == "-end" and len(parts) > 3 and ("Future Sight" in parts[3] or "Doom Desire" in parts[3]):
            s = self._side(battle, parts[2])
            if s:
                self.future[s] = 0
        elif kind == "-heal" and any("move: Wish" in p for p in parts):
            s = self._side(battle, parts[2])
            if s:
                self.wish[s] = 0
        elif kind == "-prepare" and len(parts) > 3:
            s = self._side(battle, parts[2])
            if s:
                self.prep[s] = _n(parts[3])
                self.synth[s].add(self.prep[s])
        elif kind == "-status" and len(parts) > 3:
            side, sp = self._owner(battle, parts[2])
            if side:
                self.age[side][sp] = {"status": parts[3], "age": 0}
        elif kind in ("-curestatus", "faint"):
            side, sp = self._owner(battle, parts[2])
            if side:
                self.age[side].pop(sp, None)
        elif kind == "cant" and len(parts) > 3 and parts[3] == "recharge":
            s = self._side(battle, parts[2])
            if s:
                self.synth[s].discard("mustrecharge")
        elif kind == "-mustrecharge":
            s = self._side(battle, parts[2])
            if s:
                self.synth[s].add("mustrecharge")
        elif kind == "-start" and len(parts) > 3 and parts[3] == "confusion" and "[fatigue]" in parts:
            s = self._side(battle, parts[2])
            if s:
                self.synth[s].discard("lockedmove")
        for s in ("me", "opp"):                                     # 다음 교체 때 비교할 HP
            mon = self._mon(battle, s)
            if mon is not None:
                self.hp_seen[s][_n(mon.species)] = mon.current_hp_fraction

    def _reveal(self, battle, parts, kind):
        ident0 = parts[2] if len(parts) > 2 and _IDENT.match(parts[2]) else None
        of = next((p[5:] for p in parts[3:] if p.startswith("[of] ")), None)
        if kind == "-ability" and ident0 and len(parts) > 3:
            self._set_ability(battle, ident0, parts[3])
        elif kind == "-item" and ident0 and len(parts) > 3 and not parts[3].startswith("["):
            self._set_item(battle, ident0, parts[3])
        elif kind == "-enditem" and ident0:
            self._set_item(battle, ident0, None)
            side, sp = self._owner(battle, ident0)
            mon = self._mon(battle, side) if side else None
            if mon is not None and "unburden" in _abilities(mon):    # Foul Play는 도구를 잃으면 unburden 효과를 붙임
                self.synth[side].add("unburden")
            return
        elif kind == "-activate" and ident0 and len(parts) > 4 and parts[3].lower() == "move: poltergeist":
            self._set_item(battle, ident0, parts[4])
        for p in parts[3:]:
            m = _ABIL.match(p)
            if m and kind != "-ability":
                self._set_ability(battle, of or ident0, m.group(1).strip())
            m = _ITEM.match(p)
            if m and "[consumed]" not in parts and kind != "-enditem":
                self._set_item(battle, of or ident0, m.group(1).strip())

    def _switch(self, battle, parts):
        side, sp = self._owner(battle, parts[2])
        if not side:
            return
        mon = self._mon(battle, side)
        prev = self.hp_seen[side].get(sp)
        if side == "opp" and mon is not None and prev is not None and abs(mon.current_hp_fraction - prev) > 0.005                 and "regenerator" in _abilities(mon) and not mon.ability and not self.ability["opp"].get(sp):
            self.ability["opp"][sp] = "regenerator"                    # 나갔다 들어왔더니 HP가 회복돼 있음 (Foul Play와 같은 규칙)
        self.entry[side][sp] = battle.turn
        self.used[side][sp] = set()
        self.synth[side].clear()
        self.prep[side] = None
        self.last_action[side] = ("switch", sp)
        a = self.age[side].get(sp)
        if a and a["status"] == "tox":
            a["age"] = 0                                               # 맹독 카운터는 교체하면 처음부터
        if side == "opp":
            mon = self._mon(battle, "opp")
            self.pending_boots = {"sp": sp, "lines": [], "status": mon.status if mon else None}

    def _move(self, battle, parts):
        tags = [p for p in parts[4:] if p.startswith("[from]")]
        if tags and not any("lockedmove" in t.lower() for t in tags):
            return                                         # 다른 기술이 부른 기술 (수면말하기 등)은 제외
        side, sp = self._owner(battle, parts[2])
        if not side:
            return
        mv = _n(parts[3])
        self.last_move_ident = parts[2]
        if self.prep[side]:                                # 충전이 끝나는 행동
            self.synth[side].discard(self.prep[side])
            self.prep[side] = None
        mon = self._mon(battle, side)
        if sp == "slaking" or (mon is not None and _n(mon.ability or self.ability[side].get(sp) or "") == "truant"):
            self.synth[side].add("truant")
        prev = self.last_move_sp[side]
        self.last_move_sp[side] = (sp, mv)
        self.last_action[side] = ("move", mv)
        self.used[side].setdefault(sp, set()).add(mv)
        if mv == "wish":
            self.wish[side] = 2
        try:
            locked = ((Move(mv, gen=9).entry.get("self") or {}).get("volatileStatus")) == "lockedmove"
        except Exception:
            locked = False
        if locked:
            self.synth[side].add("lockedmove")
        else:
            self.synth[side].discard("lockedmove")
        if side == "opp":
            if prev and prev[0] == sp and prev[1] != mv:                # 같은 등장 중 서로 다른 기술 -> 구애 도구가 아님
                self.can_choice[sp] = False
                if self.item["opp"].get(sp) in CHOICE and self.inferred["opp"].get(sp):
                    self.item["opp"][sp] = None
                    self.inferred["opp"][sp] = False
            if self._unlikely_choice(mv):
                self.can_choice[sp] = False

    @staticmethod
    def _unlikely_choice(mv):
        try:
            m = Move(mv, gen=9)
            return (bool(m.boosts) and m.category == MoveCategory.STATUS) or mv in ("substitute", "roost", "recover")
        except Exception:
            return False

    def _weather(self, battle, parts):
        name = _n(parts[2]) if len(parts) > 2 else "none"
        if name == "none":
            self.weather = None
        elif "[upkeep]" in parts:
            w = self.weather
            if w is not None:
                w["upkeeps"] += 1
                rock = ROCK.get(next((k for k in ROCK if k in w["name"]), ""), None)
                if w["upkeeps"] >= 5 and w["side"] == "opp" and rock and self.item["opp"].get(w["sp"]) is None \
                        and not self._known_item_by_species(battle, "opp", w["sp"]):
                    self.item["opp"][w["sp"]] = rock
                    self.inferred["opp"][w["sp"]] = True
        else:
            of = next((p[5:] for p in parts[3:] if p.startswith("[of] ")), None)
            side, sp = self._owner(battle, of or self.last_move_ident)
            self.weather = {"name": name, "upkeeps": 0, "side": side, "sp": sp}

    @staticmethod
    def _known_item_by_species(battle, side, sp):
        """poke-env가 이미 아는 도구 (Foul Play의 UNKNOWN_ITEM이 아닌 경우)"""
        mon = battle.active_pokemon if side == "me" else battle.opponent_active_pokemon
        return mon is not None and _n(mon.species) == sp and mon.item not in (None, "", "unknown_item")

    # ---- 돌격 부츠 추론 (Foul Play check_heavydutyboots): 교체 직후 줄들에서 장판 피해/상태/끈끈이가 없으면
    def _resolve_boots(self, battle):
        pend, self.pending_boots = self.pending_boots, None
        sp = pend["sp"]
        mon = battle.opponent_active_pokemon
        if mon is None or _n(mon.species) != sp or sp in self.impossible_boots or self.item["opp"].get(sp) or self._known_item_by_species(battle, "opp", sp):
            return
        abil = _abilities(mon)
        if "magicguard" in abil:
            return
        sc = battle.opponent_side_conditions
        types = {t.name.lower() for t in mon.types if t}
        lines = pend["lines"]
        flying, levitate = "flying" in types, ("levitate" in abil or _n(mon.ability or "") == "levitate")
        took = lambda tag: any(len(p) > 4 and p[1] == "-damage" and p[4] == tag for p in lines)
        if sc.get(SideCondition.STEALTH_ROCK, 0) > 0:
            hit = took("[from] Stealth Rock")
        elif sc.get(SideCondition.SPIKES, 0) > 0 and not levitate and not flying:
            hit = took("[from] Spikes")
        elif (sc.get(SideCondition.TOXIC_SPIKES, 0) > 0 and pend["status"] is None and not (types & {"flying", "poison", "steel"}) and not levitate
              and _n(mon.ability or "") not in _POISON_IMMUNE):
            hit = any(len(p) > 3 and p[1] == "-status" and p[3] in ("psn", "tox") for p in lines)
        elif sc.get(SideCondition.STICKY_WEB, 0) > 0 and not flying and "levitate" not in abil:
            hit = any(len(p) == 4 and p[1] == "-activate" and p[3] == "move: Sticky Web" for p in lines)
        else:
            return
        if hit:
            self.impossible_boots.add(sp)
        else:
            self.item["opp"][sp] = "heavydutyboots"
            self.inferred["opp"][sp] = True

    # ---- 구애 스카프 추론 (Foul Play check_choicescarf): 같은 우선도, 상대가 먼저인데 최대 속도 가정으로도 내가 더 빠름
    def _check_scarf(self, battle, lines):
        if any(l.startswith("|switch|") or l.startswith("|cant|") or (l.startswith("|-activate|") and l.endswith("confusion")) for l in lines):
            return
        moves = [l.split("|") for l in lines if l.startswith("|move|")]
        if len(moves) != 2 or self._side(battle, moves[0][2]) != "opp" or self._side(battle, moves[1][2]) != "me":
            return
        try:
            m0, m1 = Move(_n(moves[0][3]), gen=9), Move(_n(moves[1][3]), gen=9)
        except Exception:
            return
        if m0.priority != m1.priority:
            return
        opp, me = battle.opponent_active_pokemon, battle.active_pokemon
        if opp is None or me is None:
            return
        sp = _n(opp.species)
        if self.item["opp"].get(sp) or self._known_item_by_species(battle, "opp", sp) or not self.can_choice.get(sp, True):
            return
        if self._can_speed_modified(battle, opp) or self._can_priority_modified(battle, opp, m0.id) or self._can_priority_modified(battle, me, m1.id):
            return
        if _n(me.ability or "") == "unburden" and me.item in (None, ""):
            return
        tr = Field.TRICK_ROOM in battle.fields
        base, lvl = opp.base_stats["spe"], opp.level or 100
        if tr:
            stat = int((((2 * base + 31) * lvl) // 100 + 5) * 0.9)           # 느린 쪽이 먼저: 최소 속도 가정 (성격 보정 -, 노력치 0)
        else:
            stat = int((((2 * base + 31 + 63) * lvl) // 100 + 5) * 1.1)      # 최대 속도 가정 (성격 보정 +, 노력치 252)
        opp_eff = stat * _boost(opp.boosts.get("spe", 0)) * (0.5 if opp.status == Status.PAR else 1.0) \
            * (2.0 if SideCondition.TAILWIND in battle.opponent_side_conditions else 1.0) * self._pq_speed(opp)
        my_stat = (me.stats or {}).get("spe") or me.base_stats["spe"]
        me_eff = my_stat * _boost(me.boosts.get("spe", 0)) * (0.5 if me.status == Status.PAR else 1.0) \
            * (2.0 if SideCondition.TAILWIND in battle.side_conditions else 1.0) * (1.5 if me.item == "choicescarf" else 1.0) * self._pq_speed(me)
        if (opp_eff > me_eff) if tr else (me_eff > opp_eff):
            self.item["opp"][sp] = "choicescarf"
            self.inferred["opp"][sp] = True

    @staticmethod
    def _pq_speed(mon):
        names = {e.name.lower() for e in (mon.effects or {})}
        return 1.5 if names & {"quarkdrivespe", "protosynthesisspe"} else 1.0

    @staticmethod
    def _can_priority_modified(battle, mon, move_id):
        abil = _abilities(mon)
        try:
            status = Move(move_id, gen=9).category == MoveCategory.STATUS
        except Exception:
            status = False
        return "prankster" in abil or (move_id == "grassyglide" and Field.GRASSY_TERRAIN in battle.fields) or (status and "myceliummight" in abil)

    @staticmethod
    def _can_speed_modified(battle, mon):
        abil, unknown = _abilities(mon), not mon.ability
        w = battle.weather or {}
        return ((mon.item in (None, "") and "unburden" in abil)
                or (Weather.RAINDANCE in w and unknown and "swiftswim" in abil)
                or (Weather.SUNNYDAY in w and unknown and "chlorophyll" in abil)
                or (Weather.SANDSTORM in w and unknown and "sandrush" in abil)
                or ((Weather.SNOWSCAPE in w or Weather.HAIL in w) and unknown and "slushrush" in abil)
                or (Field.ELECTRIC_TERRAIN in battle.fields and unknown and "surgesurfer" in abil)
                or (mon.status == Status.PAR and unknown and "quickfeet" in abil))
