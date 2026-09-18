"""
RCT 기본 전투 AI(RCTBattleAI)를 poke-env Player로 옮긴 버전.

원본: Radical Cobblemon Trainers API 0.15.2-beta (서버 jar와 같은 버전)
      com/gitlab/srcmc/rctapi/api/ai/{RCTBattleAI, utils/PokeMath, utils/TypeChart,
      utils/MoveType, utils/ResponseBuilder, utils/BattleEffects, utils/BattleStates}.java
      Copyright (c) 2026, HDainester. Mod Community License (MCOML).
      PokeMath은 davo899의 CobblemonTrainers Gen5 AI에서 파생.

변경 사항 (MCOML에 따라 명시):
- Cobblemon 배틀 객체 대신 poke-env Battle 객체를 읽음.
- 원본은 상대의 실제 능력치·특성·도구를 읽지만 poke-env는 공개 정보만 있음.
  상대 능력치는 종족값+레벨(IV 31, EV 84, 무보정)로 추정하고, 특성은 공개됐거나 후보가 1개일 때만 사용.
- 가방 아이템(상처약 등) 사용 없음 (쇼다운에 없음).
- 원본의 턴 카운터(등장 턴, 변화기 연속 사용, 방어/희망 직후)는 choose_move 호출 기준으로 근사.
- 테라스탈은 서버 트레이너에 없어서 기본 끔 (use_tera=True로 켤 수 있음).
- 싱글 배틀만 지원.
"""
import math
import random

try:
    from poke_env.player import Player
    POKE_ENV_AVAILABLE = True
except ImportError:
    Player = object
    POKE_ENV_AVAILABLE = False


HEAL, CURE, BUFF, MALUS, STATUS, DAMAGE = "HEAL", "CURE", "BUFF", "MALUS", "STATUS", "DAMAGE"

# MoveType.MOVE_TYPES 중 HEAL/CURE/BUFF/MALUS만. 나머지는 원본 기본값처럼 분류로 STATUS/DAMAGE.
MOVE_TYPES = {m: t for t, ids in {
    HEAL: "aquaring floralhealing healorder healpulse healingwish ingrain junglehealing lifedew lunarblessing "
          "milkdrink moonlight morningsun recover refresh rest revivalblessing roost shoreup slackoff softboiled "
          "strengthsap synthesis wish",
    CURE: "aromatherapy healbell psychoshift purify",
    BUFF: "acidarmor acupressure agility amnesia aromaticmist auroraveil autotomize barrier bellydrum bulkup "
          "burningbulwark calmmind charge clangoroussoul coaching coil cosmicpower cottonguard decorate defendorder "
          "defensecurl dragoncheer dragondance extremeevoboost flowershield focusenergy gearup geomancy growth harden "
          "helpinghand honeclaws howl irondefense laserfocus lightscreen magneticflux meditate nastyplot quiverdance "
          "reflect rockpolish sharpen shellsmash shiftgear spicyextract stockpile swordsdance tailglow tailwind "
          "takeheart victorydance workup",
    MALUS: "babydolleyes captivate charm corrosivegas cottonspore eerieimpulse faketears featherdance flatter "
           "gastroacid growl leer metalsound nobleroar partingshot playnice sandattack scaryface screech smokescreen "
           "swagger tailwhip tarshot tearfullook tickle topsyturvy venomdrench",
}.items() for m in ids.split()}

# TypeChart: 방어 타입 -> {공격 타입: 배율}
CHART = {
    "normal": {"fighting": 2, "ghost": 0},
    "fighting": {"flying": 2, "rock": .5, "bug": .5, "psychic": 2, "dark": .5, "fairy": 2},
    "flying": {"fighting": .5, "ground": 0, "rock": 2, "bug": .5, "grass": .5, "electric": 2, "ice": 2},
    "poison": {"fighting": .5, "poison": .5, "ground": 2, "bug": .5, "grass": .5, "psychic": 2, "fairy": .5},
    "ground": {"poison": .5, "rock": .5, "water": 2, "grass": 2, "electric": 0, "ice": 2},
    "rock": {"normal": .5, "fighting": 2, "flying": .5, "poison": .5, "ground": 2, "steel": 2, "fire": .5,
             "water": 2, "grass": 2},
    "bug": {"fighting": .5, "flying": 2, "ground": .5, "rock": 2, "fire": 2, "grass": .5},
    "ghost": {"normal": 0, "fighting": 0, "poison": .5, "bug": .5, "ghost": 2, "dark": 2},
    "steel": {"normal": .5, "fighting": 2, "flying": .5, "poison": 0, "ground": 2, "rock": .5, "bug": .5,
              "steel": .5, "fire": 2, "grass": .5, "psychic": .5, "ice": .5, "dragon": .5, "fairy": .5},
    "fire": {"ground": 2, "rock": 2, "bug": .5, "steel": .5, "fire": .5, "water": 2, "grass": .5, "ice": .5,
             "fairy": .5},
    "water": {"steel": .5, "fire": .5, "water": .5, "grass": 2, "electric": 2, "ice": .5},
    "grass": {"flying": 2, "poison": 2, "ground": .5, "bug": 2, "fire": 2, "water": .5, "grass": .5,
              "electric": .5, "ice": 2},
    "electric": {"flying": .5, "ground": 2, "steel": .5, "electric": .5},
    "psychic": {"fighting": .5, "bug": 2, "ghost": 2, "psychic": .5, "dark": 2},
    "ice": {"fighting": 2, "rock": 2, "steel": 2, "fire": 2, "ice": .5},
    "dragon": {"fire": .5, "water": .5, "grass": .5, "electric": .5, "ice": 2, "dragon": 2, "fairy": 2},
    "dark": {"fighting": 2, "bug": 2, "ghost": .5, "psychic": 0, "dark": .5, "fairy": 2},
    "fairy": {"fighting": .5, "poison": 2, "bug": .5, "steel": 2, "dragon": 0, "dark": .5},
}

STAT_KEYS = ("atk", "def", "spa", "spd", "spe")
SINGLE_TARGETS = {"normal", "any", "adjacentfoe", "randomnormal"}
PROTECTS = {"protect", "detect", "endure", "wideguard", "quickguard", "spikyshield", "kingsshield", "banefulbunker"}
CONFUSERS = {"confuseray", "flatter", "supersonic", "swagger", "sweetkiss", "teeterdance"}
SLEEPERS = {"darkvoid", "grasswhistle", "hypnosis", "lovelykiss", "sing", "sleeppowder", "spore"}
SACRIFICES = {"explosion", "mistyexplosion", "selfdestruct"}
SACRIFICES_NO_DAMP = {"lunardance", "finalgambit", "memento", "healingwish"}
DOUBLES_ONLY = {"allyswitch", "followme", "ragepowder", "helpinghand", "afteryou", "instruct", "coaching"}


# ---------- poke-env 객체 읽기 ----------

def _id(x):
    """Enum/문자열 -> 소문자 id ('LIGHT_SCREEN' -> 'lightscreen')"""
    if x is None:
        return ""
    return str(getattr(x, "name", x)).lower().replace("_", "").replace(" ", "")


def types(mon):
    return [t for t in (_id(mon.type_1), _id(mon.type_2)) if t]


def ability(mon):
    a = mon.ability
    if not a:
        possible = list(mon.possible_abilities or [])
        a = possible[0] if len(possible) == 1 else None
    return _id(a)


def stat(mon, key):
    """자기 포켓몬은 실제 능력치, 상대는 종족값으로 추정 (원본은 상대 실제 능력치를 읽음)"""
    s = (getattr(mon, "stats", None) or {}).get(key)
    if s:
        return s
    lv = mon.level or 100
    v = (2 * mon.base_stats[key] + 31 + 21) * lv // 100
    return v + lv + 10 if key == "hp" else v + 5


def cur_hp(battle, mon):
    if mon in battle.team.values():
        return mon.current_hp
    return mon.current_hp_fraction * stat(mon, "hp")


def boost_avg(mon):
    b = mon.boosts
    # 원본 BattleEffects.Boost.avg 그대로: spd를 두 번 더하고 spe는 빠짐
    return (b.get("atk", 0) + b.get("def", 0) + b.get("spa", 0) + 2 * b.get("spd", 0) + b.get("evasion", 0)) / 36


def status_any(mon):
    return mon.status is not None and _id(mon.status) != "fnt"


def has_eff(mon, name):
    return any(_id(e) == name for e in (mon.effects or {}))


def weather(battle, *names):
    return any(_id(w) in names for w in (battle.weather or {}))


def field(battle, name):
    return any(_id(f) == name for f in (battle.fields or {}))


def side(battle, mon):
    return battle.side_conditions if mon in battle.team.values() else battle.opponent_side_conditions


def side_val(battle, mon, name):
    """장판은 겹수, 그 외는 존재하면 1 이상"""
    for k, v in (side(battle, mon) or {}).items():
        if _id(k) == name:
            return v if name in ("spikes", "toxicspikes") else 1
    return 0


def raised(battle, mon):
    if field(battle, "gravity") or has_eff(mon, "ingrain") or has_eff(mon, "smackdown"):
        return False
    item = _id(mon.item)
    if item == "ironball":
        return False
    if item == "airballoon" or ability(mon) == "levitate" or "flying" in types(mon):
        return True
    return has_eff(mon, "magnetrise") or has_eff(mon, "telekinesis")


def eff_single(battle, atk_type, def_type, defender, ab=None):
    """ab: 방어측 특성을 직접 지정 (미공개 특성을 추론값으로 판단할 때). None이면 공개 정보 사용"""
    ab = ab if ab is not None else ability(defender)
    if atk_type == "water" and ab in ("stormdrain", "waterabsorb", "dryskin"):
        return 0
    if atk_type == "electric" and ab in ("voltabsorb", "lightningrod", "motordrive"):
        return 0
    if atk_type == "ground":
        if ab == "eartheater" or raised(battle, defender):
            return 0
        if def_type == "flying":
            return 1.0
    if atk_type == "fire" and ab in ("wellbakedbody", "flashfire"):
        return 0
    if atk_type == "grass" and ab == "sapsipper":
        return 0
    return CHART.get(def_type, {}).get(atk_type, 1.0)


def eff_move(battle, move_type, defender, ab=None):
    e = math.prod(eff_single(battle, move_type, dt, defender, ab) for dt in types(defender))
    return 0 if e < 2.0 and (ab if ab is not None else ability(defender)) == "wonderguard" else e


def eff_mons(battle, attacker, defender, avg=False):
    vals = [eff_single(battle, at, dt, defender) for at in types(attacker) for dt in types(defender)]
    if avg:
        return sum(vals) / len(vals) if vals else 0.0
    return math.prod(vals)


def move_type(move):
    return MOVE_TYPES.get(move.id) or (STATUS if _id(move.category) == "status" else DAMAGE)


def damage(battle, rng, attacker, defender, move):
    """PokeMath.damage (5세대 이후 공식 + 원본의 명중률 보정)"""
    cat = _id(move.category)
    if cat == "status":
        return 0
    physical = cat == "physical"
    mt = _id(move.type)
    acc = float(move.accuracy)

    if (weather(battle, "desolateland") and mt == "water") or (weather(battle, "primordialsea") and mt == "fire"):
        return 0

    lv = attacker.level or 100
    atk = stat(attacker, "atk" if physical else "spa")
    dfn = stat(defender, "def" if physical else "spd")
    base = int(((2 * lv / 5.0 + 2) * move.base_power * atk / dfn) / 50.0) + 2
    ab = ability(attacker)
    if _id(attacker.status) == "brn" and physical and ab != "guts":
        base = int(base * 0.5)
    if status_any(attacker) and ab == "guts":
        base = int(base * 1.5)
    sun, rain = weather(battle, "sunnyday"), weather(battle, "raindance")
    if (sun and mt == "fire") or (rain and mt == "water"):
        base = int(base * 1.5)
    if (sun and mt == "water") or (rain and mt == "fire"):
        base = int(base * 0.5)

    own = types(attacker)
    tera = _id(getattr(attacker, "tera_type", None)) if getattr(attacker, "is_terastallized", False) else ""
    adapt = ab == "adaptability"
    stab = 1.0
    if tera:
        # ponytail: poke-env는 테라스탈 후 원래 타입을 따로 주지 않아 teraSame 판정이 근사치
        if mt in own or mt == tera:
            stab = (2.25 if adapt else 2.0) if tera in own else 1.5
    elif mt in own:
        stab = 2.0 if adapt else 1.5
    base = int(base * stab)
    base = int(base * eff_move(battle, mt, defender))

    dmg = max(1, base * (rng.random() * 0.15 + 0.85)) if base > 0 else base
    return math.ceil((acc + rng.random() * (1.0 - acc)) * dmg)


class RCTBattleAIPlayer(Player):
    """RCT 트레이너 기본 AI ('rct')를 흉내내는 poke-env 플레이어"""

    def __init__(self, move_bias=1.0, status_move_bias=0.85, switch_bias=0.5, max_select_margin=0.25,
                 use_tera=False, seed=None, **kwargs):
        super().__init__(**kwargs)
        self._setup(move_bias, status_move_bias, switch_bias, max_select_margin, use_tera, seed)

    def _setup(self, move_bias=1.0, status_move_bias=0.85, switch_bias=0.5, max_select_margin=0.25,
               use_tera=False, seed=None):
        self.move_bias = move_bias
        self.status_move_bias = status_move_bias
        self.switch_bias = switch_bias
        self.margin_max = max_select_margin
        self.use_tera = use_tera
        self.rng = random.Random(seed)
        self.memo = {}  # battle_tag -> 원본 BattleStates 근사용 기록

    # ----- 원본 난수 유틸 -----
    def _rsin(self):
        return math.sin(self.rng.random() * (math.pi / 2) * (math.pi / 2))

    def _range(self, lo, hi, optimistic=False):
        return lo + (self._rsin() if optimistic else 1.0 - self._rsin()) * (hi - lo)

    def _gauss(self, mean, sig3, lo=0.0, hi=1.0):
        return min(hi, max(lo, self.rng.gauss(mean, sig3 / 3.0)))

    # ----- 턴 기록 (BattleStates 근사) -----
    def _track(self, battle):
        st = self.memo.setdefault(battle.battle_tag, {"active": None, "since": 0, "status_since": None,
                                                      "last": None, "last_turn": -1})
        act = battle.active_pokemon
        if act is not None and act is not st["active"]:
            st.update(active=act, since=battle.turn - 1, status_since=None)
        return st

    def _age(self, battle):
        return battle.turn - self.memo[battle.battle_tag]["since"]

    def _used_last_turn(self, battle, ids):
        st = self.memo[battle.battle_tag]
        return st["last"] in ids and st["last_turn"] == battle.turn - 1

    # ----- MoveType.MOVE_EVALUATORS -----
    def _move_eval(self, b, f, t, mid):
        alive = sum(1 for m in b.team.values() if not m.fainted)
        if mid in CONFUSERS:
            return 0.0 if has_eff(t, "confusion") or (not raised(b, t) and field(b, "mistyterrain")) else 1.0
        if mid in PROTECTS:
            return 0.15 if self._used_last_turn(b, PROTECTS) else 1.0
        if mid in SLEEPERS:
            blocked = status_any(t) or ability(t) in ("insomnia", "vitalspirit", "purifyingsalt", "comatose")
            if mid in ("sing", "grasswhistle"):
                blocked = blocked or ability(t) == "soundproof"
            if mid in ("sleeppowder", "spore"):
                blocked = blocked or "grass" in types(t) or ability(t) == "overcoat"
            return (0.5 if mid == "darkvoid" else 0.0) if blocked else 1.0
        if mid in SACRIFICES:
            return 0.0 if ability(t) == "damp" or alive < 2 else 1.0 - 0.5 * f.current_hp_fraction
        if mid in SACRIFICES_NO_DAMP:
            return 0.0 if alive < 2 else 1.0 - 0.5 * f.current_hp_fraction
        if mid in DOUBLES_ONLY:
            return 0.1
        if mid in ("thunderwave", "stunspore", "glare"):
            immune = {"thunderwave": ("electric", "ground"), "stunspore": ("electric", "grass"),
                      "glare": ("electric",)}[mid]
            return 0.0 if status_any(t) or set(types(t)) & set(immune) or \
                ability(t) in ("limber", "comatose", "purifyingsalt") else 1.0
        if mid in ("toxic", "poisongas", "poisonpowder", "toxicthread"):
            immune = {"poison", "steel", "grass"} if mid == "poisonpowder" else {"poison", "steel"}
            blocked = status_any(t) or (set(types(t)) & immune and ability(f) != "corrosion") or \
                ability(t) in ("immunity", "comatose", "purifyingsalt", "pastelveil")
            return (0.25 if mid == "toxicthread" else 0.0) if blocked else 1.0
        if mid == "willowisp":
            return 0.0 if status_any(t) or "fire" in types(t) or \
                ability(t) in ("waterveil", "waterbubble", "comatose", "thermalexchange", "purifyingsalt") else 1.0

        rules = {
            "attract": lambda: 0.0 if has_eff(t, "attract") else 1.0,
            "curse": lambda: 0.0 if "ghost" in types(f) and has_eff(t, "curse") else 1.0,
            "leechseed": lambda: 0.0 if has_eff(t, "leechseed") else 1.0,
            "ingrain": lambda: 0.0 if has_eff(f, "ingrain") else 1.0,
            "aquaring": lambda: 0.0 if has_eff(f, "aquaring") else 1.0,
            "telekinesis": lambda: 0.0 if raised(b, t) else 1.0,
            "magnetrise": lambda: 0.0 if raised(b, f) else 1.0,
            "flowershield": lambda: 1.0 if "grass" in types(f) else 0.0,
            "auroraveil": lambda: 1.0 if not side_val(b, f, "auroraveil") and weather(b, "snow", "snowscape", "hail")
            else 0.0,
            "reflect": lambda: 0.0 if side_val(b, f, "reflect") or side_val(b, f, "auroraveil") else 1.0,
            "lightscreen": lambda: 0.0 if side_val(b, f, "lightscreen") or side_val(b, f, "auroraveil") else 1.0,
            "sunnyday": lambda: 0.0 if weather(b, "sunnyday", "desolateland", "primordialsea", "deltastream") else 1.0,
            "raindance": lambda: 0.0 if weather(b, "raindance", "desolateland", "primordialsea", "deltastream") else 1.0,
            "sandstorm": lambda: 0.0 if weather(b, "sandstorm", "desolateland", "primordialsea", "deltastream") else 1.0,
            "hail": lambda: 0.0 if weather(b, "hail", "snow", "snowscape", "desolateland", "primordialsea",
                                           "deltastream") else 1.0,
            "electricterrain": lambda: 0.0 if field(b, "electricterrain") else 1.0,
            "grassyterrain": lambda: 0.0 if field(b, "grassyterrain") else 1.0,
            "mistyterrain": lambda: 0.0 if field(b, "mistyterrain") else 1.0,
            "psychicterrain": lambda: 0.0 if field(b, "psychicterrain") else 1.0,
            "wish": lambda: 0.0 if self._used_last_turn(b, {"wish"}) else 1.0,
            "meanlook": lambda: 0.0 if has_eff(t, "trapped") else 1.0,
            "spiderweb": lambda: 0.0 if has_eff(t, "trapped") else 1.0,
            "block": lambda: 0.0 if has_eff(t, "trapped") else 1.0,
            "fakeout": lambda: 0.0 if self._age(b) > 1 else 1.75,
            "firstimpression": lambda: 0.0 if self._age(b) > 1 else 1.75,
            "tailwind": lambda: 0.0 if side_val(b, f, "tailwind") else 1.0,
            "gravity": lambda: 0.0 if field(b, "gravity") else 1.0,
            "trickroom": lambda: 0.0 if field(b, "trickroom") else 1.0,
            "spikes": lambda: (3 - side_val(b, t, "spikes")) / 3.0,
            "stealthrock": lambda: 0.0 if side_val(b, t, "stealthrock") else 1.0,
            "toxicspikes": lambda: (2 - side_val(b, t, "toxicspikes")) / 2.0,
            "stickyweb": lambda: 0.0 if side_val(b, t, "stickyweb") else 1.0,
            "taunt": lambda: 0.0 if has_eff(t, "taunt") else 1.0,
            "yawn": lambda: 0.0 if has_eff(t, "yawn") or status_any(t) or
            ability(t) in ("insomnia", "vitalspirit", "purifyingsalt", "comatose") else 1.0,
            "sleeptalk": lambda: 1.0 if _id(f.status) == "slp" or ability(f) == "comatose" else 0.0,
            "snore": lambda: 1.0 if _id(f.status) == "slp" or ability(f) == "comatose" else 0.0,
            "dreameater": lambda: 1.0 if _id(t.status) == "slp" or ability(t) == "comatose" else 0.0,
            "nightmare": lambda: 1.0 if _id(t.status) == "slp" or ability(t) == "comatose" else 0.0,
            "wakeupslap": lambda: 1.75 if _id(t.status) == "slp" or ability(t) == "comatose" else 1.0,
            "acrobatics": lambda: 1.75 if not f.item or _id(f.item) == "flyinggem" else 1.0,
            "bestow": lambda: 0.0 if not f.item else 1.0,
            "fling": lambda: 0.0 if not f.item else 1.0,
            "recycle": lambda: 1.0 if not f.item else 0.0,
            "batonpass": lambda: 1.0 if b.available_switches and boost_avg(f) > 0 else 0.0,
        }
        rule = rules.get(mid)
        return rule() if rule else 1.0

    def _type_eval(self, b, f, t, mt):
        if mt not in (MALUS, STATUS):
            return 1.0
        if mt == MALUS and ability(t) == "clearbody":
            return 0.0
        opp = b.opponent_active_pokemon
        return 1.0 / max(1.0, eff_mons(b, opp, f) if opp else 0.0)

    # ----- RCTBattleAI.evalMove -----
    def _eval_move(self, b, frm, move):
        opp = b.opponent_active_pokemon
        mt = move_type(move)
        if _id(move.target) in SINGLE_TARGETS or mt not in (HEAL, CURE, BUFF):
            to, ally = opp, False
        else:
            to, ally = frm, True
        if to is None:
            return 0.0

        m, sb = self.margin_max, self.status_move_bias
        if mt == HEAL:
            e = max(self._rsin() * m, 1.0 - to.current_hp_fraction) * sb if ally else 0
        elif mt == CURE:
            e = (self._rsin() if status_any(to) else self._rsin() * m) * sb if ally else 0
        elif mt == BUFF:
            e = (1.0 - self._rsin()) * min(1.0, max(0.0, self._rsin() - boost_avg(to) * 5)) * sb if ally else 0
            e *= frm.current_hp_fraction ** 2
        elif mt == MALUS:
            e = (1.0 - self._rsin()) * min(1.0, max(0.0, self._rsin() + boost_avg(to) * 5)) * sb if not ally else 0
            e *= to.current_hp_fraction ** 2
        elif mt == STATUS:
            clean = not status_any(to) and not to.effects
            e = ((1.0 - self._rsin()) if clean else self._rsin() * m) * sb if not ally else 0
            st = self.memo[b.battle_tag]
            used = b.turn - st["status_since"] if st["status_since"] is not None else 0
            e *= 1.0 - min(1.0, used / 1.0) * self._rsin()
        else:
            hp = cur_hp(b, to)
            dmg = min(hp, damage(b, self.rng, frm, to, move)) / hp if hp > 0 else 0
            e = min(1.0, dmg + (1.0 - self._rsin()) * m) * self.move_bias if not ally else 0

        try:
            prio = move.priority
        except Exception:
            prio = 0
        faster = ally or stat(frm, "spe") >= stat(to, "spe") or prio > 0
        speed = 1.0 if faster else self._range(frm.current_hp_fraction, 1.0, True)
        if e == 0:
            return -1.0
        return self._move_eval(b, frm, to, move.id) * self._type_eval(b, frm, to, mt) * speed * e

    # ----- RCTBattleAI.evalSwitch -----
    def _eval_switch(self, b, frm, to):
        deltas = []  # (값, 가중치)
        deltas.append((self._gauss(max(0.0, min(1.0, 0.5 - boost_avg(frm))) if frm else 0.0, 0.125), 6))
        deltas.append((self._gauss(0.5 if frm and (status_any(frm) or frm.effects) else 0.0, 0.125), 8))

        opp = b.opponent_active_pokemon
        opp_avg = 0.0
        if opp is not None and not opp.fainted:
            def eff(a, d):
                return min(1.0, eff_mons(b, a, d, avg=True) / 2.0)
            eff_from = (eff(frm, opp) + (1.0 - eff(opp, frm))) / 2 if frm else 0.0
            eff_to = (eff(to, opp) + (1.0 - eff(opp, to))) / 2
            deltas.append((self._gauss(math.sin((eff_to - eff_from) * math.pi / 2), 0.125, -1.0, 1.0), 48))
            opp_avg = sum(stat(opp, k) for k in STAT_KEYS) / 5.0

        to_avg = sum(stat(to, k) for k in STAT_KEYS) / 5.0
        stats_factor = to_avg / opp_avg if opp_avg > 0 else 1.0
        fh = frm.current_hp_fraction if frm else 0.0
        deltas.append((self._gauss(to.current_hp_fraction - fh, 0.125, -1.0, 1.0), 6))

        agef = 1.0 if (frm is None or b.turn <= 1 or self._age(b) > 1) else self._gauss(0.5, 1.0)
        dmf = self._range(0.25 ** 2, 0.25) if frm is not None and frm.is_dynamaxed else 1.0

        delta = max(0.0, sum(v * w for v, w in deltas)) / sum(w for _, w in deltas)
        biased = self._gauss(delta, self.switch_bias / 2.0, delta, 1.0)
        rating = min(1.0, biased * agef * dmf * stats_factor)
        return self._range(rating ** 1.25, rating)

    # ----- ResponseBuilder: margin 안에서 앞쪽에 치우친 무작위 선택 -----
    def _pick(self, choices, margin):
        choices = sorted(choices, key=lambda c: c[0])
        first = choices[0][0]
        pool = [c for c in choices if c[0] - first <= margin]
        picked = pool[0]
        for i, nxt in enumerate(pool[1:], start=2):
            w = (nxt[0] - first) / margin if margin > 0 else 1
            if self.rng.randrange(i + int(w * 16)) == 0:
                picked = nxt
        return picked

    def _gimmick(self, battle):
        """원본: 사용 가능한 기믹 중 하나를 턴마다 균등 무작위로 고름 (다이맥스는 메가/Z가 없을 때만)"""
        options = []
        if battle.can_mega_evolve:
            options.append("mega")
        if battle.can_z_move:
            options.append("z")
        if battle.can_dynamax and not options:
            options.append("dynamax")
        if self.use_tera and getattr(battle, "can_tera", False) and not options:
            options.append("tera")
        return self.rng.choice(options) if options else None

    def choose_move(self, battle):
        st = self._track(battle)
        active = battle.active_pokemon
        frm = active if active is not None and not active.fainted else None
        margin = self.rng.random() * self.margin_max
        gimmick = self._gimmick(battle) if frm else None
        choices = []

        if frm and not battle.force_switch:
            for mv in battle.available_moves:
                choices.append((1.0 - self._eval_move(battle, frm, mv), "move", mv))
        for mon in battle.available_switches:
            choices.append((1.0 - self._eval_switch(battle, frm, mon), "switch", mon))

        if not choices:
            return self.choose_default_move()

        _, kind, obj = self._pick(choices, margin)
        if kind == "switch":
            return self.create_order(obj)

        mt = move_type(obj)
        if mt == STATUS and st["status_since"] is None:
            st["status_since"] = battle.turn
        elif mt != STATUS:
            st["status_since"] = None
        st["last"], st["last_turn"] = obj.id, battle.turn

        z_ids = {m.id for m in frm.available_z_moves} if gimmick == "z" else set()
        return self.create_order(
            obj,
            mega=gimmick == "mega",
            z_move=obj.id in z_ids,
            dynamax=gimmick == "dynamax" and not frm.is_dynamaxed,
            terastallize=gimmick == "tera",
        )


if __name__ == "__main__":
    # poke-env 없이 계산부만 확인하는 자체 점검
    from types import SimpleNamespace as NS

    def mon(species, t1, t2=None, lv=100, base=(80, 80, 80, 80, 80, 80), hp=1.0, ab=None):
        return NS(species=species, type_1=t1, type_2=t2, level=lv, current_hp_fraction=hp, current_hp=None,
                  ability=ab, possible_abilities=[], item="", status=None, effects={}, boosts={}, stats=None,
                  base_stats=dict(zip(("hp",) + STAT_KEYS, base)), is_dynamaxed=False, is_terastallized=False,
                  fainted=False)

    garchomp = mon("garchomp", "dragon", "ground", base=(108, 130, 95, 80, 85, 102))
    rotom = mon("rotomwash", "electric", "water", ab="levitate")
    skarmory = mon("skarmory", "steel", "flying")
    battle = NS(team={}, weather={}, fields={}, side_conditions={}, opponent_side_conditions={},
                opponent_active_pokemon=rotom, turn=5, battle_tag="t")

    assert eff_move(battle, "ground", rotom) == 0            # 부유
    assert eff_move(battle, "ice", garchomp) == 4.0
    assert eff_move(battle, "ground", skarmory) == 0         # 비행 타입 = 떠 있음
    battle.fields = {"GRAVITY": 1}
    assert eff_move(battle, "ground", skarmory) == 2.0       # 중력이면 비행 1배 × 강철 2배
    battle.fields = {}
    assert stat(garchomp, "spe") == 261 and stat(garchomp, "hp") == 378  # 레벨100 IV31 EV84 추정치

    quake = NS(id="earthquake", category="PHYSICAL", type="GROUND", base_power=100, accuracy=1.0, priority=0,
               target="allAdjacent")
    p = RCTBattleAIPlayer.__new__(RCTBattleAIPlayer)
    p._setup(seed=0)
    assert damage(battle, p.rng, garchomp, rotom, quake) == 0
    dmg = damage(battle, p.rng, garchomp, mon("heatran", "fire", "steel"), quake)
    assert 400 < dmg < 800, dmg                               # 자속 4배, 난수 폭 안

    assert move_type(NS(id="swordsdance", category="STATUS")) == BUFF
    assert move_type(NS(id="thunderwave", category="STATUS")) == STATUS
    assert _id("LIGHT_SCREEN") == "lightscreen"

    picks = [p._pick([(0.1, "a", None), (0.12, "b", None), (0.9, "c", None)], 0.25)[1] for _ in range(2000)]
    assert "c" not in picks and picks.count("a") > picks.count("b")  # margin 밖 제외, 앞쪽 우대
    print("rct_player self-check OK")
