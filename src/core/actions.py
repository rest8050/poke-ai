"""22칸 행동 번호 -> poke-env 행동 (플레이어와 브리지가 같이 씀)"""


def decode_action(battle, idx):
    """22칸 행동 -> ("move", Move, z, tera) | ("switch", Pokemon) | None (실행 불가)"""
    active = battle.active_pokemon
    if idx < 8 and active:
        moves = list(active.moves.values())
        m = idx % 4
        if m < len(moves) and moves[m].id in {x.id for x in battle.available_moves}:
            gimmick = idx >= 4
            z = gimmick and battle.can_z_move and moves[m].id in {x.id for x in active.available_z_moves}
            tera = gimmick and not z and bool(getattr(battle, "can_tera", False))
            return ("move", moves[m], z, tera)
    elif 8 <= idx <= 13:
        team = list(battle.team.values())
        k = idx - 8
        if k < len(team) and team[k] in battle.available_switches:
            return ("switch", team[k])
    return None
