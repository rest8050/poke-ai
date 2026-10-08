"""torch 없이 쓰는 게임 메커니즘 상수 (matchup.py의 텐서 계산과 damage_calc.py의 스칼라 기준 계산이 같은 표를 씀)."""

# 이 특성을 가진 방어측은 해당 타입 공격기에 무효 (공중부양 계열 + 흡수/마중물 계열)
IMMUNE_ABILITIES = {"levitate": "ground", "eartheater": "ground", "flashfire": "fire", "wellbakedbody": "fire", "waterabsorb": "water",
                    "dryskin": "water", "stormdrain": "water", "voltabsorb": "electric", "lightningrod": "electric", "motordrive": "electric",
                    "sapsipper": "grass"}
