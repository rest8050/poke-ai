# 특성 EXEC 규칙 표 — 초안 v1 (상위 20종 중 규칙으로 계산 가능한 14종)

상태: **초안, 사람 검토 전.** 구현 없음. 효과 값과 조건은 `pokemon-showdown/data/abilities.ts`(및 `sim/battle-actions.ts`) 소스에서 읽었다.
이 문서는 특성을 분류하지 않는다. **특성 → (규칙, 매개변수) → 입력 특징**의 표이고, `Matchup`이 forward 때 계산해서 모델에 수치로 준다. 모델이 표를 읽지 않는다.
범위 밖: 규칙으로 값을 못 내는 반응형 효과 6종(pressure, dancer, harvest, justified, defiant, competitive)은 이 문서에서 다루지 않고 기존 훅/설명문 특징(`entity_features`)에 맡긴다. 단 defiant/competitive가 위협에 반응한 *결과*는 아래 intimidate 규칙에 포함한다.

## 1. 대상과 규모

학습에 쓰지 않은 세트(dg5 + d6_ho + 홀드아웃 fp) 활성 포켓몬 슬롯 중 공개된 특성 115,217개 기준 (미공개 토큰은 57,477개, 8장 참고).

| 구분 | 특성 | 출현 | 공개된 특성 중 비율 |
|---|---|---|---|
| 이미 일부 구현 | levitate 9,324 / waterabsorb 2,153 / sapsipper 1,427 | 12,904 | 11.2% |
| **A단계** (재빌드 없이 `Matchup`만) | intimidate 5,235 / regenerator 2,901 / sturdy 1,751 / clearbody 1,750 / chlorophyll 1,722 / technician 1,662 / swiftswim 1,541 / prankster 1,383 / thickfat 1,077 | 19,022 | 16.5% |
| **B단계** (인코더 변경 + 데이터 재빌드) | protosynthesis 4,307 / quarkdrive 3,500 | 7,807 | 6.8% |
| 합계 | 14종 | 39,733 | 34.5% |

## 2. 현재 코드 사실 (이 표가 끼워질 자리)

- `Matchup`(`src/core/matchup.py`): 공격기만 계산(변화기 4채널은 전부 0), 입력은 저장된 텐서(능력치, 랭크, 타입, 특성/도구 id, HP, 기술 위력/분류/타입)와 `field_vec`. 이미 있는 규칙: 타입 상성, 자속/테라, 화상, 부유류 면역(`IMMUNE_ABILITIES`), 풍선, 천진(공/방 랭크 무시), 장판 진입 피해(`entry_frac`).
- 스피드: `Matchup.forward`의 `spd_my/spd_opp`는 능력치 비율만 본다(날씨, 트릭룸, 마비, 도구 없음). 인코더의 `priority_speed_edge`(기술 수치 45번 열)는 우선도 등급 차이와 종족값 스피드를 비교하고 **트릭룸 반전은 반영**하지만, 랭크, 마비, 날씨 특성, 도구는 반영하지 않는다. 이 값은 npz에 저장되므로 바꾸려면 데이터 재빌드가 필요하다. 따라서 A단계의 스피드/우선도 규칙은 저장된 값을 건드리지 않고 `Matchup`이 forward 때 새로 계산한다.
- `Matchup.forward`는 현재 `field_vec`를 받지 않는다(`entry_frac`만 따로 받음). 스피드 규칙에는 날씨 플래그가 필요해서 인터페이스 변경이 필요하다.

## 3. 규칙 표

날씨 플래그: `field_vec`의 0번 열 = 쾌청/강한 쾌청, 2번 열 = 비/강한 비. 특성 id는 `data/vocab.json`의 `ability`.

| 단계 | 특성 | 소스에서 읽은 효과 | 규칙 | 삽입 지점 | 새 입력 |
|---|---|---|---|---|---|
| A | technician | 위력 60 이하 기술 ×1.5 (`onBasePower`, 보정 후 60 이하) | 공격측 특성이 technician이고 기술 위력 ≤ 60이면 위력 ×1.5 | `_pair`의 `bp` | 없음 (데미지/KO 값이 바뀜) |
| A | thickfat | 불꽃/얼음 기술의 공격 ×0.5 (`onSourceModifyAtk/SpA`, `breakable`) | 방어측 특성이 thickfat이고 기술 타입이 불/얼음이면 데미지 ×0.5 | `_pair`의 `dmg` | 없음 |
| A | sturdy | 만피에서 한 방에 쓰러지면 HP 1로 버팀, 일격필살 무효 (`breakable`) | 방어측 특성이 sturdy이고 HP 비율 ≥ 1이면 `ko = False` (데미지 비율은 그대로) | `_pair`의 `ko` | 없음 |
| A | chlorophyll | 쾌청이면 스피드 ×2 | 특성이 chlorophyll이고 쾌청이면 스피드 ×2 | `forward`의 스피드 | 없음 (`spd_*` 값이 바뀜) |
| A | swiftswim | 비이면 스피드 ×2 | 특성이 swiftswim이고 비이면 스피드 ×2 | 위와 같음 | 없음 |
| A | prankster | 변화기 우선도 +1 (`onModifyPriority`), 악 타입 대상에게는 실패 (`sim/battle-actions.ts`) | 변화기의 유효 우선도 = 저장된 우선도(기술 수치 3번 열 × 7) + 1, 대상이 악 타입이면 실패 표시 | 새 채널 "먼저 움직일 확률/방향" | `mv_extra` 1열 |
| A | intimidate | 등장 시 상대 공격 −1, 대타 상대는 무효 (`onStart`) | 교체 후보가 intimidate이면, 상대 활성의 물리 공격에 후보로 들어올 때의 공격 랭크 −1을 적용(아래 상대 특성 결과 표) | `_pair`(상대→내 후보)의 `A` | 없음 (후보의 D 채널 값이 바뀜) |
| A | clearbody | 상대가 일으키는 능력 하락 무효 (`onTryBoost`, `breakable`) | 독립 규칙이 아니라 위 intimidate의 "상대 특성 결과 표"에 포함 | | |
| A | regenerator | 교체해 나가면 HP 1/3 회복 (`onSwitchOut`) | 내 활성이 regenerator이면 `exit_heal = min(1/3, 1 − HP 비율)`을 활성 스칼라로 입력 | 종류 헤드와 교체 점수 입력 | 활성 스칼라 1열 |
| B | protosynthesis | 쾌청(또는 부스터에너지)에서 가장 높은 능력치 ×1.3(스피드면 ×1.5) (`condition`, 5325/4096) | 인코더가 관측된 휘발성 상태(`-start ... protosynthesis<능력>`)를 효과 열로 인코딩하고, `Matchup`이 해당 능력치에 배율 적용 | `EFFECT_FLAGS` 확장, `_stats` | 효과 열 추가 + 재빌드 |
| B | quarkdrive | 일렉트릭필드(또는 부스터에너지)에서 가장 높은 능력치 ×1.3(스피드면 ×1.5) | 위와 같음 (`quarkdrive<능력>`) | 위와 같음 | 위와 같음 |
| 기존 | levitate / waterabsorb / sapsipper | 땅/물/풀 무효 (`breakable`), 물/풀은 회복·공격 +1 | `IMMUNE_ABILITIES`가 면역을 이미 처리. 흡수 이득과 `breakable`(틀깨기 무시)은 미반영 | `_pair`의 `immune` | 없음 |

### intimidate의 상대 특성 결과 표 (소스 확인)

| 상대 활성의 특성 | 공격 랭크 변화 | 소스 |
|---|---|---|
| (해당 없음) | −1 | `intimidate.onStart` |
| clearbody, whitesmoke, fullmetalbody, hypercutter | 0 (차단) | `onTryBoost` |
| innerfocus, oblivious, owntempo, scrappy | 0 (위협 면역) | 각 특성의 `onTryBoost`(이름 'Intimidate') |
| guarddog | +1 | `guarddog.onTryBoost` |
| defiant | −1 후 +2 = **+1** | `defiant.onAfterEachBoost` |
| competitive | 공격 −1, 특공 +2 | `competitive.onAfterEachBoost` |
| mirrorarmor | 0 (반사: 내 후보의 공격 −1) | `mirrorarmor.onTryBoost` |
| rattled | 공격 −1, 스피드 +1 | `rattled`(이름 'Intimidate') |
| 상대가 대타를 쓰는 중 | 0 | `intimidate.onStart` |

상대 특성이 미공개이면 8장의 처리를 따른다.

## 4. 새 입력 열 (초안)

| 열 | 형태 | 의미 |
|---|---|---|
| `fm_edge` | 기술 토큰별 1 (`mv_extra` +1) | 이 기술이 상대 활성보다 먼저 나갈지: +1 먼저 / −1 나중 / 0 동률·불명. 유효 우선도(prankster 포함)와 보정된 스피드(날씨 특성) 기준. 트릭룸 반전 포함 |
| `exit_heal` | 활성 스칼라 1 | 지금 활성이 교체되어 나갈 때 회복량(regenerator) |

그 외 규칙은 기존 D/O 채널(배율, 데미지 비율, KO)과 `spd_my/spd_opp` 값만 정확하게 바꾼다 (열이 늘지 않음).
입력 열이 늘고 값 분포가 바뀌므로 **모델은 재학습**이 필요하다. A단계는 데이터 재빌드가 필요 없다.

## 5. 구현 순서 (제안)

1. `Matchup.forward`에 `field_vec`를 넘기는 인터페이스 정리.
2. A단계 규칙 구현, 규칙마다 소스 경계값 단위 테스트.
3. B단계: 인코더 효과 열 확장 + 데이터 재빌드 + 규칙 구현.
4. 같은 데이터로 `v6`(합법 비트 수정만 반영) 대조군과 비교.

## 6. 테스트 (소스 경계값)

- technician: 위력 60 → ×1.5, 위력 65 → 변화 없음.
- thickfat: 불꽃/얼음 ×0.5, 물 변화 없음.
- sturdy: 만피에서 KO 가능한 데미지도 `ko=False`, HP 99%에서는 `ko=True`.
- chlorophyll/swiftswim: 쾌청/비에서만 ×2, 다른 날씨에서는 ×1.
- prankster: 변화기 우선도 +1, 공격기 +0, 악 타입 대상은 실패 표시.
- intimidate: 후보 입장 시 공격 −1(물리 데미지 ×2/3), clearbody 상대는 0, defiant 상대는 +1.
- regenerator: HP 100%면 0, HP 50%면 1/3, HP 80%면 0.2.

## 7. 통과선 (사전 등록)

- 날씨 연동 특성 묶음의 일치율 격차 −0.043이 줄어드는지 (쓱쓱/엽록소 규칙의 직접 효과).
- 변화기를 교사가 고른 결정의 일치율 0.535가 오르는지 (prankster `fm_edge`의 직접 효과는 일부만).
- 교체 선택이 가장 약하므로(일치율 0.452), intimidate/regenerator 규칙이 교체 결정 일치율을 올리는지.
- 전체 KL이 나빠지지 않을 것 (+0.005 이내), 다른 특성 묶음이 나빠지지 않을 것.
- 특성 소거 민감도가 오르는지 (지금 0.026~0.046).
- 비교는 항상 같은 데이터로 학습한 `v6` 대조군 대비이며, 승률은 시드 변동(약 ±5pp) 때문에 최종 확인용으로만 쓴다.

## 8. 미결정 (검토 요청)

1. **상대 특성 미공개(전체 슬롯의 약 33%)**: A단계는 **공개된 것만** 적용하는 단순안을 기본으로 한다. 신념(`hid_ability`) 확률 기대값 적용은 두 번째 실험으로 분리해서 효과를 따로 본다.
2. **스피드 규칙의 범위**: 이 표는 날씨 특성만 다룬다. 마비, 랭크, 도구(스카프), 순풍, 트릭룸의 스피드 보정은 특성이 아니라 별도 규칙이지만 `fm_edge` 계산에 같이 넣는 게 자연스러움. 범위를 같이 넓힐지.
3. **intimidate 규칙의 위치**: 후보의 D 채널 값을 바꾸는 방식(새 열 없음)으로 정했지만, 위협 효과를 명시 열(`entry_foe_atk_drop`)로도 줄지.
4. **fm_edge 입력 위치**: 기술 토큰별로만 줄지, 상대 활성의 기술 토큰에도 줄지.

## 9. 예약 (이 표의 확장)

`IGNORE_TARGET_ABILITY`(moldbreaker: `breakable` 면역/보정 무효화), `STATUS_IMMUNE`(불면 등), `REFLECT_STATUS`(magicbounce), `ATTACK_MULT`(hugepower, guts), `DAMAGE_MULT_COND`(multiscale 등). 상위 20종 밖이고 변화기 적용 가능성 표와 함께 별도 문서로 다룬다.
