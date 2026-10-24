# 특성 효과 원시 요소 어휘 — 초안 v0 (상위 20종)

상태: **초안, 사람 검토 전.** 효과 내용은 `pokemon-showdown/data/abilities.ts` 소스에서 읽었다(값과 조건은 소스 기준). 구현은 아직 없음.
목적: 특성을 "분류"하지 않고 **효과 원시 요소의 조합**으로 적어 두고, (1) `Matchup`이 실행하는 것, (2) 모델이 태그로 읽는 것으로 나눈다.

## 1. 대상: 출현 빈도 상위 20종

학습에 쓰지 않은 세트(dg5 + d6_ho + 홀드아웃 fp, 활성 포켓몬 슬롯 115,217개 중 공개된 특성만) 기준. 미공개 특성 토큰(id 318)은 57,477회로 별도(주로 상대 쪽).

| 순위 | 특성 | 출현 | 소스에서 읽은 효과 | 원시 요소 | 처리 | 현재 코드 |
|---|---|---|---|---|---|---|
| 1 | levitate | 9,324 | 땅 기술 무효 (`breakable`) | `IMMUNE_TYPE(ground)` | EXEC_NOW | 있음 (`IMMUNE_ABILITIES`), 무시 가능 조건 없음 |
| 2 | intimidate | 5,235 | 등장 시 상대 공격 −1 (대타 상대는 무효, clearbody류가 막음) | `ENTRY_FOE_STAT(atk,−1)` | EXEC_ENTRY | 없음 |
| 3 | pressure | 4,610 | 상대의 기술 PP 소모 +1 | `PP_DRAIN_FOE(+1)` | TAG | 없음 |
| 4 | protosynthesis | 4,307 | 쾌청이면(또는 부스터에너지) 가장 높은 능력치 ×1.3 (스피드면 ×1.5) | `BEST_STAT_BOOST(sun)` | EXEC_STATE | 없음 |
| 5 | quarkdrive | 3,500 | 일렉트릭필드이면(또는 부스터에너지) 가장 높은 능력치 ×1.3 (스피드면 ×1.5) | `BEST_STAT_BOOST(electric terrain)` | EXEC_STATE | 없음 |
| 6 | regenerator | 2,901 | 교체해 나가면 HP 1/3 회복 | `EXIT_HEAL(1/3)` | EXEC_ENTRY | 없음 |
| 7 | waterabsorb | 2,153 | 물 기술 무효 + HP 1/4 회복 (`breakable`) | `IMMUNE_TYPE(water)` + `ABSORB(heal 1/4)` | EXEC_NOW | 면역만 있음 |
| 8 | defiant | 1,785 | 상대 때문에 능력이 떨어지면 공격 +2 | `ON_FOE_DROP_BOOST(atk,+2)` | TAG + 상호작용 | 없음 |
| 9 | sturdy | 1,751 | 만피에서 한 방에 쓰러지면 HP 1로 버팀, 일격필살 무효 (`breakable`) | `FULLHP_SURVIVE` | EXEC_NOW | 없음 (KO 판정에 안 들어감) |
| 10 | clearbody | 1,750 | 상대가 일으키는 능력 하락 무효 (`breakable`) | `BLOCK_FOE_DROP(all)` | EXEC_NOW | 없음 |
| 11 | chlorophyll | 1,722 | 쾌청이면 스피드 ×2 | `SPEED_MULT(sun,×2)` | EXEC_NOW | 없음 (스피드 비교가 날씨를 안 봄) |
| 12 | dancer | 1,714 | 다른 포켓몬이 춤 기술을 쓰면 따라 씀 | `COPY_DANCE` | TAG | 없음 |
| 13 | technician | 1,662 | 위력 60 이하 기술 ×1.5 | `BASEPOWER_MULT(bp<=60,×1.5)` | EXEC_NOW | 없음 |
| 14 | swiftswim | 1,541 | 비이면 스피드 ×2 | `SPEED_MULT(rain,×2)` | EXEC_NOW | 없음 |
| 15 | justified | 1,441 | 악 타입 기술에 맞으면 공격 +1 | `ON_HIT_BOOST(dark,atk,+1)` | TAG | 없음 |
| 16 | sapsipper | 1,427 | 풀 기술 무효 + 공격 +1 (`breakable`) | `IMMUNE_TYPE(grass)` + `ABSORB(atk +1)` | EXEC_NOW | 면역만 있음 |
| 17 | prankster | 1,383 | 변화기 우선도 +1 (악 타입 상대에게는 실패) | `PRIORITY_BONUS(status,+1)` | EXEC_NOW | 없음 |
| 18 | harvest | 1,125 | 턴 끝에 먹은 열매 복구 (쾌청이면 확정, 아니면 1/2) | `RESIDUAL_RESTORE_BERRY` | TAG | 없음 |
| 19 | competitive | 1,122 | 상대 때문에 능력이 떨어지면 특공 +2 | `ON_FOE_DROP_BOOST(spa,+2)` | TAG + 상호작용 | 없음 |
| 20 | thickfat | 1,077 | 불꽃/얼음 기술의 공격 ×0.5 (`breakable`) | `ATTACK_MULT_VS(fire\|ice,×0.5)` | EXEC_NOW | 없음 |

참고: 21위 unaware(천진)는 v6에서 이미 규칙으로 처리됨, 22위 moldbreaker는 `breakable` 면역 무시용으로 예약(아래 6장).

## 2. 처리 클래스

| 클래스 | 뜻 | 모델 쪽 연결 |
|---|---|---|
| EXEC_NOW | 이번 턴의 데미지, 스피드, 우선도, 면역, KO를 즉시 수치로 바꾸는 효과 | `Matchup`이 계산해서 기존 채널(배율, 데미지, KO, 스피드)에 반영 |
| EXEC_ENTRY | 교체 선택이라는 행동의 비용/이득 (등장 시 효과, 퇴장 시 회복) | 교체 후보별 입력(진입 피해 `entry_frac`와 같은 자리)에 열 추가 |
| EXEC_STATE | 조건이 충족되는 동안 켜지는 상태 | 관측된 휘발성 상태(`-start ... protosynthesisatk`)를 효과 열로 인코딩해 정확히 읽음 (가장 높은 능력치를 추정하지 않음) |
| TAG | 규칙으로 값을 못 내는 반응형 효과 | 원시 요소 멀티핫을 특성 임베딩 옆에 입력, 판단은 학습 |

## 3. 원시 요소 어휘 (초안 15개)

| 원시 요소 | 트리거 | 대상 | 매개변수 | 소스 플래그 |
|---|---|---|---|---|
| `IMMUNE_TYPE` | 피격 | 자신 | 타입 | breakable |
| `ABSORB` | 면역 발동 시 | 자신 | 회복 비율 또는 능력 변화 | breakable |
| `FULLHP_SURVIVE` | 피격 | 자신 | – | breakable |
| `BLOCK_FOE_DROP` | 상대가 일으키는 능력 하락 | 자신 | 능력 범위(전체/일부) | breakable |
| `SPEED_MULT` | 상시 | 자신 | 날씨/지형 조건, 배율 | |
| `BASEPOWER_MULT` | 기술 사용 | 자신 | 위력 조건, 배율 | |
| `ATTACK_MULT_VS` | 피격 | 자신(방어측) | 공격 기술 타입, 배율 | breakable |
| `PRIORITY_BONUS` | 기술 사용 | 자신 | 기술 분류, 우선도 변화 | |
| `BEST_STAT_BOOST` | 날씨/지형 조건 | 자신 | 조건, 배율(스피드는 별도) | |
| `ENTRY_FOE_STAT` | 등장 | 상대 | 능력, 변화량 | |
| `EXIT_HEAL` | 퇴장 | 자신 | HP 비율 | |
| `ON_FOE_DROP_BOOST` | 상대가 능력 하락을 일으킴 | 자신 | 능력, 변화량 | |
| `ON_HIT_BOOST` | 피격 | 자신 | 기술 타입, 능력, 변화량 | |
| `PP_DRAIN_FOE` | 상대 기술 사용 | 상대 | PP 양 | |
| `COPY_DANCE` | 상대/아군 춤 기술 | 자신 | – | |
| `RESIDUAL_RESTORE_BERRY` | 턴 끝 | 자신 | 확률/날씨 조건 | |

## 4. 커버리지 (상위 20종)

- 공개된 특성 슬롯 115,217개 중 상위 20종이 51,530개 = **44.7%**.
- 이 중 EXEC 클래스(NOW, ENTRY, STATE) 14종이 39,733개 = 상위 20종 출현의 **77.1%**, 공개된 특성 전체의 **34.5%**. TAG 6종이 11,797개 = 22.9%.
- 미공개 토큰(57,477회)은 별도: 위 효과는 상대의 특성이 공개되었거나 신념의 예측 확률이 있을 때만 적용할 수 있다.

## 5. 설계 판단이 필요한 곳 (검토 요청)

1. **상대 특성 미공개**: 전체 활성 슬롯의 약 33%. EXEC 효과를 신념(`hid_ability` 로짓) 확률 기대값으로 계산할지, 공개된 것만 계산할지. 기술 쪽에서 쓰는 "상위 12 후보 혼합" 방식의 특성판이 필요함.
2. **BEST_STAT_BOOST**: 가장 높은 능력치를 규칙으로 추정하기보다, 관측 프로토콜의 휘발성 상태 이름(`protosynthesisatk` 등)을 효과 열로 읽는 쪽이 정확함. 인코더 효과 열(`EFFECT_FLAGS`)에 추가가 필요하고, 데이터 재빌드가 필요함.
3. **intimidate와 defiant/competitive/clearbody의 상호작용**: 위협으로 공격이 떨어지면 분발은 +2, 클리어바디는 무효. 교체 후보 평가 시 "후보가 위협이고 상대 특성이 이 중 하나"인 경우를 규칙으로 계산할지, 태그로 둘지.
4. **EXEC_ENTRY의 입력 자리**: 교체 후보별 열에 `exit_heal`(내 활성)과 `entry_foe_atk_drop`(후보)를 추가하면 후보 특징 폭이 바뀜.
5. **변화기 적용 가능성**: prankster 악 타입 실패, 위협의 대타 무효 같은 규칙은 이 어휘 하나로 안 끝나고 "기술이 먹히는가" 판정표와 함께 만들어야 함 (별도 문서).

## 6. 이 어휘로 바로 확장되는 예약 원시 요소 (상위 20종 밖)

`IGNORE_TARGET_ABILITY`(moldbreaker 22위: `breakable` 면역/보정을 무효화), `STATUS_IMMUNE(status)`(불면 등), `REFLECT_STATUS`(magicbounce), `ATTACK_MULT`(hugepower, guts), `DAMAGE_MULT_COND`(multiscale 등), `ON_CONTACT_STEAL_ITEM`(magician).

## 7. 다음 단계 제안

1. 위 15개 어휘와 5장 결정 사항을 검토해 확정.
2. EXEC 14종을 구현 순서로 정함: EXEC_NOW(스피드/위력/면역/KO) → EXEC_ENTRY → EXEC_STATE.
3. 각 규칙마다 소스 값과 대조하는 단위 테스트를 같이 작성 (예: technician은 위력 60 경계, swiftswim은 비 계열 날씨만).
4. 통과선(사전 등록): 날씨 연동 특성 묶음의 일치율 격차(−4.3pp) 축소, 특성 소거 민감도 상승, 변화기 선택 일치율(0.535) 개선, 전체 KL 악화 없음(+0.005 이내), 같은 데이터로 합법 비트 수정만 한 `v6` 대조군과 비교.
