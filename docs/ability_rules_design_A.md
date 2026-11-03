# A단계 구현 설계: 특성 EXEC 규칙 (`matchup_v7` + `model_v7`)

상태: **구현 완료(코드와 테스트), 학습·평가 전.** 아래 "10. 구현 현황" 참고. 규칙의 내용과 근거는 [ability_rules_draft.md](ability_rules_draft.md)(v1.1). 이 문서는 어떤 파일에 어떻게 구현하는지를 정한다.

## 1. 원칙

- 모든 규칙은 `torch.no_grad()` 안의 계산이다(파라미터 없음). 세트 추정 출력은 이미 detach이므로 학습 경로가 늘지 않는다.
- 입력은 **저장된 관측 텐서와 세트 추정 출력뿐**: 특성/도구 id, 상태이상, 랭크, 기술 위력/분류/타입/우선도, `field_vec`. 데이터 재빌드 없음, 서버 인코더 변경 없음.
- 현행 `v6_full`(서버 탑재)이 `matchup.py`와 `model_v6.py`를 쓰므로 그 출력은 **바뀌면 안 된다**. 그래서 기존 파일을 고치지 않고 새 파일로 만든다.

## 2. 파일 배치

| 파일 | 내용 | 비고 |
|---|---|---|
| `src/core/matchup_v7.py` (신규) | `Matchup` 독립 복사본 + A단계 규칙. 상속 없음 | v6 퇴역 시 `matchup.py`는 삭제 |
| `src/core/model_v7.py` (신규) | `model_v6.py` 독립 복사본 + 새 입력 열 + 세트 추정 배관 + 합법 비트 수정 | cfg `"model": "v7"`, `build_model`에 분기 추가 |
| `tests/test_matchup_v7.py` (신규) | 규칙별 소스 경계값 + 중립 회귀 테스트 | |
| `tests/test_model_v7.py` (신규) | 롤아웃 == 시퀀스, 새 열이 출력에 닿음, 체크포인트 왕복 | v6 테스트에서 복사 후 수정 |
| `deploy/*.sh` | 파일 목록에 `matchup_v7`, `model_v7` 추가 | 배포 시점에 |

v7이 서버에 배포되고 v6가 퇴역하면 `model_v6.py`, `matchup.py`를 지운다.

## 3. `MatchupV7`의 API

```python
class MatchupV7(nn.Module):
    def __init__(self, vocab_path, chart_path)         # entry 옵션 없음: v7은 항상 진입/천진/A단계 규칙을 씀
    def forward(self, my_cat, my_num, my_mv, opp_cat, opp_num, opp_mv,
                field_vec, opp_ab_prob, opp_item_prob, opp_mult=None):
        -> {"def": [B,6,24,4], "off": [B,6,24,4],            # 규칙이 반영된 배율/데미지/KO(확률)/공격기
            "spd_my": [B,6], "spd_opp": [B,6],               # 보정된 스피드 비율
            "fm_my": [B,6,4], "fm_opp": [B,6,4],             # 기술 칸별 선공 기대값
            "entry_opp": [B,6,3],                            # 후보별 상대 활성의 공격/특공/스피드 배율
            "entry_frac": [B,6],                             # 기존 장판 진입 피해 (v6와 같은 값)
            "exit_heal": [B]}                                # 내 활성 교체 시 회복량
```

- `opp_ab_prob` [B,6,A]: 상대 6마리 각각의 특성 분포(원핫 또는 세트 추정). `opp_item_prob` [B,6]: 상대 각 마리의 구애스카프 확률. 둘 다 모델이 만들어 넘긴다(7장).
- 내 쪽 특성/도구는 `my_cat`의 id로 원핫을 안에서 만든다(항상 공개).
- `entry`(장판 진입 피해)는 기존 `entry_frac`을 그대로 복사해 `forward`가 같이 계산한다. 별도 인자를 받던 구조를 없애서 호출부를 단순화한다.

## 4. 규칙 표 버퍼 (어휘 크기 `A = vocab ability + 3`)

생성 시 `data/vocab.json`의 이름으로 한 번 채우는 비학습 버퍼:

| 버퍼 | 형태 | 값 |
|---|---|---|
| `F_def` | [A, T] | 방어측 특성 × 기술 타입 배율. 기본 1. 면역(`levitate`, `eartheater`, `flashfire`, `wellbakedbody`, `waterabsorb`, `dryskin`, `stormdrain`, `voltabsorb`, `lightningrod`, `motordrive`, `sapsipper`)은 해당 타입 0, `thickfat`은 불꽃/얼음 0.5 |
| `is_tech`, `is_sturdy`, `is_prank`, `is_intim`, `is_regen` | [A] | 0/1 |
| `speed_w` | [A, W] | 날씨별 스피드 배율. W = (쾌청, 비). `chlorophyll`은 쾌청 2, `swiftswim`은 비 2, 나머지 1 |
| `intim_d` | [A, 3, 4] | 위협 입장 시 상대 특성이 a일 때 (공격/특공/스피드)의 랭크 변화가 `{−1, 0, +1, +2}` 중 무엇인지 원핫 (기본: 공격 −1). 표는 draft 3장 "상대 특성 결과 표" |
| `bad_ids` | [A] | 특수 칸(빈칸, `<unk>`, 숨김): 확률 계산에서 제외 |

`F_def`가 기존 `IMMUNE_ABILITIES`의 일반화이므로 v6와 동일한 입력(공개 특성)에서는 면역 결과가 같다. 다른 점은 상대 특성이 미공개일 때 확률 기대값이 들어간다는 것.

## 5. 규칙별 구현 (위치와 식)

### 5-1. `_pair` (공격측 → 방어측 6마리, 기존 구조)

기존 흐름(`A`, `D`, `stab`, `eff`, `dmg`, `ko`)에서 바뀌는 곳:

1. **상성/면역/두꺼운지방**: `eff = chart(...)`까지는 같고, 기존 `immune` 처리를 아래로 교체.
   ```
   Mt[b,i,t] = Σ_a q_def[b,i,a] · F_def[a,t]          # [B,6,T], 방어측 6마리 × 기술 타입
   eff = eff * Mt.gather(-1, mt_expand)                # 면역은 0, thickfat은 0.5, 확률이면 기대값
   ```
   `q_def`: 내 쪽이 방어측이면 원핫(`my_cat`), 상대 쪽이면 `opp_ab_prob`.
2. **technician**: `p_tech = q_att · is_tech` ([B,6]) → `bp_eff = bp * (1 + 0.5 * p_tech_rep * (bp <= 60))`.
3. **sturdy**: `p_surv = (q_def · is_sturdy) * (def_hp_frac >= 1.0)`. `ko = (dm & dmg >= cur) * (1 − p_surv)` (확률 값).
4. **intimidate 후보 보정**: 상대 → 내 후보 `_pair`에서 `A`(상대 공격측 능력치)에 후보별 배율을 곱한다. 배율은 5-3의 `entry_opp[...,0:2]`(물리/특수).

### 5-2. 스피드/선공 (`fm_*`, `spd_*`)

```
eff_spe(side) = base_spe(_stats, 세트 추정 mult) * stage(spe 랭크) * (par ? 0.5 : 1)
              * (scarf ? 1.5 : 1) * (tailwind(side) ? 2 : 1) * weather_ability_mult
```
- 내 쪽은 확정값. 상대 쪽 불확실 요인 3개(스카프 `p_s`, 날씨 스피드 특성 `p_w`, prankster `p_p`)를 **이진 조합 8개**로 열거하고 조합 확률의 곱으로 가중한다. 텐서 형태 `[B,6,4,8]`.
- 비교: 유효 우선도(기술 우선도 + prankster 변화기 +1)가 다르면 등급만으로 결정, 같으면 유효 스피드 비교(트릭룸이면 반전), 동률은 0.
- `fm_my[b,i,j]`: 내 포켓몬 i의 기술 j가 **상대 활성**보다 먼저일 기대값. `fm_opp[b,o,j]`: 상대 포켓몬 o의 드러난 기술 j가 **내 활성**보다 먼저일 기대값. 안 드러난 기술 칸은 0.
- prankster 변화기가 악 타입 대상이면 0(실패).
- `spd_my/spd_opp`: 같은 유효 스피드(기대값 조합)로 계산한 log2 비율.

### 5-3. 위협 `entry_opp` [B,6,3]

각 교체 후보 i(비활성)에 대해: 후보가 intimidate면 상대 활성의 특성 분포 `q_opp_act`로 `P(δ) = q @ intim_d[:, k, :]` (k = 공격/특공/스피드, δ ∈ {−1,0,+1,+2}), 현재 랭크 s에서 배율 `R(δ) = stage(s+δ)/stage(s)` → `entry_opp[b,i,k] = Σ_δ P(δ) R(δ)`. 후보가 intimidate가 아니면 1. 활성 포켓몬은 1.

### 5-4. `exit_heal` [B]
`exit_heal[b] = (q_my_active · is_regen) * min(1/3, 1 − hp_frac_active)`.

## 6. 새 입력 열과 차원 변화 (`model_v7`)

| 항목 | v6 | v7 |
|---|---|---|
| 기술 토큰 추가 수치 `mv_extra` (토큰마다) | 3 (배율, 데미지, KO) | + `fm` 1 → `MV_EXTRA` 4 (`move_proj` 입력 +1) |
| 교체 후보 특징 `CAND_N` | 12 | + `entry_opp` 3 = 15 |
| 전역 요약 `G_DIM` | 합법 요약 25 + 효과 14 = 39 | + `exit_heal` 1 = 40, 단 합법 비트의 교체 슬롯 6칸 제거(−6): 합법 요약은 비트 8(기술 4 + 테라 4, 교체는 개수만) + 6 플래그 + 5 = 19 → 최종 19 + 14 + 1 = 34 |
| 포켓몬 토큰 `MON_X` | 16 | 변화 없음 |

합법 비트 수정: 합법 비트 14개(기술 4, 테라 4, 교체 6)에서 교체 6칸 슬롯 순서를 빼고 이미 있는 `교체 개수`로 대체한다. 슬롯별 교체 가능 여부는 포켓몬 토큰의 "교체 가능 플래그"가 이미 주므로 정보 손실은 없다. 이렇게 하면 `my_pos=False`의 순열 대칭이 모든 마스크에서 정확히 성립한다(테스트 추가).

## 7. 세트 추정 배관 (`model_v7._encode_turn`)

```
logits = belief(opp_cat, opp_num)["ability"]               # [B,6,A'] (detach된 출력)
q = softmax(logits with bad_ids masked)                    # 특수 칸 제외
revealed = opp_cat[..., 1] 이 어휘 안의 정상 id
opp_ab_prob = where(revealed, onehot(opp_cat[...,1]), q)   # 공개/종 후보 1개는 원핫
p_scarf = softmax(item logits, 특수 칸 제외)[choicescarf]   # [B,6]
```
- 이미 `_top_guess`가 기술에 대해 같은 방식을 쓰고 있어 구조가 일치한다.
- 새 학습 대상이나 손실은 없다.

## 8. 구현 순서와 확인

1. `matchup_v7.py` 작성 + `tests/test_matchup_v7.py`: 먼저 **중립 회귀 테스트**(표를 전부 중립으로 두면 `def/off/spd`가 `matchup.py`의 v6 출력과 같다), 그다음 규칙별 소스 경계값(draft 6장).
2. `model_v7.py`: v6 복사본 + 새 열 + 배관 + 합법 비트 수정. `tests/test_model_v7.py`(롤아웃 == 시퀀스, 순열 대칭, 새 열 도달, 체크포인트 왕복).
3. `build_model`에 `"v7"` 분기, `train_fp_distill.py` 기본 구조를 v7로, 학습 데이터 호환 확인(필요 열 모두 기존 npz에 있음).
4. 소규모 학습 스모크, 이어서 `v7_full`(전체 데이터, 6에폭, 약 2.5시간).
5. 평가: draft 7장 통과선 + 입력 소거 민감도 + 변화기/교체 선택 일치율 + `tools/field_eval.py`/`entropy_eval.py`.
6. 통과하면 서버 배포(같은 절차), v6 퇴역 계획 수립 (v4 legacy는 2026-10-25에 이미 삭제).

## 9. 위험과 한계

- **비교의 불순함**: v7 = A단계 규칙 + 합법 비트 수정 + 새 열. 대조군 `v6_full`과 한 번에 여러 가지가 다르다. 규칙 효과만 따로 보려면 규칙을 중립으로 둔 v7(= 합법 비트 수정만)을 한 번 더 학습해야 한다(2.5시간 추가). 먼저 v7 한 번으로 방향을 보고, 효과가 있으면 분리 학습한다.
- **저장된 45번 열(`priority_speed_edge`) 중복**: 새 `fm_edge`와 겹치지만 첫 v7에서는 그대로 둔다(재빌드 없이 규칙 효과를 분리하기 위해). 다음 데이터 재빌드 때 제거.
- **스카프 규칙은 효과 기대 없음**: 상대 도구 세트 추정이 스카프를 거의 못 짚는다(draft 3-3장). 기대값 형태로만 넣는다.
- **지연**: 새 계산은 `[B,6,24,4]`와 `[B,6,4,8]` 수준의 작은 텐서라 서버 응답(50~100ms)에 몇 ms 추가될 것으로 예상(측정 필요).
- **표 정확성**: `F_def`, `intim_d`, `speed_w` 표는 소스와 대조한 값만 넣는다. 값이 틀리면 정확한 규칙이 틀린 정보가 되므로 테스트로 고정한다.
- **미구현으로 남는 규칙**: B단계(고대활성/쿼크차지, 인코더 변경과 재빌드), 흡수 이득(물먹기 회복 등), `breakable`(틀깨기 무시).

## 10. 구현 현황 (2026-10-25)

구현됨: `src/core/matchup_v7.py`, `src/core/model_v7.py`, `build_model`의 `"v7"` 분기, `tests/test_matchup_v7.py`, `tests/test_model_v7.py`. 기본 학습 구조(`train_fp_distill.py`)는 아직 v6 그대로(채택 전). 배포 파일 목록에도 아직 없음.
학습 명령: `ARCH='{"model":"v7","my_pos":false}' bash logs/train_full.sh v7_full v6_full` (또는 `train_fp_distill.py --arch '{"model":"v7","my_pos":false}'`).

설계와 달라진 점:
- 특성 표 폭은 `<unk> + 2`(세트 추정 특성 로짓 폭 319)로 맞춤 (처음에 `max id + 3 = 320`으로 해서 세트 추정과 크기가 어긋났음).
- 정밀기술 위력 경계는 반올림 후 비교 (위력/200 후 ×200의 부동소수 오차로 60이 60.000004가 되어 규칙이 안 걸렸음).

검증:
- 테스트 28개 전부 통과. 중립 회귀(표를 중립으로 두면 v6 `Matchup` 출력과 같음), 규칙별 경계값, 세트 추정 기대값, `my_pos=False` 순열 대칭이 부분 마스크에서도 정확히 성립함을 확인.
- 기존 `v1_full_r3`, `v4_full_tv`(이후 삭제), `v6_full` 출력 차이 0.0 (새 파일만 추가했고 기존 파일 동작 불변).
- 실제 데이터(dg5) 점검: 모든 규칙 출력이 유한하고 범위 안. 8,000개 결정에서 내 활성이 재생력인 경우 129개(HP가 깎여 회복량 > 0인 경우 84개), 위협 교체 후보(`entry_opp` ≠ 1) 629개, `fm_my`는 +1 46.4% / −1 52.6% / 중간 1.0%.
- 규모: 파라미터 8.038M (v6 8.040M). 한 턴 CPU 1스레드 추론: v6 21.7ms → v7 24.9ms (+3.2ms, 이 PC 기준, 서버에서는 따로 측정 필요).
- 1에폭 학습 스모크(d6_ho): v6와 v7 모두 정상 학습.

**후속 (2026-10-25):** v7_full은 학습/평가 결과 v6_full과 같았음 (SPRT 50.6%, Foul Play 500판 36.4% 대 37.8%). 이후 model_v7.py/	ests/test_model_v7.py는 model_v8.py/	ests/test_model_v8.py로 이름을 바꾸고 숨은 특성/도구 기대 임베딩과 융합층 활성 토큰 입력을 더함 (uild_model의 키도 8). matchup_v7.py는 그대로. 기존 v7_full 체크포인트는 구조가 달라 더 이상 로드되지 않음.
