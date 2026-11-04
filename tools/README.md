# tools/ 자주 쓰는 명령어

모든 명령은 `poke-ai` 폴더에서 bash로 실행 (경로는 `C:/Users/lsh/Desktop/orcl_server`, 파이썬은 kosa conda 환경 `/c/Users/lsh/anaconda3/envs/kosa/python.exe`).
Foul Play(Docker) 관련 스크립트는 Docker 실행 중 + 이미지 `fp-runner` 필요. Showdown 서버(:8000)는 `fp_selfplay.sh`·`arena.py` 실행 전에 미리 수동으로 켜둬야 함.

## 0. Showdown 서버 켜기/끄기 (수동 필요할 때)

```bash
cd pokemon-showdown && nohup node pokemon-showdown start --no-security > ../logs/showdown.log 2>&1 &
cd ..

netstat -ano | grep LISTENING | grep ":8000"     # 켜졌는지 확인

for pid in $(netstat -ano | grep LISTENING | grep ":8000" | awk '{print $5}' | sort -u); do
  taskkill //PID $pid //F
done                                              # 끄기
```
작업 끝나면 항상 꺼둘 것.

## 1. 팀 풀 준비 (한 번만, 또는 풀 바뀔 때만)

```bash
node tools/filter_teams.js --dir data/metamon/teams/gen9ou --format gen9ou --holdout 500 --seed 42
# → data/team_pool_metamon.json (학습), data/team_pool_metamon_holdout.json (평가 A)

node tools/make_party_pools.js --n 500 --seed 7
# → data/team_pool_rare.json, data/team_pool_randomset.json (평가 전용 B/C)

node tools/make_party_pools.js --seed 11 --suffix _train --plain \
  --avoid data/team_pool_rare.json,data/team_pool_randomset.json --train-a 2000
# → data/team_pool_rare_train.json, data/team_pool_randomset_train.json, data/team_pool_train_a.json (학습용, 평가 풀과 안 겹침)
```

새 학습 풀 (2026-09-28, 기존 풀이 팀당 10~43회 재사용돼서 교체. 평가 풀 및 이전 학습 풀과 겹치지 않음을 확인함):
```bash
node tools/make_party_pools.js --n 2000 --seed 17 --suffix _train3 --plain --train-a 3000 --train-a-out data/team_pool_train_a2.json   --avoid data/team_pool_rare.json,data/team_pool_randomset.json,data/team_pool_rare_train.json,data/team_pool_rare_train2.json,data/team_pool_train_a.json
# → data/team_pool_train_a2.json(3000) / rare_train3.json(2000) / randomset_train3.json(2000)
python tools/export_fp_teams.py data/team_pool_train_a2.json train_a2
python tools/export_fp_teams.py data/team_pool_rare_train3.json rare_train3
python tools/export_fp_teams.py data/team_pool_randomset_train3.json randomset_train3
```
옛 randomset_train/randomset_train2 풀 파일과 FP 폴더는 삭제함(C는 랜덤 세트라 재생성). train_a/rare_train/rare_train2는 비교 실증용으로 유지.

Foul Play가 team-preview에 쓸 팀 폴더로 내보내기 (풀 새로 만들 때마다):
```bash
python tools/export_fp_teams.py data/team_pool_train_a.json train_a
python tools/export_fp_teams.py data/team_pool_rare_train.json rare_train
python tools/export_fp_teams.py data/team_pool_randomset_train.json randomset_train
```

엔티티(도구/특성/기술) 특징 표 재생성 (Showdown 데이터 바뀌었을 때만):
```bash
node tools/dump_entities.js
python tools/build_entity_features.py
```

## 2. BC 학습

BC 학습 코드(`train_bc.py`)는 정리해서 삭제함. 현재 계보는 Foul Play 증류만 이어가며 BC 원본 체크포인트도 삭제함. 코드가 필요하면 GitHub 커밋 `ee0df15`에 있음.

## 3. Foul Play 데이터 수집 (탐색 교사 데이터)

**2026-09-28부터 탐색 300ms로만 수집한다.** 100ms는 라벨이 가끔 틀린다(예: 필드에 트릭룸이 이미 켜져 있는데
또 쓰는 걸 100ms 교사가 가끔 고름 → 300ms로 재생하면 정정됨, `tests/test_trickroom_guard.py` 관련 조사 참고).
기존 100ms 누적 데이터(`data/fp_selfplay/ms100/`)는 이미 쌓인 양이 많아서 버리지 않고 계속 학습에 섞지만,
**새로 수집하는 건 전부 ms300**. 시간이 3배라 판 수는 줄여서 조정.

FP vs FP 자가대전 (약한 상대 편향 없는 강한 데이터):
```bash
# 풀마다 한 번씩 실행. A는 판이 길고(평균 39턴 vs 25~30턴) 1쌍이라 전체 소요시간을 결정함 → A는 B/C의 절반 남짓 판수(1쌍x400판 ≈ 4.2h < B/C 6.5h라 병목 아님, 판수 비중 ~7%) (sp12: A 667판이 7h, B/C 667판x4쌍은 6h)
# (2026-09-28 풀 교체: train_a→train_a2 / rare_train2→rare_train3 / randomset_train*→randomset_train3, 아래 "새 학습 풀" 참고)
bash tools/fp_selfplay.sh 1 400 300 sp13a train_a2
bash tools/fp_selfplay.sh 4 700 300 sp13b rare_train3
bash tools/fp_selfplay.sh 4 700 300 sp13c randomset_train3
```


### Uber/AG 종 커버 데이터 (OU에서 금지된 종, 2026-09-29 준비)
Metamon 등 공개 데이터에는 Gen9 Ubers/AG가 없어서 직접 만든다. 형식은 `gen9anythinggoes`(Uber+AG 종 모두 합법). 팀은 랜덤배틀 세트 방식(팀당 Uber/AG 종 3마리 이상).
```bash
node tools/make_party_pools.js --n 300 --seed 22 --suffix _uber_eval --plain --format gen9anythinggoes --uber-min 3 --only-randomset 1
node tools/make_party_pools.js --n 2000 --seed 23 --suffix _uber_train --plain --format gen9anythinggoes --uber-min 3 --only-randomset 1 --avoid data/team_pool_randomset_uber_eval.json
python tools/export_fp_teams.py data/team_pool_randomset_uber_train.json uber_train
python tools/export_fp_teams.py data/team_pool_randomset_uber_eval.json uber_eval
# 수집 (FORMAT 환경변수로 형식 지정; 평가용 uber_eval은 학습에 쓰지 말 것)
FORMAT=gen9anythinggoes bash tools/fp_selfplay.sh 4 <판수> 300 <태그> uber_train
```
- 주의: Foul Play가 부가 데이터(`data.foulplay.cc/gen9anythinggoes/...`)를 못 받아서(404) 상대 세트 추정이 OU보다 약함. Smogon AG 사용률 통계(2026-08)는 자동 다운로드됨.
- 풀에 없는 금지 종 8개는 전부 폼 변화(Dialga-Origin, Palkia-Origin, Giratina-Origin, Magearna-Original, Zacian-Crowned, Zamazenta-Crowned, Palafin-Hero, Ogerpon-Hearthflame). 기본 종은 들어 있음.

## 4. 증류 데이터셋 빌드 + 학습

```bash
# 자가대전은 컨테이너 자신이 protocol 레코드를 남기므로 상대편 기록이 필요 없음
python src/training/build_fp_dataset.py --decisions "data/fp_selfplay/ms300/dec_sp12[abc]_*.jsonl" \
  --out data/fp_selfplay/ms300/fp_data_sp12.npz

python src/training/train_fp_distill.py --data "data/fp_selfplay/ms300/fp_data*.npz,data/fp_selfplay/ms100/fp_data*.npz" \
  --out checkpoints/supervised_v2_fp_next.pt   # 처음부터 학습: 피크 lr 1e-3 + 웜업 5% + 코사인 감쇠, 6에폭, 구조 v8. 세트 추정은 checkpoints/belief_<NUM_DIM>.pt(pretrain_belief.py)를 불러옴
```
수집이 끝난 원본 결정기록(`dec_*.jsonl`)은 `python tools/compress_raw.py <폴더>`로 압축해 보관한다(해시 검증 후 원본 삭제, 약 12배 작아짐). 빌더는 같은 패턴(`dec_sp12[abc]_*.jsonl`)으로 `.gz`도 읽는다.
npz 데이터셋은 `savez_compressed`로 저장한다(빌더·holdout 기본). 예전 무압축 npz는 `python tools/compress_npz.py <파일들>`로 무손실 변환(배열 동일 확인 후 교체, 약 60배 작아짐, 로드 1~2초).

증류는 누적 전체 데이터로 처음부터 학습한다 (워밍스타트와 KL 항은 지움: v6/v7/v8 모두 처음부터, KL 0). `--out`이 기존 체크포인트 이름이면 덮어쓰니 다른 이름을 쓸 것.

## 5. 평가

```bash
# 모델끼리 승률 비교 (SPRT): python tools/arena.py <새모델> <기준모델> --delta 0.03  (README 본문 '평가 체계' 참고)
```

평가는 항상 holdout(A)/rare(B)/randomset(C) 풀에만. 학습 풀(train_a*, rare_train*, randomset_train3)로 평가하지 말 것.

### 5-1. 범주별 평가 (v5 합법성 입력, v6 진입 후 매치업·효과 입력)

데이터는 build_fp_dataset이 서버 요청(`|request|`)으로 만든 합법 마스크를 쓰고, 행동 제약 효과 열(field_vec 뒤쪽 14열)도 같이 만든다. 옛 npz(마스크가 규칙 기반, 효과 열 없음)로는 v5/v6의 입력이 비어 있으니 다시 빌드할 것.
```bash
# 합법성 범주(기술 4개/2~3개/1개만/교체 봉쇄) + 제약 효과 범주(도발·앵콜 등)별 교사/학생 교체 확률, 종류 KL, 총 KL, AUC와 기준 모델 대비 차이의 판 단위 부트스트랩 95% 구간
python tools/legal_eval.py --models v6_full,v8_full --base v6_full --sets sp4=data/fp_selfplay/ms100/fp_data_sp4.npz,dg5=data/dagger/dg5.npz
# 진입 후 매치업 범주(상대 +2 이상 & 천진 후보, 스텔스록 & 약점/저항 후보): 특징 포켓몬으로의 교체 질량이 교사와 얼마나 같은가 (편향·평균 |오차|)
python tools/entry_eval.py --models v6_full,v8_full --base v6_full --sets sp4=data/fp_selfplay/ms100/fp_data_sp4.npz,dg5=data/dagger/dg5.npz
# 필드/도구/상태 범주(트릭룸, 순풍, 벽, 날씨, 지형, 구애·증폭 도구, 마비)별 KL과 일치율을 나머지와 비교 (학습에 안 쓴 학생 상태 세트만 쓸 것)
python tools/field_eval.py --models v6_full --sets data/dagger/dg5.npz,data/dagger/d6_ho.npz
# 교사 확신도(최상위 확률)별·교사 선택 종류별 KL/일치 (학생의 격차가 확신하는 결정에 몰려 있는지)
python tools/entropy_eval.py --models v6_full --sets data/dagger/dg5.npz,data/dagger/d6_ho.npz
# regret: 학생 argmax가 교사 탐색의 행동별 가치(Q)로 얼마나 손해인가 (일치율은 근소한 차이 선택도 감점이라 가치 손실로 재측정). 교사 argmax 기준선 포함, 2개 모델이면 판 단위 쌍 비교
# q_mean/q_vis가 든 npz만 가능 (build_fp_dataset이 항상 저장. 옛 홀드아웃은 원본 결정기록으로 재빌드)
python tools/regret_eval.py --models v6_full,v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz
# regret 분해: 학생 regret이 어떤 결정에 몰려 있나 (교사 선택 종류/턴/벤치 수/강제 교체/결정 중요도/교사 확신/드러난 상대 기술 수별 평균 regret과 전체 몫)
python tools/regret_breakdown.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz
# 기능별 혼동표: 교사 대 학생의 행동 기능(공격/셋업/회복/설치/보호/교체/테라 ...)별 정책 질량, 불일치율, regret 몫, 기능 간 혼동 (자발 결정, 날카로운 결정은 따로). CPU로 됨
python tools/move_confusion.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz
# 사례 분석: 교사가 확신한 교체/테라를 학생이 맞힌 결정과 놓친 결정을 한 수 앞 규칙 상황(KO 위협, 선공, KO 가능, 벤치 상성 차)으로 비교하고, 규칙 점수와 학생의 교체 판별 AUC를 견줌. CPU로 됨
python tools/case_mine.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz
# 한 수 앞 규칙 대 학생: 최대 데미지 규칙 기준선과 학생의 기술 일치/regret 비교, 규칙이 확실한 상황(선공 확정 KO, 반감 기술 회피)에 개입했을 때 학생 대비 regret 쌍 비교. CPU로 됨
python tools/rule_baseline.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz
# 실패 사례 읽기: 학생이 크게 틀린 결정(regret >= 0.15)을 무작위로 N개 뽑아 활성/벤치/필드/교사와 학생의 상위 행동과 Q를 읽을 수 있게 출력. CPU로 됨
python tools/failure_cases.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz --n 30 --seed 0
# 교체 이유 분석: 상대 활성의 공개 기술 수별 교체 판별 AUC/regret, 세계(상대 세트 표본)별 교체 이득의 일관성. 세계별 가치는 world_q.py가 원본 결정기록에서 따로 만듦 (d7/d6만 가능)
python tools/world_q.py --decisions "data/dagger/raw/d7/lab_rec_*_h*.jsonl" --npz data/dagger/d7_ho.npz --out data/dagger/d7_ho_worlds.npz
python tools/switch_why.py --models v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz
# 쌍 비교: 같은 결정에서 두 모델의 KL/일치/regret 차이를 "규칙이 닿는 결정" 범주(특성, 위협 후보, 스피드 상황)별로 판 단위 부트스트랩 (묶음별 표보다 민감). q가 든 npz만 가능
python tools/paired_eval.py --a v6_full --b v8_full --sets data/dagger/d7_ho.npz,data/dagger/d6_ho.npz,data/dagger/dg5.npz
# 교사 잡음 상한: 같은 상태를 Foul Play가 두 번 라벨링한 결정기록 두 벌의 일치율/KL/regret (학생이 낼 수 있는 값의 기준). 재라벨은 logs/ceiling_run.sh (탐색이 시간 기반이라 다른 CPU 작업 금지)
python tools/teacher_ceiling.py --a "data/dagger/raw/d7/lab_rec_*_h*.jsonl" --b "data/dagger/raw/d7b/lab_rec_*_h*.jsonl"
```

### 5-2. 행동 사건 추출 + 손계산 오차 진단 (결과 예측기 1단계)

자가대전 기록에서 기술/등장 사건(직전 상태 → 결과)을 뽑고, 모델의 손계산(Matchup과 같은 근사)이 실제 피해와 어디서 어긋나는지 특성/도구/기술/날씨/벽/랭크별로 집계한다.
메커니즘 표에 무엇부터 넣을지, 학습된 보정이 얼마나 필요한지를 여기서 정한다. 사건 추출은 표준 라이브러리만 써서 빠르고, 진단만 poke-env가 필요함.
```bash
python tools/extract_events.py --decisions "data/fp_selfplay/ms300/dec_sp12[abc]_*.jsonl" --out data/events/sp12.jsonl.gz   # --max-battles 500 으로 빠르게 맛보기
python tools/damage_residual_report.py data/events/sp12.jsonl.gz --min-n 30 --tsv logs/damage_residual_sp12.tsv              # --stats standard: 노력치 표준 가정과 비교
```
읽는 법: 그룹의 "실제/손계산"이 1에서 멀고 n이 클수록 손계산이 놓친 메커니즘 (개별 |r| 0.1 안팎은 난수/HP 반올림 잡음). KO는 따로("최대 난수 손계산이 KO를 예측한 비율"), 장판 피해와 위협 발동 결과도 함께 찍힘.

## 6. 검증 표준 (학생 상태 홀드아웃 + 직접 대전)

교사 상태 검증(학습 스크립트가 찍는 값)은 강도를 따라가지 못함 → 1차 검증은 **학생 상태 홀드아웃**, 최종 판정은 **직접 대전**.
```bash
# 1차 (약 30초): 학생이 방문한 상태에서 교사와의 일치율/CE (높을수록/낮을수록 좋음). 기준: all5d300 0.485/1.5596, all5x 0.497/1.5500, all5h 0.500/1.5335
python tools/eval_on_npz.py checkpoints/supervised_v2_fp_XXX.pt data/holdout/holdout_all.npz    # 그룹(dg2 dg3 dg4 dg5 fp)별 한 줄. 홀드아웃은 이 통합 파일 하나: 새 라운드는 python tools/merge_holdouts.py dgN=data/dagger/dgN_ho.npz 로 추가

# 최종 (Showdown 서버는 없으면 알아서 띄움): 후보 대 기준 SPRT 대전. 결론이 나는 즉시 멈추고(δ 3%p, 상한 6000판), 풀별 판 수를 맞춘 데이터로 판정. 기록은 results/ledger.jsonl
python tools/arena.py <후보> <기준> --delta 0.03 --cap 6000    # 이름 = checkpoints/supervised_v2_fp_<이름>.pt

# DAgger 데이터 만들기: tools/record_games.py (학생 자가대전 기록) → foul-play/fp/replay_label.py (컨테이너에서 재생 라벨링) → build_fp_dataset.py
```

### 타겟팅 DAgger (약점 정조준)

무작위 자가대전 DAgger는 저번에 일반 자가대전 데이터와 차이가 없었음(학생이 원래 자주 가는 상태만 도니까 겹침).
이번엔 (1) 니치 아키타입 팀 풀로 그 상황을 강제로 많이 만들거나, (2) 학생이 스스로 불리하다고 느낀(연속으로 value 음수) 판만 골라서 재라벨링에 씀.

```bash
# (1) 트릭룸: 전용 팀 풀로 자가대전 (data/team_pool_dagger_tr.json, 트릭룸 비중 22.8%)
python tools/record_games.py <ckpt> <ckpt> <판수> <태그> data/team_pool_dagger_tr.json <출력접두사>

# (2) 불리한 판만 필터링: value가 threshold 이하로 min_consecutive턴 이상 연속인 판만 통과
python tools/filter_dagger_hard.py <출력접두사>_a.jsonl <출력접두사>_a_hard.jsonl -0.2 3
python tools/filter_dagger_hard.py <출력접두사>_b.jsonl <출력접두사>_b_hard.jsonl -0.2 3
# 이후는 기존 DAgger와 동일: replay_label.py로 재라벨링 → build_fp_dataset.py
```

## 7. 강화학습 미세조정 (PPO, 2026-10-29)
증류된 정책에서 시작해 자가대전 승패로 개선. 롤아웃(`tools/rl_rollout.py`, 현재 정책 샘플링) → PPO 갱신(`src/training/train_rl.py`) → 체크포인트 저장을 반복하고, `--eval-every`마다 시작 모델과 arena SPRT 대전.
```bash
python src/training/train_rl.py --run rl1 --iters 20 --games 2000 --kl 0.1 --lr 3e-5   # Showdown은 없으면 직접 띄움. 출력: checkpoints/rl/rl1/it<k>.pt
```
- 이어 학습: `--start <이전 it*.pt> --base <원래 시작 모델>` (KL 앵커와 arena 기준이 원래 시작 모델로 유지됨, 옵티마이저 상태와 과거 상대 목록은 새로 시작)
- 손실: PPO 클립 + β·KL(현재 ‖ 시작 정책), advantage = 할인된 승패 − V(s_t). 세트 추정은 얼림, v6/v8(합법 마스크 입력)만 지원. 손실 self-check: `PYTHONPATH=. python tests/test_rl.py`
- 학습 팀은 train_a2/randomset_train3/rare_train3, 평가(arena)는 holdout/rare/randomset 풀. 작업자 일부(`--snap-frac`)는 과거 반복 체크포인트와 대전.
- 스모크 측정: 240판 약 2.2분(약 110판/분), 갱신 VRAM 최대 4.9GB(`--batch 16`, 긴 판이 많으면 `--batch`를 줄일 것).
- FP 상대 확인은 별도: `logs/hybrid_eval.sh` 모드 student (`CK=<ckpt>`). FP를 상대로 한 롤아웃은 아직 없음.

## 8. 사람 데이터 BC와 FP 변환기 (2026-10-30)
설계와 결과는 `logs/summary_20261030.md`, `logs/human_report.md`, 메모리 `poke-ai-fp-converter`. 한 번 돌리고 끝난 실행 스크립트는 `logs/archive/` (README에 목록).

### 사람 데이터 BC
```bash
# 파싱: Metamon 샤드 0 -> 양쪽 시점 시퀀스 npz (r1400 약 25분/21GB, r1600 약 10분, 이미 있는 조각은 건너뜀)
PYTHONPATH=. python tools/parse_human.py --shard 0 --min-rating 1400 [--out-dir data/human]
python tools/parse_diff.py <옛 폴더> <새 폴더> r1600 0 1      # 파서나 규칙을 바꿨을 때 옛 조각과 새 조각의 배열별 차이
# 학습 (v8, 사람 행동 22칸 NLL): logs/human_<태그>.txt, 시험 결정별 NLL logs/human_<태그>_test.npz
PYTHONPATH=. python src/training/train_human.py --tag <태그> [--init <ckpt>] [--epochs N] [--lr 1e-3] [--data "data/human/gen9ou_s0_r1400_*.npz"] [--arch '{"model":"v8","my_pos":false}']
python tools/human_compare.py <태그A> <태그B>     # 판 단위 쌍부트스트랩 (양수 = B가 더 좋음), 시험 판이 같은 것끼리만
python tools/human_table.py                       # logs/human_*.txt 전체 표
python tools/human_agree.py                       # FP로 증류한 학생과 사람 행동의 일치
```
태그에 대소문자만 다른 이름을 쓰지 말 것 (윈도우가 같은 파일로 봐서 결과를 덮어씀).

### FP 변환기 (사람 관전자 로그 -> FP가 재생하는 플레이어 시점 기록)
- `tools/human_to_fp.py`: `convert(log, side, user, tag, oracle=None, world=None)`. 내 쪽 세트(드러난 기술 + 사용률/세트 표), 절대 HP, `|request|`를 복원.
- `tools/set_posterior.py`: FP 완결 세트 표(`pokemon_full_sets.json`)를 로그 증거로 거른 세트 사후분포, `make_world`(월드 샘플). 단독 실행하면 진단.
- `tools/proxy_accuracy.py`: 사람 로그에서 드러난 도구/특성/테라 대 사용률 최빈 추정의 정확도 (정답을 아는 풀 팀과 나란히).
- `tools/fp_fidelity.py`: 충실도 측정.
```bash
python tools/fp_fidelity.py convert --src data/dagger/raw/d7 --recs rec_fp_h0,rec_fp_h1 --out data/fp_fidelity_<팔> [--oracle all|moves,item,ability,tera,stats | --map | --worlds 4] [--group 사람형]
# FP 재라벨: docker run --rm -v foul-play:/foul-play -v <out>:/rec -e FP_DECISION_LOG=/rec/lab_<rec>.jsonl --entrypoint python3 fp-runner -m fp.replay_label /rec/<rec>.jsonl 300
#   (foul-play 마운트 필수, 실행 중 다른 CPU 작업 금지, 예는 logs/archive/fp_fidelity*.sh)
python tools/fp_fidelity.py compare --a data/dagger/raw/d7 --a2 data/dagger/raw/d7b --b data/fp_fidelity_<팔> --recs rec_fp_h0,rec_fp_h1 [--worlds 0,1,2,3] [--only-group 사람형]
```
  기준선 A2 = 같은 상태의 FP 재라벨(교사 잡음). 지표는 top-1 일치, 평활 KL, argmax/평균 정책 regret, B/A2 비.
- `tools/critic_gate.py`: FP-Q 비판자 오프라인 관문 (얼린 v8_full latent 위 Q 헤드, `phi_check.py`와 같은 상관 정의). 결과: 정책 자체의 이점(r +0.17)이 비판자보다 강함.
- `tools/td_critic_gate.py`: 승패만으로 TD 학습한 Q 헤드(K=1 대 K=4 앙상블) 관문, 결과 logs/td_critic_gate.log (실패: 정책 logπ r +0.172 대 TD +0.02~0.06)
- `tests/test_hindsight.py`: `replay_dataset._hindsight` 회귀 테스트 (합성 로그 + 정답을 아는 판). 이 함수나 파서를 고치면 `PYTHONPATH=. python tests/test_hindsight.py`로 확인.
- `tools/human_vs_fp.py` (+ `logs/human_vs_fp.sh`): 사람 고수/중위 결정을 FP(300ms)로 재라벨해 사람 행동이 FP 분포/Q에서 어디에 놓이는지 잼. 기준선 = FP 자기 선택(같은 변환 잡음). 결과 `logs/human_vs_fp.txt`, 메모리 `poke-ai-human-vs-fp`.
- `tools/human_martingale.py`: 사람 행동의 Δ = V_FP(다음 결정) − Q_FP(둔 행동)를 regret 구간별로, 학생 행동 기준선과 비교 (FP가 사람 행동을 과소평가하는지). 결과 `logs/human_martingale.txt`.
- `tools/future_probe.py`: 동결 프로브 — v8_full 잠재가 t+1/3/6의 생존 수·HP 우위와 승패를 현재 요약/원시 특징 대비 얼마나 담는가 (학생 대 FP300 판, 사전 등록 판정). 결과 `logs/future_probe.txt`.
- `tools/where_lost.py`: 학생이 FP300에게 지는 판의 진행도별 V 곡선/regret 몫, V 0.4~0.6 결정의 남은 길이별 승률 부족분. 결과 `logs/where_lost.txt`.
- `tools/lost_cases.py`: 학생이 FP300에게 진 판에서 "실제 수가 크게 틀리고 다음 결정에서 FP 승률이 떨어진" 결정을 진행도 고르게 N개 읽을 수 있게 출력. 결과 `logs/lost_cases.txt`.
- `tools/forcing_eval.py`: 앙코르/도발/트집/사슬묶기 사용과 위협(Intimidate) 교체를 학생이 교사만큼 쓰는가 (질량, AUC, 일치, regret 몫). 결과 `logs/forcing_eval.txt`.
- `tools/decode_eval.py`: 칸 argmax / 종류 우선 디코딩 / 표집 regret 비교 (교체 약점이 디코딩 때문인가) + 위임 곡선 (불확실한 결정 상위 x%를 교사로 바꿀 때 regret 감소). 결과 `logs/decode_eval.txt`.
- `tools/kind_probe.py`: 기술/교체 종류 결정(type_head)이 후보별 정보를 잃는지 동결 프로브 (type_head 입력 vs + 후보 점수/은닉 풀링 AUC). 결과 `logs/kind_probe.txt`.
- `tools/equivariance_check.py`: 내/상대 팀 칸 순서를 섞어 정책이 같은지 점검 (내 칸은 정확히 등변이어야 함, 상대 칸은 공개 순서라 의도적으로 민감). 결과 `logs/equivariance_check.txt`.
- `tools/matchup_check.py` + `tools/sd_damage.js`: 매치업 모듈(MatchupV7)의 최대 난수 데미지를 Showdown `getDamage`(급소 없음, 같은 능력치/랭크/HP/도구/특성/테라/필드 입력)와 대조, 변형(필드 제외/도구 제외)으로 모듈이 안 쓰는 요인을 분리. 결과 `logs/matchup_check.txt`.
- `tools/gen_matchup_tables.js`: Showdown 데이터에서 `data/matchup_tables.json` 생성 (기술 태그 slicing/punch/sound/contact/bite/pulse/recoil/secondary/바디프레스/속임수/사이코쇼크, 진화 전 종, 종 기본 타입). 매치업 모듈이 읽음.
- `tools/build_dynamics_data.sh`: 자기 대전 원본 로그를 data/dyn/에 변환 (결정 상태/FP 정책·Q + turn, kind, opp_action, opp_move). 동역학 모델/상대 행동 예측용, 다음 상태는 같은 배틀의 다음 행.

### 장면 테스트 세트 (docs/scenario_suite_draft.md)
- `tools/scenario_gen.js`: 양쪽 행동을 스크립트로 고정해 Showdown 시뮬레이터로 위치까지 진행, p1 시점 메시지를 `fp.replay_label`용 rec jsonl로 저장 (rqid를 붙이고 마지막에 가짜 `|win|` 마무리). 마지막 단계 뒤 p1의 결정 요청이 대기 중인 상태가 테스트 장면.
- `data/scenarios/lib.py`: 세트 사전 + `scene()`/`write()` 도우미. 정의 파일 `data/scenarios/<이름>_defs.py`가 `<이름>_in.json`을 씀.
- `tools/scenario_run.sh <이름> [ckpt ...]`: 정의 -> rec -> FP 300ms 재생 라벨(docker) -> npz -> `tools/scenario_eval.py`(마지막 결정의 교사 분포 대 모델 분포). FP 라벨링은 시간 기반이라 다른 CPU 작업이 없을 때만.
