# tools/ 자주 쓰는 명령어

모든 명령은 `poke-ai` 폴더에서 bash로 실행 (경로는 `C:/Users/lsh/Desktop/orcl_server`, 파이썬은 kosa conda 환경 `/c/Users/lsh/anaconda3/envs/kosa/python.exe`).
Foul Play(Docker) 관련 스크립트는 Docker 실행 중 + 이미지 `fp-runner` 필요. Showdown 서버(:8000)는 `fp_selfplay.sh`·`analyze_model.py`·`arena.py` 실행 전에 미리 수동으로 켜둬야 함.

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

BC 학습 코드(`train_bc.py`)는 정리해서 삭제함. 현재 계보는 Foul Play 증류만 이어가며 BC 원본 체크포인트는 `checkpoints/supervised_v2.pt`로 보존. 코드가 필요하면 GitHub 커밋 `ee0df15`에 있음.

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
# 자가대전은 --protocols가 필요 없음 (컨테이너 자신이 protocol 레코드를 이미 남김)
python src/training/build_fp_dataset.py --decisions "data/fp_selfplay/ms300/dec_sp12[abc]_*.jsonl" \
  --out data/fp_selfplay/ms300/fp_data_sp12.npz

python src/training/train_fp_distill.py --data "data/fp_selfplay/ms300/fp_data*.npz,data/fp_selfplay/ms100/fp_data*.npz" \
  --init checkpoints/supervised_v2_fp_all9.pt --out checkpoints/supervised_v2_fp_all11.pt   # 기본값: 피크 lr 2e-3 + 웜업 5% + 코사인 감쇠, 4에폭. 데이터가 ~150만 결정을 넘으면 --fp16-store
```
증류는 BC에서 다시 시작하지 말고 직전 최신 모델에서 누적 전체 데이터로 이어서 학습한다 (KL 기본 0.05). KL 계수는 최적해가 (교사 + 계수 × 시작정책)/(1 + 계수) 혼합이라 크면 시작 정책 쪽으로 끌려간다 (0.3이면 교사 77% + 시작 정책 23%). `--init`과 `--out`이 같으면 덮어쓰니 다른 이름을 쓸 것.

## 5. 평가

```bash
# 모델끼리 승률 비교 (SPRT): python tools/arena.py <새모델> <기준모델> --delta 0.03  (README 본문 '평가 체계' 참고)

# RCT/휴리스틱 상대로 홀드아웃 풀 평가
python src/evaluation/analyze_model.py --ckpt checkpoints/supervised_v2.pt --mode argmax \
  --battles 1000 --opponent rct --pool data/team_pool_metamon_holdout.json --workers 6
```

평가는 항상 holdout(A)/rare(B)/randomset(C) 풀에만. 학습 풀(train_a*, rare_train*, randomset_train3)로 평가하지 말 것.

## 6. 검증 표준 (학생 상태 홀드아웃 + 직접 대전)

교사 상태 검증(학습 스크립트가 찍는 값)은 강도를 따라가지 못함 → 1차 검증은 **학생 상태 홀드아웃**, 최종 판정은 **직접 대전**.
```bash
# 1차 (약 30초): 학생이 방문한 상태에서 교사와의 일치율/CE (높을수록/낮을수록 좋음). 기준: all5d300 0.485/1.5596, all5x 0.497/1.5500, all5h 0.500/1.5335
python tools/eval_on_npz.py checkpoints/supervised_v2_fp_XXX.pt data/dagger/dg2_ho.npz

# 최종 (약 15분, Showdown 서버 먼저 켤 것): 모델끼리 3000판. 표준오차 약 0.9pp
bash logs/h2h_run.sh <모델A> <모델B> 125    # checkpoints/supervised_v2_fp_<이름>.pt, 결과 logs/h2h_summary.txt

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
