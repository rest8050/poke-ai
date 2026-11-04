# PokeAI

포켓몬 쇼다운 9세대 OU 싱글 배틀 AI입니다.
탐색 봇 [Foul Play](https://github.com/pmariglia/foul-play)(몬테카를로 트리 탐색)의 판단을 트랜스포머 정책망으로 증류해서, 탐색 없이 한 번의 추론으로 행동을 고릅니다.
학습한 모델은 마인크래프트 Cobblemon 서버의 트레이너 AI로 실제 게임에서 돌아갑니다.

## 결과

모델 비교는 순차 검정(SPRT)으로 합니다. 학습에 안 쓴 팀으로 대전하고, 승률이 50%보다 3%p 높은지 낮은지 결론이 날 때까지 둡니다. 모든 판은 `results/ledger.jsonl`에 남습니다.

| 모델 | 추가한 것 | 직전 모델 대비 승률 | 판정까지 판 수 |
|---|---|---|---|
| v3 | 엔티티 토큰 트랜스포머 + 손계산 데미지·KO 특징 | 52.3% | 3,477 |
| v4 | 상대 숨은 기술의 위협 + 노력치·성격 예측, 교사 승률 증류 | 53.2% | 1,452 |
| v6 | 실제 가능한 행동(구애 고정, 도발 등) 입력, 천진·장판 진입 계산 | 50.1% (차이 없음) | 2,016 |
| v8 | 특성 규칙(정밀기술, 옹골참, 위협, 스피드·선공), 숨은 특성·도구의 기대 임베딩 | 53.8% (v6 대비) | 1,089 |

v8은 2%p 재확인 대국에서 52.0%(3,048판)까지 두고 결론 전에 멈췄습니다. Foul Play 상대 승률은 v6 37.8%, 특성 규칙만 더한 v7 36.4%입니다(각 500판).

## 동작 방식

1. **교사 데이터**: Foul Play끼리 대전시키고, 결정마다 탐색 분포와 승률 추정을 기록합니다. 대전 기록을 재생해 그 시점의 상태를 만들고, 서버 요청으로 실제 가능한 행동을 복원합니다.
2. **학생 모델**: 포켓몬 12개, 기술 48개, 필드 3개, 요약 1개의 토큰을 받는 트랜스포머입니다. 손계산 매치업(데미지·KO, 특성 규칙, 선공)을 토큰 입력과 어텐션 편향에 넣고, 따로 학습한 세트 추정 모듈이 상대의 숨은 기술·도구·특성·노력치를 예측합니다. 숨은 정보는 그 확률의 기대값으로 계산합니다.
3. **학습**: 교사 분포를 따라가는 증류 손실과 승률을 맞추는 가치 손실로 처음부터 학습합니다. 학생이 실제로 가는 상태는 Foul Play로 다시 라벨링해 추가합니다(DAgger). 자가대전 PPO 미세조정은 실험 중입니다.
4. **평가**: 학습에 안 쓴 팀 풀 3종(holdout, rare, randomset)에서 SPRT 대전으로 채택 여부를 정합니다.
5. **배포**: Cobblemon 모드(`mod/pokeai`)가 배틀 메시지를 파이썬 브리지 서버로 보내면 모델이 행동을 돌려줍니다. 브리지가 응답하지 않으면 기존 AI가 대신 두고, 모델은 서버 재시작 없이 교체됩니다.

## 폴더 구조

```
src/core/       모델(model_v8 학습 기본, model_v6 서버 탑재), 인코더, 손계산 매치업, 세트 추정, 브리지 서버
src/training/   데이터셋 빌드, 증류 학습, 세트 추정 사전학습, 강화학습(PPO)
tools/          데이터 수집, 평가, 진단 도구 (명령어 모음: tools/README.md)
docs/           설계 문서와 실험 전에 정한 판정 기준
deploy/         서버 배포 묶음과 설치 스크립트
mod/pokeai/     Cobblemon 모드 (Java)
tests/          테스트 (PYTHONPATH=. python tests/<파일>.py)
results/        SPRT 대전 기록
```

## 실행

```bash
pip install -r requirements.txt

# 1. 데이터 빌드: Foul Play 대전 기록 → 학습 데이터
#    (대전 기록 수집은 로컬 Showdown 서버와 Foul Play 설정이 필요: tools/README.md 3장)
python src/training/build_fp_dataset.py --decisions "data/fp_selfplay/ms300/dec_*.jsonl" --out data/fp_selfplay/ms300/fp_data.npz

# 2. 학습: 처음부터, 기본 구조 v8, 6에폭. 모델 이름 = checkpoints/supervised_v2_fp_<이름>.pt
#    (세트 추정 모듈 사전학습이 먼저 필요: src/training/pretrain_belief.py)
python src/training/train_fp_distill.py --data "data/fp_selfplay/ms300/fp_data*.npz" --out checkpoints/supervised_v2_fp_new.pt

# 3. 평가: 새 모델과 기준 모델의 SPRT 대전
python tools/arena.py new <기준> --delta 0.03

# 4. 배포 묶음 만들기 (서버 코드는 현재 v6 구조)
bash deploy/build_package.sh v6_full
```

## 평가 체계

- `python tools/arena.py <후보> <기준> [--delta 0.03] [--cap 10000]`: 세 팀 풀을 고르게 섞어 대전하고 판마다 장부에 기록합니다. 종료 코드 0=낫다, 1=나쁘다, 2=차이 없음, 3=상한 도달.
- `python tools/sprt.py <a> <b>`: 장부에 쌓인 결과로 현재 판정을 봅니다.
- `python tools/bt_report.py --ref v1_full_boot`: 장부 전체의 Elo 레이팅과 95% 구간.

기본 관문은 3%p(평균 약 2,400판)입니다. 통과한 후보는 2%p(평균 약 5,000판)로 새 판에서 한 번 더 확인한 뒤 기준 모델로 올립니다.

## 한계와 다음 단계

- **교체 판단**: 교사 대비 손해(regret)의 약 44%가 교체 결정에서 나옵니다. 교체할지 말지를 가리는 판단이 아직 교사보다 약합니다.
- **탐색 없음**: 순전파 한 번으로 탐색 교사를 흉내 내는 구조라 여러 턴을 내다보는 계산에 한계가 있습니다. Foul Play 상대 승률이 40% 아래에 머무는 이유입니다.
- **진행 중**: 기술 가방과 기술 x 포켓몬 쌍 모듈을 더한 후보는 교체 결정 regret이 12% 줄었고 대국 검증 전입니다. 자가대전 PPO 미세조정도 결론 전입니다.
