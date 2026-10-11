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

v5는 실제로 가능한 행동(구애 고정, 도발, PP 소진 등)을 입력으로 받습니다. 파일럿에서 "기술이 하나만 가능한" 상황의 교체 판단 오차(KL)가 0.045 줄었습니다(95% 구간 0.036~0.054).

## 동작 방식

1. **교사 데이터**: Foul Play끼리 대전시키고, 결정마다 탐색 분포와 승률 추정을 기록합니다. 대전 기록을 재생해 그 시점의 상태를 만들고, 서버 요청으로 실제 가능한 행동을 복원합니다.
2. **학생 모델**: 포켓몬 12개, 기술 48개, 필드 3개, 요약 1개의 토큰을 받는 트랜스포머입니다. 손계산 데미지·KO 특징을 어텐션 편향으로 넣고, 따로 학습한 신념 모듈이 상대의 숨은 기술·도구·특성·노력치를 예측합니다.
3. **학습**: 교사 분포를 따라가는 증류 손실과 교사 승률을 맞추는 가치 손실로 학습합니다. 학생이 실제로 가는 상태는 Foul Play로 다시 라벨링해 추가합니다(DAgger).
4. **평가**: 학습에 안 쓴 팀 풀 3종(holdout, rare, randomset)에서 SPRT 대전으로 채택 여부를 정합니다.
5. **배포**: Cobblemon 모드(`mod/pokeai`)가 배틀 메시지를 파이썬 브리지 서버로 보내면 모델이 행동을 돌려줍니다. 브리지가 응답하지 않으면 기존 AI가 대신 두고, 모델은 서버 재시작 없이 교체됩니다.

## 폴더 구조

```
src/core/       모델(model_v3~v6), 인코더, 손계산 매치업, 신념 모듈, 브리지 서버
src/training/   데이터셋 빌드, 증류 학습, 신념 모듈 사전학습
tools/          데이터 수집, 평가, 진단 도구 (명령어 모음: tools/README.md)
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

# 2. 학습 (기본 구조 v5). 모델 이름 = checkpoints/supervised_v2_fp_<이름>.pt
python src/training/train_fp_distill.py --data "data/fp_selfplay/ms300/fp_data*.npz" \
  --init checkpoints/supervised_v2_fp_<기준>.pt --out checkpoints/supervised_v2_fp_new.pt

# 3. 평가: 새 모델과 기준 모델의 SPRT 대전
python tools/arena.py new <기준> --delta 0.03

# 4. 배포 묶음 만들기
bash deploy/build_package.sh new
```

## 평가 체계

- `python tools/arena.py <후보> <기준> [--delta 0.03] [--cap 10000]`: 세 팀 풀을 고르게 섞어 대전하고 판마다 장부에 기록합니다. 종료 코드 0=낫다, 1=나쁘다, 2=차이 없음, 3=상한 도달.
- `python tools/sprt.py <a> <b>`: 장부에 쌓인 결과로 현재 판정을 봅니다.
- `python tools/bt_report.py --ref v1_full_boot`: 장부 전체의 Elo 레이팅과 95% 구간.

기본 관문은 3%p(평균 약 2,400판)입니다. 통과한 후보는 2%p(평균 약 5,000판)로 새 판에서 한 번 더 확인한 뒤 기준 모델로 올립니다.

## 한계와 다음 단계

- **교체 판단**: 학생이 교체를 고르는 비율은 약 13%, 교사는 약 23%입니다. 교체할지 말지를 가리는 판단이 아직 교사보다 약합니다.
- **진행 중(v6)**: 천진 특성과 장판 피해를 반영한 매치업 계산, 도발·앵콜 같은 행동 제약 효과를 입력으로 추가해 파일럿 중입니다.
