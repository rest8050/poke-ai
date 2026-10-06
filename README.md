# 🎮 PokeAI - 포켓몬 배틀 딥러닝 & 강화학습 AI 프로젝트

자기 포켓몬 샘플(특성, 도구, 상성)을 이해하고 상대의 행동(스킬/교체)을 예측하는 PyTorch 기반 포켓몬 배틀 AI 템플릿 프로젝트입니다.

---

## 📁 프로젝트 구조

```
poke-ai/
├── requirements.txt   # 필요한 파이썬 패키지 목록
├── model.py           # PyTorch 신경망 아키텍처 (Policy + Opponent Predictor + Value)
├── player.py          # poke-env & RCT JSON 전황 데이터를 상태 벡터로 인코딩
├── train.py           # 상대 행동 예측(Auxiliary Loss) 보조 학습 루프 템플릿
└── README.md          # 프로젝트 사용 가이드
```

---

## ⚙️ 1. 환경 설치

```bash
cd poke-ai
pip install -r requirements.txt
```

---

## 🧠 2. 모델 학습 (Train)

```bash
python train.py
```
* 학습이 완료되면 `checkpoint.pt` 가중치 파일이 생성됩니다.
* `poke-env` 라이브러리를 이용해 로컬 포켓몬 쇼다운 서버와 셀프 플레이(Self-Play) 방식으로 수만 판 훈련시킬 수 있습니다.

---

## 🚀 3. AI 추론 서버

마인크래프트 모드 연동 서버는 삭제됨 (텐서 입력 방식이라 현재 모델과 어긋남). 프로토콜 원문 입력 방식으로 추후 새로 작성.

---


## 평가 체계 (SPRT)
모델 비교는 고정 판 수 대신 순차 검정(SPRT)으로 한다: 승률 50% 대비 ±δ를 α=β=5%로 가리고, 결론이 나는 즉시 멈춘다.
- `python tools/arena.py <후보> <챔피언> [--delta 0.03] [--cap 10000]` : 풀(holdout/rare/randomset)을 균등 혼합해 대전, 판마다 `results/ledger.jsonl`에 기록. 종료 코드 0=낫다 1=나쁘다 2=차이 없음 3=상한/중단. 판당 제한 시간 초과는 `void`(집계 제외)
- `python tools/sprt.py <a> <b>` : 장부에 쌓인 결과로 현재 판정 조회 (옛 결과를 합친 건 참고용)
- `python tools/bt_report.py --ref v1_full_boot` : 장부 전체의 Bradley–Terry 레이팅(Elo)과 95% 구간
- `python tools/ledger.py backfill` : 기존 h2h 요약 파일을 장부에 이식
기본 관문은 δ=3%p(평균 약 2.4천 판), 통과한 후보는 δ=2%p(평균 약 5천 판)로 새 판에서 한 번 더 확인한 뒤 챔피언으로 올린다. 검증: `PYTHONPATH=. python tests/test_sprt.py`
