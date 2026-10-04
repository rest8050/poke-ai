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


