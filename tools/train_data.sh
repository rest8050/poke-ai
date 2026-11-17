# 표준 학습 데이터 목록 (쉼표 구분). 학습 스크립트는 이 파일을 source해서 TRAIN_DATA를 쓸 것.
# (2026-11-18) 새 인코더(ENC_VERSION 2)로 재빌드한 25개 파일 (logs/rebuild_enc1.sh). 평가 파일(d7_ho, d6_ho, dg5)은 data/enc1/eval에 따로 있음.
# 새 수집분/DAgger는 data/enc1/npz에 넣으면 자동 포함 (구 v1 데이터는 data/base_v1에 격리)
TRAIN_DATA=$(ls data/enc1/npz/*.npz | paste -sd, -)
