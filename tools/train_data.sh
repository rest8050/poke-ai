# 표준 학습 데이터 목록 (쉼표 구분). 학습 스크립트는 이 파일을 source해서 TRAIN_DATA를 쓸 것 — 무제한룰(AG, uber1)도 항상 포함.
# 새 수집분/DAgger는 여기에 추가 (예외: 구조 비교용 소규모 파일럿은 데이터를 고정하므로 따로 지정)
D100=data/fp_selfplay/ms100
TRAIN_DATA="$D100/fp_data_sp0.npz,$D100/fp_data_sp23.npz,$D100/fp_data_sp4.npz,$D100/fp_data_sp5.npz,$D100/fp_data_sp6.npz,data/fp_selfplay/ms300/fp_data_sp7.npz,data/dagger/dg2_tr.npz,$D100/fp_data_sp81.npz,$D100/fp_data_sp82.npz,$D100/fp_data_sp83.npz,$D100/fp_data_sp91.npz,$D100/fp_data_sp92.npz,$D100/fp_data_sp101.npz,$D100/fp_data_sp102.npz,$D100/fp_data_sp103.npz,data/dagger/dg3_tr.npz,data/fp_selfplay/ms300/fp_data_sp12.npz,data/dagger/dg4_tr.npz,data/fp_selfplay/uber300/fp_data_uber1.npz"
TRAIN_DATA="$TRAIN_DATA${EXTRA_DATA:+,$EXTRA_DATA}"   # 환경변수 EXTRA_DATA(쉼표 구분 npz)로 이번 학습만 데이터를 더함
