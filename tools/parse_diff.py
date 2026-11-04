"""사람 데이터 재파싱(_hindsight 도구 주인 수정) 검증: 같은 이름의 옛 조각과 새 조각을 배열별로 비교한다.
기대: 라벨/마스크/상대 쪽/필드는 그대로이고, 내 팀 도구 열(my_team_cat의 0번 열)만 일부 포켓몬에서 달라짐. 다른 곳이 달라지면 알려 줌.
사용: python tools/parse_diff.py data/human data/human_v2 r1600 [조각 번호들, 기본 앞 3개]"""
import sys

import numpy as np

old_dir, new_dir, tag = sys.argv[1:4]
ids = [int(x) for x in sys.argv[4:]] or [0, 1, 2]
for i in ids:
    name = f"gen9ou_s0_{tag}_{i:03d}.npz"
    a, b = np.load(f"{old_dir}/{name}"), np.load(f"{new_dir}/{name}")
    print(f"== {name}: 시퀀스 {len(a['lens'])} → {len(b['lens'])}, 결정 {len(a['action'])} → {len(b['action'])}")
    if len(a["action"]) != len(b["action"]):
        print("   ❌ 결정 수가 다름 → 같은 판/순서가 아님")
        continue
    for k in a.files:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"   {k}: 모양이 다름 {x.shape} vs {y.shape}")
        elif not np.array_equal(x, y, equal_nan=True):
            ch = (x != y)
            rows = ch.reshape(len(x), -1).any(1)
            msg = f"   {k}: 달라진 결정 {rows.mean():.3f}"
            if k == "my_team_cat" and ch.ndim == 3:
                msg += f" | 열별 달라진 포켓몬 칸 수 {dict(enumerate(map(int, ch.sum((0, 1)))))}"
            print(msg)
