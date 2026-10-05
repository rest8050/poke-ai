"""양방향 SPRT(승/패 베르누이, 무승부·무효 제외). 귀무 p=0.5, 대립 p=0.5±δ 두 개.
- 이긴다(better): 'p=0.5+δ' 쪽 로그우도비가 상한 A=ln((1-β)/α)에 닿음
- 진다(worse): 'p=0.5-δ' 쪽이 상한에 닿음
- 차이 없음(equal): 두 로그우도비가 모두 하한 B=ln(β/(1-α)) 아래
- 그 외: continue (더 돌려야 함)
단일 검정의 오류율 보장은 사전에 정한 (δ, α, β)로 처음부터 이어서 본 경우에만 성립 — 장부의 옛 결과를 합쳐 보는 건 참고용.
사용: python tools/sprt.py <a> <b> [--delta 0.03] [--alpha 0.05] [--beta 0.05] [--run RUN_ID] [--ledger PATH]"""
import argparse
import math

NAMES = {"better": "후보가 낫다", "worse": "후보가 나쁘다", "equal": "차이 없음(±δ 이내)", "continue": "계속"}


def decide(wins, losses, delta=0.03, alpha=0.05, beta=0.05):
    A, B = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    p1, p2 = 0.5 + delta, 0.5 - delta
    up = wins * math.log(2 * p1) + losses * math.log(2 * (1 - p1))
    dn = wins * math.log(2 * p2) + losses * math.log(2 * (1 - p2))
    d = "better" if up >= A else "worse" if dn >= A else "equal" if up <= B and dn <= B else "continue"
    return dict(decision=d, n=wins + losses, wins=wins, losses=losses, up=up, dn=dn, A=A, B=B)


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = w / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


if __name__ == "__main__":
    from tools.ledger import LEDGER, pair_counts, read
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--delta", type=float, default=0.03); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.05); ap.add_argument("--run", default=None); ap.add_argument("--ledger", default=LEDGER)
    a = ap.parse_args()
    rows = [r for r in read(a.ledger) if a.run is None or r.get("run") == a.run]
    w, l, d, v = pair_counts(rows, a.a, a.b)
    s = decide(w, l, a.delta, a.alpha, a.beta)
    lo, hi = wilson(w, w + l)
    print(f"{a.a} 대 {a.b}: {w}승 {l}패 (무 {d}, 무효 {v}) 승률 {w / max(1, w + l):.1%} [95% {lo:.1%}~{hi:.1%}]")
    print(f"로그우도비 +δ {s['up']:+.2f} / -δ {s['dn']:+.2f} (경계 {s['A']:.2f} / {s['B']:.2f}) -> {NAMES[s['decision']]}")
