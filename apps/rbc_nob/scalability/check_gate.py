#!/usr/bin/env python3
"""허용오차 게이트 — 두 런 로그의 모니터 줄(step, t, dt, θ_c, Nu, Tmin, Tmax, Nu_b, Nu_t, D_x, |U| bot/top)을 비교.
   python3 check_gate.py <ref.log> <new.log> [rel_tol=1e-12]
   max|div| 열(인덱스 3)과 sec/step(마지막) 은 뺀다 (리덕션 순서 잡음·시간)."""
import re, sys
ref, new = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-12
def rows(path):
    out = {}
    for line in open(path):
        if re.match(r"^\s+\d+\s", line):
            f = line.split()
            out[int(f[0])] = [float(x) for x in f[1:3] + f[4:13]]
    return out
A, B = rows(ref), rows(new)
worst = 0.0; worst_at = None
for step in sorted(set(A) & set(B)):
    for k, (x, y) in enumerate(zip(A[step], B[step])):
        rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
        if rel > worst:
            worst, worst_at = rel, (step, k, x, y)
print(f"steps compared {len(set(A) & set(B))}, worst relative diff {worst:.3e} at {worst_at}")
print("GATE PASS" if worst <= tol else "GATE FAIL")
sys.exit(0 if worst <= tol else 1)
