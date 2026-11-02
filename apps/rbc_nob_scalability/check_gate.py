#!/usr/bin/env python3
"""게이트 — 두 런 로그의 모니터 줄(t, dt, θ_c, Nu, Tmin, Tmax, Nu_b, Nu_t, D_x, |U| bot/top)을 비교한다.
   python3 check_gate.py <ref.log> <new.log> [rel_tol=1e-12]
   job_scal.sh 가 rbc_nob(ref)과 rbc_nob_scalability(new)를 같은 계산으로 돌려 부른다 — 측정 앱의 시간
   루프가 본 앱과 같은 계산을 하는지 확인. max|div| 열(리덕션 순서 잡음)과 sec/step(시간)은 뺀다.
   (apps/rbc_nob/scalability/check_gate.py 와 같은 비교)"""
import re
import sys

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
common = sorted(set(A) & set(B))
if not common:
    print(f"no common monitor steps (ref {len(A)}, new {len(B)} lines)")
    print("GATE FAIL")
    sys.exit(1)
worst, worst_at = 0.0, None
for step in common:
    for k, (x, y) in enumerate(zip(A[step], B[step])):
        rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
        if rel != rel:  # NaN 은 통과시키지 않는다
            rel = float("inf")
        if rel > worst:
            worst, worst_at = rel, (step, k, x, y)
if worst_at is None:
    print(f"steps compared {len(common)}, identical")
else:
    print(f"steps compared {len(common)}, worst relative diff {worst:.3e} at (step, column, ref, new) {worst_at}")
print("GATE PASS" if worst <= tol else "GATE FAIL")
sys.exit(0 if worst <= tol else 1)
