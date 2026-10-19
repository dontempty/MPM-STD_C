#!/usr/bin/env python3
"""RBC(rbc_nob) 통계 전/후 비교: 로그 FINAL 과 모니터 창 평균(θ_c·Nu), 최종 rbc_nob_stats 헤더 비교."""
import glob, re, sys
def final(log):
    m = [l for l in open(log) if l.startswith('# FINAL')]; return m[-1].strip() if m else '(no FINAL yet)'
def monitor_mean(log, t0):
    th, nu, n = 0.0, 0.0, 0
    for l in open(log):
        if re.match(r'^\s*\d+\s', l):
            f = l.split()
            if float(f[1]) >= t0: th += float(f[4]); nu += float(f[5]); n += 1
    return (th / n, nu / n, n) if n else (float('nan'),) * 3
for name, log in (('orig', sys.argv[1]), ('new', sys.argv[2])):
    th, nu, n = monitor_mean(log, float(sys.argv[3]) if len(sys.argv) > 3 else 16900.0)
    print(f"{name:5s} {final(log)}  | monitor mean (t ≥ window): theta_c {th:.5f} Nu {nu:.4f} (n={n})")
