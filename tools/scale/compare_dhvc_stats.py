#!/usr/bin/env python3
"""DHVC 통계 전/후 비교: 최종 누적 dhvc_stats_*.dat 의 헤더 스칼라(Nu·Re·u_tau)와 z-프로파일 상대차."""
import glob, re, sys, math
def last(d): return sorted(glob.glob(f"{d}/dhvc_stats_*.dat"))[-1]
def load(p):
    hdr, rows = {}, []
    for l in open(p):
        if l.startswith('#'):
            for k, v in re.findall(r'(\w+)=([-+0-9.eE]+)', l): hdr[k] = float(v)
        elif l.strip(): rows.append([float(x) for x in l.split()])
    return hdr, rows
a_dir, b_dir = sys.argv[1], sys.argv[2]
(ha, ra), (hb, rb) = load(last(a_dir)), load(last(b_dir))
keys = ['Nu_hot', 'Nu_cold', 'Re_dstar', 'Re_rms', 'u_tau_hot', 'u_tau_cold', 'U_rms_delta_max', 'delta_star', 'mean_dPdx']
print(f"{'quantity':16s} {'orig':>14s} {'new':>14s} {'rel diff':>9s}   (window t {ha['t_start']:.0f}–{ha['t_start']+ha['sum_dt']:.0f} vs {hb['t_start']:.0f}–{hb['t_start']+hb['sum_dt']:.0f})")
for k in keys:
    x, y = ha[k], hb[k]; print(f"{k:16s} {x:14.6g} {y:14.6g} {100*(y-x)/abs(x) if x else float('nan'):8.2f}%")
cols = ['U', 'V', 'W', 'T', 'urms', 'vrms', 'wrms', 'trms', 'uw', 'uT', 'wT']
print("\nz-profile max |diff| / max |orig| per column:")
for c, i in zip(cols, range(1, 12)):
    xa = [r[i] for r in ra]; xb = [r[i] for r in rb]; m = max(abs(v) for v in xa) or 1.0
    print(f"  {c:5s} {max(abs(p - q) for p, q in zip(xa, xb)) / m * 100:7.3f}%")
