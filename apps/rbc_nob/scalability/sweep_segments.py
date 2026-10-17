#!/usr/bin/env python3
"""조각 수 p 스윕 집계: 꼬리표 <hw>s<p> 런의 tdma_x+y(momentum)·scalar_solve·step_total 을 p 별로 한 표에.
   python3 sweep_segments.py h200   (out/ 기준)"""
import csv, glob, re, sys
hw = sys.argv[1]
runs = {}
for p in glob.glob(f"out/scal_*/phase_timing_{hw}s*_*.csv"):
    m = re.search(rf"scal_(strong|weak)_(\d+)_np(\d+)/phase_timing_{hw}s(\d+)_", p)
    if not m: continue
    k = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
    ph = {r[0]: float(r[3]) for r in csv.reader(l for l in open(p) if not l.startswith("#")) if r[0] != "phase"}
    runs.setdefault(k, []).append(ph)
by = {k: sorted(v, key=lambda r: r["step_total"])[len(v) // 2] for k, v in runs.items()}
txy = lambda ph: sum(v for n, v in ph.items() if n.endswith(".tdma_x") or n.endswith(".tdma_y"))
for (mode, n, np_) in sorted({k[:3] for k in by}):
    ps = sorted(k[3] for k in by if k[:3] == (mode, n, np_))
    lines = n * n * (n // np_) // n  # ny * nz_local for x sweep
    print(f"\n[{hw}] {mode} {n}^3 np{np_}  (x-sweep lines = {n * (n // np_)}, rows = {n})")
    print(f"  {'p':>3s} {'tdma_xy':>8s} {'per call':>9s} {'scalar':>8s} {'momentum':>9s} {'step':>8s}")
    for p in ps:
        ph = by[(mode, n, np_, p)]
        print(f"  {p:3d} {txy(ph):8.2f} {txy(ph)/6:9.3f} {ph['scalar_solve']:8.2f} {ph['momentum_solve']:9.2f} {ph['step_total']:8.2f}")
