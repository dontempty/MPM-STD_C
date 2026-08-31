#!/usr/bin/env python3
"""Quantify 2-dx (Nyquist) checkerboard amplitude near the two walls.

Restart .bin layout: C-order double, axes (x, y, z), z fastest; z=0 is the
hot wall (bc.z.minus, T=+0.5), z=N-1 the cold wall (T=-0.5).

Metric per horizontal slice u(x,y):
  nyq_frac = E_nyq / E_tot, where E_tot is the fluctuation energy (mean removed)
  and E_nyq the energy in 2D-FFT modes whose kx or ky is the Nyquist mode N/2
  (the pure 2-dx checkerboard family). Reported alongside rms.
"""
import numpy as np, sys, os

N = 256
CASES = {
    "baseline dT=40 th=1.0": "restart_h200_deltat40_256_cfl03_theta1_short",
    "exp1     dT=0.01 th=1.0": "restart_h200_exp1_deltat001",
    "exp2     dT=40 th=0.5(BLK)": "restart_h200_exp2_dt40_blk05",
    "exp3     dT=40 dt=0.0015": "restart_h200_exp3_dtsmall",
    "exp4     dT=40 dt=0.006": "restart_h200_exp4_dtlarge",
}
FIELDS = ["U", "V", "W", "T"]
BOT = [2, 4, 8]            # z indices near hot wall
TOP = [N-3, N-5, N-9]      # z indices near cold wall

def load(path):
    a = np.fromfile(path, dtype=np.float64)
    assert a.size == N**3, f"{path}: {a.size}"
    return a.reshape(N, N, N)   # (x, y, z)

def nyq_frac(sl):
    f = sl - sl.mean()
    F = np.fft.rfft2(f)                       # (kx: 0..N-1 wrapped, ky: 0..N/2)
    E = np.abs(F)**2
    tot = E.sum() - E[0, 0]
    nyq = E[N//2, :].sum() + E[:, N//2].sum() - E[N//2, N//2]
    return (nyq / tot if tot > 0 else 0.0), f.std()

base = os.path.dirname(os.path.abspath(__file__))
print(f"{'case':28s} {'fld':3s} | {'bot nyq%':>9s} {'bot rms':>9s} | {'top nyq%':>9s} {'top rms':>9s}")
for label, d in CASES.items():
    dd = os.path.join(base, d)
    if not os.path.isfile(os.path.join(dd, "U.bin")):
        print(f"{label:28s} --- missing ({d})"); continue
    for fld in FIELDS:
        a = load(os.path.join(dd, f"{fld}.bin"))
        bn = np.mean([nyq_frac(a[:, :, k])[0] for k in BOT])
        br = np.mean([nyq_frac(a[:, :, k])[1] for k in BOT])
        tn = np.mean([nyq_frac(a[:, :, k])[0] for k in TOP])
        tr = np.mean([nyq_frac(a[:, :, k])[1] for k in TOP])
        print(f"{label:28s} {fld:3s} | {100*bn:8.3f}% {br:9.2e} | {100*tn:8.3f}% {tr:9.2e}")
    print()
