#!/usr/bin/env python3
"""compare_dhvc_analytical.py — Compare MPM-STD_C DHVC T and U dumps against
analytical (steady-state) solution for differentially heated vertical channel.

Physical setup:
  x = streamwise/vertical  (periodic, buoyancy direction)
  y = spanwise             (periodic)
  z = wall-normal          (hot wall at z=0, cold wall at z=Lz)

Analytical solution (free-fall ND, H=Lz=1):
  T(z) = 0.5 - z                           (linear)
  U(z) = z*(2z-1)*(z-1) / (12*nu)         (cubic, zero at both walls)
  where nu = sqrt(Pr/Ra)

Usage:
  python3 compare_dhvc_analytical.py  [T_bin] [U_bin]
"""

import sys
import math

T_BIN = sys.argv[1] if len(sys.argv) > 1 else "dump_mpm_T.bin"
U_BIN = sys.argv[2] if len(sys.argv) > 2 else "dump_mpm_U.bin"

# ── simulation parameters ─────────────────────────────────────────────────────
Ra    = 100.0
Pr    = 1.0
nu    = math.sqrt(Pr / Ra)   # = 0.1
Lz    = 1.0
n1m   = 32   # stream
n2m   = 32   # span
n3m   = 16   # wall-normal (z)
gamma = 2.6  # tanh stretching parameter for z

# ── compute tanh-stretched cell-centre z coordinates ──────────────────────────
def tanh_faces(n, L, g):
    hg = 0.5 * g
    denom = math.tanh(hg)
    return [0.5 * L * (1.0 + math.tanh(hg * (2.0 * i / n - 1.0)) / denom)
            for i in range(n + 1)]

zf = tanh_faces(n3m, Lz, gamma)
zc = [0.5 * (zf[k] + zf[k + 1]) for k in range(n3m)]

# ── load binary dumps ─────────────────────────────────────────────────────────
def load_f64(path):
    import struct
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 8
    return list(struct.unpack(f"<{n}d", data))

print(f"Loading {T_BIN} and {U_BIN} ...")
T_flat = load_f64(T_BIN)
U_flat = load_f64(U_BIN)

assert len(T_flat) == n1m * n3m * n2m, f"T size mismatch: {len(T_flat)} vs {n1m*n3m*n2m}"
assert len(U_flat) == n1m * n3m * n2m, f"U size mismatch: {len(U_flat)} vs {n1m*n3m*n2m}"

# Layout in dump: T[ii][kk][jj]  (ii=stream, kk=wall, jj=span)
# Average over ii (stream) and jj (span) to get z-profile
def z_profile(flat):
    prof = [0.0] * n3m
    cnt  = n1m * n2m
    for ii in range(n1m):
        for kk in range(n3m):
            for jj in range(n2m):
                idx = ii * n3m * n2m + kk * n2m + jj
                prof[kk] += flat[idx]
    return [v / cnt for v in prof]

T_num = z_profile(T_flat)
U_num = z_profile(U_flat)

# ── analytical solution ───────────────────────────────────────────────────────
T_ana = [0.5 - z / Lz for z in zc]
U_ana = [z * (2*z/Lz - 1.0) * (z/Lz - 1.0) / (12.0 * nu) for z in zc]

# ── print comparison table ────────────────────────────────────────────────────
print(f"\n{'k':>3}  {'z':>8}  {'T_num':>12}  {'T_ana':>12}  {'err_T':>10}  "
      f"{'U_num':>12}  {'U_ana':>12}  {'err_U':>10}")
print("-" * 95)

T_errs, U_errs = [], []
for k in range(n3m):
    et = T_num[k] - T_ana[k]
    eu = U_num[k] - U_ana[k]
    T_errs.append(abs(et))
    U_errs.append(abs(eu))
    print(f"{k:3d}  {zc[k]:8.5f}  {T_num[k]:12.6f}  {T_ana[k]:12.6f}  {et:10.3e}  "
          f"{U_num[k]:12.6f}  {U_ana[k]:12.6f}  {eu:10.3e}")

print("-" * 95)
linf_T = max(T_errs)
linf_U = max(U_errs)
ref_T  = max(abs(v) for v in T_ana) or 1e-15
ref_U  = max(abs(v) for v in U_ana) or 1e-15
print(f"\nL∞(T_num - T_ana) = {linf_T:.3e}   relative: {linf_T/ref_T:.3e}")
print(f"L∞(U_num - U_ana) = {linf_U:.3e}   relative: {linf_U/ref_U:.3e}")

if linf_T < 1e-6 and linf_U < 1e-3:
    print("\nPASS  (T: L∞<1e-6, U: L∞<1e-3 — DHVC steady state verified)")
elif linf_T < 1e-4 and linf_U < 5e-2:
    print("\nWARN  (not fully converged or mesh too coarse)")
else:
    print("\nFAIL  (L∞ too large — check buoyancy direction or T BCs)")
