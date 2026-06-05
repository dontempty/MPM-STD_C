#!/usr/bin/env python3
"""compare_pascal_tcs.py — Compare MPM-STD_C T dump against PaScaL_TCS T dump.

Usage:
    python3 compare_pascal_tcs.py  [pascal_bin] [mpm_bin]

Both files are raw little-endian float64 arrays with identical shape
(n1m, n2m_wall, n3m_span) and layout:
    outer loop  i  = streamwise    (PaScaL n1  / MPM X)
    middle loop j  = wall-normal   (PaScaL n2  / MPM Z)
    inner loop  k  = spanwise      (PaScaL n3  / MPM Y)

Sign convention: T_PaScaL = T_MPM (DHVC: both codes use +0.5 at hot wall, -0.5 at cold wall).
"""

import sys
import numpy as np

PASCAL_BIN = sys.argv[1] if len(sys.argv) > 1 else "dump_pascal_T.bin"
MPM_BIN    = sys.argv[2] if len(sys.argv) > 2 else "dump_mpm_T.bin"

# ── load ──────────────────────────────────────────────────────────────────────
t_pascal = np.fromfile(PASCAL_BIN, dtype="<f8")
t_mpm    = np.fromfile(MPM_BIN,    dtype="<f8")

if t_pascal.size != t_mpm.size:
    print(f"ERROR: size mismatch — pascal {t_pascal.size}  vs  mpm {t_mpm.size}")
    sys.exit(1)

n = t_pascal.size
print(f"Loaded {n} doubles from each file.")

# ── norms ─────────────────────────────────────────────────────────────────────
diff   = t_pascal - t_mpm
linf   = np.max(np.abs(diff))
l2     = np.sqrt(np.mean(diff**2))
ref    = max(np.max(np.abs(t_pascal)), 1e-15)

print(f"L∞(T_pascal - T_mpm) = {linf:.3e}   (relative: {linf/ref:.3e})")
print(f"L2(T_pascal - T_mpm) = {l2:.3e}   (relative: {l2/ref:.3e})")

if linf < 1e-10:
    print("PASS  (L∞ < 1e-10)")
elif linf < 1e-6:
    print("WARN  (1e-10 ≤ L∞ < 1e-6)")
else:
    print("FAIL  (L∞ ≥ 1e-6)")

# ── spot-check: wall-normal profile at (i=0, k=0) ────────────────────────────
# Infer n2m (wall-normal) from the total size — needs shape knowledge.
# For the standard small test: n1m=32, n2m=16, n3m=32 → n=32*16*32=16384
# We'll just print the first 'slice' of 16 values (one column at i=0, k=0)
# by assuming the dump is C-order (i,j,k) with j the middle index.
# Detect n_wall heuristically: print first few and last few values.
print("\nFirst 4 T_pascal values:", t_pascal[:4])
print("First 4 T_mpm    values:", t_mpm[:4])
print("Last  4 T_pascal values:", t_pascal[-4:])
print("Last  4 T_mpm    values:", t_mpm[-4:])
