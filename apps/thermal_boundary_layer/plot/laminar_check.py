#!/usr/bin/env python3
"""층류 열 경계층 검증 — bl_scalar<n>_stats 의 St(x) 를 상사해와 비교한다 (트립 없는 런).
사용: python3 laminar_check.py stats/laminar_check
  isoscalar: St = α θ'(0) √(U∞/(ν x_abs)) / U∞ = θ'(0) / (Pr √Re_x)         (Pohlhausen; Pr 0.71 → θ'(0) ≈ 0.2927)
  isoflux  : θ_w = 1 − √(x/x₀) g(0)/1.7208,  St = α /(U∞ (1 − θ_w)) = 1.7208 /(Pe g(0) √(x/x₀))  (Pe = Re_δ*₀ Pr)
x_abs = x₀ + x, x₀ = (1/1.7208)² Re_δ*₀ (δ* = 1 인 Blasius 위치)."""
import glob, os, re, sys
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "stats/laminar_check"
RE = 450.0
X0 = (1 / 1.7208) ** 2 * RE


def read_scalar_stats(path):
    header = {}
    rows = []
    for line in open(path):
        if line.startswith("#"):
            for key, value in re.findall(r"(\w+)=([-\d.eE+]+)", line): header[key] = float(value)
            m = re.search(r"wall=(\w+)", line)
            if m: header["wall"] = m.group(1)
            if line.startswith("# section 2"): break
            continue
        if line.strip(): rows.append([float(v) for v in line.split()])
    return header, np.array(rows)


log = sorted(glob.glob(os.path.join(os.path.dirname(folder.rstrip("/")) or ".", "..", "logs", "laminar_check_*.log")))
coeff = {}
if log:
    for line in open(log[-1]):
        m = re.match(r"# scalar (\d+): Pr=([\d.]+) wall=(\w+)\s+laminar wall coefficient ([\d.]+)", line)
        if m: coeff[int(m.group(1))] = (float(m.group(2)), m.group(3), float(m.group(4)))
for n in sorted(coeff):
    pr, wall, c = coeff[n]
    expect = (0.332 * pr ** (1 / 3) if wall == "isoscalar" else 1 / (0.453 * pr ** (1 / 3)))
    print(f"scalar {n}: Pr {pr} {wall:9s} wall coefficient {c:.4f}  (approx. theory {expect:.4f}, {100*(c/expect-1):+.1f}%)")

for path in sorted(glob.glob(os.path.join(folder, "bl_scalar*_stats_*.dat"))):
    n = int(re.search(r"bl_scalar(\d+)_", path).group(1))
    h, d = read_scalar_stats(path)
    if n not in coeff: continue
    pr, wall, c = coeff[n]
    x, st = d[:, 0], d[:, 2]
    x_abs = X0 + x
    re_x = x_abs * RE
    if wall == "isoscalar": st_theory = c / (pr * np.sqrt(re_x))
    else: st_theory = 1.7208 / (RE * pr * c * np.sqrt(x_abs / X0))
    inner = (x > 0.1 * x.max()) & (x < 0.9 * x.max())
    err = 100 * (st[inner] / st_theory[inner] - 1)
    print(f"{os.path.basename(path)}: Pr {pr} {wall:9s}  St error vs similarity: mean {err.mean():+.2f}%  max |{np.abs(err).max():.2f}|%"
          f"  (St at x=Lx/2: {np.interp(0.5*x.max(), x, st):.4e} vs theory {np.interp(0.5*x.max(), x, st_theory):.4e})")
