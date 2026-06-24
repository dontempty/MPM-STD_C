#!/usr/bin/env python3
"""Visualize a DHVC restart (.bin = raw float64, C-order, no header) — slice in a
chosen direction, ALL variables (U,V,W,T,P).

Usage:
    python3 plot_restart.py <restart_dir> <axis> <pos>
      <axis> : x | y | z              (slice-normal direction)
      <pos>  : 0..1  -> fraction of the domain along that axis (0.5 = middle)
               >1    -> explicit integer cell index along that axis
Examples:
    python3 plot_restart.py apps/dhvc/restart_1e7_t150 z 0.5     # mid-gap x-y plane
    python3 plot_restart.py apps/dhvc/restart_1e7_t150 y 0.5     # mid-span x-z plane
    python3 plot_restart.py apps/dhvc/restart_1e7_t150 x 0.25    # x-quarter y-z plane

Field layout: F[i,j,k], i=x(vertical/stream), j=y(span), k=z(wall-normal). All three
axes here are periodic-or-wall as in DHVC; domain = Lx×Ly×Lz = 8×4×1.
"""
import numpy as np, matplotlib, sys, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d    = sys.argv[1] if len(sys.argv) > 1 else "apps/dhvc/restart_1e7_t150"
axis = (sys.argv[2] if len(sys.argv) > 2 else "z").lower()
pos  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
assert axis in ("x", "y", "z"), "axis must be x, y or z"

# dims from meta.txt (n1m,n2m,n3m); fallback to 384x96x192
n = {"n1m": 384, "n2m": 96, "n3m": 192}
mp = os.path.join(d, "meta.txt")
if os.path.exists(mp):
    for line in open(mp):
        t = line.split()
        if len(t) == 2 and t[0] in n:
            n[t[0]] = int(float(t[1]))
n1, n2, n3 = n["n1m"], n["n2m"], n["n3m"]
L  = {"x": 8.0, "y": 4.0, "z": 1.0}            # DHVC domain
na = {"x": n1, "y": n2, "z": n3}[axis]
idx = int(round(pos * (na - 1))) if 0.0 <= pos <= 1.0 else int(pos)
idx = max(0, min(na - 1, idx))
phys = idx / (na - 1) * L[axis]

def load(name):
    a = np.fromfile(os.path.join(d, f"{name}.bin"), dtype=np.float64)
    if a.size != n1 * n2 * n3:
        raise SystemExit(f"{name}.bin size {a.size} != {n1*n2*n3} (check meta dims)")
    return a.reshape(n1, n2, n3)                # [x, y, z]

# slice → (sl, vert_axis, horiz_axis). sl is indexed [vert_idx, horiz_idx]; the two
# remaining axes are taken in (x<y<z) order = (vertical, horizontal).
def take(F):
    if axis == "x": return F[idx, :, :], "y", "z"   # y-z plane (vert=y, horiz=z)
    if axis == "y": return F[:, idx, :], "x", "z"   # x-z plane (vert=x, horiz=z)
    return F[:, :, idx], "x", "y"                   # x-y plane (vert=x, horiz=y)
coord = {"x": np.linspace(0, L["x"], n1),
         "y": np.linspace(0, L["y"], n2),
         "z": np.linspace(0, L["z"], n3)}

varnames = ["U", "V", "W", "T", "P"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle(f"{d}   slice {axis}={idx} (phys {axis}={phys:.3g})   grid {n1}x{n2}x{n3}", fontsize=12)
for a, name in zip(axes.flat, varnames):
    F = load(name)
    sl, vert, horiz = take(F)                  # sl[vert_idx, horiz_idx]
    cmap = "coolwarm" if name == "T" else "RdBu_r"
    # pcolormesh(X=horiz, Y=vert, C=sl): C shape (len(vert), len(horiz)) — matches
    im = a.pcolormesh(coord[horiz], coord[vert], sl, shading="auto", cmap=cmap)
    a.set(title=f"{name}  [{sl.min():.3g}, {sl.max():.3g}]", xlabel=horiz, ylabel=vert)
    fig.colorbar(im, ax=a, fraction=0.046)
axes.flat[-1].axis("off")                       # 6th panel empty (5 vars)

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(d, f"slice_{axis}{idx}.png")
fig.savefig(out, dpi=110)
print(f"saved {out}  (grid {n1}x{n2}x{n3}, slice {axis}={idx}, phys={phys:.3g})")
