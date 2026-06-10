#!/usr/bin/env python3
"""Plot Fig-9 (NOB glycerol RBC) from the extracted CSVs.

  python3 plot_fig9.py          # writes fig9.png (Theta_c vs Delta) + fig9_convergence.png

Reads fig9_thetac_vs_delta.csv (the curve) and timeseries_D*.csv (convergence).
Only rows with status==OK are plotted as solid points; DIVERGED cases are
annotated, not plotted as data.
"""
import csv, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))

def load_master():
    rows = []
    with open(os.path.join(HERE, "fig9_thetac_vs_delta.csv")) as f:
        for line in f:
            if line.startswith("#") or line.startswith("Delta"):
                continue
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            D = float(p[0]); tc = p[1]; status = p[2]
            rows.append((D, (float(tc) if tc else None), status))
    return rows

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available — CSVs are ready; plot them with your tool of choice.")
    raise SystemExit(0)

rows = load_master()
ok = [(D, tc) for D, tc, s in rows if s == "OK" and tc is not None]
bad = [D for D, tc, s in rows if s != "OK"]

# (1) the Fig-9 curve: Theta_c vs Delta
fig, ax = plt.subplots(figsize=(5.5, 4.2))
if ok:
    xs = [d for d, _ in ok]; ys = [t for _, t in ok]
    ax.plot(xs, ys, "o-", color="C3", lw=1.6, ms=7, label="MPM-STD_C (full NOB)")
for D in bad:
    ax.axvline(D, ls=":", color="0.6")
    ax.text(D, ax.get_ylim()[1] if ok else 0.07, f"Δ={D:.0f}\ndiverged", ha="center",
            va="top", fontsize=8, color="0.4")
ax.axhline(0.0, ls="--", color="0.7", lw=1, label="OB (Θ_c=0)")
ax.set_xlabel("Δ = DeltaT  (temperature gap)")
ax.set_ylabel(r"$\Theta_c$  (central-plane mean temp deviation)")
ax.set_title("Fig 9 (proxy): NOB glycerol RBC\nPr=7, Ra=1e5, 64³, full NOB")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig9.png"), dpi=130)
print("wrote fig9.png")

# (2) convergence of the running mean per case
fig2, ax2 = plt.subplots(figsize=(6, 4))
for ts in sorted(glob.glob(os.path.join(HERE, "timeseries_D*.csv"))):
    D = os.path.basename(ts).replace("timeseries_D", "").replace(".csv", "")
    t, th = [], []
    with open(ts) as f:
        r = csv.DictReader(f)
        for row in r:
            t.append(float(row["t"])); th.append(float(row["Theta_c_runmean"]))
    if t:
        ax2.plot(t, th, lw=1.2, label=f"Δ={D}")
ax2.set_xlabel("t (free-fall times)"); ax2.set_ylabel(r"$\Theta_c$ (running mean)")
ax2.set_title("Convergence of Θ_c running mean")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(os.path.join(HERE, "fig9_convergence.png"), dpi=130)
print("wrote fig9_convergence.png")
