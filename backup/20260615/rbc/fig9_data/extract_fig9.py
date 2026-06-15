#!/usr/bin/env python3
"""Extract Fig-9 (NOB glycerol RBC) data from the rbc solver logs.

Fig 9 plots the central-plane mean temperature deviation Theta_c vs the
temperature gap Delta (=DeltaT). For each Delta case this script parses the
monitor log (step, t, dt, div, Theta_c) into a per-case time series and
collects the converged Theta_c (the solver's running mean over the averaging
window, printed on the final line) into a master CSV the user can plot.

Theta_c printed by the solver IS the running mean tc_sum/tc_n over steps
>= nstat_start (main.cpp:238-244); a flat tail ⇒ converged.

Config of this dataset (CPU-feasible proxy, NOT the exact Fig-9 run):
  Pr=7, Ra=1e5, 64^3, Gamma=2 (Lx=Ly=2, Lz=1), x,y periodic, z wall (gravity),
  hot bottom theta=+0.5 / cold top theta=-0.5, full NOB (var-mu + var-kappa +
  NOB buoyancy), glycerol property polynomials scaled by DeltaT.
  The exact Fig 9 is Pr=2547, Ra=1e6, Gamma=6, 128^3 (GPU territory).
"""
import re, os, glob, math

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "..", "logs")

MON = re.compile(r"step=(\d+)\s+t=(\S+)\s+dt=(\S+)\s+div=(\S+)\s+Theta_c=(\S+)")
DONE = re.compile(r"done:.*Theta_c=([0-9eE.+-]+)")
HEAD = re.compile(r"Ra=(\S+)\s+Pr=(\S+).*n=(\d+)x(\d+)x(\d+)")

def parse(logpath):
    """Return (series, meta) where series=list of (step,t,dt,div,theta) and
    meta has final theta_c, final div, final t, grid, Ra, Pr, status."""
    series, meta = [], {"Ra": None, "Pr": None, "grid": None,
                        "theta_c": None, "div_final": None, "t_final": None,
                        "status": "no-log"}
    if not os.path.exists(logpath):
        return series, meta
    with open(logpath) as f:
        for line in f:
            h = HEAD.search(line)
            if h:
                meta["Ra"], meta["Pr"] = h.group(1), h.group(2)
                meta["grid"] = f"{h.group(3)}x{h.group(4)}x{h.group(5)}"
            m = MON.search(line)
            if m:
                step = int(m.group(1)); t = float(m.group(2))
                dt = float(m.group(3)); dv = float(m.group(4)); th = float(m.group(5))
                series.append((step, t, dt, dv, th))
            d = DONE.search(line)
            if d:
                meta["theta_c"] = float(d.group(1))
    if series:
        meta["div_final"] = series[-1][3]
        meta["t_final"] = series[-1][1]
        # diverged ⇒ dt collapsed / div exploded near the end
        last_dt, last_div = series[-1][2], series[-1][3]
        if last_dt < 1e-12 or last_div > 1e-3:
            meta["status"] = "DIVERGED"
            # converged theta_c is meaningless for a diverged run
            meta["theta_c_note"] = "run diverged (stiffness); value unreliable"
        else:
            meta["status"] = "OK"
    return series, meta

# Δ -> log file. Stable converged runs use fig9_<D>.log; the diverged Δ=60
# uses the earlier sw60.log (default-CFL) trajectory.
cases = [(20, "fig9_20.log"), (30, "fig9_30.log"),
         (40, "fig9_40.log"), (50, "fig9_50.log"), (60, "sw60.log")]

master = []
for D, fn in cases:
    series, meta = parse(os.path.join(LOGDIR, fn))
    # per-case time series
    tspath = os.path.join(HERE, f"timeseries_D{D}.csv")
    with open(tspath, "w") as f:
        f.write("step,t,dt,div,Theta_c_runmean\n")
        for s in series:
            f.write(f"{s[0]},{s[1]:.6g},{s[2]:.6g},{s[3]:.3e},{s[4]:.6f}\n")
    # convergence diagnostic: slope of Theta_c over the last 25% of samples
    tail = [s[4] for s in series[int(0.75*len(series)):]] if series else []
    drift = (max(tail) - min(tail)) if tail else float("nan")
    master.append((D, meta, drift, len(series)))
    print(f"Delta={D:3d}  status={meta['status']:8s}  "
          f"Theta_c={meta['theta_c']}  t_final={meta['t_final']}  "
          f"div={meta['div_final']}  tail_drift={drift:.4g}  ({fn})")

# master Fig-9 CSV
mpath = os.path.join(HERE, "fig9_thetac_vs_delta.csv")
with open(mpath, "w") as f:
    f.write("# Fig-9 NOB glycerol RBC: central-plane mean temp deviation vs DeltaT\n")
    f.write("# CPU proxy: Pr=7, Ra=1e5, 64^3, Gamma=2, full NOB. Exact Fig9 = Pr=2547,Ra=1e6,128^3 (GPU).\n")
    f.write("# Theta_c = running mean <theta>_{x,y} at mid-height over averaging window; tail_drift = spread over last 25% (smaller=more converged).\n")
    f.write("Delta,Theta_c,status,t_final,div_final,tail_drift,Ra,Pr,grid\n")
    for D, meta, drift, n in master:
        tc = "" if meta["theta_c"] is None else f"{meta['theta_c']:.5f}"
        f.write(f"{D},{tc},{meta['status']},{meta['t_final']},"
                f"{meta['div_final']},{drift:.4g},{meta['Ra']},{meta['Pr']},{meta['grid']}\n")
print(f"\nwrote {mpath}")
print(f"wrote per-case timeseries_D*.csv in {HERE}")
