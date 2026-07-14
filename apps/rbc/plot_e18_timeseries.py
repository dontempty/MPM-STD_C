#!/usr/bin/env python3
"""
E18 buoyancy-fix re-equilibration: Theta_c and Nu vs physical time t, from the
monitor.csv the run actually wrote (apps/rbc/e18_buoyfix_equil/monitor.csv).

monitor.csv's Theta_c/Nu columns are CUMULATIVE means seeded with the restart's
carried-over stats (tc_sum=3465.27, tc_n=60001, wt_sum=786865.81 @ t=17312,
nstat_start=step 2160000/t~18673 is when THIS run started adding new samples on
top of that). Plotted alongside the raw cumulative curve is the LOCAL windowed
value (10 sequential, non-overlapping chunks of the new-accumulation region,
each chunk's own average via consecutive-cumulative-sum differencing) — this is
what actually reveals the monotonic Theta_c drift / noisy-flat Nu discussed in
this session, which the raw cumulative curve visually understates (it's an
average of an average, lags and smooths any real trend).

Usage:  python3 apps/rbc/plot_e18_timeseries.py
Output: apps/rbc/e18_timeseries.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv, os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MON = os.path.join(APP_DIR, 'e18_buoyfix_equil/monitor.csv')

TC_N_OLD = 60001
NSTAT_START_STEP = 2160000
alpha_T = 1.9815e-05
nxny = 256 * 256

steps, ts, tcs, nus = [], [], [], []
with open(MON) as f:
    r = csv.DictReader(f)
    for row in r:
        steps.append(int(row['step'])); ts.append(float(row['t']))
        tcs.append(float(row['Theta_c'])); nus.append(float(row['Nu']))
steps = np.array(steps); ts = np.array(ts); tcs = np.array(tcs); nus = np.array(nus)

def tc_n_at(step):
    if step < NSTAT_START_STEP:
        return TC_N_OLD
    return TC_N_OLD + (step - NSTAT_START_STEP) + 1

tc_n = np.array([tc_n_at(s) for s in steps])
is_new = steps >= NSTAT_START_STEP

# ── local (non-cumulative) chunk averages over the new-accumulation region ──
new_idx = np.where(is_new)[0]
n_chunks = 10
chunk_bounds = np.array_split(new_idx, n_chunks)
chunk_t, chunk_tc, chunk_nu = [], [], []
for idx in chunk_bounds:
    lo, hi = idx[0], idx[-1]
    n_lo, n_hi = tc_n[lo], tc_n[hi]
    sum_lo, sum_hi = tcs[lo]*n_lo, tcs[hi]*n_hi
    local_tc = (sum_hi - sum_lo) / (n_hi - n_lo) if n_hi > n_lo else tcs[hi]
    wt_lo = (nus[lo]-1)*n_lo*alpha_T*nxny
    wt_hi = (nus[hi]-1)*n_hi*alpha_T*nxny
    local_nu = 1 + ((wt_hi-wt_lo)/nxny/(n_hi-n_lo))/alpha_T if n_hi > n_lo else nus[hi]
    chunk_t.append(0.5*(ts[lo]+ts[hi]))
    chunk_tc.append(local_tc)
    chunk_nu.append(local_nu)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

t_switch = ts[is_new][0]

# Theta_c panel
ax1.plot(ts[~is_new], tcs[~is_new], '--', color='0.6', lw=1.5,
         label='cumulative (pre-accumulation, frozen restart baseline)')
ax1.plot(ts[is_new], tcs[is_new], '-', color='tab:blue', lw=2.5, zorder=4,
         label='cumulative mean (as printed by monitor.csv)')
ax1.plot(chunk_t, chunk_tc, 'o-', color='darkorange', lw=2.5, markersize=7,
         zorder=5, label='local window avg (10 sequential chunks — the real trend)')
ax1.axhline(0.071, ls=':', color='green', lw=1.3, label='reference (Pan 2022, ΔT=20K): 0.071')
ax1.axhline(0.052, ls=':', color='gray', lw=1.0, label='pre-fix equilibrium: 0.052')
ax1.axvline(t_switch, ls=':', color='k', lw=0.8, alpha=0.5)
ax1.set_ylabel(r'$\Theta_c$')
ax1.set_title(r'E18: $\Theta_c$ — clean monotonic rise, still climbing at run end', fontsize=11)
ax1.legend(fontsize=8, loc='best')
ax1.grid(True, alpha=0.3)

# Nu panel
ax2.plot(ts[~is_new], nus[~is_new], '--', color='0.6', lw=1.2)
ax2.plot(ts[is_new], nus[is_new], '-', color='tab:red', lw=1.4,
         label='cumulative mean (as printed by monitor.csv)')
ax2.plot(chunk_t, chunk_nu, 'o-', color='darkorange', lw=2.2, markersize=6,
         zorder=5, label='local window avg (10 sequential chunks)')
ax2.axhline(9.61, ls=':', color='green', lw=1.3, label='reference (Pan 2022, ΔT=20K): 9.61')
ax2.axvline(t_switch, ls=':', color='k', lw=0.8, alpha=0.5)
ax2.set_xlabel('t (free-fall time units)')
ax2.set_ylabel('Nu')
ax2.set_title('E18: Nu — noisy, no clear trend (already ~stationary)', fontsize=11)
ax2.legend(fontsize=8, loc='best')
ax2.grid(True, alpha=0.3)

fig.suptitle('E18 buoyancy-fix (Report/34) re-equilibration — ΔT=20K, Ra=1e6, 256³, np=1\n'
             f'warm-start t=17312 → t={ts[-1]:.0f}  (+{ts[-1]-17312.49:.0f} t.u., 400k steps)',
             fontsize=11, y=1.0)
plt.tight_layout()
out = os.path.join(APP_DIR, 'e18_timeseries.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
