#!/usr/bin/env python3
"""
Combined E18 (long re-equilibration, contaminated cumulative stats -> local-window
corrected) + E19 (short stats_reset=1 continuation, clean cumulative stats) view of
Theta_c and Nu vs physical time. Answers: did the E18 monotonic Theta_c rise
continue into E19, or has it plateaued?

Usage:  python3 apps/rbc/plot_e18_e19_combined.py
Output: apps/rbc/e18_e19_combined.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv, os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
alpha_T = 1.9815e-05
nxny = 256 * 256

def read_monitor(path):
    steps, ts, tcs, nus = [], [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row['step'])); ts.append(float(row['t']))
            tcs.append(float(row['Theta_c'])); nus.append(float(row['Nu']))
    return np.array(steps), np.array(ts), np.array(tcs), np.array(nus)

def local_chunks(steps, ts, tcs, nus, tc_n_fn, n_chunks):
    idx = np.arange(len(steps))
    bounds = np.array_split(idx, n_chunks)
    out_t, out_tc, out_nu = [], [], []
    for b in bounds:
        lo, hi = b[0], b[-1]
        n_lo, n_hi = tc_n_fn(steps[lo]), tc_n_fn(steps[hi])
        sum_lo, sum_hi = tcs[lo]*n_lo, tcs[hi]*n_hi
        out_tc.append((sum_hi-sum_lo)/(n_hi-n_lo) if n_hi > n_lo else tcs[hi])
        wt_lo = (nus[lo]-1)*n_lo*alpha_T*nxny
        wt_hi = (nus[hi]-1)*n_hi*alpha_T*nxny
        out_nu.append(1 + ((wt_hi-wt_lo)/nxny/(n_hi-n_lo))/alpha_T if n_hi > n_lo else nus[hi])
        out_t.append(0.5*(ts[lo]+ts[hi]))
    return np.array(out_t), np.array(out_tc), np.array(out_nu)

# ── E18: long run, cumulative stats seeded with restart_e14_h256's old baseline
TC_N_OLD, NSTAT_START_E18 = 60001, 2160000
e18_steps, e18_t, e18_tc, e18_nu = read_monitor(os.path.join(APP_DIR, 'e18_buoyfix_equil/monitor.csv'))
e18_new = e18_steps >= NSTAT_START_E18
tc_n_e18 = lambda step: TC_N_OLD + (step - NSTAT_START_E18) + 1
e18c_t, e18c_tc, e18c_nu = local_chunks(e18_steps[e18_new], e18_t[e18_new],
                                        e18_tc[e18_new], e18_nu[e18_new], tc_n_e18, 10)

# ── E19: short continuation, stats_reset=1 -> clean cumulative (no dilution)
NSTAT_START_E19 = 2360000
e19_steps, e19_t, e19_tc, e19_nu = read_monitor(os.path.join(APP_DIR, 'e19_stats_reset/monitor.csv'))
tc_n_e19 = lambda step: (step - NSTAT_START_E19) // 200 * 200 + 200  # matches nmonitor=200 cadence
e19c_t, e19c_tc, e19c_nu = local_chunks(e19_steps, e19_t, e19_tc, e19_nu, tc_n_e19, 5)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=False)

for ax, e18_local, e19_local, ref, label in (
        (ax1, e18c_tc, e19c_tc, 0.071, r'$\Theta_c$'),
        (ax2, e18c_nu, e19c_nu, 9.61, 'Nu')):
    ax.plot(e18c_t, e18_local, 'o-', color='darkorange', lw=2.2, markersize=6,
            label='E18 local window avg (10 chunks, +2688 t.u.)')
    ax.plot(e19c_t, e19_local, 's-', color='mediumblue', lw=2.2, markersize=6,
            label='E19 local window avg (5 chunks, +135 t.u., stats_reset=1 — clean)')
    ax.axhline(ref, ls=':', color='green', lw=1.3, label=f'reference (Pan 2022, ΔT=20K): {ref}')
    ax.axvline(e19c_t[0] - (e19c_t[1]-e19c_t[0])/2, ls='--', color='k', lw=0.8, alpha=0.4)
    ax.set_ylabel(label)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

ax1.set_title(r'$\Theta_c$: E18 steady climb $\to$ E19 flattens around 0.0647 (plateau signal)', fontsize=11)
ax2.set_title('Nu: noisy in both legs, no sustained trend', fontsize=11)
ax2.set_xlabel('t (free-fall time units)')

fig.suptitle('E18 → E19: Theta_c/Nu local-window trend, RBC NOB buoyancy-fix (Report/34,35)\n'
             r'$Ra=10^6$, $Pr=2547$, $\Delta T=20$K, 256³',
             fontsize=11, y=1.0)
plt.tight_layout()
out = os.path.join(APP_DIR, 'e18_e19_combined.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')

print(f"\nE18 last chunk theta_c: {e18c_tc[-1]:.5f}")
print(f"E19 full-window theta_c (clean): {e19_tc[-1]:.5f}")
print(f"E19 chunk theta_c range: {e19c_tc.min():.5f} - {e19c_tc.max():.5f}")
print(f"E18 last chunk Nu: {e18c_nu[-1]:.4f}")
print(f"E19 full-window Nu (clean): {e19_nu[-1]:.4f}")
print(f"E19 chunk Nu range: {e19c_nu.min():.4f} - {e19c_nu.max():.4f}")
