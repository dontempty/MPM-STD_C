#!/usr/bin/env python3
"""
Fig 9 axes, with the REAL production re-equilibration result (E18) added:
Report/34 NOB-buoyancy face-interpolation fix, warm-started from the pre-fix
equilibrium restart (restart_e14_h256, t=17312, Theta_c=0.052) and run
+2688 time units (dt=0.01->0.006 adaptive, 400k steps, single A100, ~11.6h,
2026-07-10/11) at DeltaT=20K, Ra=1e6, Pr=2547, Gamma=6, 256^3.

Theta_c printed by the binary is a CUMULATIVE mean seeded with the restart's
carried-over stats (tc_sum=3465.27, tc_n=60001) — see the project's own
"판독 요령" note (monitor Theta_c is a running mean; must back out the
instantaneous/windowed value). This script does that explicitly:
  new_window_avg = (tc_sum_final - tc_sum_old) / (tc_n_final - tc_n_old)
      -- average over ALL 200001 samples accumulated during E18 (fair,
         unbiased by the diluted old baseline)
  tail_window_avg = same but only the last 10% of E18's own samples
      -- shows whether it's still drifting (it is, slightly, upward)

Usage:  python3 apps/rbc/plot_fig9_e18_production.py
Output: apps/rbc/fig9_e18_production.png (separate file, does not touch
        fig9_gpu_validation.png or fig9_buoyancy_fix_ab.png)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv, os, re

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(APP_DIR, '../..'))
DATA_DIR = os.path.join(APP_DIR, 'fig9_data')

# ── Old (pre-fix) production point at DeltaT=20K.
# fig9_nob_gpu_out_20_x/monitor.csv (the source used earlier this session, via
# plot_rbc_fig9.py, to get Theta_c=0.0404/n=451/last_step=500000) was
# ACCIDENTALLY TRUNCATED to its header by a later debug toml
# (input_e18_calib5_128np2.toml) that copied dir_stats from the production toml
# without overriding it. Re-citing the value captured before the truncation
# (see this session's transcript / Report/34) rather than re-deriving it.
old_tc = 0.0404
old_y = 20.0 * old_tc
print(f'[old, pre-fix, ~500k steps, RE-CITED — source monitor.csv truncated by debug-run accident] '
      f'Theta_c={old_tc:.4f} -> y={old_y:.3f} K')

# ── E18: parse step/Theta_c/tc_n from the production log, back out windowed avgs
LOG = os.path.join(APP_DIR, 'logs/rbc_e18_buoyfix_equil.log')
TC_SUM_OLD, TC_N_OLD = 3465.2701613522427, 60001   # restart_e14_h256/stats.txt

rows = []
with open(LOG) as f:
    buf_step = None
    for line in f:
        m1 = re.search(r'step=(\d+).*?Theta_c=([\d.]+)', line)
        if m1:
            buf_step = (int(m1.group(1)), float(m1.group(2)))
            continue
        m2 = re.search(r'tc_n=(\d+)\)', line)
        if m2 and buf_step is not None:
            rows.append((buf_step[0], buf_step[1], int(m2.group(1))))
            buf_step = None

step_f, tc_f, n_f = rows[-1]
tc_sum_f = tc_f * n_f
new_n = n_f - TC_N_OLD
new_avg = (tc_sum_f - TC_SUM_OLD) / new_n

target_n = TC_N_OLD + int(0.9 * new_n)
step_m, tc_m, n_m = min(rows, key=lambda r: abs(r[2] - target_n))
tail_n = n_f - n_m
tail_avg = (tc_sum_f - tc_m * n_m) / tail_n

new_y, tail_y = 20.0 * new_avg, 20.0 * tail_avg
print(f'[E18, post-fix] full new-window (n={new_n}, {2688:.0f} t.u. total): '
      f'Theta_c={new_avg:.4f} -> y={new_y:.3f} K')
print(f'[E18, post-fix] last-10% tail window (n={tail_n}): '
      f'Theta_c={tail_avg:.4f} -> y={tail_y:.3f} K  (still rising -> not hard-converged)')

# ── Reference curves ─────────────────────────────────────────────────────────
def load_reference_csv(path):
    dt_arr, y_arr = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                dt_arr.append(float(parts[0])); y_arr.append(float(parts[1]))
    return np.array(dt_arr), np.array(y_arr)

pascal_dt, pascal_y = load_reference_csv(os.path.join(DATA_DIR, 'PaScaL_TCS.csv'))
horm_dt,   horm_y   = load_reference_csv(os.path.join(DATA_DIR, 'Horm_et_al.csv'))
ref_20 = float(np.interp(20.0, pascal_dt, pascal_y))
print(f'[reference] PaScaL_TCS DNS @ DeltaT=20K: {ref_20:.3f} K')
print(f'  old error: {100*(old_y-ref_20)/ref_20:+.1f}%   '
      f'new(full) error: {100*(new_y-ref_20)/ref_20:+.1f}%   '
      f'new(tail) error: {100*(tail_y-ref_20)/ref_20:+.1f}%')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.plot(horm_dt, horm_y, 'k^-', markersize=5, linewidth=1.2, zorder=2,
        label='Horm et al. (experiment)')
ax.plot(pascal_dt, pascal_y, 'bs', markersize=8, markeredgecolor='k',
        markeredgewidth=0.6, linewidth=1.5, zorder=4, label='PaScaL_TCS DNS (Ra=1e6)')

ax.plot(20, old_y, 'o', color='lightcoral', markersize=10, markeredgecolor='k',
        markeredgewidth=0.8, zorder=5, label='MPM-STD GPU, PRE-fix (~500k steps, cell-centered buoyancy)')
ax.annotate(f'{old_y:.2f} K\n({100*(old_y-ref_20)/ref_20:+.0f}% vs DNS)',
            xy=(20, old_y), xytext=(23, old_y*0.75),
            fontsize=8, color='firebrick', fontweight='bold')

# E18 post-fix: plot as a vertical range (full-window -> tail-window) since it's
# still slowly drifting upward — not a single converged dot.
ax.plot([20, 20], [new_y, tail_y], '-', color='darkorange', lw=2.5, zorder=5)
ax.plot(20, new_y, 'D', color='darkorange', markersize=9, markeredgecolor='k',
        markeredgewidth=0.8, zorder=6,
        label='MPM-STD GPU, POST-fix (E18, +2688 t.u. from pre-fix equilibrium)\nfull-window avg -> tail-window avg (still rising)')
ax.plot(20, tail_y, '^', color='darkorange', markersize=9, markeredgecolor='k',
        markeredgewidth=0.8, zorder=6)
ax.annotate(f'{new_y:.2f}-{tail_y:.2f} K\n({100*(new_y-ref_20)/ref_20:+.0f}% to {100*(tail_y-ref_20)/ref_20:+.0f}% vs DNS)\nstill rising, not plateaued',
            xy=(20, tail_y), xytext=(23, tail_y*1.02),
            fontsize=8, color='darkorange', fontweight='bold')

ax.annotate('', xy=(20, tail_y), xytext=(20, old_y),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2, ls='--'), zorder=3)

ax.axhline(0.0, ls='--', color='0.7', lw=1.0, label='OB baseline')

ax.set_xlabel(r'$\Delta T$ (K)', fontsize=13)
ax.set_ylabel(r'$\Delta(\Theta_c - \Theta_m)$  (K)', fontsize=12)
ax.set_title('Fig 9 — NOB RBC (glycerol): E18 buoyancy-fix production re-equilibration\n'
             r'$Ra=10^6$, $Pr=2547$, $\Gamma=6$, 256³, single ΔT=20K point',
             fontsize=11)
ax.legend(fontsize=7.5, loc='upper left')
ax.set_xlim(0, 75)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

plt.tight_layout()
out = os.path.join(APP_DIR, 'fig9_e18_production.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
