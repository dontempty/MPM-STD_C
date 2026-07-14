#!/usr/bin/env python3
"""
Fig 9 axes (same as plot_rbc_fig9.py / fig9_gpu_validation.png), extended with the
Report/34 NOB-buoyancy face-interpolation fix.

IMPORTANT — what the new markers are and are NOT:
  The BEFORE/AFTER triangles are from a short isolated A/B smoke test
  (32x32x48, DeltaT=20K, t=0..60, single rank — see Report/34 sec.4), run ONLY to
  confirm the fix changes the physics and stays stable. They are NOT converged
  equilibrium points and are NOT comparable in rigor to the red circles (which are
  200k-500k step production runs). Plotted small, low zorder, explicitly labeled
  "transient t=60, NOT equilibrium" so they cannot be mistaken for a re-validated
  Fig 9 point. A real re-validation needs full production reruns (deferred; see
  Report/34 sec.5).

Usage:  python3 apps/rbc/plot_fig9_buoyancy_fix_ab.py
Output: apps/rbc/fig9_buoyancy_fix_ab.png   (separate file — does NOT overwrite
        fig9_gpu_validation.png, which stays the pre-fix production record)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv, os, re

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(APP_DIR, '../..'))
DATA_DIR = os.path.join(APP_DIR, 'fig9_data')

# ── Old (pre-buoyancy-fix) production GPU points — same data pipeline as
# plot_rbc_fig9.py. DeltaT=60K's source log was cleaned up after the original
# fig9_gpu_validation.png was rendered (2026-06-14/24); its value (4.10 K,
# not converged) is re-cited from that existing PNG, not recomputed.
def parse_monitor_csv(path, nstat_start=0):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                step = int(r['step']); tc = float(r['Theta_c'])
                if step >= nstat_start and tc != 0.0:
                    rows.append((step, tc))
            except (ValueError, KeyError):
                pass
    return rows

def tail_mean(vals, frac=0.1):
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]))

OLD_CASES = {
    20: ('apps/rbc/fig9_nob_gpu_out_20_x/monitor.csv', 50000),
    40: ('apps/rbc/fig9_nob_gpu_out_40_x/monitor.csv', 50000),
}
old_dt, old_y, old_note = [], [], []
for delta_t, (rel, nstat_start) in sorted(OLD_CASES.items()):
    rows = parse_monitor_csv(os.path.join(ROOT_DIR, rel), nstat_start)
    if rows:
        tc = tail_mean([v for _, v in rows])
        old_dt.append(delta_t); old_y.append(delta_t * tc)
        old_note.append(f'{delta_t*tc:.2f} K')
        print(f'[old production, pre-fix] DeltaT={delta_t}K: Theta_c={tc:.4f} -> y={delta_t*tc:.2f} K '
              f'(n={len(rows)}, last_step={rows[-1][0]})')
# 60K: log cleaned up — re-cite the value from the existing validated PNG.
old_dt.append(60); old_y.append(4.10); old_note.append('4.10 K\n(step 200K, not converged;\nvalue re-cited, log deleted)')
old_dt, old_y = np.array(old_dt), np.array(old_y)

# ── New: Report/34 buoyancy-fix isolation A/B (short transient, NOT equilibrium)
def parse_run_log(path):
    rows = []
    pat = re.compile(r't=([\d.]+).*?Theta_c=([\d.e+-]+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                rows.append((float(m.group(1)), float(m.group(2))))
    return rows

DELTA_T_AB = 20.0  # input_verify_buoy_ab.toml DeltaT
before_rows = parse_run_log(os.path.join(APP_DIR, 'verify_buoy_ab_BEFORE_cellcentered.log'))
after_rows  = parse_run_log(os.path.join(APP_DIR, 'verify_buoy_ab_AFTER_faceinterp.log'))
before_t  = np.array([t for t, _ in before_rows]); before_tc = np.array([tc for _, tc in before_rows])
after_t   = np.array([t for t, _ in after_rows]);  after_tc  = np.array([tc for _, tc in after_rows])
before_y  = DELTA_T_AB * before_tc
after_y   = DELTA_T_AB * after_tc
print(f'[A/B transient] BEFORE(cell-centered) final t={before_t[-1]:.0f}: y={before_y[-1]:.3f} K')
print(f'[A/B transient] AFTER(face-interp)    final t={after_t[-1]:.0f}: y={after_y[-1]:.3f} K')

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

# ── Plot (same axes/style as plot_rbc_fig9.py) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.plot(horm_dt, horm_y, 'k^-', markersize=5, linewidth=1.2, zorder=2,
        label='Horm et al. (experiment)')
ax.plot(pascal_dt, pascal_y, 'bs', markersize=8, markeredgecolor='k',
        markeredgewidth=0.6, linewidth=1.5, zorder=4, label='PaScaL_TCS DNS (Ra=1e6)')
ax.plot(old_dt, old_y, 'ro', markersize=10, markeredgecolor='k', markeredgewidth=0.8,
        linewidth=1.5, zorder=5, label='MPM-STD GPU, pre-fix production\n(200k-500k steps, cell-centered NOB buoyancy)')
for dt, y, note in zip(old_dt, old_y, old_note):
    ax.annotate(note, xy=(dt, y), xytext=(dt + 1.5, y * 1.05),
                fontsize=7.5, color='crimson', fontweight='bold')

# New A/B markers — deliberately small/muted, offset off the DeltaT=20 line so they
# don't visually compete with the real (converged) red circle.
AB_X = DELTA_T_AB - 3.5
ax.plot(AB_X, before_y[-1], marker='x', color='gray', markersize=9, markeredgewidth=2,
        zorder=6, label='cell-centered buoyancy (pre-fix code), t=60 transient')
ax.plot(AB_X, after_y[-1], marker='^', color='darkorange', markersize=9,
        markeredgecolor='k', markeredgewidth=0.6, zorder=6,
        label='face-interpolated buoyancy (Report/34 fix), t=60 transient')
ax.annotate('', xy=(AB_X, after_y[-1]), xytext=(AB_X, before_y[-1]),
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5), zorder=6)
ax.annotate('t=60 ONLY\n(NOT equilibrium—illustrative)',
            xy=(AB_X, max(before_y[-1], after_y[-1])),
            xytext=(AB_X - 9.5, max(before_y[-1], after_y[-1]) + 0.9),
            fontsize=7.5, color='#555', style='italic',
            arrowprops=dict(arrowstyle='-', color='#999', lw=0.8))

ax.axhline(0.0, ls='--', color='0.7', lw=1.0, label='OB baseline')

ax.set_xlabel(r'$\Delta T$ (K)', fontsize=13)
ax.set_ylabel(r'$\Delta(\Theta_c - \Theta_m)$  (K)', fontsize=12)
ax.set_title('Fig 9 — NOB RBC (glycerol), with Report/34 buoyancy-fix A/B overlay\n'
             r'$Ra=10^6$, $Pr=2547$, $\Gamma=6$: $\Delta(\Theta_c-\Theta_m)$ vs $\Delta T$',
             fontsize=11)
ax.legend(fontsize=7.5, loc='upper left')
ax.set_xlim(0, 75)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

plt.tight_layout()
out = os.path.join(APP_DIR, 'fig9_buoyancy_fix_ab.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
