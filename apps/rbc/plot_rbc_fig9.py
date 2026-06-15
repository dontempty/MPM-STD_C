#!/usr/bin/env python3
"""
Fig 9 — NOB glycerol RBC: Theta_c vs DeltaT
  Ra=1e6, Pr=2547, Gamma=6

Data priority per case:
  1. {dir_stats}/monitor.csv (new binary — columns: step,t,dt,div,Theta_c,Nu)
  2. apps/rbc/logs/{logfile}   (old binary — regex parse)

Reference data (both in percent → divide by 100):
  - apps/rbc/fig9_data/PaScaL_TCS.csv   : (DeltaT, Theta_c%)
  - apps/rbc/fig9_data/Horm_et_al.csv   : (DeltaT, Theta_c%)

Usage:  python3 apps/rbc/plot_rbc_fig9.py
Output: apps/rbc/fig9_gpu_validation.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv, os, re

APP_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(APP_DIR, '../..'))
LOG_DIR  = os.path.join(APP_DIR, 'logs')
DATA_DIR = os.path.join(APP_DIR, 'fig9_data')

# ── Case definitions: DeltaT → (dir_stats, fallback_logs, nstat_start)
# dir_stats: where monitor.csv is written by new binary
# fallback_logs: list of log filenames tried in order if monitor.csv absent
# nstat_start: skip samples from stats accumulation warm-up
CASES = {
    20: ('apps/rbc/fig9_nob_gpu_out_20_x/',   ['g25a_fig9_gpu_20_x.log', 'g23_fig9_gpu_20_long.log'],  50000),
    40: ('apps/rbc/fig9_nob_gpu_out_40_x/',    ['g25c_fig9_gpu_40_x.log', 'g20_fig9_gpu_40.log'],        50000),
    60: ('apps/rbc/fig9_nob_gpu_out_60_x/',    ['g25b_fig9_gpu_60_x.log', 'g24_fig9_gpu_60_long.log'], 100000),
}

def _tail_mean(vals, frac=0.1):
    """Average over the last `frac` of a list (Theta_c is already a running mean)."""
    n = max(1, int(len(vals) * frac))
    return float(np.mean(vals[-n:]))

def parse_monitor_csv(path, nstat_start=0):
    """Read monitor.csv → list of (step, Theta_c). Returns [] if file absent."""
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

def parse_log_file(path, nstat_start=0):
    """Regex-parse old-style log → list of (step, Theta_c). Returns [] if absent."""
    rows = []
    if not os.path.exists(path):
        return rows
    pat = re.compile(r'step=(\d+).*?Theta_c=([\d.e+-]+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                step = int(m.group(1)); tc = float(m.group(2))
                if step >= nstat_start and tc != 0.0:
                    rows.append((step, tc))
    return rows

def load_reference_csv(path):
    """Load (DeltaT_K, y_K) CSV → (DeltaT_arr, y_K_arr).
    y_K = Fig-9 y-axis = Δ(Θ_c − Θ_m) = ΔT × Θ_c  [Kelvin]."""
    dt_arr, tc_arr = [], []
    if not os.path.exists(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    dt_arr.append(float(parts[0]))
                    tc_arr.append(float(parts[1]))   # raw Kelvin — no conversion
                except ValueError:
                    pass
    return np.array(dt_arr), np.array(tc_arr)

# ── 1. Load GPU results ────────────────────────────────────────────────────────
gpu_dt, gpu_tc, gpu_info = [], [], []
for delta_t, (dir_rel, logfiles, nstat_start) in sorted(CASES.items()):
    dir_abs = os.path.join(ROOT_DIR, dir_rel)
    mon_path = os.path.join(dir_abs, 'monitor.csv')

    rows = parse_monitor_csv(mon_path, nstat_start)
    src = 'monitor.csv'
    if not rows:
        for logfile in logfiles:
            log_path = os.path.join(LOG_DIR, logfile)
            rows = parse_log_file(log_path, nstat_start)
            if rows:
                src = os.path.basename(log_path)
                break

    if rows:
        tc_vals = [tc for _, tc in rows]
        tc_mean = _tail_mean(tc_vals)
        last_step = rows[-1][0]
        gpu_dt.append(delta_t)
        gpu_tc.append(tc_mean)
        gpu_info.append((src, last_step, len(tc_vals)))
        print(f'DeltaT={delta_t}K: Theta_c={tc_mean:.4f}  '
              f'(n={len(tc_vals)}, last_step={last_step}, src={src})')
    else:
        print(f'DeltaT={delta_t}K: no data (log={log_path}, mon={mon_path})')

gpu_dt = np.array(gpu_dt)
gpu_tc = np.array(gpu_tc)
# Convert dimensionless Θ_c → dimensional [K]:  y = ΔT × Θ_c
gpu_y  = gpu_dt * gpu_tc

# ── 2. Load reference data ────────────────────────────────────────────────────
pascal_dt, pascal_y = load_reference_csv(os.path.join(DATA_DIR, 'PaScaL_TCS.csv'))
horm_dt,   horm_y   = load_reference_csv(os.path.join(DATA_DIR, 'Horm_et_al.csv'))

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

if len(horm_dt):
    ax.plot(horm_dt, horm_y, 'k^-', markersize=5, linewidth=1.2, zorder=2,
            label='Horm et al. (experiment)')

if len(pascal_dt):
    ax.plot(pascal_dt, pascal_y, 'bs', markersize=8, markeredgecolor='k',
            markeredgewidth=0.6, linewidth=1.5, zorder=4,
            label='PaScaL_TCS DNS (Ra=1e6)')

if len(gpu_dt):
    ax.plot(gpu_dt, gpu_y, 'ro', markersize=10, markeredgecolor='k',
            markeredgewidth=0.8, linewidth=1.5, zorder=5,
            label='MPM-STD GPU (this work)')

    # Annotations: value in K + convergence note
    for dt, y, (src, last_step, n_pts) in zip(gpu_dt, gpu_y, gpu_info):
        conv_note = ''
        if last_step < 350000:
            conv_note = f'\n(step {last_step//1000}K, not converged)'
        ax.annotate(f'{y:.2f} K{conv_note}',
                    xy=(dt, y),
                    xytext=(dt + 1.5, y * 1.08),
                    fontsize=8.5, color='crimson', fontweight='bold')

    # Error vs PaScaL_TCS if available
    if len(pascal_dt):
        pascal_interp = np.interp(gpu_dt, pascal_dt, pascal_y,
                                   left=np.nan, right=np.nan)
        for dt, y, ref in zip(gpu_dt, gpu_y, pascal_interp):
            if not np.isnan(ref) and ref > 0:
                err = (y - ref) / ref * 100
                sign = '+' if err >= 0 else ''
                ax.annotate(f'({sign}{err:.1f}% vs DNS)',
                            xy=(dt, y),
                            xytext=(dt + 1.5, y * 0.88),
                            fontsize=7.5, color='gray', style='italic')

ax.axhline(0.0, ls='--', color='0.7', lw=1.0, label='OB baseline')

ax.set_xlabel(r'$\Delta T$ (K)', fontsize=13)
ax.set_ylabel(r'$\Delta(\Theta_c - \Theta_m)$  (K)', fontsize=12)
ax.set_title('Fig 9 — NOB RBC (glycerol)\n'
             r'$Ra=10^6$, $Pr=2547$, $\Gamma=6$: $\Delta(\Theta_c-\Theta_m)$ vs $\Delta T$',
             fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(0, 75)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=10)

plt.tight_layout()
out = os.path.join(APP_DIR, 'fig9_gpu_validation.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
