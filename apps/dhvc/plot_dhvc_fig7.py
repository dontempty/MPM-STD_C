import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re, csv
import os

APP_DIR  = os.path.dirname(os.path.abspath(__file__))   # apps/dhvc/
ROOT_DIR = os.path.normpath(os.path.join(APP_DIR, '../..'))
LOG_DIR  = os.path.join(APP_DIR, 'logs')

# ── Case definitions: Ra → (dir_stats, log_file, nstat_start) ─────────────────
CASES = {
    1e5: ('apps/dhvc/fig7_gpu_out_1e5/', 'dhvc_gpu_1e5.log', 6000),
    1e6: ('apps/dhvc/fig7_gpu_out_1e6/', 'dhvc_gpu_1e6.log', 20000),
    1e7: ('apps/dhvc/fig7_gpu_out_1e7/', 'dhvc_gpu_1e7.log', 20000),
}

def parse_monitor_csv_dhvc(path, nstat_start=0):
    """Read monitor.csv → list of (step, Re_dstar). Returns [] if absent."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                step = int(r['step']); val = float(r['Re_dstar'])
                if step >= nstat_start and val > 0:
                    rows.append((step, val))
            except (ValueError, KeyError):
                pass
    return rows

def parse_re_dstar(logpath, nstat_start=0):
    """Extract (step, Re_dstar) pairs from DHVC GPU log. Returns list of (step, val)."""
    pattern = re.compile(r'step=(\d+).*?Re_dstar=([\d.]+)')
    results = []
    if not os.path.exists(logpath):
        return results
    with open(logpath) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1)); val = float(m.group(2))
                if step >= nstat_start and val > 0:
                    results.append((step, val))
    return results

# ── 1. Load data: monitor.csv first, log fallback ─────────────────────────────
dns_ra, dns_re = [], []
last_step = {}
for ra, (dir_rel, logfile, nstat_start) in sorted(CASES.items()):
    mon_path = os.path.join(ROOT_DIR, dir_rel, 'monitor.csv')
    log_path = os.path.join(LOG_DIR, logfile)

    data = parse_monitor_csv_dhvc(mon_path, nstat_start)
    src  = 'monitor.csv'
    if not data:
        data = parse_re_dstar(log_path, nstat_start)
        src  = logfile

    if data:
        step, val = data[-1]
        dns_ra.append(ra)
        dns_re.append(val)
        last_step[ra] = step
        print(f'Ra={ra:.0e}: Re_dstar={val:.3f}  last step={step}  ({src})')
    else:
        print(f'Ra={ra:.0e}: no data (mon={mon_path}, log={log_path})')

dns_ra = np.array(dns_ra)
dns_re = np.array(dns_re)

# ── 2. PaScaL_TCS reference DNS data ─────────────────────────────────────────
csv_path = os.path.join(APP_DIR, 'PaScaL_TCS.csv')
pascal_ra, pascal_re = [], []
with open(csv_path) as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 2:
            pascal_ra.append(float(parts[0]))
            pascal_re.append(float(parts[1]))
pascal_ra = np.array(pascal_ra)
pascal_re = np.array(pascal_re)

# ── 3. Ng et al. (2015) correlation ──────────────────────────────────────────
Ra_line = np.logspace(4, 11, 400)
Re_ng   = 0.23 * Ra_line**0.28

# ── 4. Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(Ra_line, Re_ng, 'k-', linewidth=2.0,
          label=r'Ng et al. (2015): $0.23\,Ra^{0.28}$')

ax.loglog(pascal_ra, pascal_re, 'bs', markersize=8, markeredgecolor='k',
          markeredgewidth=0.6, zorder=4, label='PaScaL_TCS DNS')

ax.loglog(dns_ra, dns_re, 'ro', markersize=10, markeredgecolor='k',
          markeredgewidth=0.8, zorder=5, label='MPM-STD GPU (this work)')

# Error annotations vs Ng correlation
Re_target = 0.23 * dns_ra**0.28
for ra, re, tgt in zip(dns_ra, dns_re, Re_target):
    err = (re - tgt) / tgt * 100
    sign = '+' if err >= 0 else '−'
    ax.annotate('%s%.1f%%' % (sign, abs(err)),
                xy=(ra, re), xytext=(ra * 1.5, re * 1.12),
                fontsize=9, color='crimson', fontweight='bold')

# Running note for Ra=1e6 if job still running (last step < n_steps)
if 1e6 in last_step and last_step[1e6] < 49000:
    ax.annotate('(step %dK / 50K)' % (last_step[1e6] // 1000),
                xy=(1e6, dns_re[dns_ra == 1e6][0]),
                xytext=(1.9e6, dns_re[dns_ra == 1e6][0] * 0.86),
                fontsize=8, color='gray', style='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

ax.set_xlabel(r'$Ra$', fontsize=13)
ax.set_ylabel(r'$Re_{\delta^*}$', fontsize=13)
ax.set_title(r'DHVC Validation: $Re_{\delta^*}$ vs $Ra$  (Pr = 0.7)', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(3e4, 5e7)
ax.set_ylim(2.5, 35)
ax.grid(True, which='both', alpha=0.3)
ax.tick_params(labelsize=10)
ax.yaxis.set_minor_locator(ticker.LogLocator(subs=[2,3,4,5,6,7,8,9]))
ax.xaxis.set_minor_locator(ticker.LogLocator(subs=[2,3,4,5,6,7,8,9]))

plt.tight_layout()
out = os.path.join(APP_DIR, 'dhvc_fig7_validation.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
