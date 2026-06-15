#!/usr/bin/env python3
"""
PaScaL_TCS Fig 9 재현 비교 플롯
NOB glycerol RBC: Ra=1e6, Pr=2547, Γ=6
Θ_c vs ΔT — 64³ vs 128³ vs 논문 참조값
"""
import re, os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LOG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../log'))

# ─── 논문 참조값 (PaScaL_TCS Kim et al. Fig 9, Ra=1e6) ──────────────────────
# 논문 Fig 9 에서 읽은 근사값 (glycerol, 128³, Γ=6)
paper = {20: 0.022, 40: 0.051, 60: 0.132}

# ─── 로그 파싱 ───────────────────────────────────────────────────────────────
def parse(path):
    steps, ts, tcs = [], [], []
    pat = re.compile(r'step=(\d+)\s+t=([\d.e+-]+).*?Theta_c=([\d.e+-]+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                steps.append(int(m.group(1)))
                ts.append(float(m.group(2)))
                tcs.append(float(m.group(3)))
    return np.array(steps), np.array(ts), np.array(tcs)

def final_from_done(path):
    pat = re.compile(r'\[rbc\] done.*Theta_c=([\d.e+-]+)')
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m: return float(m.group(1))
    return None

# 64³ 결과
n64 = {}
for dt, fname in [(20,'fig9_64_20_29270.out'), (40,'fig9_64_40_29271.out')]:
    p = os.path.join(LOG_DIR, fname)
    if os.path.exists(p):
        v = final_from_done(p)
        if v: n64[dt] = v

# 128³ 시계열
n128 = {}
for dt, fname in [(20,'fig9_128_20_29278.out'), (40,'fig9_128_40_29279.out'), (60,'fig9_128_60_29280.out')]:
    p = os.path.join(LOG_DIR, fname)
    if os.path.exists(p):
        steps, ts, tcs = parse(p)
        if len(steps):
            n128[dt] = (steps, ts, tcs)

# 128³ 시간평균값 계산 (통계 시작 이후)
NSTAT = 20000
def mean128(dt):
    if dt not in n128: return float('nan')
    steps, ts, tcs = n128[dt]
    mask = steps >= NSTAT
    if mask.any(): return float(tcs[mask].mean())
    return float(tcs[-max(1,len(tcs)//4):].mean())

# ─── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

clr = {20:'#2196F3', 40:'#FF9800', 60:'#F44336'}

# ── Panel 1: 128³ Θ_c 시계열 ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for dt, (steps, ts, tcs) in sorted(n128.items()):
    ax1.plot(ts, tcs, color=clr[dt], lw=1.3, alpha=0.85,
             label=fr'$\Delta T={dt}$ K  (running mean)')
    # 통계 구간 표시
    mask = steps >= NSTAT
    if mask.any():
        ax1.axvline(ts[mask][0], color=clr[dt], ls=':', lw=0.8, alpha=0.5)
        mu = tcs[mask].mean()
        ax1.axhline(mu, color=clr[dt], ls='--', lw=1.1)
        ax1.text(ts[-1]+2, mu, fr'  $\mu$={mu:.4f}',
                 color=clr[dt], va='center', fontsize=8.5)

ax1.set_xlabel('time (free-fall units)', fontsize=11)
ax1.set_ylabel(r'$\Theta_c = \langle\theta\rangle_{x,y,t}$', fontsize=11)
ax1.set_title('128³ NOB RBC  —  running mean $\Theta_c$\n'
              r'$Ra=10^6$, $Pr=2547$, $\Gamma=6$ (glycerol)', fontsize=10)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.25)
ax1.set_xlim(left=0)

# 통계 창 설명
if any((n128[dt][0] >= NSTAT).any() for dt in n128):
    ax1.text(0.02, 0.97, f'Stat. avg. from step {NSTAT}', transform=ax1.transAxes,
             fontsize=8, va='top', color='0.5')

# ── Panel 2: 막대 비교 ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
DTs = [20, 40, 60]
x   = np.arange(len(DTs))
w   = 0.25

# 논문
bp = ax2.bar(x - w, [paper.get(d, np.nan) for d in DTs], w,
             label='PaScaL_TCS\n(paper)', color='#90A4AE', edgecolor='k', lw=0.8)
# 64³
b64 = ax2.bar(x, [n64.get(d, np.nan) for d in DTs], w,
              label='MPM-STD_C\n64³ (done)', color='#42A5F5', edgecolor='k', lw=0.8)
# 128³
b128 = ax2.bar(x + w, [mean128(d) for d in DTs], w,
               label='MPM-STD_C\n128³ (partial)', color='#EF5350', edgecolor='k', lw=0.8)

for bars in (bp, b64, b128):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h) or h <= 0: continue
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=90)

ax2.set_xticks(x)
ax2.set_xticklabels([f'{d} K' for d in DTs], fontsize=10)
ax2.set_xlabel(r'$\Delta T$', fontsize=11)
ax2.set_ylabel(r'$\Theta_c$', fontsize=11)
ax2.set_title(r'$\Theta_c$ vs $\Delta T$', fontsize=10)
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.25, axis='y')
ax2.set_ylim(bottom=0)

fig.suptitle('PaScaL_TCS Fig 9  —  NOB RBC glycerol '
             r'($Ra=10^6$, $Pr=2547$, $\Gamma=6$)', fontsize=12, y=1.01)

out = os.path.join(os.path.dirname(__file__), 'fig9_comparison.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'saved → {out}')

# ─── 수치 요약 ─────────────────────────────────────────────────────────────────
print('\n┌─────────────────────────────────────────────────────────────┐')
print('│  ΔT   Paper   64³(done)   128³(partial)   ratio(128³/paper) │')
print('├─────────────────────────────────────────────────────────────┤')
for dt in DTs:
    p   = paper.get(dt, float('nan'))
    s64 = n64.get(dt, float('nan'))
    s128= mean128(dt)
    r   = s128/p if not np.isnan(s128) and p>0 else float('nan')
    print(f'│  {dt:2d}K  {p:.4f}  {s64:9.4f}   {s128:13.4f}   {r:17.3f} │')
print('└─────────────────────────────────────────────────────────────┘')
