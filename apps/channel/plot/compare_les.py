#!/usr/bin/env python3
"""채널 LES 검증: LES 통계(channel_stats_*.dat, nut_over_nu 열 포함) 를 DNS 참조(stats_re180_ref 최종 파일)와 비교.
사용: python3 compare_les.py --ref stats_re180_ref/channel_stats_00060000.dat stats_les64_none/channel_stats_X.dat ... [--png out.png]
지표(docs/Design/20261006_les_design.md §4): Re_τ 오차, 로그영역(30<z⁺<0.3Re_τ) U⁺ RMS 차, rms⁺ 피크, −uw⁺ 피크,
총전단 수지 ν_eff dU/dz − uw = u_τ²(1−z/δ), ν_t 벽 멱지수(log-log 기울기, z⁺<10)."""
import re, sys
import numpy as np

def read_stats(path):
    meta = {}
    for ln in open(path):
        if ln.startswith('#'):
            for k, v in re.findall(r'(\w+)=([0-9.eE+-]+)', ln): meta[k] = float(v)
            m = re.search(r'les=(\w+)', ln)
            if m: meta['les'] = m.group(1)
    d = np.loadtxt(path, comments='#')
    cols = dict(z=d[:, 0], zp=d[:, 1], U=d[:, 2], Up=d[:, 3], urms=d[:, 4], vrms=d[:, 5], wrms=d[:, 6], uw=d[:, 7])
    cols['nut'] = d[:, 8] if d.shape[1] > 8 else np.zeros(len(d))
    return meta, cols

def half(c, n):  # 하반(벽 z=0 쪽) 프로파일 + 상반 대칭 평균
    nz = len(c['z']); k = np.arange(nz // 2)
    def sym(a, sign=1): return 0.5 * (a[k] + sign * a[nz - 1 - k])
    return dict(zp=c['zp'][k], Up=sym(c['Up']), urms=sym(c['urms']), vrms=sym(c['vrms']), wrms=sym(c['wrms']), uw=sym(c['uw'], -1), nut=sym(c['nut']), z=c['z'][k])

def main():
    argv = sys.argv[1:]; png = None; ref = None
    if '--png' in argv: k = argv.index('--png'); png = argv[k + 1]; argv = argv[:k] + argv[k + 2:]
    if '--ref' in argv: k = argv.index('--ref'); ref = argv[k + 1]; argv = argv[:k] + argv[k + 2:]
    rm, rc = read_stats(ref); rh = half(rc, len(rc['z']))
    print(f"ref: {ref}  Re_tau={rm['Re_tau']:.2f} u_tau={rm['u_tau']:.5f}")
    print(f"{'case':<28} {'Re_tau':>7} {'err%':>6} {'U+ logRMS':>10} {'U+ bias':>8} {'urms pk':>8} {'(ref)':>7} {'vrms pk':>8} {'(ref)':>7} {'wrms pk':>8} {'(ref)':>7} {'-uw pk':>7} {'(ref)':>7} {'max nut/nu':>10} {'nut~y^n':>8}")
    rows = []
    for path in argv:
        m, c = read_stats(path); h = half(c, len(c['z'])); ut = m['u_tau']; retau = m['Re_tau']
        mask = (h['zp'] > 30) & (h['zp'] < 0.3 * retau)
        Uref = np.interp(h['zp'][mask], rh['zp'], rh['Up'])
        logrms = np.sqrt(np.mean((h['Up'][mask] - Uref) ** 2)) if mask.sum() > 2 else float('nan')
        bias = np.mean(h['Up'][mask] - Uref) if mask.sum() > 2 else float('nan')
        pk = lambda a: a.max() / ut
        uwp = -(h['uw'] / ut ** 2); rpk = lambda a: a.max() / rm['u_tau']
        # ν_t 벽 멱지수: z+ in (1, 10) log-log slope
        msk = (h['zp'] > 1) & (h['zp'] < 10) & (h['nut'] > 1e-12)
        slope = np.polyfit(np.log(h['zp'][msk]), np.log(h['nut'][msk]), 1)[0] if msk.sum() > 3 else float('nan')
        name = path.split('/')[-2]
        print(f"{name:<28} {retau:7.2f} {100*(retau/rm['Re_tau']-1):6.2f} {logrms:10.3f} {bias:8.3f} {pk(h['urms']):8.3f} {rpk(rh['urms']):7.3f} {pk(h['vrms']):8.3f} {rpk(rh['vrms']):7.3f} {pk(h['wrms']):8.3f} {rpk(rh['wrms']):7.3f} {uwp.max():7.3f} {(-(rh['uw'])/rm['u_tau']**2).max():7.3f} {h['nut'].max():10.3f} {slope:8.2f}")
        rows.append((name, m, h))
    if png:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))
        ax[0].semilogx(rh['zp'], rh['Up'], 'k-', lw=2, label=f"DNS 256^3 (Re_tau {rm['Re_tau']:.0f})")
        zp = np.logspace(0, np.log10(200), 50); ax[0].semilogx(zp, np.log(zp) / 0.41 + 5.2, 'k:', lw=1, label='log law')
        for name, m, h in rows: ax[0].semilogx(h['zp'], h['Up'], label=f"{name} ({m['Re_tau']:.0f})")
        ax[0].set_xlabel('z+'); ax[0].set_ylabel('U+'); ax[0].legend(fontsize=7)
        for key, a in [('urms', ax[1]), ('uw', ax[2])]:
            if key == 'uw': a.plot(rh['zp'], -rh['uw'] / rm['u_tau']**2, 'k-', lw=2)
            else: a.plot(rh['zp'], rh[key] / rm['u_tau'], 'k-', lw=2)
            for name, m, h in rows:
                if key == 'uw': a.plot(h['zp'], -h['uw'] / m['u_tau']**2, label=name)
                else: a.plot(h['zp'], h[key] / m['u_tau'], label=name)
            a.set_xlabel('z+'); a.set_ylabel('urms+' if key == 'urms' else '-uw+'); a.set_xlim(0, 180); a.legend(fontsize=7)
        for name, m, h in rows: ax[3].loglog(h['zp'], np.maximum(h['nut'], 1e-9), label=name)
        ax[3].loglog(zp, 1e-4 * zp**3, 'k:', label='y^3'); ax[3].set_xlabel('z+'); ax[3].set_ylabel('nu_t/nu'); ax[3].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(png, dpi=130); print('saved', png)

if __name__ == '__main__':
    main()
