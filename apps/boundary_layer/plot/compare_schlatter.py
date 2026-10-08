#!/usr/bin/env python3
"""단계 B 후처리: bl_stats 파일을 Schlatter & Örlü 2010 (KTH) 및 Torroja (Jiménez) DNS 와 비교.
사용: python3 compare_schlatter.py stats/<case>/bl_stats_XXXXXXXX.dat [--png out.png]
  - C_f(Re_θ), H(Re_θ), Re_τ(Re_θ) 를 참조점(파일 헤더) + 상관식과 비교
  - 각 스테이션 U⁺(z⁺)·rms⁺·uw⁺ 를 가장 가까운 Re_θ 의 참조 프로파일과 비교 (로그 영역 RMS 차, 피크)"""
import glob, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'ref')

def read_kth(path):
    h = open(path).read()
    g = lambda key: float(re.search(key + r'\s*=\s*([0-9.eE+-]+)', h).group(1))
    meta = dict(src='KTH', re_theta=g(r'Re_\{\\theta\}'), re_tau=g(r'Re_\{\\tau\}'), H=g(r'H_\{12\}'), cf=g(r'c_f'))
    d = np.loadtxt(path, comments='%')
    return meta, dict(yd=d[:, 0], yp=d[:, 1], U=d[:, 2], urms=d[:, 3], vrms=d[:, 5], wrms=d[:, 4], uv=d[:, 6])  # KTH v=벽수직·w=스팬 → 우리 v=스팬·w=벽수직

def read_torroja(path):
    h = open(path).read()
    g = lambda key: float(re.search(key + r'\s*=\s*([0-9.eE+-]+)', h).group(1))
    ut = g(r'u_\{tau\}'); dstar = g(r'delta\^\*'); theta = g(r'theta ')
    meta = dict(src='Torroja', re_theta=g(r'Re_\{theta\}'), re_tau=g(r'Re_\{tau\}'), H=dstar / theta, cf=2 * ut**2)
    d = np.loadtxt(path, comments='%')
    return meta, dict(yd=d[:, 0], yp=d[:, 1], U=d[:, 6], urms=d[:, 2], vrms=d[:, 4], wrms=d[:, 3], uv=d[:, 5])  # Torroja 도 같은 교환

def load_refs():
    refs = []
    for f in sorted(glob.glob(os.path.join(REF, 'vel_*_dns.prof'))): refs.append(read_kth(f))
    for f in sorted(glob.glob(os.path.join(REF, 'torroja', 'Re_theta.*.prof'))): refs.append(read_torroja(f))
    return refs

def read_stats(path):
    lines = open(path).read().split('\n')
    sec1, stations, cur = [], [], None
    for ln in lines:
        if ln.startswith('# section 2'):
            m = dict(re.findall(r'(\w+)=([0-9.eE+-]+)', ln))
            cur = dict(meta={k: float(v) for k, v in m.items()}, rows=[]); stations.append(cur); continue
        if ln.startswith('#') or not ln.strip(): continue
        vals = [float(v) for v in ln.split()]
        if cur is None: sec1.append(vals)
        else: cur['rows'].append(vals)
    s1 = np.array(sec1)
    integ = dict(x=s1[:, 0], cf=s1[:, 2], d99=s1[:, 3], dstar=s1[:, 4], theta=s1[:, 5], H=s1[:, 6], re_theta=s1[:, 7], re_tau=s1[:, 8], u_tau=s1[:, 9])
    for st in stations:
        r = np.array(st['rows']); st['prof'] = dict(z=r[:, 0], yp=r[:, 1], U=r[:, 2], urms=r[:, 3], vrms=r[:, 4], wrms=r[:, 5], uv=r[:, 6])
    return integ, stations

def cf_correlations(re_theta):
    smits = 0.024 * re_theta ** (-0.25)                               # Smits et al. 1983 (S&Ö 2010 이 사용)
    cf_cf = 2.0 / (np.log(re_theta) / 0.384 + 4.127) ** 2               # Coles–Fernholz (Nagib et al. 2007)
    return smits, cf_cf

def main():
    argv = sys.argv[1:]; png = None
    if '--png' in argv:
        k = argv.index('--png'); png = argv[k + 1]; argv = argv[:k] + argv[k + 2:]
    refs = load_refs()
    for path in argv:
        integ, stations = read_stats(path)
        print(f"\n=== {path}")
        print("--- C_f / H / Re_tau at reference Re_theta (ours = interpolation along x; valid if inside our Re_theta range)")
        print(f"{'src':>8} {'Re_theta':>9} {'Cf_ref':>9} {'Cf_ours':>9} {'err%':>7} {'H_ref':>7} {'H_ours':>7} {'err%':>7} {'Retau_ref':>9} {'Retau_ours':>10} {'err%':>7}  | corr: Smits  Coles-Fernholz")
        lo, hi = integ['re_theta'].min(), integ['re_theta'].max()
        order = np.argsort(integ['re_theta'])
        for meta, _ in refs:
            rt = meta['re_theta']
            if rt < lo or rt > hi: continue
            cf = np.interp(rt, integ['re_theta'][order], integ['cf'][order]); H = np.interp(rt, integ['re_theta'][order], integ['H'][order]); rtau = np.interp(rt, integ['re_theta'][order], integ['re_tau'][order])
            sm, cfz = cf_correlations(rt)
            print(f"{meta['src']:>8} {rt:9.1f} {meta['cf']:9.6f} {cf:9.6f} {100*(cf/meta['cf']-1):7.2f} {meta['H']:7.4f} {H:7.4f} {100*(H/meta['H']-1):7.2f} {meta['re_tau']:9.1f} {rtau:10.1f} {100*(rtau/meta['re_tau']-1):7.2f}  | {sm:.6f} {cfz:.6f}")
        print(f"(our Re_theta range {lo:.0f}..{hi:.0f})")
        for st in stations:
            m = st['meta']; p = st['prof']
            ref = min(refs, key=lambda r: abs(r[0]['re_theta'] - m['Re_theta']))
            rm, rp = ref
            print(f"\n--- station Re_theta={m['Re_theta']:.1f} (Re_tau {m['Re_tau']:.1f}, Cf {m['Cf']:.6f}, H {m['H']:.4f}) vs {rm['src']} Re_theta={rm['re_theta']:.1f} (Re_tau {rm['re_tau']:.1f}, Cf {rm['cf']:.6f}, H {rm['H']:.4f})")
            yp = p['yp']; mask = (yp > 30) & (yp < 0.3 * m['Re_tau'])
            if mask.sum() >= 3:
                Uref = np.interp(yp[mask], rp['yp'], rp['U'])
                print(f"    log region 30<z+<0.3Re_tau: RMS(U+ - U+_ref) = {np.sqrt(np.mean((p['U'][mask]-Uref)**2)):.3f}  (U+ bias {np.mean(p['U'][mask]-Uref):+.3f})")
            for key in ['urms', 'vrms', 'wrms']:
                k_o = int(np.argmax(p[key])); k_r = int(np.argmax(rp[key]))
                print(f"    {key}+ peak: ours {p[key][k_o]:.3f} @z+={yp[k_o]:.1f}   ref {rp[key][k_r]:.3f} @z+={rp['yp'][k_r]:.1f}   ({100*(p[key][k_o]/rp[key][k_r]-1):+.1f}%)")
            k_o = int(np.argmin(p['uv'])); k_r = int(np.argmin(rp['uv']))
            print(f"    uw+ min: ours {p['uv'][k_o]:.3f} @z+={yp[k_o]:.1f}   ref {rp['uv'][k_r]:.3f} @z+={rp['yp'][k_r]:.1f}")
        if png:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            n = len(stations); fig, ax = plt.subplots(2, max(2, n), figsize=(5 * max(2, n), 8))
            rt = np.linspace(max(200, lo * 0.8), hi * 1.1, 100); sm, cfz = cf_correlations(rt)
            ax[0, 0].plot(integ['re_theta'], integ['cf'], 'k-', label='MPM-STD'); ax[0, 0].plot(rt, sm, 'g:', label='0.024 Re_theta^-1/4'); ax[0, 0].plot(rt, cfz, 'm--', label='Coles-Fernholz')
            for meta, _ in refs: ax[0, 0].plot(meta['re_theta'], meta['cf'], 'rs' if meta['src'] == 'KTH' else 'b^')
            ax[0, 0].set_xlabel('Re_theta'); ax[0, 0].set_ylabel('C_f'); ax[0, 0].legend(fontsize=7); ax[0, 0].set_title('C_f (red: KTH, blue: Torroja)')
            ax[0, 1].plot(integ['re_theta'], integ['H'], 'k-')
            for meta, _ in refs: ax[0, 1].plot(meta['re_theta'], meta['H'], 'rs' if meta['src'] == 'KTH' else 'b^')
            ax[0, 1].set_xlabel('Re_theta'); ax[0, 1].set_ylabel('H'); ax[0, 1].set_title('shape factor')
            for c, st in enumerate(stations):
                m = st['meta']; p = st['prof']; rm, rp = min(refs, key=lambda r: abs(r[0]['re_theta'] - m['Re_theta']))
                a = ax[1, c]; a.semilogx(p['yp'], p['U'], 'k-', label=f"ours Re_theta {m['Re_theta']:.0f}"); a.semilogx(rp['yp'], rp['U'], 'r--', label=f"{rm['src']} {rm['re_theta']:.0f}")
                for key, col in [('urms', 'b'), ('vrms', 'g'), ('wrms', 'c')]:
                    a.semilogx(p['yp'], p[key], col + '-'); a.semilogx(rp['yp'], rp[key], col + '--')
                a.semilogx(p['yp'], -p['uv'], 'm-'); a.semilogx(rp['yp'], -rp['uv'], 'm--')
                a.set_xlim(0.3, 2 * m['Re_tau']); a.set_xlabel('z+'); a.set_ylabel('U+, rms+, -uw+'); a.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(png, dpi=130); print('saved', png)

if __name__ == '__main__':
    main()
