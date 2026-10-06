#!/usr/bin/env python3
"""단계 A 후처리: bl_field 덤프 → C_f(x), δ*(x), θ(x), H 와 프로파일을 Blasius 와 비교.
사용: python3 blasius_check.py stats/blasius/bl_field_XXXXXXXX.dat [--png out.png]"""
import sys
import numpy as np

def blasius_table(eta_max=12.0, d=1e-3):
    n = int(eta_max / d) + 1
    def integrate(s):
        y = np.array([0.0, 0.0, s]); f = np.zeros(n); fp = np.zeros(n)
        rhs = lambda y: np.array([y[1], y[2], -0.5 * y[0] * y[2]])
        for i in range(1, n):
            k1 = rhs(y); k2 = rhs(y + 0.5 * d * k1); k3 = rhs(y + 0.5 * d * k2); k4 = rhs(y + d * k3)
            y = y + d / 6 * (k1 + 2 * k2 + 2 * k3 + k4); f[i] = y[0]; fp[i] = y[1]
        return y[1], f, fp
    s0, s1 = 0.33, 0.34
    g0 = integrate(s0)[0] - 1; g1, f, fp = integrate(s1); g1 -= 1
    for _ in range(30):
        if abs(g1) < 1e-12: break
        s2 = s1 - g1 * (s1 - s0) / (g1 - g0); s0, g0 = s1, g1; s1 = s2; g1, f, fp = integrate(s1); g1 -= 1
    return np.arange(n) * d, f, fp, s1

def read_dump(path):
    head = open(path).readline().replace('#', '').split()
    meta = dict(kv.split('=') for kv in head[3:])
    nx, nz = int(meta['nx']), int(meta['nz']); nu = float(meta['nu']); uinf = float(meta['u_inf']); x0 = float(meta['x0'])
    data = np.loadtxt(path, comments='#')
    nU = (nx + 1) * nz
    U = data[:nU, 2].reshape(nx + 1, nz); xf = data[:nU, 0].reshape(nx + 1, nz)[:, 0]; zc = data[:nU, 1].reshape(nx + 1, nz)[0]
    W = data[nU:, 2].reshape(nx, nz + 1); xc = data[nU:, 0].reshape(nx, nz + 1)[:, 0]; zf = data[nU:, 1].reshape(nx, nz + 1)[0]
    return dict(nx=nx, nz=nz, nu=nu, uinf=uinf, x0=x0, U=U, xf=xf, zc=zc, W=W, xc=xc, zf=zf, t=float(meta['t']))

def integrals(d):
    U, zc, zf, uinf, nu, x0 = d['U'], d['zc'], d['zf'], d['uinf'], d['nu'], d['x0']
    dz = np.diff(zf)
    cf = np.zeros(d['nx'] + 1); dstar = np.zeros_like(cf); theta = np.zeros_like(cf)
    z1, z2 = zc[0], zc[1]
    for i in range(d['nx'] + 1):
        u = U[i]
        dudz = u[0] * z2 / (z1 * (z2 - z1)) - u[1] * z1 / (z2 * (z2 - z1))
        cf[i] = 2 * nu * dudz / uinf**2
        dstar[i] = np.sum((1 - u / uinf) * dz)            # 셀 중점 적분 (벽~상단)
        theta[i] = np.sum((u / uinf) * (1 - u / uinf) * dz)
    x = x0 + d['xf']; rex = uinf * x / nu
    return dict(x=d['xf'], rex=rex, cf=cf, cf_b=0.664 / np.sqrt(rex), dstar=dstar, dstar_b=1.7208 * np.sqrt(nu * x / uinf),
                theta=theta, theta_b=0.66412 * np.sqrt(nu * x / uinf), H=dstar / theta)

def main():
    png = None
    argv = sys.argv[1:]
    if '--png' in argv:
        k = argv.index('--png'); png = argv[k + 1]; argv = argv[:k] + argv[k + 2:]
    args = [a for a in argv if not a.startswith('--')]
    eta_t, f_t, fp_t, shear = blasius_table()
    for path in args:
        d = read_dump(path); r = integrals(d); lx = d['xf'][-1]
        print(f"\n=== {path}  t={d['t']:.1f}  Re_dstar0={d['uinf']/d['nu']:.0f}  x0={d['x0']:.2f}  f''(0)={shear:.6f}")
        print(f"{'x/Lx':>6} {'Re_x':>9} {'Cf':>10} {'Cf_bl':>10} {'err%':>7} {'dstar':>8} {'bl':>8} {'err%':>7} {'theta':>8} {'bl':>8} {'err%':>7} {'H':>7}")
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            i = int(round(frac * d['nx']))
            e = lambda a, b: 100 * (a / b - 1)
            print(f"{frac:6.2f} {r['rex'][i]:9.0f} {r['cf'][i]:10.6f} {r['cf_b'][i]:10.6f} {e(r['cf'][i], r['cf_b'][i]):7.2f} "
                  f"{r['dstar'][i]:8.4f} {r['dstar_b'][i]:8.4f} {e(r['dstar'][i], r['dstar_b'][i]):7.2f} "
                  f"{r['theta'][i]:8.4f} {r['theta_b'][i]:8.4f} {e(r['theta'][i], r['theta_b'][i]):7.2f} {r['H'][i]:7.4f}")
        sel = (d['xf'] > 0.2 * lx) & (d['xf'] < 0.8 * lx)
        print(f"0.2<x/Lx<0.8 max|err|: Cf {100*np.max(np.abs(r['cf'][sel]/r['cf_b'][sel]-1)):.2f}%  dstar {100*np.max(np.abs(r['dstar'][sel]/r['dstar_b'][sel]-1)):.2f}%  theta {100*np.max(np.abs(r['theta'][sel]/r['theta_b'][sel]-1)):.2f}%  H range {r['H'][sel].min():.4f}..{r['H'][sel].max():.4f} (Blasius 2.5911)")
        # 프로파일 vs f'(η) at 3 stations
        for frac in [0.25, 0.5, 0.75]:
            i = int(round(frac * d['nx'])); x = d['x0'] + d['xf'][i]
            eta = d['zc'] * np.sqrt(d['uinf'] / (d['nu'] * x)); fp_num = d['U'][i] / d['uinf']
            fp_ref = np.interp(eta, eta_t, fp_t, right=1.0)
            m = eta < 8
            print(f"profile x/Lx={frac:.2f}: max|u/U - f'(eta)| (eta<8) = {np.max(np.abs(fp_num[m]-fp_ref[m])):.2e}")
        if png:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(14, 4))
            ax[0].plot(r['rex'], r['cf'], 'k-', label='MPM-STD'); ax[0].plot(r['rex'], r['cf_b'], 'r--', label='Blasius 0.664/sqrt(Re_x)'); ax[0].set_xlabel('Re_x'); ax[0].set_ylabel('C_f'); ax[0].legend()
            ax[1].plot(r['x'], r['dstar'], 'k-', label='delta*'); ax[1].plot(r['x'], r['dstar_b'], 'r--'); ax[1].plot(r['x'], r['theta'], 'b-', label='theta'); ax[1].plot(r['x'], r['theta_b'], 'r--'); ax[1].set_xlabel('x'); ax[1].legend()
            for frac in [0.25, 0.5, 0.75]:
                i = int(round(frac * d['nx'])); x = d['x0'] + d['xf'][i]; eta = d['zc'] * np.sqrt(d['uinf'] / (d['nu'] * x))
                ax[2].plot(d['U'][i] / d['uinf'], eta, '.', ms=3, label=f'x/Lx={frac}')
            ax[2].plot(fp_t, eta_t, 'r-', lw=1, label="Blasius f'"); ax[2].set_ylim(0, 8); ax[2].set_xlabel('u/U'); ax[2].set_ylabel('eta'); ax[2].legend()
            plt.tight_layout(); plt.savefig(png, dpi=130); print('saved', png)

if __name__ == '__main__':
    main()
