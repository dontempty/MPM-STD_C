#!/usr/bin/env python3
"""
BL-theory prediction of the NOB center temperature theta_c (Report/25 P1).

Two-boundary-layer model of Ahlers et al. 2006 (JFM 569) / Horn & Shishkina 2013,
evaluated with the REPO glycerol property polynomials (PaScaL_TCS module_global.f90
l.129-132 == src/physics/properties/cpu/properties_cpu.cpp). Purpose: decide whether
the repo polynomials reproduce the reference curve theta_c(DeltaT) ~ 0.074/0.135/0.190
(Kim et al. 2023 Fig.9 CSV; Horn et al. CSV) independent of any DNS.

Model (each plate, Prandtl-Blasius similarity, eta = z*sqrt(U/(nu_c x))):
    momentum:  (mu~ f'')' + 1/2 rho~ f f''            = 0
    energy  :  (lam~ th')' + (Pr_c/2) rho~ cp~ f th'  = 0    ["conductivity" form,
                                                    faithful to the DNS energy eq.]
    BCs: f(0)=f'(0)=0, f'(inf)=1;  th(0)=theta_wall, th(inf)=theta_c
    ~ = property normalized by its value at the center temperature theta_c.
Flux balance picks theta_c:   lam~(hot wall)|th'(0)|_hot = lam~(cold wall)|th'(0)|_cold.
Variant V1 uses the diffusivity form (kap~ th')' + (Pr_c/2) f th' = 0 with
kap~(wall)|th'(0)| flux (Ahlers 2006 as written) — reported for sensitivity.

theta in [-0.5, +0.5], t = DeltaT*theta [K around 40 C]. Pr_m = 2547 at t=0.
Outside the thermal BL (thickness ~ Pr^{-1/3}) the energy equation is stiff for
explicit RK4; once th' has decayed to ~0 the exact solution keeps th'=0, so th'
is frozen there.
"""

import math

# ── repo glycerol polynomials (t in K, deviation from 40 C) ─────────────────
A = (-4.789e-4, -0.3795e-6, 0.0, 0.0, 0.0)                          # rho~ = 1+..
B = (22.511e-4, 0.0, 0.0, 0.0, 0.0)                                 # cp~
C = (3.863e-4, 0.0, 0.0, 0.0, 0.0)                                  # lam~
D = (-702.83e-4, 2393.1e-6, -6923.0e-8, 33131.3e-10, -71517.5e-12)  # nu~

def npoly(k, t):
    return 1.0 + t*(k[0] + t*(k[1] + t*(k[2] + t*(k[3] + t*k[4]))))

def dnpoly(k, t):
    return k[0] + t*(2*k[1] + t*(3*k[2] + t*(4*k[3] + t*5*k[4])))

PR_M = 2547.0

def raw_props(t):
    rho, cp, lam, nu = npoly(A, t), npoly(B, t), npoly(C, t), npoly(D, t)
    drho, dcp, dlam, dnu = dnpoly(A, t), dnpoly(B, t), dnpoly(C, t), dnpoly(D, t)
    mu, dmu = rho*nu, drho*nu + rho*dnu
    rcp, drcp = rho*cp, drho*cp + rho*dcp
    kap = lam/rcp
    dkap = (dlam*rcp - lam*drcp)/(rcp*rcp)
    return dict(rho=rho, mu=mu, dmu=dmu, lam=lam, dlam=dlam,
                rcp=rcp, kap=kap, dkap=dkap)


class PlateODE:
    """Coupled BL ODEs with properties normalized at the center temperature."""

    def __init__(self, dT, tc, form):
        self.dT, self.form = dT, form
        pc = raw_props(dT*tc)
        self.mu_c, self.rho_c, self.rcp_c = pc['mu'], pc['rho'], pc['rcp']
        self.lam_c, self.kap_c = pc['lam'], pc['kap']
        self.pr_c = PR_M*(pc['mu']/pc['rho'])/pc['kap']

    def wall_flux_weight(self, th_w):
        p = raw_props(self.dT*th_w)
        return p['lam']/self.lam_c if self.form == 'conductivity' else p['kap']/self.kap_c

    def rhs(self, f, fp, fpp, th, thp):
        t = self.dT*th
        p = raw_props(t)
        mu_h, dmu_h = p['mu']/self.mu_c, p['dmu']/self.mu_c
        rho_h = p['rho']/self.rho_c
        if self.form == 'conductivity':
            dif_h, ddif_h = p['lam']/self.lam_c, p['dlam']/self.lam_c
            adv_h = p['rcp']/self.rcp_c
        else:
            dif_h, ddif_h = p['kap']/self.kap_c, p['dkap']/self.kap_c
            adv_h = 1.0
        dfpp = (-0.5*rho_h*f*fpp - dmu_h*self.dT*thp*fpp) / mu_h
        dthp = (-0.5*self.pr_c*adv_h*f*thp - ddif_h*self.dT*thp*thp) / dif_h
        return fp, fpp, dfpp, thp, dthp

    def integrate(self, s, g, th_w, eta1=1.0, eta_max=8.0, h1=1e-3, h2=1e-2):
        f, fp, fpp, th, thp = 0.0, 0.0, s, th_w, g
        gtol = 1e-13*max(1.0, abs(g))
        eta, rhs = 0.0, self.rhs
        while eta < eta_max - 1e-12:
            h = h1 if eta < eta1 else h2
            a1, b1, c1, d1, e1 = rhs(f, fp, fpp, th, thp)
            a2, b2, c2, d2, e2 = rhs(f+0.5*h*a1, fp+0.5*h*b1, fpp+0.5*h*c1,
                                     th+0.5*h*d1, thp+0.5*h*e1)
            a3, b3, c3, d3, e3 = rhs(f+0.5*h*a2, fp+0.5*h*b2, fpp+0.5*h*c2,
                                     th+0.5*h*d2, thp+0.5*h*e2)
            a4, b4, c4, d4, e4 = rhs(f+h*a3, fp+h*b3, fpp+h*c3, th+h*d3, thp+h*e3)
            f += (h/6.0)*(a1+2*a2+2*a3+a4)
            fp += (h/6.0)*(b1+2*b2+2*b3+b4)
            fpp += (h/6.0)*(c1+2*c2+2*c3+c4)
            th += (h/6.0)*(d1+2*d2+2*d3+d4)
            thp += (h/6.0)*(e1+2*e2+2*e3+e4)
            eta += h
            if abs(thp) < gtol:
                thp = 0.0          # exact outside the thermal BL; avoids stiffness
            if not (math.isfinite(fp) and math.isfinite(th)) or abs(fp) > 50 or abs(th) > 5:
                return f, fp, fpp, th, thp, False
        return f, fp, fpp, th, thp, True


def solve_plate(ode, th_w, tc, guess=None):
    """2D shooting: s=f''(0), g=th'(0) such that f'(inf)=1, th(inf)=tc."""
    if guess is not None:
        s, g = guess
    else:
        mu_wall = raw_props(ode.dT*th_w)['mu']/ode.mu_c
        s = 0.4696/math.sqrt(mu_wall)
        g = -(th_w - tc)*0.339*ode.pr_c**(1.0/3.0)

    def resid(s_, g_):
        _, fp, _, th, _, ok = ode.integrate(s_, g_, th_w)
        return (fp - 1.0, th - tc, ok)

    for _ in range(40):
        r1, r2, ok = resid(s, g)
        if not ok:
            s *= 0.7; g *= 0.7
            continue
        if abs(r1) < 1e-9 and abs(r2) < 1e-11:
            return s, g
        ds, dg = 1e-6*max(1.0, abs(s)), 1e-6*max(1.0, abs(g))
        r1s, r2s, _ = resid(s+ds, g)
        r1g, r2g, _ = resid(s, g+dg)
        j11, j21 = (r1s-r1)/ds, (r2s-r2)/ds
        j12, j22 = (r1g-r1)/dg, (r2g-r2)/dg
        det = j11*j22 - j12*j21
        if det == 0:
            raise RuntimeError('singular shooting Jacobian')
        st_s = ( j22*r1 - j12*r2)/det
        st_g = (-j21*r1 + j11*r2)/det
        lam = 1.0
        base = abs(r1) + 10*abs(r2)
        while lam > 1e-3:
            n1, n2, okn = resid(s - lam*st_s, g - lam*st_g)
            if okn and (abs(n1) + 10*abs(n2)) < base:
                break
            lam *= 0.5
        s, g = s - lam*st_s, g - lam*st_g
    raise RuntimeError(f'shooting failed th_w={th_w} tc={tc}')


def flux_imbalance(tc, dT, form, cache):
    ode = PlateODE(dT, tc, form)
    cache['hot'] = solve_plate(ode, +0.5, tc, cache.get('hot'))
    cache['cold'] = solve_plate(ode, -0.5, tc, cache.get('cold'))
    q_hot = ode.wall_flux_weight(+0.5)*abs(cache['hot'][1])
    q_cold = ode.wall_flux_weight(-0.5)*abs(cache['cold'][1])
    return q_hot - q_cold


def theta_c(dT, form='conductivity', verbose=False):
    cache = {}
    x0, x1 = 0.02, 0.10
    f0 = flux_imbalance(x0, dT, form, cache)
    f1 = flux_imbalance(x1, dT, form, cache)
    for it in range(30):                     # secant
        if f1 == f0:
            break
        x2 = x1 - f1*(x1 - x0)/(f1 - f0)
        x2 = min(max(x2, 0.0), 0.35)
        f2 = flux_imbalance(x2, dT, form, cache)
        if verbose:
            print(f'    secant {it}: tc={x2:.6f} imbalance={f2:+.3e}')
        x0, f0, x1, f1 = x1, f1, x2, f2
        if abs(x1 - x0) < 1e-7:
            break
    return x1


if __name__ == '__main__':
    # self-checks: OB limit -> Blasius f''(0)=0.46960 and symmetric fluxes
    ode0 = PlateODE(0.0, 0.0, 'conductivity')
    s0, g0 = solve_plate(ode0, +0.5, 0.0)
    imb0 = flux_imbalance(0.0, 0.0, 'conductivity', {})
    print(f"[check] OB Blasius f''(0) = {s0:.5f}  (ref 0.33206 for f'''+f f''/2=0)")
    print(f'[check] OB flux imbalance at tc=0: {imb0:+.2e}  (ref 0)')
    print()
    print(f'{"DeltaT":>7} {"nu_c/nu_h":>10} {"theta_c(cond)":>14} {"theta_c(diff)":>14}'
          f' {"ref Kim23":>10} {"ref Horn":>9}')
    refs = {20: (0.0744, 0.0775), 40: (0.1350, 0.145), 60: (0.1900, float("nan"))}
    for dT in (20.0, 40.0, 60.0):
        r = npoly(D, -dT/2)*npoly(A, -dT/2)/(npoly(D, dT/2)*npoly(A, dT/2))
        tc1 = theta_c(dT, 'conductivity')
        tc2 = theta_c(dT, 'diffusivity')
        k, h = refs[int(dT)]
        print(f'{dT:7.0f} {r:10.2f} {tc1:14.4f} {tc2:14.4f} {k:10.4f} {h:9.4f}')
