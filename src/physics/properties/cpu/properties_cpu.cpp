#include "physics/properties/properties.hpp"
#include "common/macros.hpp"

#include <cstddef>

// NOB glycerol property update, ported from the reference PaScaL_TCS
// (module_global.f90 coefficients; module_solve_momentum/thermal evaluation).
// Properties are polynomials in Δθ = DeltaT·θ. Computed over the FULL array
// (incl. halo, since T is already synced) so the face interpolations + ghosts
// downstream are valid.

namespace mpmstd::physics {

// Horner: k[0]·(1 + k[1]t + k[2]t² + k[3]t³ + k[4]t⁴ + k[5]t⁵)
static inline double poly(const real_t k[6], double t) {
  return double(k[0]) * (1.0 + t * (k[1] + t * (k[2] + t * (k[3] + t * (k[4] + t * k[5])))));
}
// normalized polynomial 1 + k[1]t + … (i.e. poly/k[0])
static inline double npoly(const real_t k[6], double t) {
  return 1.0 + t * (k[1] + t * (k[2] + t * (k[3] + t * (k[4] + t * k[5]))));
}

PropertyModel PropertyModel::glycerol(real_t DeltaT) {
  PropertyModel m;

  // ── OB baseline (active by default) ─────────────────────────────────────────
  // Mirrors PaScaL_TCS module_global.f90 l.117-120: all coefficients = 1/0 so
  // every property ≡ 1 regardless of DeltaT → equivalent to constant-property OB.
  m.a[0]=1.0;       m.a[1]=0;          m.a[2]=0;          m.a[3]=0; m.a[4]=0; m.a[5]=0; // ρ
  m.b[0]=1.0;       m.b[1]=0;          m.b[2]=0;          m.b[3]=0; m.b[4]=0; m.b[5]=0; // Cp
  m.c[0]=1.0;       m.c[1]=0;          m.c[2]=0;          m.c[3]=0; m.c[4]=0; m.c[5]=0; // κ
  m.d[0]=1.0;       m.d[1]=0;          m.d[2]=0;          m.d[3]=0; m.d[4]=0; m.d[5]=0; // μ-factor

  // ── Glycerol NOB — uncomment block below to activate ────────────────────────
  // Mirrors PaScaL_TCS module_global.f90 l.129-132.
  m.a[0]=1.2477e3;  m.a[1]=-4.789e-4;  m.a[2]=-0.3795e-6; m.a[3]=0;          m.a[4]=0;           m.a[5]=0;            // ρ
  m.b[0]=2.5108e3;  m.b[1]=22.511e-4;  m.b[2]=0;          m.b[3]=0;          m.b[4]=0;           m.b[5]=0;            // Cp
  m.c[0]=2.9351e-3; m.c[1]=3.863e-4;   m.c[2]=0;          m.c[3]=0;          m.c[4]=0;           m.c[5]=0;            // κ
  m.d[0]=238.71e-6; m.d[1]=-702.83e-4; m.d[2]=2393.1e-6;  m.d[3]=-6923.0e-8; m.d[4]=33131.3e-10; m.d[5]=-71517.5e-12; // μ-factor

  m.DeltaT = DeltaT;
  return m;
}

void update_properties_cpu(CpuPropertyFields& p, const core::CpuField& T, const PropertyModel& m) {
  const auto nt = T.n_total();
  const std::size_t n = static_cast<std::size_t>(nt[0]) * nt[1] * nt[2];
  const real_t* Tp = T.data();
  real_t* mu = p.mu.data(); real_t* irho = p.inv_rho.data();
  real_t* kpp = p.kappa.data(); real_t* dkpp = p.dkappa.data();
  real_t* ircp = p.inv_rhocp.data(); real_t* dircp = p.dinv_rhocp.data();

  const double Mu0 = double(m.a[0]) * m.d[0];
  const real_t* a = m.a; const real_t* b = m.b; const real_t* c = m.c; const real_t* d = m.d;
  const double Dt = m.DeltaT;

  for (std::size_t i = 0; i < n; ++i) {
    const double t = Dt * double(Tp[i]);                           // Δθ

    mu[i]   = static_cast<real_t>(poly(a, t) * poly(d, t) / Mu0);   // μ̂ = ρ_poly·d_poly / Mu0
    irho[i] = static_cast<real_t>(1.0 / npoly(a, t));              // 1/ρ̂ = 1/(1+a1 t+…)
    kpp[i]  = static_cast<real_t>(npoly(c, t));                    // κ̂ = (1+c1 t+…)
    dkpp[i] = static_cast<real_t>(c[1] + t * (2.0*c[2] + t * (3.0*c[3] + t * (4.0*c[4] + t * 5.0*c[5])))); // dκ̂/d(Δθ)

    const double ia = 1.0 + t * (a[1] + t * (a[2] + t * a[3]));    // (1+a1 t+a2 t²+a3 t³)
    const double ib = 1.0 + t * (b[1] + t * (b[2] + t * b[3]));
    ircp[i] = static_cast<real_t>(1.0 / (ia * ib));               // 1/(ρ̂Ĉp)
    // d[1/(ρ̂Ĉp)]/d(Δθ) — reference analytic form (≤cubic), module_solve_thermal.f90:142
    const double t2 = t * t, t3 = t2 * t;
    const double num = b[1] * (1.0 + 3.0*a[2]*t2 + 4.0*a[3]*t3)
                     + a[1] * (1.0 + 2.0*b[1]*t + 3.0*b[2]*t2 + 4.0*b[3]*t3)
                     + t * ( b[2] * (2.0 + 5.0*a[3]*t3)
                           + a[2] * (2.0 + 4.0*b[2]*t2 + 5.0*b[3]*t3)
                           + 3.0*t * (a[3] + b[3] + 2.0*a[3]*b[3]*t3) );
    dircp[i] = static_cast<real_t>(-num / (ia * ia * ib * ib));
  }
}

} // namespace mpmstd::physics
