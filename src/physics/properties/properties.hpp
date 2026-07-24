#pragma once

#include "common/macros.hpp"       // MPMSTD_HD
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/gpu_binding.hpp"   // Subdomain

namespace mpmstd::physics::properties {

// NOB temperature-dependent properties (PaScaL_TCS reference module_global.f90).
// All properties are polynomials in the argument Δθ = DeltaT·θ (θ∈[−0.5,0.5]);
// the reference temperature θm is baked into the coefficients (normalized out by
// the leading *[0] term). OB limit = only *[0] nonzero → every property ≡ 1.
//   ρ_poly(t) = a[0]·(1 + a[1]t + a[2]t² + … + a[5]t⁵),  similarly Cp(b), κ(c),
//   and the viscosity shape d; μ = ρ_poly·d_poly.  Normalizers Rho0=a0, Cp0=b0,
//   KPP0=c0, Mu0=a0·d0.  (rev.2 §9; docs/NOB_RBC_SPEC.md)
struct PropertyModel {
  real_t a[6] = {1,0,0,0,0,0};   // density ρ
  real_t b[6] = {1,0,0,0,0,0};   // specific heat Cp
  real_t c[6] = {1,0,0,0,0,0};   // thermal conductivity κ
  real_t d[6] = {1,0,0,0,0,0};   // viscosity factor (μ = ρ_poly · d_poly)
  real_t DeltaT = 0;             // dimensional ΔT — the polynomial argument scale

  // Fig-9 glycerol coefficients (PaScaL_TCS module_global.f90 l.123-126).
  // Note a default-constructed PropertyModel (only *[0]=1) is already the
  // constant-property OB model; independently, at DeltaT=0 the polynomial
  // argument t = DeltaT·θ vanishes, so even glycerol reduces exactly to OB.
  static PropertyModel glycerol(real_t DeltaT) {
    PropertyModel m;
    m.a[0]=1.2477e3;  m.a[1]=-4.789e-4;  m.a[2]=-0.3795e-6; m.a[3]=0;          m.a[4]=0;           m.a[5]=0;            // ρ
    m.b[0]=2.5108e3;  m.b[1]=22.511e-4;  m.b[2]=0;          m.b[3]=0;          m.b[4]=0;           m.b[5]=0;            // Cp
    m.c[0]=2.9351e-3; m.c[1]=3.863e-4;   m.c[2]=0;          m.c[3]=0;          m.c[4]=0;           m.c[5]=0;            // κ
    m.d[0]=238.71e-6; m.d[1]=-702.83e-4; m.d[2]=2393.1e-6;  m.d[3]=-6923.0e-8; m.d[4]=33131.3e-10; m.d[5]=-71517.5e-12; // μ-factor
    m.DeltaT = DeltaT;
    return m;
  }
};

// Effective-coefficient fields produced from T (all cell-centered, full array
// incl. halo). Consumed by the variable-μ momentum, variable-κ energy, and NOB
// buoyancy. Templated on field type for Cpu/Gpu symmetry.
// ── Shared one-point evaluation (single implementation for cpu/ AND gpu/) ────

// Horner: k[0]·(1 + k[1]t + k[2]t² + k[3]t³ + k[4]t⁴ + k[5]t⁵)
MPMSTD_HD double poly(const real_t k[6], double t) {
  return double(k[0]) * (1.0 + t * (k[1] + t * (k[2] + t * (k[3] + t * (k[4] + t * k[5])))));
}

struct PropertyValues { real_t Mu, invRho, KPP, dKPP, invRhoCp, dinvRhoCp; };

// Property evaluation at one point, line-for-line against PaScaL_TCS
// (module_solve_momentum.f90:165-168, module_solve_thermal.f90:155-168).
// Normalizers Rho0=a0, Cp0=b0, KPP0=c0, Mu0=a0·d0 (module_global.f90:134-137).
MPMSTD_HD PropertyValues evaluate_properties(const PropertyModel& m, real_t theta) {
  const real_t* a = m.a; const real_t* b = m.b; const real_t* c = m.c; const real_t* d = m.d;
  const double Rho0 = double(a[0]);
  const double Cp0  = double(b[0]);
  const double KPP0 = double(c[0]);
  const double Mu0  = double(a[0]) * d[0];
  const double t = double(m.DeltaT) * double(theta);   // Δθ (TCS tmp1)

  PropertyValues v;
  // module_solve_momentum.f90:165-168
  v.Mu     = static_cast<real_t>(poly(a, t) * poly(d, t) / Mu0);
  // invRho = Rho0/(a10·(1 + a11*t + …))
  v.invRho = static_cast<real_t>(Rho0 / poly(a, t));

  // module_solve_thermal.f90:155-156:
  //   KPP  = c10*(1 + c11*t +   c12*t² + …)/KPP0
  //   dKPP = c10*(c11 + 2*c12*t + 3*c13*t² + …)/KPP0
  v.KPP  = static_cast<real_t>(poly(c, t) / KPP0);
  v.dKPP = static_cast<real_t>(double(c[0]) * (c[1] + t * (2.0*c[2] + t * (3.0*c[3] + t * (4.0*c[4] + t * 5.0*c[5])))) / KPP0);

  // module_solve_thermal.f90:158-160:
  //   invRhoCp = Rho0*Cp0 /(a10·(1 + a11*t + …)) /(b10·(1 + b11*t + …))
  v.invRhoCp = static_cast<real_t>(Rho0 * Cp0 / poly(a, t) / poly(b, t));

  // module_solve_thermal.f90:162-168 — analytic derivative; the reference
  // itself truncates the denominators at cubic: /(1+a11 t+a12 t²+a13 t³)²/(1+b11 t+…)²
  const double ia = 1.0 + t * (a[1] + t * (a[2] + t * a[3]));
  const double ib = 1.0 + t * (b[1] + t * (b[2] + t * b[3]));
  const double t2 = t * t, t3 = t2 * t;
  const double num = b[1] * (1.0 + 3.0*a[2]*t2 + 4.0*a[3]*t3)
                   + a[1] * (1.0 + 2.0*b[1]*t + 3.0*b[2]*t2 + 4.0*b[3]*t3)
                   + t * ( b[2] * (2.0 + 5.0*a[3]*t3)
                         + a[2] * (2.0 + 4.0*b[2]*t2 + 5.0*b[3]*t3)
                         + 3.0*t * (a[3] + b[3] + 2.0*a[3]*b[3]*t3) );
  v.dinvRhoCp = static_cast<real_t>(-num / (ia * ia) / (ib * ib));
  return v;
}

template <class FieldT>
struct PropertyFieldsT {
  // Field names follow the PaScaL_TCS reference arrays (module_solve_momentum/
  // thermal.f90): Mu, invRho, KPP, dKPP, invRhoCp, dinvRhoCp.
  FieldT Mu, invRho, KPP, dKPP, invRhoCp, dinvRhoCp;
  explicit PropertyFieldsT(const parallel::mpi::Subdomain& sub)
    : Mu(sub, "Mu"), invRho(sub, "invRho"), KPP(sub, "KPP"),
      dKPP(sub, "dKPP"), invRhoCp(sub, "invRhoCp"), dinvRhoCp(sub, "dinvRhoCp") {}
};
using CpuPropertyFields = PropertyFieldsT<core::CpuField>;
using GpuPropertyFields = PropertyFieldsT<core::GpuField>;

void update_properties_cpu(CpuPropertyFields& props, const core::CpuField& T, const PropertyModel& m);
void update_properties_gpu(GpuPropertyFields& props, const core::GpuField& T, const PropertyModel& m);

} // namespace mpmstd::physics::properties
