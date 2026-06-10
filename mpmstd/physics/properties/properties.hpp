#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"   // Subdomain

namespace mpmstd::physics {

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

  static PropertyModel glycerol(real_t DeltaT);   // Fig-9 glycerol coefficients
};

// Effective-coefficient fields produced from T (all cell-centered, full array
// incl. halo). Consumed by the variable-μ momentum, variable-κ energy, and NOB
// buoyancy. Templated on field type for Cpu/Gpu symmetry.
template <class FieldT>
struct PropertyFieldsT {
  FieldT mu, inv_rho, kappa, dkappa, inv_rhocp, dinv_rhocp;
  explicit PropertyFieldsT(const core::Subdomain& sub)
    : mu(sub, "mu"), inv_rho(sub, "inv_rho"), kappa(sub, "kappa"),
      dkappa(sub, "dkappa"), inv_rhocp(sub, "inv_rhocp"), dinv_rhocp(sub, "dinv_rhocp") {}
};
using CpuPropertyFields = PropertyFieldsT<core::CpuField>;
using GpuPropertyFields = PropertyFieldsT<core::GpuField>;

void update_properties_cpu(CpuPropertyFields& props, const core::CpuField& T, const PropertyModel& m);
void update_properties_gpu(GpuPropertyFields& props, const core::GpuField& T, const PropertyModel& m);

} // namespace mpmstd::physics
