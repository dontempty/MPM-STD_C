#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/domain.hpp"   // Domain (grid metrics for face-interpolated buoyancy T)

namespace mpmstd::physics {

// OB/NOB Boussinesq buoyancy. Composed in main AFTER momentum assemble (adds to
// the RHS b). Omit the call ⇒ no buoyancy. (rev.2 §9)
struct BuoyancyParams {
  int    gravity_axis = 2;   // wall/gravity axis is case-dependent (channel z, DHVC streamwise, RBC vertical)
  real_t coeff        = 0;   // Ra/Pr-derived scaling (set per case)
  real_t T_ref        = 0;
  // NOB density-nonlinearity (RBC glycerol, NOB_RBC_SPEC.md §5):
  //   F = invRho · (θ + a12pera11·θ²·DeltaT),  Cmt=1, no T_ref (θ already centred).
  // a12pera11 = a12/a11 of the density polynomial; 0 ⇒ OB (then F = invRho·θ).
  real_t a12pera11    = 0;
  real_t delta_T      = 0;
};

// OB Boussinesq: F = coeff·(Tc − T_ref) on the gravity-axis component, where Tc
// is T interpolated onto that component's staggered face (PaScaL_TCS f90:348).
void add_buoyancy_force_cpu(const core::Domain& domain, core::CpuMomentumSystem& mom, const core::CpuField& T, const BuoyancyParams& p, real_t dt);
void add_buoyancy_force_gpu(const core::Domain& domain, core::GpuMomentumSystem& mom, const core::GpuField& T, const BuoyancyParams& p, real_t dt);

// NOB (variable density): F = invRhoc·(Tc + a12pera11·Tc²·DeltaT), with Tc and
// invRhoc both face-interpolated onto the gravity-axis component (like the OB
// function above). With invRho≡1, a12pera11=0 this reduces to the OB form
// with coeff=1, T_ref=0.
void add_buoyancy_force_nob_cpu(const core::Domain& domain, core::CpuMomentumSystem& mom, const core::CpuField& T, const core::CpuField& inv_rho, const BuoyancyParams& p, real_t dt);
void add_buoyancy_force_nob_gpu(const core::Domain& domain, core::GpuMomentumSystem& mom, const core::GpuField& T, const core::GpuField& inv_rho, const BuoyancyParams& p, real_t dt);

} // namespace mpmstd::physics
