#pragma once

#include "equation/momentum/linear_system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/domain.hpp"   // Domain (grid metrics for face-interpolated buoyancy T)
#include "physics/properties/properties.hpp"   // PropertyModel (from_property_model factory)

#include <cmath>   // std::fabs

namespace mpmstd::physics::buoyancy {

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

  // NOB-only factory: derive the density-nonlinearity params from a property
  // model — a12pera11 = a[2]/a[1] (with the a[1]≈0 guard), delta_T = model.DeltaT,
  // coeff=1, T_ref=0. OB cases (channel/DHVC) set `coeff` per-case by hand and
  // do not use this. Body moved verbatim from the RBC mains.
  static BuoyancyParams from_property_model(const properties::PropertyModel& model,
                                            int gravity_axis) {
    BuoyancyParams p;
    p.gravity_axis = gravity_axis;
    p.coeff        = 1;
    p.T_ref        = 0;
    p.a12pera11    = (std::fabs(model.a[2]) < real_t{1e-14}) ? real_t{0}
                                                            : model.a[2] / model.a[1];
    p.delta_T      = model.DeltaT;
    return p;
  }
};

// OB Boussinesq: F = coeff·(Tc − T_ref) on the gravity-axis component, where Tc
// is T interpolated onto that component's staggered face (PaScaL_TCS f90:348).
void add_buoyancy_force_constant_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom, const core::CpuField& T, const BuoyancyParams& p, real_t dt);
void add_buoyancy_force_constant_property_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom, const core::GpuField& T, const BuoyancyParams& p, real_t dt);

// NOB (variable density): F = invRhoc·(Tc + a12pera11·Tc²·DeltaT), with Tc and
// invRhoc both face-interpolated onto the gravity-axis component (like the OB
// function above). With invRho≡1, a12pera11=0 this reduces to the OB form
// with coeff=1, T_ref=0.
void add_buoyancy_force_variable_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom, const core::CpuField& T, const core::CpuField& inv_rho, const BuoyancyParams& p, real_t dt);
void add_buoyancy_force_variable_property_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom, const core::GpuField& T, const core::GpuField& inv_rho, const BuoyancyParams& p, real_t dt);

} // namespace mpmstd::physics::buoyancy
