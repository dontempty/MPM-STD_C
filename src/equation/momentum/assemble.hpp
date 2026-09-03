#pragma once

#include "core/domain.hpp"
#include "core/fields.hpp"
#include "equation/momentum/linear_system.hpp"

namespace mpmstd::post { class TimingRecorder; }   // scaling-run timing (post/timing.hpp)

namespace mpmstd::equation::momentum {

// Momentum assemble — T-independent (rev.2 §9). Reads the velocity fields + the
// nu constant from `Fields`, builds the explicit MPM-STD BW-ADI RHS per component
// into the MomentumSystem (compute_mpmstd_rhs). Pressure gradient is applied by
// projection (+ channel forcing), NOT here.
//   const_visc : scalar nu (Fields constant), NO cross-stress  → channel/cavity/DHVC (P1)
//   var_visc   : mu field,  WITH cross-stress → NOB/LES (P7)
void assemble_momentum_constant_property_cpu(const core::Domain& domain, core::CpuFields& fields,
                                      equation::momentum::CpuLinearSystem& mom, real_t dt);
void assemble_momentum_variable_property_cpu  (const core::Domain& domain, core::CpuFields& fields,
                                      const core::CpuField& mu, const core::CpuField& inv_rho,
                                      equation::momentum::CpuLinearSystem& mom, real_t dt);

void assemble_momentum_constant_property_gpu(const core::Domain& domain, core::GpuFields& fields,
                                      equation::momentum::GpuLinearSystem& mom, real_t dt);

// Kernel-level timing sub-events of assemble_momentum_variable_property_gpu
// (apps/rbc/timing_plan.md §6): one k_mom_rhs_var launch per velocity component.
enum GpuAssembleTimingEvent {
    kAssembleTimingRhsU = 0, kAssembleTimingRhsV, kAssembleTimingRhsW,
    kGpuAssembleTimingEventCount
};
inline const char* const kGpuAssembleTimingEventNames[kGpuAssembleTimingEventCount] = {
    "momentum_rhs_u", "momentum_rhs_v", "momentum_rhs_w",
};
constexpr int kGpuAssembleTimingMarksPerCall = 3;

// `recorder` (nullptr = no instrumentation, identical execution) records the
// sub-events above at `timing_event_base + GpuAssembleTimingEvent`.
void assemble_momentum_variable_property_gpu  (const core::Domain& domain, core::GpuFields& fields,
                                      const core::GpuField& mu, const core::GpuField& inv_rho,
                                      equation::momentum::GpuLinearSystem& mom, real_t dt,
                                      post::TimingRecorder* recorder = nullptr, int timing_event_base = 0);

} // namespace mpmstd::equation::momentum
