#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/fields.hpp"
#include "equation/scalar/linear_system.hpp"

namespace mpmstd::post { class TimingRecorder; }   // scaling-run timing (post/timing.hpp)

namespace mpmstd::equation::scalar {

// Kernel-level timing sub-events of solve_scalar_variable_property_gpu
// (apps/rbc/timing_plan.md §6).  The caller appends `kGpuTimingEventNames` to
// its recorder's event list and passes the block's base index; the solver marks
// `base + <event>`.  Directions are distinguished (only z carries distributed
// TDMA/MPI); repeats accumulate.
enum GpuTimingEvent {
    kTimingBuildX = 0, kTimingBuildY, kTimingBuildZ,      // k_build_A_rhs_nob per sweep
    kTimingWallRows,                                       // k_modify_tdma_row_gpu (wall dirs)
    kTimingTdmaX, kTimingTdmaY, kTimingTdmaZ,              // (P)TDMA solve; z = distributed/MPI
    kTimingScatterX, kTimingScatterY, kTimingScatterZ,     // k_scatter_from_tdma per sweep
    kTimingUpdate,                                         // k_add_increment (T += delta)
    kGpuTimingEventCount
};
inline const char* const kGpuTimingEventNames[kGpuTimingEventCount] = {
    "scalar_build_x", "scalar_build_y", "scalar_build_z",
    "scalar_wall_rows",
    "scalar_tdma_x", "scalar_tdma_y", "scalar_tdma_z",
    "scalar_scatter_x", "scalar_scatter_y", "scalar_scatter_z",
    "scalar_update",
};
/// mark() calls the GPU solver issues per invocation with the RBC topology
/// (x,y periodic; z wall): X/Y 3 each, Z 4 (adds wall_rows), update 1.
constexpr int kGpuTimingMarksPerCall = 11;

// Solve the assembled scalar ADI system: 3-sweep ADI (sweep order from BC,
// distributed (P)TDMA via Domain::tdma, wall rows via modify_tdma_row_cpu) then
// add the increment back into Fields[Var::T]. Caller syncs T afterward. (rev.2 §5)
void solve_scalar_constant_property_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                         core::CpuFields& fields, equation::scalar::CpuLinearSystem& scalar_system, real_t dt);
void solve_scalar_constant_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                         core::GpuFields& fields, equation::scalar::GpuLinearSystem& scalar_system, real_t dt);

// Variable-κ̂ (NOB) twin — FAITHFUL PaScaL_TCS delta-form 3-factor ADI, fused
// build_A_RHS: each sweep builds bands AND packs the RHS in one pass
// (build_A_rhs_nob / k_build_A_rhs_nob) carrying that axis's diffusion + dκ̂
// linearization + 0.25 conservative-face semi-implicit convection; stage 1
// computes the explicit residual R inline and packs the M·θⁿ delta-form RHS
// (no separate coef step). Needs dKPP + dinvRhoCp. With props≡1 the
// diffusion reduces to solve_scalar_constant_property_cpu's. Ct=alpha_T (=1/√(Ra·Pr)).
void solve_scalar_variable_property_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                          core::CpuFields& fields,
                          const core::CpuField& KPP, const core::CpuField& dKPP,
                          const core::CpuField& invRhoCp, const core::CpuField& dinvRhoCp,
                          equation::scalar::CpuLinearSystem& scalar_system, real_t dt);
// `recorder` (nullptr = no instrumentation, identical execution) records the
// kernel-level sub-events above at `timing_event_base + GpuTimingEvent`.
void solve_scalar_variable_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                          core::GpuFields& fields,
                          const core::GpuField& KPP, const core::GpuField& dKPP,
                          const core::GpuField& invRhoCp, const core::GpuField& dinvRhoCp,
                          equation::scalar::GpuLinearSystem& scalar_system, real_t dt,
                          post::TimingRecorder* recorder = nullptr, int timing_event_base = 0);

} // namespace mpmstd::equation::scalar
