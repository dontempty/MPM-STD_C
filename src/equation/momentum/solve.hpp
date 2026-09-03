#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/fields.hpp"
#include "equation/momentum/linear_system.hpp"

namespace mpmstd::post { class TimingRecorder; }   // scaling-run timing (post/timing.hpp)

namespace mpmstd::equation::momentum {

// rev.2 M2: solve_momentum does U,V,W + the block lower-triangular velocity
// coupling in one (MPM-STD fortran core_momentum). Increments are written into
// the MomentumSystem's own dU,dV,dW (created inside momentum). Distributed
// (P)TDMA via Domain::tdma; BC via core::modify_tdma_row_cpu + sync.
void solve_momentum_constant_property_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                        core::CpuFields& fields, equation::momentum::CpuLinearSystem& mom, real_t dt);
void solve_momentum_constant_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                        core::GpuFields& fields, equation::momentum::GpuLinearSystem& mom, real_t dt);

// Variable-viscosity (NOB) twin: same 3-sweep ADI + block coupling but with the
// face-μ + 1/ρ̂ variable-viscosity bands (build_adi_bands_var). With μ≡1, invρ≡1
// it reproduces solve_momentum_constant_property_cpu.
void solve_momentum_variable_property_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                            core::CpuFields& fields, const core::CpuField& mu, const core::CpuField& inv_rho,
                            equation::momentum::CpuLinearSystem& mom, real_t dt);
// Kernel-level timing sub-events of solve_momentum_variable_property_gpu
// (apps/rbc/timing_plan.md §6).  Directions distinguished (z = distributed
// TDMA/MPI); the three velocity components accumulate into the same events.
enum GpuSolveTimingEvent {
    kSolveTimingCrossCouple = 0,   // k_dd_dV_var / k_dd_dW_var (+k_zero_dw_wall_face)
    kSolveTimingBuildX, kSolveTimingBuildY, kSolveTimingBuildZ,   // k_mom_adi_var
    kSolveTimingWallRows,          // k_modify_tdma_row_gpu (+k_w_wallface_identity_row)
    kSolveTimingTdmaX, kSolveTimingTdmaY, kSolveTimingTdmaZ,      // (P)TDMA; z = MPI
    kSolveTimingScatterX, kSolveTimingScatterY, kSolveTimingScatterZ,   // k_m_scatter
    kSolveTimingCopyOut,           // k_m_copy_field (delta out)
    kSolveTimingHalo,              // sync_field(dU/dV/dW) inside the solve
    kSolveTimingBlockCouple,       // k_block_dV_var / k_block_dU_var
    kGpuSolveTimingEventCount
};
inline const char* const kGpuSolveTimingEventNames[kGpuSolveTimingEventCount] = {
    "momentum_cross_couple",
    "momentum_build_x", "momentum_build_y", "momentum_build_z",
    "momentum_wall_rows",
    "momentum_tdma_x", "momentum_tdma_y", "momentum_tdma_z",
    "momentum_scatter_x", "momentum_scatter_y", "momentum_scatter_z",
    "momentum_copy_out",
    "momentum_halo",
    "momentum_block_couple",
};
/// mark() calls per invocation with the RBC topology (x,y periodic; z wall):
/// per component adi 11 (x/y: build+tdma+scatter, z: +wall_rows, +copy_out) +
/// halo 1 = 12; V and W add one cross_couple each; block stage adds
/// block_couple, halo(dV), block_couple: 12+13+13+3 = 41.  (The lower-wall
/// k_zero_dw_wall_face and the env-gated debug stages are deliberately
/// unmarked — they are rank-/env-conditional and would break the fixed
/// pattern; they are absorbed into the following interval.)
constexpr int kGpuSolveTimingMarksPerCall = 41;

// `recorder` (nullptr = no instrumentation, identical execution) records the
// sub-events above at `timing_event_base + GpuSolveTimingEvent`.
void solve_momentum_variable_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                            core::GpuFields& fields, const core::GpuField& mu, const core::GpuField& inv_rho,
                            equation::momentum::GpuLinearSystem& mom, real_t dt,
                            post::TimingRecorder* recorder = nullptr, int timing_event_base = 0);

// pseudo-update: U += dU (etc.); caller syncs U,V,W afterwards.
void update_velocity_cpu(core::CpuFields& fields, const equation::momentum::CpuLinearSystem& mom);
void update_velocity_gpu(core::GpuFields& fields, const equation::momentum::GpuLinearSystem& mom);

// Report/30 §6 instrumentation: prints max|f| over the top-wall slab (last 6
// interior k-planes) vs the remaining interior. No-op unless MPMSTD_MOMDBG>=1.
// With MPMSTD_MOMDBG>=2 and argmax_tag != 0, also prints the interior
// locations holding the field max (spike locator).
void debug_slabmax_gpu(const char* tag, const core::GpuField& f, const core::Domain& domain,
                       char argmax_tag = 0);

} // namespace mpmstd::equation::momentum
