#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/fields.hpp"
#include "equation/momentum/linear_system.hpp"
#include "equation/pressure/linear_system.hpp"

namespace mpmstd::post { class TimingRecorder; }   // scaling-run timing (post/timing.hpp)

namespace mpmstd::equation::pressure {

// Pressure step (rev.2 §7), bundled: div(U*) → pencil-FFT/transpose/z-TDMA
// Poisson → dP unpack into Fields[P] → project Fields[U,V,W] divergence-free.
// Engine (FFTW plans/buffers) held by PressureSystem, built lazily from Domain+BC.
// inv_rho == nullptr → OB (constant-density) path.  Non-null → NOB: adds the
// variable-density ExtraTerm to the Poisson RHS and applies the (invRhoc-1)*∇P̂
// correction in the velocity projection (PaScaL_TCS mpi_pressure_RHS §214-225 +
// mpi_pressure_Projection §804-817).
void solve_pressure_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                        core::CpuFields& fields, equation::pressure::LinearSystem& poi, real_t dt,
                        const real_t* inv_rho = nullptr);
void solve_pressure_constant_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                        core::GpuFields& fields, equation::pressure::LinearSystem& poi, real_t dt);

// Kernel-level timing sub-events of solve_pressure_variable_property_gpu
// (apps/rbc/timing_plan.md §6), in execution order of PressureSolverGpu::solve_nob
// on the production fast path (np1==np2==1, GPU-resident z-TDMA).  On other
// paths the extra transposes/H<->D copies land in the neighbouring intervals.
enum GpuPressureTimingEvent {
    kPressureTimingDivergence = 0,   // k_divergence_rhs_pres
    kPressureTimingExtraTerm,        // k_extra_term_nob_pres (NOB)
    kPressureTimingFftXForward,      // cuFFT D2Z
    kPressureTimingFftYForward,      // cuFFT Z2Z forward
    kPressureTimingTdmaSplit,        // k_tdma_split_hat_Y (complex -> two real bands)
    kPressureTimingTdmaZ,            // distributed z-TDMA (MPI)
    kPressureTimingTdmaMerge,        // k_tdma_merge_hat_Y
    kPressureTimingFftYBackward,     // cuFFT Z2Z inverse
    kPressureTimingFftXBackward,     // cuFFT Z2D + k_gpu_scale
    kPressureTimingUnpack,           // k_unpack_dp_C_to_P
    kPressureTimingProject,          // k_project_nob_pres
    kPressureTimingDphatExtrapolate, // k_dphat_extrapolate
    kPressureTimingHalo,             // dPhat + dU/dV/dW halo exchanges (MPI) + final sync
    kGpuPressureTimingEventCount
};
inline const char* const kGpuPressureTimingEventNames[kGpuPressureTimingEventCount] = {
    "pressure_divergence", "pressure_extra_term",
    "pressure_fft_x_forward", "pressure_fft_y_forward",
    "pressure_tdma_split", "pressure_tdma_z", "pressure_tdma_merge",
    "pressure_fft_y_backward", "pressure_fft_x_backward",
    "pressure_unpack", "pressure_project",
    "pressure_dphat_extrapolate", "pressure_halo",
};
constexpr int kGpuPressureTimingMarksPerCall = kGpuPressureTimingEventCount;   // each once

// NOB GPU pressure solve: adds ExtraTerm + NOB projection, updates internal dPhat.
// `recorder` (nullptr = no instrumentation, identical execution) records the
// sub-events above at `timing_event_base + GpuPressureTimingEvent`.
void solve_pressure_variable_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                             core::GpuFields& fields, equation::pressure::LinearSystem& poi, real_t dt,
                             const core::GpuField& inv_rho,
                             post::TimingRecorder* recorder = nullptr, int timing_event_base = 0);

// ── Incremental pressure-correction (van Kan) ───────────────────────────────────
// Predictor coupling: subtract dt·∇P_acc (the accumulated pressure Pⁿ) from the
// momentum RHS dU/dV/dW — same staggered-face gradient as the projection — matching
// PaScaL_TCS module_solve_momentum.f90:344-345. Call AFTER assemble+buoyancy, BEFORE
// the momentum solve. Pairs with accumulate_pressure_* to form 2nd-order incremental
// projection (vs the previous 1st-order non-incremental fractional step).
void add_pressure_gradient_constant_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom,
                               const core::CpuField& P_acc, real_t dt);
void add_pressure_gradient_constant_property_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom,
                               const core::GpuField& P_acc, real_t dt);
// NOB twin: subtract dt·(1/ρ̂)_face·∇P_acc with the PaScaL dx-weighted face invRho
// (mom.f90:331/343). The OB version omits the (1/ρ̂) weight (≡1 there).
void add_pressure_gradient_variable_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom,
                                   const core::CpuField& P_acc, const core::CpuField& inv_rho,
                                   real_t dt);
void add_pressure_gradient_variable_property_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom,
                                   const core::GpuField& P_acc, const core::GpuField& inv_rho,
                                   real_t dt);
// Accumulate the pressure increment δp (Fields[P], written by solve_pressure_*) into
// the persistent P_acc:  P_acc += δp  over ALL cells (ghost/halo are linear copies →
// stay consistent, no re-exchange). Matches PaScaL_TCS module_solve_pressure.f90:818.
void accumulate_pressure_cpu(core::CpuFields& fields, const equation::pressure::LinearSystem& poi);
void accumulate_pressure_gpu(core::GpuFields& fields, const equation::pressure::LinearSystem& poi);

} // namespace mpmstd::equation::pressure
