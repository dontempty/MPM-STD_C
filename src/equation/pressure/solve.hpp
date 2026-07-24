#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/variables.hpp"
#include "equation/momentum/linear_system.hpp"
#include "equation/pressure/linear_system.hpp"

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

// NOB GPU pressure solve: adds ExtraTerm + NOB projection, updates internal dPhat.
void solve_pressure_variable_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                             core::GpuFields& fields, equation::pressure::LinearSystem& poi, real_t dt,
                             const core::GpuField& inv_rho);

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
