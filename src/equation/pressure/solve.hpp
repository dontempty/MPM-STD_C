#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"

namespace mpmstd::equation {

// Pressure step (rev.2 §7), bundled: div(U*) → pencil-FFT/transpose/z-TDMA
// Poisson → dP unpack into Fields[P] → project Fields[U,V,W] divergence-free.
// Engine (FFTW plans/buffers) held by PressureSystem, built lazily from Domain+BC.
// inv_rho == nullptr → OB (constant-density) path.  Non-null → NOB: adds the
// variable-density ExtraTerm to the Poisson RHS and applies the (invRhoc-1)*∇P̂
// correction in the velocity projection (PaScaL_TCS mpi_pressure_RHS §214-225 +
// mpi_pressure_Projection §804-817).
void solve_pressure_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::CpuFields& fields, core::PressureSystem& poi, real_t dt,
                        const real_t* inv_rho = nullptr);
void solve_pressure_gpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::GpuFields& fields, core::PressureSystem& poi, real_t dt);

} // namespace mpmstd::equation
