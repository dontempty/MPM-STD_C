#pragma once

#include "core/domain.hpp"
#include "core/boundary.hpp"
#include "core/variables.hpp"
#include "core/system.hpp"

namespace mpmstd::equation {

// rev.2 M2: solve_momentum does U,V,W + the block lower-triangular velocity
// coupling in one (MPM-STD fortran core_momentum). Increments are written into
// the MomentumSystem's own dU,dV,dW (created inside momentum). Distributed
// (P)TDMA via Domain::tdma; BC via core::modify_tdma_row_cpu + sync.
void solve_momentum_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::CpuFields& fields, core::CpuMomentumSystem& mom, real_t dt);
void solve_momentum_gpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                        core::GpuFields& fields, core::GpuMomentumSystem& mom, real_t dt);

// Variable-viscosity (NOB) twin: same 3-sweep ADI + block coupling but with the
// face-μ + 1/ρ̂ variable-viscosity bands (build_adi_bands_var). With μ≡1, invρ≡1
// it reproduces solve_momentum_cpu.
void solve_momentum_var_cpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                            core::CpuFields& fields, const core::CpuField& mu, const core::CpuField& inv_rho,
                            core::CpuMomentumSystem& mom, real_t dt);
void solve_momentum_var_gpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                            core::GpuFields& fields, const core::GpuField& mu, const core::GpuField& inv_rho,
                            core::GpuMomentumSystem& mom, real_t dt);

// pseudo-update: U += dU (etc.); caller syncs U,V,W afterwards.
void update_velocity_cpu(core::CpuFields& fields, const core::CpuMomentumSystem& mom);
void update_velocity_gpu(core::GpuFields& fields, const core::GpuMomentumSystem& mom);

// Report/30 §6 instrumentation: prints max|f| over the top-wall slab (last 6
// interior k-planes) vs the remaining interior. No-op unless MPMSTD_MOMDBG>=1.
// With MPMSTD_MOMDBG>=2 and argmax_tag != 0, also prints the interior
// locations holding the field max (spike locator).
void debug_slabmax_gpu(const char* tag, const core::GpuField& f, const core::Domain& domain,
                       char argmax_tag = 0);

} // namespace mpmstd::equation
