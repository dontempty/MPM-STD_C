#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"

// P0 skeleton stubs (no-op). P1 ports the MPM-STD delta-form Beam-Warming ADI
// (own ×2 implicit) + block lower-triangular coupling (block_couple_dV/dU) for
// const_visc; P7 adds the full var_visc cross-stress.

namespace mpmstd::equation {

void assemble_momentum_const_visc_cpu(core::MomentumSystem&, const core::CpuField&, const core::CpuField&,
                                      const core::CpuField&, const core::CpuField&, real_t,
                                      const core::Grid&, const core::Boundary&, real_t) { /* TODO(P1) */ }

void assemble_momentum_var_visc_cpu(core::MomentumSystem&, const core::CpuField&, const core::CpuField&,
                                    const core::CpuField&, const core::CpuField&, const core::CpuField&,
                                    const core::Grid&, const core::Boundary&, real_t) { /* TODO(P7) */ }

void solve_momentum_cpu(core::MomentumSystem&, const core::CpuField&, const core::CpuField&, const core::CpuField&,
                        core::CpuField&, core::CpuField&, core::CpuField&, const core::Subdomain&) { /* TODO(P1) */ }

void update_velocity_cpu(core::CpuField&, core::CpuField&, core::CpuField&,
                         const core::CpuField&, const core::CpuField&, const core::CpuField&) { /* TODO(P1) */ }

} // namespace mpmstd::equation
