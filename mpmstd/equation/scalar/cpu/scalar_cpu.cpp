#include "equation/scalar/assemble.hpp"
#include "equation/scalar/solve.hpp"

// P0 skeleton stubs (no-op). P1 ports the MPM-STD scalar ADI (convection
// explicit + CN diffusion) using solve/banded_solver; P6/P7 add OB/NOB κ.

namespace mpmstd::equation {

void assemble_scalar_system_cpu(core::ScalarSystem& /*sys*/,
                                const core::CpuField& /*T*/,
                                const core::CpuField& /*U*/, const core::CpuField& /*V*/, const core::CpuField& /*W*/,
                                const core::CpuField& /*kappa*/,
                                const core::Grid& /*grid*/, const core::Boundary& /*bc*/, real_t /*dt*/) {
  // TODO(P1)
}

void solve_scalar_cpu(core::ScalarSystem& /*sys*/, core::CpuField& /*T*/, const core::Subdomain& /*sub*/) {
  // TODO(P1)
}

} // namespace mpmstd::equation
