#include "equation/scalar/assemble.hpp"
#include "equation/scalar/solve.hpp"

// P0 skeleton stubs (no-op), GPU build only. P5 ports the device kernels.

namespace mpmstd::equation {

void assemble_scalar_system_gpu(core::ScalarSystem& /*sys*/,
                                const core::GpuField& /*T*/,
                                const core::GpuField& /*U*/, const core::GpuField& /*V*/, const core::GpuField& /*W*/,
                                const core::GpuField& /*kappa*/,
                                const core::Grid& /*grid*/, const core::Boundary& /*bc*/, real_t /*dt*/) {
  // TODO(P5)
}

void solve_scalar_gpu(core::ScalarSystem& /*sys*/, core::GpuField& /*T*/, const core::Subdomain& /*sub*/) {
  // TODO(P5)
}

} // namespace mpmstd::equation
