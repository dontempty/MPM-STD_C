#include "equation/scalar/assemble.hpp"
#include "equation/scalar/solve.hpp"

// P5 stubs (no-op), GPU build only. Device scalar ADI kernels land at P5.

namespace mpmstd::equation {

void assemble_scalar_const_diff_gpu(const core::Domain&, core::GpuFields&, core::GpuScalarSystem&, real_t) {
  // TODO(P5)
}
void solve_scalar_gpu(const core::Domain&, const core::BoundaryCondition&, core::GpuFields&, core::GpuScalarSystem&, real_t) {
  // TODO(P5)
}

} // namespace mpmstd::equation
