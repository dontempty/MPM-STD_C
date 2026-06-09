#include "driver/cfl.hpp"

// P1/P5 stub (GPU build only). Device max-reduction at P5.
namespace mpmstd::driver {
real_t compute_cfl_dt_gpu(const core::Domain&, const core::GpuFields&, real_t /*max_cfl*/, real_t dt_cap) {
  return dt_cap;  // TODO(P5)
}
} // namespace mpmstd::driver
