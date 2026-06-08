#include "driver/cfl.hpp"

// P0 skeleton stub (GPU build only). P5 computes the CFL min on device.
namespace mpmstd::driver {
real_t compute_cfl_dt_gpu(const core::GpuField&, const core::GpuField&, const core::GpuField&,
                          const core::Grid&, const core::Subdomain&, real_t dt_cap) {
  return dt_cap;  // TODO(P5)
}
} // namespace mpmstd::driver
