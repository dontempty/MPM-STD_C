#include "grid/grid_device.hpp"
#include "parallel/cuda/cuda_memory.hpp"

namespace mpmstd::grid {

GridDevice::GridDevice(const Grid& g) : host_grid_(g) {
#ifdef MPMSTD_BACKEND_CUDA
  for (int a = 0; a < 3; ++a) {
    const auto d = static_cast<Direction>(a);
    const std::size_t n_bytes = g.dx(d).size() * sizeof(real_t);

    xc_owned_ [a] = reinterpret_cast<real_t*>(parallel::cuda_helpers::device_alloc(n_bytes));
    dx_owned_ [a] = reinterpret_cast<real_t*>(parallel::cuda_helpers::device_alloc(n_bytes));
    dmx_owned_[a] = reinterpret_cast<real_t*>(parallel::cuda_helpers::device_alloc(n_bytes));

    parallel::cuda_helpers::copy_host_to_device(xc_owned_ [a], g.xc (d).data(), n_bytes);
    parallel::cuda_helpers::copy_host_to_device(dx_owned_ [a], g.dx (d).data(), n_bytes);
    parallel::cuda_helpers::copy_host_to_device(dmx_owned_[a], g.dmx(d).data(), n_bytes);

    xc_ [a] = xc_owned_ [a];
    dx_ [a] = dx_owned_ [a];
    dmx_[a] = dmx_owned_[a];
  }
#else
  // CPU build: just alias the host arrays.
  for (int a = 0; a < 3; ++a) {
    const auto d = static_cast<Direction>(a);
    xc_ [a] = g.xc (d).data();
    dx_ [a] = g.dx (d).data();
    dmx_[a] = g.dmx(d).data();
  }
#endif
}

GridDevice::~GridDevice() {
#ifdef MPMSTD_BACKEND_CUDA
  for (int a = 0; a < 3; ++a) {
    parallel::cuda_helpers::device_free(xc_owned_ [a]);
    parallel::cuda_helpers::device_free(dx_owned_ [a]);
    parallel::cuda_helpers::device_free(dmx_owned_[a]);
  }
#endif
}

} // namespace mpmstd::grid
