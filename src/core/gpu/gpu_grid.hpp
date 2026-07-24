#pragma once

// Device-resident copies of grid metric arrays (dx, dmx per axis).
// Backend-agnostic header: no <cuda_runtime.h> included here.
// Definitions are in gpu_grid.cu (CUDA build only).

#include "core/grid.hpp"  // Grid = grid::Grid — type alias, cannot be forward-declared

namespace mpmstd::core {

// Holds device pointers to per-axis dx/dmx arrays.
// Constructed from a host Grid; freed in destructor.
// Only usable in CUDA builds.
class GpuGridArrays {
public:
  explicit GpuGridArrays(const grid::Grid& host_grid);
  ~GpuGridArrays();

  GpuGridArrays(const GpuGridArrays&)            = delete;
  GpuGridArrays& operator=(const GpuGridArrays&) = delete;

  double* d_dx1()  const { return d_dx_[0]; }
  double* d_dmx1() const { return d_dmx_[0]; }
  double* d_dx2()  const { return d_dx_[1]; }
  double* d_dmx2() const { return d_dmx_[1]; }
  double* d_dx3()  const { return d_dx_[2]; }
  double* d_dmx3() const { return d_dmx_[2]; }

private:
  double* d_dx_[3]  = {};
  double* d_dmx_[3] = {};
};

} // namespace mpmstd::core
