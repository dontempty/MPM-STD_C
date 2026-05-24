#pragma once

#include "common/types.hpp"
#include "common/direction.hpp"
#include "grid/grid.hpp"

namespace mpmstd::grid {

// Device-side mirror of Grid metrics (xc, dx, dmx).
// CPU build : owns no device memory; raw pointers fall back to host arrays.
// CUDA build: cudaMalloc + Memcpy of each metric array (M5').
//
// For M0 the implementation is the CPU-equivalent: ptrs alias the host buffers.

class GridDevice {
public:
  explicit GridDevice(const Grid& g);
  ~GridDevice();

  GridDevice(const GridDevice&) = delete;
  GridDevice& operator=(const GridDevice&) = delete;

  const real_t* xc (Direction d) const { return xc_ [to_int(d)]; }
  const real_t* dx (Direction d) const { return dx_ [to_int(d)]; }
  const real_t* dmx(Direction d) const { return dmx_[to_int(d)]; }

private:
  const Grid&   host_grid_;
  const real_t* xc_ [3] = {nullptr, nullptr, nullptr};
  const real_t* dx_ [3] = {nullptr, nullptr, nullptr};
  const real_t* dmx_[3] = {nullptr, nullptr, nullptr};

  // Owned device buffers (CUDA build only).
  real_t* xc_owned_ [3] = {nullptr, nullptr, nullptr};
  real_t* dx_owned_ [3] = {nullptr, nullptr, nullptr};
  real_t* dmx_owned_[3] = {nullptr, nullptr, nullptr};
};

} // namespace mpmstd::grid
