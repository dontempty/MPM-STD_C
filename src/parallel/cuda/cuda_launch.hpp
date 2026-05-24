#pragma once

#include "common/macros.hpp"
#include "common/types.hpp"

namespace mpmstd::parallel::cuda_helpers {

// 3D block/grid sizing helper. CPU build: returned values are not used in any
// kernel launch (no kernels compiled). CUDA build: passed to <<<grid,block>>>.
struct LaunchExtent3D {
  int n1, n2, n3;
};

struct BlockShape3D {
  int x = 8;
  int y = 8;
  int z = 4;
};

struct GridShape3D {
  int x, y, z;
};

inline GridShape3D compute_grid_shape(LaunchExtent3D ext, BlockShape3D block) {
  GridShape3D g;
  g.x = (ext.n1 + block.x - 1) / block.x;
  g.y = (ext.n2 + block.y - 1) / block.y;
  g.z = (ext.n3 + block.z - 1) / block.z;
  return g;
}

} // namespace mpmstd::parallel::cuda_helpers
