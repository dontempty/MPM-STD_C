#include "core/gpu/gpu_grid.hpp"
#include "core/grid.hpp"

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>

namespace mpmstd::core::gpu {

static void cuda_check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "GpuGridArrays CUDA error in %s: %s\n",
                 msg, cudaGetErrorString(e));
    std::abort();
  }
}

static double* upload_vec(const std::vector<real_t>& v) {
  double* d = nullptr;
  cuda_check(cudaMalloc(&d, v.size() * sizeof(double)), "cudaMalloc grid");
  cuda_check(cudaMemcpy(d, v.data(), v.size() * sizeof(double),
                        cudaMemcpyHostToDevice), "cudaMemcpy grid");
  return d;
}

GpuGridArrays::GpuGridArrays(const Grid& g) {
  for (int a = 0; a < 3; ++a) {
    d_dx_[a]  = upload_vec(g.dx (static_cast<Direction>(a)));
    d_dmx_[a] = upload_vec(g.dmx(static_cast<Direction>(a)));
  }
}

GpuGridArrays::~GpuGridArrays() {
  for (int a = 0; a < 3; ++a) {
    if (d_dx_[a])  cudaFree(d_dx_[a]);
    if (d_dmx_[a]) cudaFree(d_dmx_[a]);
  }
}

} // namespace mpmstd::core::gpu
