#include "parallel/cuda/cuda_runtime.hpp"
#include "common/macros.hpp"

#ifdef MPMSTD_BACKEND_CUDA
  #include <cuda_runtime.h>
#endif

namespace mpmstd::parallel::cuda_helpers {

int initialize_device(int rank_in_node) {
#ifdef MPMSTD_BACKEND_CUDA
  int n_dev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n_dev));
  if (n_dev == 0) {
    return -1;
  }
  const int dev = rank_in_node % n_dev;
  CUDA_CHECK(cudaSetDevice(dev));
  return dev;
#else
  MPMSTD_UNUSED(rank_in_node);
  return -1;
#endif
}

void synchronize_device() {
#ifdef MPMSTD_BACKEND_CUDA
  CUDA_CHECK(cudaDeviceSynchronize());
#endif
}

int device_count() {
#ifdef MPMSTD_BACKEND_CUDA
  int n = 0;
  CUDA_CHECK(cudaGetDeviceCount(&n));
  return n;
#else
  return 0;
#endif
}

} // namespace mpmstd::parallel::cuda_helpers
