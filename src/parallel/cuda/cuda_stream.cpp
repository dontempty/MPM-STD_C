#include "parallel/cuda/cuda_stream.hpp"
#include "common/macros.hpp"

#ifdef MPMSTD_BACKEND_CUDA
  #include <cuda_runtime.h>
#endif

namespace mpmstd::parallel::cuda_helpers {

Stream::Stream() {
#ifdef MPMSTD_BACKEND_CUDA
  cudaStream_t s = nullptr;
  CUDA_CHECK(cudaStreamCreate(&s));
  native_ = reinterpret_cast<void*>(s);
#endif
}

Stream::~Stream() {
#ifdef MPMSTD_BACKEND_CUDA
  if (native_) {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(native_));
  }
#endif
}

void Stream::synchronize() {
#ifdef MPMSTD_BACKEND_CUDA
  if (native_) {
    CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(native_)));
  }
#endif
}

Stream& default_stream() {
  // A single shared instance representing the default (legacy) stream.
  // On CPU build this is just a Stream with native_ = nullptr.
  static Stream s;
  return s;
}

} // namespace mpmstd::parallel::cuda_helpers
