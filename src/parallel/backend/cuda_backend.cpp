#include "parallel/backend/cuda_backend.hpp"
#include "parallel/cuda/cuda_memory.hpp"
#include "parallel/cuda/cuda_runtime.hpp"

#include <stdexcept>

namespace mpmstd::parallel {

CudaBackend::CudaBackend() : name_("cuda") {
#ifndef MPMSTD_BACKEND_CUDA
  throw std::runtime_error(
      "CudaBackend instantiated in a CPU build. "
      "Rebuild with BACKEND=cuda or use CpuBackend.");
#endif
}

CudaBackend::~CudaBackend() = default;

void* CudaBackend::alloc(std::size_t bytes) {
  return cuda_helpers::device_alloc(bytes);
}

void CudaBackend::free(void* ptr) {
  cuda_helpers::device_free(ptr);
}

void CudaBackend::synchronize() {
  cuda_helpers::synchronize_device();
}

} // namespace mpmstd::parallel
