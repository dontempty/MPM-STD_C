#pragma once

#include "parallel/backend/backend.hpp"
#include "parallel/backend/cpu_backend.hpp"
#include "parallel/backend/cuda_backend.hpp"

#include <memory>

namespace mpmstd::parallel {

// Factory: returns the backend that matches the current build.
// CPU build : CpuBackend
// CUDA build: CudaBackend
inline std::unique_ptr<Backend> make_default_backend() {
#ifdef MPMSTD_BACKEND_CUDA
  return std::make_unique<CudaBackend>();
#else
  return std::make_unique<CpuBackend>();
#endif
}

} // namespace mpmstd::parallel
