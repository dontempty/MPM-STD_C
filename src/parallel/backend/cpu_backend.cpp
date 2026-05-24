#include "parallel/backend/cpu_backend.hpp"
#include "parallel/cuda/cuda_memory.hpp"

namespace mpmstd::parallel {

CpuBackend::CpuBackend() : name_("cpu") {}

void* CpuBackend::alloc(std::size_t bytes) {
  // Reuse the cuda_helpers allocator: on CPU build this is just aligned malloc.
  return cuda_helpers::device_alloc(bytes);
}

void CpuBackend::free(void* ptr) {
  cuda_helpers::device_free(ptr);
}

void CpuBackend::synchronize() {
  // CPU build: nothing to do.
}

} // namespace mpmstd::parallel
