#include "parallel/cuda/cuda_memory.hpp"
#include "common/macros.hpp"

#include <cstdlib>
#include <cstring>

#ifdef MPMSTD_BACKEND_CUDA
  #include <cuda_runtime.h>
#endif

namespace mpmstd::parallel::cuda_helpers {

void* device_alloc(std::size_t bytes) {
#ifdef MPMSTD_BACKEND_CUDA
  void* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, bytes));
  return p;
#else
  // CPU build: fall back to aligned host allocation so that callers can treat
  // "device" pointers uniformly with host pointers.
  void* p = nullptr;
  if (posix_memalign(&p, 64, bytes) != 0) return nullptr;
  std::memset(p, 0, bytes);
  return p;
#endif
}

void device_free(void* ptr) {
  if (!ptr) return;
#ifdef MPMSTD_BACKEND_CUDA
  CUDA_CHECK(cudaFree(ptr));
#else
  std::free(ptr);
#endif
}

void copy_host_to_device(void* device_dst, const void* host_src, std::size_t bytes) {
#ifdef MPMSTD_BACKEND_CUDA
  CUDA_CHECK(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice));
#else
  std::memcpy(device_dst, host_src, bytes);
#endif
}

void copy_device_to_host(void* host_dst, const void* device_src, std::size_t bytes) {
#ifdef MPMSTD_BACKEND_CUDA
  CUDA_CHECK(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost));
#else
  std::memcpy(host_dst, device_src, bytes);
#endif
}

void copy_device_to_device(void* device_dst, const void* device_src, std::size_t bytes) {
#ifdef MPMSTD_BACKEND_CUDA
  CUDA_CHECK(cudaMemcpy(device_dst, device_src, bytes, cudaMemcpyDeviceToDevice));
#else
  std::memcpy(device_dst, device_src, bytes);
#endif
}

void copy_host_to_device_async(void* device_dst, const void* host_src, std::size_t bytes, void* stream) {
#ifdef MPMSTD_BACKEND_CUDA
  CUDA_CHECK(cudaMemcpyAsync(device_dst, host_src, bytes,
                              cudaMemcpyHostToDevice,
                              reinterpret_cast<cudaStream_t>(stream)));
#else
  MPMSTD_UNUSED(stream);
  std::memcpy(device_dst, host_src, bytes);
#endif
}

void copy_device_to_host_async(void* host_dst, const void* device_src, std::size_t bytes, void* stream) {
#ifdef MPMSTD_BACKEND_CUDA
  CUDA_CHECK(cudaMemcpyAsync(host_dst, device_src, bytes,
                              cudaMemcpyDeviceToHost,
                              reinterpret_cast<cudaStream_t>(stream)));
#else
  MPMSTD_UNUSED(stream);
  std::memcpy(host_dst, device_src, bytes);
#endif
}

} // namespace mpmstd::parallel::cuda_helpers
