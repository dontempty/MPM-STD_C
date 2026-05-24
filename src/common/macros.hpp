#pragma once

// =============================================================================
// host/device function qualifier
// CUDA build : __host__ __device__ inline
// CPU build  : inline
// =============================================================================
#ifdef MPMSTD_BACKEND_CUDA
  #define MPMSTD_HD     __host__ __device__ inline
  #define MPMSTD_DEVICE __device__ inline
  #define MPMSTD_HOST   __host__ inline
#else
  #define MPMSTD_HD     inline
  #define MPMSTD_DEVICE inline
  #define MPMSTD_HOST   inline
#endif

// =============================================================================
// restrict qualifier (both compilers support __restrict__)
// =============================================================================
#define MPMSTD_RESTRICT __restrict__

// =============================================================================
// CUDA error check macro
// CPU build : no-op
// =============================================================================
#ifdef MPMSTD_BACKEND_CUDA
  #include <cuda_runtime.h>
  #include <cstdio>
  #include <cstdlib>
  #define CUDA_CHECK(call)                                                     \
    do {                                                                       \
      cudaError_t _err = (call);                                               \
      if (_err != cudaSuccess) {                                               \
        std::fprintf(stderr, "[CUDA] error at %s:%d : %s\n",                   \
                     __FILE__, __LINE__, cudaGetErrorString(_err));            \
        std::abort();                                                          \
      }                                                                        \
    } while (0)
#else
  #define CUDA_CHECK(call) ((void)0)
#endif

// =============================================================================
// silence unused parameter warnings (often used in CPU build stubs)
// =============================================================================
#define MPMSTD_UNUSED(x) ((void)(x))

// =============================================================================
// compiler hints
// =============================================================================
#if defined(__GNUC__) || defined(__clang__) || defined(__NVCOMPILER)
  #define MPMSTD_LIKELY(x)   __builtin_expect(!!(x), 1)
  #define MPMSTD_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define MPMSTD_LIKELY(x)   (x)
  #define MPMSTD_UNLIKELY(x) (x)
#endif

// =============================================================================
// halo width (project-wide constant)
// =============================================================================
namespace mpmstd {
inline constexpr int kHaloWidth = 1;
} // namespace mpmstd
