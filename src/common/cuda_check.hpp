#pragma once
// CUDA 호출 검사 — 조용한 실패 금지 (D10). GPU 빌드 전용 헤더
// (cuda_runtime.h 를 끌어오므로 .cu / GPU 전용 TU 에서만 include 한다).

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace mpmstd {

inline void cuda_check(const cudaError_t status, const char* what) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA failed: ") + what + " -> "
                             + cudaGetErrorString(status));
}

}  // namespace mpmstd

#define MPMSTD_CUDA_CHECK(expr) ::mpmstd::cuda_check((expr), #expr)

// 커널 런치 직후의 비동기 에러 수거용 (블로킹 없음).
#define MPMSTD_CUDA_CHECK_LAUNCH() ::mpmstd::cuda_check(cudaGetLastError(), "kernel launch")
