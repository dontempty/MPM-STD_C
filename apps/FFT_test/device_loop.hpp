#pragma once
// 원소별 루프 하나를 CPU/GPU 양쪽에 — 앱 폴더의 .cu 는 Makefile 이 집지 않으므로(plan.md §4) 커널을 .cpp 안에 둔다.
// GPU 빌드(nvc++ -cuda)는 모든 TU 를 CUDA 로 컴파일하므로 템플릿 __global__ 이 그대로 된다.
//
//   struct Scale { real_t* p; real_t s; MPMSTD_HD void operator()(long long i) const { p[i] *= s; } };
//   fft_test::for_each(n, Scale{ptr, 2.0});
//
// 함수객체는 POD(포인터·정수·실수만)여야 한다 — 값 복사로 커널 인자가 된다.
// GPU 런치는 기본 스트림에서 비동기다. MPI·D2H 전에 동기화는 호출자(교환 함수)가 한다.

#include <cstdio>

#include "common/hd.hpp"
#include "common/types.hpp"

#ifdef MPMSTD_GPU
#include "common/cuda_check.hpp"
#endif

namespace fft_test {

using mpmstd::real_t;

#ifdef MPMSTD_GPU

template <class Function>
__global__ void for_each_kernel(const long long n, const Function function) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i < n) function(i);
}

template <class Function>
void for_each(const long long n, const Function& function) {
  if (n <= 0) return;
  constexpr int threads = 256;
  const long long blocks = (n + threads - 1) / threads;
  for_each_kernel<<<static_cast<unsigned int>(blocks), threads>>>(n, function);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

inline void synchronize_device() { MPMSTD_CUDA_CHECK(cudaDeviceSynchronize()); }

#else

template <class Function>
void for_each(const long long n, const Function& function) {
  for (long long i = 0; i < n; ++i) function(i);
}

inline void synchronize_device() {}

#endif

}  // namespace fft_test
