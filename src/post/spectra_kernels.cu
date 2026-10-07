#include "post/spectra_kernels.hpp"

#include "common/cuda_check.hpp"

// GPU 백엔드 — spectra_kernels.cpp 의 쌍둥이.

namespace mpmstd::post {

namespace {

__global__ void power_kernel(real_t* power, const real_t* spectrum, const int n_lines,
                             const int n_mode, const int n, const real_t scale) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_lines) * n_mode) return;
  const int m = static_cast<int>(idx % n_mode);
  const real_t re = spectrum[2 * idx], im = spectrum[2 * idx + 1];
  const real_t one_sided = (m == 0 || (n % 2 == 0 && m == n / 2)) ? real_t(1) : real_t(2);
  power[idx] += scale * one_sided * (re * re + im * im);
}

}  // namespace

void accumulate_line_power(real_t* power, const real_t* spectrum, const int n_lines,
                           const int n_mode, const int n, const real_t scale) {
  const long long total = static_cast<long long>(n_lines) * n_mode;
  const int threads = 256;
  power_kernel<<<static_cast<int>((total + threads - 1) / threads), threads>>>(
      power, spectrum, n_lines, n_mode, n, scale);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void zero_buffer(real_t* data, const std::size_t n) {
  MPMSTD_CUDA_CHECK(cudaMemset(data, 0, n * sizeof(real_t)));
}

}  // namespace mpmstd::post
