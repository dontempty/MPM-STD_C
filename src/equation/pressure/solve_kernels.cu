#include "equation/pressure/solve_kernels.hpp"

#include <cstddef>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// GPU 백엔드 — solve_kernels.cpp 의 쌍둥이.

namespace mpmstd::equation::pressure {

namespace {

enum : int { kThreads = 256 };

inline int blocks_for(const long long n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

__global__ void pack_rhs_kernel(real_t* lines, const field::ConstView r, const int h,
                                const int n_x, const int n_y, const int nz_local) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_x) * n_y * nz_local) return;
  const int ii = static_cast<int>(idx % n_x);
  const int kk = static_cast<int>((idx / n_x) % nz_local);
  const int jj = static_cast<int>(idx / (static_cast<long long>(n_x) * nz_local));
  lines[(static_cast<std::size_t>(jj) * nz_local + kk) * n_x + ii]
      = r(ii + h, jj + h, kk + h);
}

__global__ void reorder_x_to_y_kernel(real_t* sxy, const real_t* sx, const int n_y,
                                      const int nz_local, const int n_half_x) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_y) * nz_local * n_half_x) return;
  const int mx = static_cast<int>(idx % n_half_x);
  const int kk = static_cast<int>((idx / n_half_x) % nz_local);
  const int jy = static_cast<int>(idx / (static_cast<long long>(n_half_x) * nz_local));
  const std::size_t dst = 2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
  const std::size_t src = 2 * ((static_cast<std::size_t>(jy) * nz_local + kk) * n_half_x + mx);
  sxy[dst] = sx[src];
  sxy[dst + 1] = sx[src + 1];
}

__global__ void reorder_y_to_x_kernel(real_t* sx, const real_t* sxy, const int n_y,
                                      const int nz_local, const int n_half_x) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_y) * nz_local * n_half_x) return;
  const int mx = static_cast<int>(idx % n_half_x);
  const int kk = static_cast<int>((idx / n_half_x) % nz_local);
  const int jy = static_cast<int>(idx / (static_cast<long long>(n_half_x) * nz_local));
  const std::size_t dst = 2 * ((static_cast<std::size_t>(jy) * nz_local + kk) * n_half_x + mx);
  const std::size_t src = 2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
  sx[dst] = sxy[src];
  sx[dst + 1] = sxy[src + 1];
}

__global__ void pack_bands_kernel(real_t* D, const real_t* sxy, const int nz_local,
                                  const int n_half_x, const int n_y,
                                  const bool pin_mean_row) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(nz_local) * n_half_x * n_y) return;
  const int jy = static_cast<int>(idx % n_y);
  const int mx = static_cast<int>((idx / n_y) % n_half_x);
  const int kk = static_cast<int>(idx / (static_cast<long long>(n_y) * n_half_x));
  const int nsys = 2 * n_half_x * n_y;
  const std::size_t q = 2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
  const std::size_t p = static_cast<std::size_t>(kk) * nsys
                        + 2 * (static_cast<std::size_t>(mx) * n_y + jy);
  const bool pinned = pin_mean_row && kk == 0 && mx == 0 && jy == 0;
  D[p] = pinned ? 0 : sxy[q];
  D[p + 1] = pinned ? 0 : sxy[q + 1];
}

__global__ void unpack_bands_kernel(real_t* sxy, const real_t* D, const int nz_local,
                                    const int n_half_x, const int n_y) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(nz_local) * n_half_x * n_y) return;
  const int jy = static_cast<int>(idx % n_y);
  const int mx = static_cast<int>((idx / n_y) % n_half_x);
  const int kk = static_cast<int>(idx / (static_cast<long long>(n_y) * n_half_x));
  const int nsys = 2 * n_half_x * n_y;
  const std::size_t q = 2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
  const std::size_t p = static_cast<std::size_t>(kk) * nsys
                        + 2 * (static_cast<std::size_t>(mx) * n_y + jy);
  sxy[q] = D[p];
  sxy[q + 1] = D[p + 1];
}

__global__ void scatter_lines_kernel(const field::View p, const real_t* lines,
                                     const real_t scale, const int h, const int n_x,
                                     const int n_y, const int nz_local) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(n_x) * n_y * nz_local) return;
  const int ii = static_cast<int>(idx % n_x);
  const int kk = static_cast<int>((idx / n_x) % nz_local);
  const int jj = static_cast<int>(idx / (static_cast<long long>(n_x) * nz_local));
  p(ii + h, jj + h, kk + h) =
      lines[(static_cast<std::size_t>(jj) * nz_local + kk) * n_x + ii] * scale;
}

__global__ void interior_sum_kernel(const field::ConstView p, const int h,
                                    const int i1, const int i2, const int i3,
                                    real_t* total) {
  __shared__ real_t partial[kThreads];
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  real_t mine = 0;
  if (idx < static_cast<long long>(i1) * i2 * i3) {
    const int k = h + static_cast<int>(idx % i3);
    const int j = h + static_cast<int>((idx / i3) % i2);
    const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
    mine = p(i, j, k);
  }
  partial[threadIdx.x] = mine;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] += partial[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(total, partial[0]);
}

__global__ void subtract_kernel(const field::View p, const int h, const int i1,
                                const int i2, const int i3, const real_t value) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  p(i, j, k) -= value;
}



}  // namespace

void pack_rhs_to_lines(real_t* lines, const field::Field& rhs,
                       const parallel::Subdomain& sub, const int n_x, const int n_y,
                       const int nz_local) {
  const long long n = static_cast<long long>(n_x) * n_y * nz_local;
  pack_rhs_kernel<<<blocks_for(n), kThreads>>>(lines, rhs.view(), halo_width, n_x,
                                               n_y, nz_local);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void reorder_x_to_y(real_t* sxy, const real_t* sx, const int n_y, const int nz_local,
                    const int n_half_x) {
  const long long n = static_cast<long long>(n_y) * nz_local * n_half_x;
  reorder_x_to_y_kernel<<<blocks_for(n), kThreads>>>(sxy, sx, n_y, nz_local, n_half_x);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void reorder_y_to_x(real_t* sx, const real_t* sxy, const int n_y, const int nz_local,
                    const int n_half_x) {
  const long long n = static_cast<long long>(n_y) * nz_local * n_half_x;
  reorder_y_to_x_kernel<<<blocks_for(n), kThreads>>>(sx, sxy, n_y, nz_local, n_half_x);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void pack_spectrum_to_bands(real_t* D, const real_t* sxy, const int nz_local,
                            const int n_half_x, const int n_y,
                            const bool pin_mean_row) {
  const long long n = static_cast<long long>(nz_local) * n_half_x * n_y;
  pack_bands_kernel<<<blocks_for(n), kThreads>>>(D, sxy, nz_local, n_half_x, n_y,
                                                 pin_mean_row);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void unpack_bands_to_spectrum(real_t* sxy, const real_t* D, const int nz_local,
                              const int n_half_x, const int n_y) {
  const long long n = static_cast<long long>(nz_local) * n_half_x * n_y;
  unpack_bands_kernel<<<blocks_for(n), kThreads>>>(sxy, D, nz_local, n_half_x, n_y);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void scatter_lines_to_field(field::Field& phi, const real_t* lines, const real_t scale,
                            const parallel::Subdomain& sub, const int n_x,
                            const int n_y, const int nz_local) {
  const long long n = static_cast<long long>(n_x) * n_y * nz_local;
  scatter_lines_kernel<<<blocks_for(n), kThreads>>>(phi.view(), lines, scale,
                                                    halo_width, n_x, n_y, nz_local);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

real_t interior_sum(const field::Field& phi, const parallel::Subdomain& sub) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  real_t* total = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&total), sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(total, 0, sizeof(real_t)));
  interior_sum_kernel<<<blocks_for(n), kThreads>>>(phi.view(), halo_width, i1, i2, i3,
                                                   total);
  MPMSTD_CUDA_CHECK_LAUNCH();
  real_t host = 0;
  MPMSTD_CUDA_CHECK(cudaMemcpy(&host, total, sizeof(real_t), cudaMemcpyDeviceToHost));
  cudaFree(total);
  return host;
}

void subtract_constant(field::Field& phi, const parallel::Subdomain& sub,
                       const real_t value) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  subtract_kernel<<<blocks_for(n), kThreads>>>(phi.view(), halo_width, i1, i2, i3,
                                               value);
  MPMSTD_CUDA_CHECK_LAUNCH();
}






}  // namespace mpmstd::equation::pressure
