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

__global__ void correct_kernel(const field::View u, const field::View v,
                               const field::View w, const field::ConstView p,
                               const real_t* dmx, const real_t* dmy,
                               const real_t* dmz, const int h, const int i1,
                               const int i2, const int i3, const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  u(i, j, k) -= dt * stencil::gradient_x(p, i, j, k, dmx[i]);
  v(i, j, k) -= dt * stencil::gradient_y(p, i, j, k, dmy[j]);
  w(i, j, k) -= dt * stencil::gradient_z(p, i, j, k, dmz[k]);
}

__global__ void accumulate_kernel(real_t* p, const real_t* d, const long long n) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q < n) p[q] += d[q];
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

void correct_velocity(field::Field& U, field::Field& V, field::Field& W,
                      const field::Field& phi, const grid::Grid& grid,
                      const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("correct_velocity", U, sub);
  check_field_matches("correct_velocity", V, sub);
  check_field_matches("correct_velocity", W, sub);
  check_field_matches("correct_velocity", phi, sub);
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  correct_kernel<<<blocks_for(n), kThreads>>>(
      U.view(), V.view(), W.view(), phi.view(), grid.x.device_center_distance.data(),
      grid.y.device_center_distance.data(), grid.z.device_center_distance.data(),
      halo_width, i1, i2, i3, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void accumulate(field::Field& P, const field::Field& phi) {
  if (P.size() != phi.size())
    throw std::invalid_argument("pressure::accumulate: P and phi differ in shape");
  const long long n = static_cast<long long>(P.size());
  accumulate_kernel<<<blocks_for(n), kThreads>>>(P.data(), phi.data(), n);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

namespace {

__global__ void correct_velocity_variable_kernel(
    real_t* u, real_t* v, real_t* w, const real_t* p, const real_t* ph,
    const real_t* ir, const real_t* dx1, const real_t* dmx1, const real_t* dx2,
    const real_t* dmx2, const real_t* dx3, const real_t* dmx3, const int h,
    const int ni, const int nj, const int nk, const int ny_total, const int nz_total,
    const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * nj * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j = h + static_cast<int>((idx / nk) % nj);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * nj));
  const auto at = [ny_total, nz_total](int a, int b, int c) {
    return (static_cast<long long>(a) * ny_total + b) * nz_total + c;
  };
  const long long c = at(i, j, k);
  const long long cxm = at(i - 1, j, k);
  const long long cym = at(i, j - 1, k);
  const long long czm = at(i, j, k - 1);
  const real_t irc_x = real_t(0.5) / dmx1[i] * (dx1[i - 1] * ir[c] + dx1[i] * ir[cxm]);
  const real_t irc_y = real_t(0.5) / dmx2[j] * (dx2[j - 1] * ir[c] + dx2[j] * ir[cym]);
  const real_t irc_z = real_t(0.5) / dmx3[k] * (dx3[k - 1] * ir[c] + dx3[k] * ir[czm]);
  u[c] -= dt * ((p[c] - p[cxm]) / dmx1[i] + (irc_x - 1) * (ph[c] - ph[cxm]) / dmx1[i]);
  v[c] -= dt * ((p[c] - p[cym]) / dmx2[j] + (irc_y - 1) * (ph[c] - ph[cym]) / dmx2[j]);
  w[c] -= dt * ((p[c] - p[czm]) / dmx3[k] + (irc_z - 1) * (ph[c] - ph[czm]) / dmx3[k]);
}

__global__ void advance_predictor_kernel(real_t* ph, real_t* p0, const real_t* dp,
                                         const long long n) {
  const long long q = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (q >= n) return;
  const real_t new_previous = real_t(0.5) * (ph[q] + p0[q]);
  p0[q] = new_previous;
  ph[q] = 2 * dp[q] - new_previous;
}

}  // namespace

void correct_velocity_variable(field::Field& U, field::Field& V, field::Field& W,
                               const field::Field& phi,
                               const field::Field& pressure_predictor,
                               const field::Field& inverse_density,
                               const grid::Grid& grid, const parallel::Subdomain& sub,
                               const real_t dt) {
  check_field_matches("correct_velocity_variable", U, sub);
  check_field_matches("correct_velocity_variable", phi, sub);
  check_field_matches("correct_velocity_variable", pressure_predictor, sub);
  check_field_matches("correct_velocity_variable", inverse_density, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  correct_velocity_variable_kernel<<<blocks, threads>>>(
      U.data(), V.data(), W.data(), phi.data(), pressure_predictor.data(),
      inverse_density.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, ni, nj, nk,
      sub.y().n_total, sub.z().n_total, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void advance_pressure_predictor(field::Field& pressure_predictor,
                                field::Field& previous_predictor,
                                const field::Field& phi) {
  if (pressure_predictor.size() != phi.size()
      || previous_predictor.size() != phi.size())
    throw std::invalid_argument(
        "advance_pressure_predictor: fields differ in shape");
  const long long n = static_cast<long long>(phi.size());
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  advance_predictor_kernel<<<blocks, threads>>>(pressure_predictor.data(),
                                                previous_predictor.data(), phi.data(),
                                                n);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation::pressure
