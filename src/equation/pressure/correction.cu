#include "equation/pressure/correction.hpp"

#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// GPU 백엔드 — correction.cpp 의 쌍둥이.

namespace mpmstd::equation::pressure {

namespace {

enum : int { kThreads = 256 };
inline int blocks_for(const long long n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
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

void correct_velocity_constant_property(field::Field& U, field::Field& V, field::Field& W,
                      const field::Field& phi, const grid::Grid& grid,
                      const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("correct_velocity_constant_property", U, sub);
  check_field_matches("correct_velocity_constant_property", V, sub);
  check_field_matches("correct_velocity_constant_property", W, sub);
  check_field_matches("correct_velocity_constant_property", phi, sub);
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

void correct_velocity_variable_property(field::Field& U, field::Field& V, field::Field& W,
                               const field::Field& phi,
                               const field::Field& pressure_predictor,
                               const field::Field& inverse_density,
                               const grid::Grid& grid, const parallel::Subdomain& sub,
                               const real_t dt) {
  check_field_matches("correct_velocity_variable_property", U, sub);
  check_field_matches("correct_velocity_variable_property", phi, sub);
  check_field_matches("correct_velocity_variable_property", pressure_predictor, sub);
  check_field_matches("correct_velocity_variable_property", inverse_density, sub);
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
