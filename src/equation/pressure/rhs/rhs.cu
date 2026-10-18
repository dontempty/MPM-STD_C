#include "equation/pressure/rhs/rhs.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// GPU 백엔드 — rhs.cpp 의 쌍둥이. 수식은 pressure_stencil.hpp.

namespace mpmstd::equation::pressure {

namespace {

enum : int { kThreads = 256 };

__global__ void divergence_kernel(const field::View r, const field::ConstView u,
                                  const field::ConstView v, const field::ConstView w,
                                  const real_t* dx, const real_t* dy, const real_t* dz,
                                  const int h, const int i1, const int i2,
                                  const int i3, const real_t inv_dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  r(i, j, k) += inv_dt * stencil::divergence(u, v, w, i, j, k, dx[i], dy[j], dz[k]);
}

__global__ void gradient_kernel(const field::View ru, const field::View rv,
                                const field::View rw, const field::ConstView p,
                                const real_t* dmx, const real_t* dmy,
                                const real_t* dmz, const int h, const int i1,
                                const int i2, const int i3, const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  ru(i, j, k) -= dt * stencil::gradient_x(p, i, j, k, dmx[i]);
  rv(i, j, k) -= dt * stencil::gradient_y(p, i, j, k, dmy[j]);
  rw(i, j, k) -= dt * stencil::gradient_z(p, i, j, k, dmz[k]);
}

inline int blocks_for(const parallel::Subdomain& sub) {
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

// [최적화 2026-10-18] 전용 Poisson 우변 빌드 커널 — divergence_kernel + add_extra_term_kernel 의 식을 한 스레드에서
//   (0 + div) + extra 순서로(패스 10 → 6). 셀당 나눗셈 18회는 격자 역수(idx·idmx) 곱으로 → 허용오차 게이트.
//   (패스 융합만으로는 2.92 → 2.33 ms 뿐 — 커널이 대역폭이 아니라 FP64 나눗셈에 묶여 있었다.)
__global__ void build_rhs_kernel(real_t* __restrict__ rhs, const real_t* __restrict__ u,
                                 const real_t* __restrict__ v, const real_t* __restrict__ w,
                                 const real_t* __restrict__ ph, const real_t* __restrict__ ir,
                                 const real_t* __restrict__ dx1, const real_t* __restrict__ idx1,
                                 const real_t* __restrict__ idmx1,
                                 const real_t* __restrict__ dx2, const real_t* __restrict__ idx2,
                                 const real_t* __restrict__ idmx2,
                                 const real_t* __restrict__ dx3, const real_t* __restrict__ idx3,
                                 const real_t* __restrict__ idmx3,
                                 const int h, const int ni, const int nj, const int nk,
                                 const int ny_total, const int nz_total, const real_t inv_dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * nj * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j = h + static_cast<int>((idx / nk) % nj);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * nj));
  const auto at = [ny_total, nz_total](int a, int b, int c) {
    return (static_cast<long long>(a) * ny_total + b) * nz_total + c;
  };
  const long long c = at(i, j, k);
  const long long cxm = at(i - 1, j, k), cxp = at(i + 1, j, k);
  const long long cym = at(i, j - 1, k), cyp = at(i, j + 1, k);
  const long long czm = at(i, j, k - 1), czp = at(i, j, k + 1);
  // ∇·u* (divergence 와 같은 식, 나눗셈 → 역수 곱)
  real_t r = inv_dt * ((u[cxp] - u[c]) * idx1[i] + (v[cyp] - v[c]) * idx2[j] + (w[czp] - w[c]) * idx3[k]);
  const real_t ir_xlo = real_t(0.5) * idmx1[i] * (dx1[i - 1] * ir[c] + dx1[i] * ir[cxm]);
  const real_t ir_xhi = real_t(0.5) * idmx1[i + 1] * (dx1[i + 1] * ir[c] + dx1[i] * ir[cxp]);
  const real_t ir_ylo = real_t(0.5) * idmx2[j] * (dx2[j - 1] * ir[c] + dx2[j] * ir[cym]);
  const real_t ir_yhi = real_t(0.5) * idmx2[j + 1] * (dx2[j + 1] * ir[c] + dx2[j] * ir[cyp]);
  const real_t ir_zlo = real_t(0.5) * idmx3[k] * (dx3[k - 1] * ir[c] + dx3[k] * ir[czm]);
  const real_t ir_zhi = real_t(0.5) * idmx3[k + 1] * (dx3[k + 1] * ir[c] + dx3[k] * ir[czp]);
  const real_t gx_lo = (ph[c] - ph[cxm]) * idmx1[i];
  const real_t gx_hi = (ph[cxp] - ph[c]) * idmx1[i + 1];
  const real_t gy_lo = (ph[c] - ph[cym]) * idmx2[j];
  const real_t gy_hi = (ph[cyp] - ph[c]) * idmx2[j + 1];
  const real_t gz_lo = (ph[c] - ph[czm]) * idmx3[k];
  const real_t gz_hi = (ph[czp] - ph[c]) * idmx3[k + 1];
  r += ((1 - ir_xhi) * gx_hi - (1 - ir_xlo) * gx_lo) * idx1[i]
       + ((1 - ir_yhi) * gy_hi - (1 - ir_ylo) * gy_lo) * idx2[j]
       + ((1 - ir_zhi) * gz_hi - (1 - ir_zlo) * gz_lo) * idx3[k];
  rhs[c] = r;
}

}  // namespace

void build_variable_property_rhs(field::Field& rhs, const field::Field& U, const field::Field& V,
                                 const field::Field& W, const field::Field& pressure_predictor,
                                 const field::Field& inverse_density, const grid::Grid& grid,
                                 const parallel::Subdomain& sub, const real_t dt) {
  const char* who = "pressure::build_variable_property_rhs";
  const field::Field* fields[] = {&rhs, &U, &V, &W, &pressure_predictor, &inverse_density};
  for (const field::Field* f : fields) check_field_matches(who, *f, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  build_rhs_kernel<<<static_cast<int>((n + kThreads - 1) / kThreads), kThreads>>>(
      rhs.data(), U.data(), V.data(), W.data(), pressure_predictor.data(),
      inverse_density.data(),
      grid.x.device_cell_width.data(), grid.x.device_inverse_cell_width.data(), grid.x.device_inverse_center_distance.data(),
      grid.y.device_cell_width.data(), grid.y.device_inverse_cell_width.data(), grid.y.device_inverse_center_distance.data(),
      grid.z.device_cell_width.data(), grid.z.device_inverse_cell_width.data(), grid.z.device_inverse_center_distance.data(),
      halo_width, ni, nj, nk, sub.y().n_total, sub.z().n_total, real_t(1) / dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void add_divergence(field::Field& rhs, const field::Field& U, const field::Field& V,
                    const field::Field& W, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("add_divergence", rhs, sub);
  check_field_matches("add_divergence", U, sub);
  check_field_matches("add_divergence", V, sub);
  check_field_matches("add_divergence", W, sub);
  divergence_kernel<<<blocks_for(sub), kThreads>>>(
      rhs.view(), U.view(), V.view(), W.view(), grid.x.device_cell_width.data(),
      grid.y.device_cell_width.data(), grid.z.device_cell_width.data(), halo_width,
      sub.x().n_local, sub.y().n_local, sub.z().n_local, 1 / dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void add_gradient_constant_property(field::Field& rhs_u, field::Field& rhs_v, field::Field& rhs_w,
                  const field::Field& P, const grid::Grid& grid,
                  const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("add_gradient_constant_property", rhs_u, sub);
  check_field_matches("add_gradient_constant_property", rhs_v, sub);
  check_field_matches("add_gradient_constant_property", rhs_w, sub);
  check_field_matches("add_gradient_constant_property", P, sub);
  gradient_kernel<<<blocks_for(sub), kThreads>>>(
      rhs_u.view(), rhs_v.view(), rhs_w.view(), P.view(),
      grid.x.device_center_distance.data(), grid.y.device_center_distance.data(),
      grid.z.device_center_distance.data(), halo_width, sub.x().n_local,
      sub.y().n_local, sub.z().n_local, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

namespace {

__global__ void add_extra_term_kernel(real_t* rhs, const real_t* ph, const real_t* ir,
                                      const real_t* dx1, const real_t* dmx1,
                                      const real_t* dx2, const real_t* dmx2,
                                      const real_t* dx3, const real_t* dmx3,
                                      const int h, const int ni, const int nj,
                                      const int nk, const int ny_total,
                                      const int nz_total) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * nj * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j = h + static_cast<int>((idx / nk) % nj);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * nj));
  const auto at = [ny_total, nz_total](int a, int b, int c) {
    return (static_cast<long long>(a) * ny_total + b) * nz_total + c;
  };
  const long long c = at(i, j, k);
  const long long cxm = at(i - 1, j, k), cxp = at(i + 1, j, k);
  const long long cym = at(i, j - 1, k), cyp = at(i, j + 1, k);
  const long long czm = at(i, j, k - 1), czp = at(i, j, k + 1);
  const real_t ir_xlo = real_t(0.5) / dmx1[i] * (dx1[i - 1] * ir[c] + dx1[i] * ir[cxm]);
  const real_t ir_xhi = real_t(0.5) / dmx1[i + 1] * (dx1[i + 1] * ir[c] + dx1[i] * ir[cxp]);
  const real_t ir_ylo = real_t(0.5) / dmx2[j] * (dx2[j - 1] * ir[c] + dx2[j] * ir[cym]);
  const real_t ir_yhi = real_t(0.5) / dmx2[j + 1] * (dx2[j + 1] * ir[c] + dx2[j] * ir[cyp]);
  const real_t ir_zlo = real_t(0.5) / dmx3[k] * (dx3[k - 1] * ir[c] + dx3[k] * ir[czm]);
  const real_t ir_zhi = real_t(0.5) / dmx3[k + 1] * (dx3[k + 1] * ir[c] + dx3[k] * ir[czp]);
  const real_t gx_lo = (ph[c] - ph[cxm]) / dmx1[i];
  const real_t gx_hi = (ph[cxp] - ph[c]) / dmx1[i + 1];
  const real_t gy_lo = (ph[c] - ph[cym]) / dmx2[j];
  const real_t gy_hi = (ph[cyp] - ph[c]) / dmx2[j + 1];
  const real_t gz_lo = (ph[c] - ph[czm]) / dmx3[k];
  const real_t gz_hi = (ph[czp] - ph[c]) / dmx3[k + 1];
  rhs[c] += ((1 - ir_xhi) * gx_hi - (1 - ir_xlo) * gx_lo) / dx1[i]
            + ((1 - ir_yhi) * gy_hi - (1 - ir_ylo) * gy_lo) / dx2[j]
            + ((1 - ir_zhi) * gz_hi - (1 - ir_zlo) * gz_lo) / dx3[k];
}

}  // namespace

void add_extra_term_variable_property(field::Field& rhs_field,
                             const field::Field& pressure_predictor,
                             const field::Field& inverse_density,
                             const grid::Grid& grid, const parallel::Subdomain& sub) {
  check_field_matches("add_extra_term_variable_property", rhs_field, sub);
  check_field_matches("add_extra_term_variable_property", pressure_predictor, sub);
  check_field_matches("add_extra_term_variable_property", inverse_density, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_extra_term_kernel<<<blocks, threads>>>(
      rhs_field.data(), pressure_predictor.data(), inverse_density.data(),
      grid.x.device_cell_width.data(), grid.x.device_center_distance.data(),
      grid.y.device_cell_width.data(), grid.y.device_center_distance.data(),
      grid.z.device_cell_width.data(), grid.z.device_center_distance.data(),
      halo_width, ni, nj, nk, sub.y().n_total, sub.z().n_total);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

namespace {

__global__ void add_gradient_variable_kernel(
    real_t* du, real_t* dv, real_t* dw, const real_t* p, const real_t* ir,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t* dx3, const real_t* dmx3, const int h, const int ni, const int nj,
    const int nk, const int ny_total, const int nz_total, const real_t dt) {
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
  const real_t irx = real_t(0.5) * (dx1[i - 1] * ir[c] + dx1[i] * ir[cxm]) / dmx1[i];
  const real_t iry = real_t(0.5) * (dx2[j - 1] * ir[c] + dx2[j] * ir[cym]) / dmx2[j];
  const real_t irz = real_t(0.5) * (dx3[k - 1] * ir[c] + dx3[k] * ir[czm]) / dmx3[k];
  du[c] -= dt * irx * (p[c] - p[cxm]) / dmx1[i];
  dv[c] -= dt * iry * (p[c] - p[cym]) / dmx2[j];
  dw[c] -= dt * irz * (p[c] - p[czm]) / dmx3[k];
}

}  // namespace

void add_gradient_variable_property(field::Field& rhs_u, field::Field& rhs_v,
                           field::Field& rhs_w, const field::Field& P,
                           const field::Field& inverse_density,
                           const grid::Grid& grid, const parallel::Subdomain& sub,
                           const real_t dt) {
  check_field_matches("add_gradient_variable_property", rhs_u, sub);
  check_field_matches("add_gradient_variable_property", P, sub);
  check_field_matches("add_gradient_variable_property", inverse_density, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_gradient_variable_kernel<<<blocks, threads>>>(
      rhs_u.data(), rhs_v.data(), rhs_w.data(), P.data(), inverse_density.data(),
      grid.x.device_cell_width.data(), grid.x.device_center_distance.data(),
      grid.y.device_cell_width.data(), grid.y.device_center_distance.data(),
      grid.z.device_cell_width.data(), grid.z.device_center_distance.data(),
      halo_width, ni, nj, nk, sub.y().n_total, sub.z().n_total, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation::pressure
