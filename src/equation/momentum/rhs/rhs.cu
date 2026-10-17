#include "equation/momentum/rhs/rhs.hpp"

#include "common/cuda_check.hpp"
#include "common/field_checks.hpp"
#include "equation/momentum/momentum_stencil.hpp"

// GPU 백엔드 — rhs.cpp 의 쌍둥이. 수식은 momentum_stencil.hpp.

namespace mpmstd::equation::momentum {

namespace {

// component: 0=U, 1=V, 2=W (성분별 셀 함수 선택만 — 수식은 스텐실 한 곳)
template <class MuArray, class RhoArray>
__global__ void add_kernel(real_t* rhs, const real_t* U, const real_t* V,
                           const real_t* W, const MuArray viscosity,
                           const RhoArray inverse_density, const real_t* dx1,
                           const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
                           const real_t* dx3, const real_t* dmx3, const real_t* idx1,
                           const real_t* idmx1, const real_t* idx2, const real_t* idmx2,
                           const real_t* idx3, const real_t* idmx3, const int h,
                           const int ni, const int nj, const int nk,
                           const int ny_total, const int nz_total, const real_t Cmu,
                           const real_t dt, const int component) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * nj * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j = h + static_cast<int>((idx / nk) % nj);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * nj));
  real_t value;
  if (component == 0)
    value = stencil::rhs_u_cell(i, j, k, ny_total, nz_total, U, V, W,
                                         viscosity, inverse_density, dx1, dmx1, dx2,
                                         dmx2, dx3, dmx3, idx1, idmx1, idx2, idmx2, idx3, idmx3, Cmu, dt);
  else if (component == 1)
    value = stencil::rhs_v_cell(i, j, k, ny_total, nz_total, U, V, W,
                                         viscosity, inverse_density, dx1, dmx1, dx2,
                                         dmx2, dx3, dmx3, idx1, idmx1, idx2, idmx2, idx3, idmx3, Cmu, dt);
  else
    value = stencil::rhs_w_cell(i, j, k, ny_total, nz_total, U, V, W,
                                         viscosity, inverse_density, dx1, dmx1, dx2,
                                         dmx2, dx3, dmx3, idx1, idmx1, idx2, idmx2, idx3, idmx3, Cmu, dt);
  rhs[stencil::vindex(i, j, k, ny_total, nz_total)] += value;
}

void add_impl(const char* who, field::Field& rhs, const field::Field& U,
              const field::Field& V, const field::Field& W,
              const field::Field& viscosity, const field::Field& inverse_density,
              const grid::Grid& grid, const parallel::Subdomain& sub, const real_t Cmu,
              const real_t dt, const int component) {
  check_field_matches(who, rhs, sub);
  check_field_matches(who, U, sub);
  check_field_matches(who, V, sub);
  check_field_matches(who, W, sub);
  check_field_matches(who, viscosity, sub);
  check_field_matches(who, inverse_density, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_kernel<<<blocks, threads>>>(
      rhs.data(), U.data(), V.data(), W.data(), viscosity.data(),
      inverse_density.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), grid.x.device_inverse_cell_width.data(),
      grid.x.device_inverse_center_distance.data(), grid.y.device_inverse_cell_width.data(),
      grid.y.device_inverse_center_distance.data(), grid.z.device_inverse_cell_width.data(),
      grid.z.device_inverse_center_distance.data(), halo_width, ni, nj, nk,
      sub.y().n_total, sub.z().n_total, Cmu, dt, component);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void add_impl_constant(const char* who, field::Field& rhs, const field::Field& U,
                       const field::Field& V, const field::Field& W,
                       const grid::Grid& grid, const parallel::Subdomain& sub,
                       const real_t viscosity, const real_t dt, const int component) {
  check_field_matches(who, rhs, sub);
  check_field_matches(who, U, sub);
  check_field_matches(who, V, sub);
  check_field_matches(who, W, sub);
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  add_kernel<<<blocks, threads>>>(
      rhs.data(), U.data(), V.data(), W.data(), equation::UnitArray{},
      equation::UnitArray{}, grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), grid.x.device_inverse_cell_width.data(),
      grid.x.device_inverse_center_distance.data(), grid.y.device_inverse_cell_width.data(),
      grid.y.device_inverse_center_distance.data(), grid.z.device_inverse_cell_width.data(),
      grid.z.device_inverse_center_distance.data(), halo_width, ni, nj, nk,
      sub.y().n_total, sub.z().n_total, viscosity, dt, component);
  MPMSTD_CUDA_CHECK_LAUNCH();
}


// ── 전용 RHS 빌드 커널 (2026-10-17, 운동량 RHS 최적화 1·2·3) ──────────────────────────────────────
//   3: 성분당 한 패스 — rhs_q = dt·R_q(이송·확산) [+ 부력(W)] − dt·(1/ρ̂)∂P/∂q. 예전 main 의 fill·add·부력·구배 8 패스를 3 패스로.
//      덧셈 순서는 예전과 같다 ((0 + adv) + buoy) − grad 이고 0 + adv == adv 라 그 부분은 비트 동일.
//   2: 스레드 하나가 j 방향 연속 셀 kCoarsenJ 개 — 이웃 스레드는 k 방향(병합 읽기 유지). j±1 이웃 로드는 컴파일러가
//      (__restrict__ + unroll) 레지스터로 넘겨 재사용한다. 셀 수는 같고 스레드 수만 1/kCoarsenJ.
//   1: 격자 간격 나눗셈 → 역수 곱 (momentum_stencil.hpp rhs_*_cell, 부력·구배 식도 같이). 비트 동일 아님 → 허용오차 게이트.
constexpr int kCoarsenJ = 1;   // V100 실측(2026-10-17): K 1/2/4 = 7.02/7.14/7.77 ms — coarsening 은 레지스터(164)만 늘려 손해. 1 = 끔

template <int Component, int K, class MuArray, class RhoArray>
__global__ void build_rhs_kernel(real_t* __restrict__ rhs, const real_t* __restrict__ U,
                                 const real_t* __restrict__ V, const real_t* __restrict__ W,
                                 const real_t* __restrict__ T, const real_t* __restrict__ P,
                                 const MuArray mu, const RhoArray invrho,
                                 const real_t* __restrict__ dx1, const real_t* __restrict__ dmx1,
                                 const real_t* __restrict__ dx2, const real_t* __restrict__ dmx2,
                                 const real_t* __restrict__ dx3, const real_t* __restrict__ dmx3,
                                 const real_t* __restrict__ idx1, const real_t* __restrict__ idmx1,
                                 const real_t* __restrict__ idx2, const real_t* __restrict__ idmx2,
                                 const real_t* __restrict__ idx3, const real_t* __restrict__ idmx3,
                                 const int h, const int ni, const int nj, const int nk,
                                 const int ny_total, const int nz_total, const real_t Cmu,
                                 const real_t density_quadratic_ratio,
                                 const real_t delta_temperature, const real_t dt) {
  const int njg = (nj + K - 1) / K;                      // j 그룹 수
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * njg * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j0 = h + K * static_cast<int>((idx / nk) % njg);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * njg));
#pragma unroll
  for (int s = 0; s < K; ++s) {
    const int j = j0 + s;
    if (j - h >= nj) break;
    const long long c = stencil::vindex(i, j, k, ny_total, nz_total);
    real_t r;
    if (Component == 0) {
      r = stencil::rhs_u_cell(i, j, k, ny_total, nz_total, U, V, W, mu, invrho, dx1, dmx1, dx2,
                              dmx2, dx3, dmx3, idx1, idmx1, idx2, idmx2, idx3, idmx3, Cmu, dt);
      const long long cxm = stencil::vindex(i - 1, j, k, ny_total, nz_total);
      const real_t irx = real_t(0.5) * (dx1[i - 1] * invrho[c] + dx1[i] * invrho[cxm]) * idmx1[i];
      r -= dt * irx * (P[c] - P[cxm]) * idmx1[i];
    } else if (Component == 1) {
      r = stencil::rhs_v_cell(i, j, k, ny_total, nz_total, U, V, W, mu, invrho, dx1, dmx1, dx2,
                              dmx2, dx3, dmx3, idx1, idmx1, idx2, idmx2, idx3, idmx3, Cmu, dt);
      const long long cym = stencil::vindex(i, j - 1, k, ny_total, nz_total);
      const real_t iry = real_t(0.5) * (dx2[j - 1] * invrho[c] + dx2[j] * invrho[cym]) * idmx2[j];
      r -= dt * iry * (P[c] - P[cym]) * idmx2[j];
    } else {
      r = stencil::rhs_w_cell(i, j, k, ny_total, nz_total, U, V, W, mu, invrho, dx1, dmx1, dx2,
                              dmx2, dx3, dmx3, idx1, idmx1, idx2, idmx2, idx3, idmx3, Cmu, dt);
      const long long czm = stencil::vindex(i, j, k - 1, ny_total, nz_total);
      // 부력 (physics/buoyancy.cu buoyancy_variable_kernel 과 같은 식; z 면 체적가중 보간)
      const real_t T_face = real_t(0.5) * (dx3[k - 1] * T[c] + dx3[k] * T[czm]) * idmx3[k];
      const real_t ir_face = real_t(0.5) * (dx3[k - 1] * invrho[c] + dx3[k] * invrho[czm]) * idmx3[k];
      r += dt * ir_face * (T_face + density_quadratic_ratio * T_face * T_face * delta_temperature);
      r -= dt * ir_face * (P[c] - P[czm]) * idmx3[k];   // 구배의 irz 는 ir_face 와 같은 식
    }
    rhs[c] = r;
  }
}

template <int Component>
void launch_build_rhs(real_t* rhs, const field::Field& U, const field::Field& V,
                      const field::Field& W, const field::Field& T, const field::Field& P,
                      const field::Field& viscosity, const field::Field& inverse_density,
                      const grid::Grid& grid, const parallel::Subdomain& sub, const real_t Cmu,
                      const real_t density_quadratic_ratio, const real_t delta_temperature,
                      const real_t dt) {
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const int njg = (nj + kCoarsenJ - 1) / kCoarsenJ;
  const long long n = static_cast<long long>(ni) * njg * nk;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  build_rhs_kernel<Component, kCoarsenJ><<<blocks, threads>>>(
      rhs, U.data(), V.data(), W.data(), T.data(), P.data(), viscosity.data(),
      inverse_density.data(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), grid.x.device_inverse_cell_width.data(),
      grid.x.device_inverse_center_distance.data(), grid.y.device_inverse_cell_width.data(),
      grid.y.device_inverse_center_distance.data(), grid.z.device_inverse_cell_width.data(),
      grid.z.device_inverse_center_distance.data(), halo_width, ni, nj, nk, sub.y().n_total,
      sub.z().n_total, Cmu, density_quadratic_ratio, delta_temperature, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace

void add_advection_diffusion_u_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_impl("add_advection_diffusion_u_variable_property", rhs, U, V, W, viscosity,
           inverse_density, grid, sub, Cmu, dt, 0);
}

void add_advection_diffusion_v_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_impl("add_advection_diffusion_v_variable_property", rhs, U, V, W, viscosity,
           inverse_density, grid, sub, Cmu, dt, 1);
}

void add_advection_diffusion_w_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_impl("add_advection_diffusion_w_variable_property", rhs, U, V, W, viscosity,
           inverse_density, grid, sub, Cmu, dt, 2);
}

void add_advection_diffusion_u_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_impl_constant("add_advection_diffusion_u_constant_property", rhs, U, V, W,
                    grid, sub, viscosity, dt, 0);
}

void add_advection_diffusion_v_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_impl_constant("add_advection_diffusion_v_constant_property", rhs, U, V, W,
                    grid, sub, viscosity, dt, 1);
}

void add_advection_diffusion_w_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_impl_constant("add_advection_diffusion_w_constant_property", rhs, U, V, W,
                    grid, sub, viscosity, dt, 2);
}

void build_variable_property_rhs(field::Field& rhs_u, field::Field& rhs_v, field::Field& rhs_w,
                                 const field::Field& U, const field::Field& V,
                                 const field::Field& W, const field::Field& T,
                                 const field::Field& P, const field::Field& viscosity,
                                 const field::Field& inverse_density, const grid::Grid& grid,
                                 const parallel::Subdomain& sub, const real_t Cmu,
                                 const real_t density_quadratic_ratio,
                                 const real_t delta_temperature, const real_t dt) {
  const char* who = "build_variable_property_rhs";
  const field::Field* fields[] = {&rhs_u, &rhs_v, &rhs_w, &U, &V, &W, &T, &P, &viscosity, &inverse_density};
  for (const field::Field* f : fields) check_field_matches(who, *f, sub);
  launch_build_rhs<0>(rhs_u.data(), U, V, W, T, P, viscosity, inverse_density, grid, sub, Cmu,
                      density_quadratic_ratio, delta_temperature, dt);
  launch_build_rhs<1>(rhs_v.data(), U, V, W, T, P, viscosity, inverse_density, grid, sub, Cmu,
                      density_quadratic_ratio, delta_temperature, dt);
  launch_build_rhs<2>(rhs_w.data(), U, V, W, T, P, viscosity, inverse_density, grid, sub, Cmu,
                      density_quadratic_ratio, delta_temperature, dt);
}

}  // namespace mpmstd::equation::momentum
