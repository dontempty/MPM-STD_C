#include "equation/scalar/system/system.hpp"

#include <cstddef>

#include <mpi.h>

#include "common/cuda_check.hpp"
#include "equation/scalar/scalar_stencil.hpp"

// GPU 백엔드 — system.cpp 의 쌍둥이. 수식은 scalar_stencil.hpp.

namespace mpmstd::equation::scalar {

namespace {

// sweep_axis 별 (셀 → 밴드 슬롯 p) 매핑만 커널 셋으로 나눈다 (CPU 루프와 동일).
template <class KA, class DKA, class HA, class DHA>
__global__ void build_x_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* T, const real_t* U, const real_t* V,
                               const real_t* W, const KA conductivity,
                               const DKA conductivity_derivative,
                               const HA inverse_heat_capacity,
                               const DHA inverse_heat_capacity_derivative,
                               const real_t* dx_x, const real_t* dmx_x,
                               const real_t* dx_y, const real_t* dmx_y,
                               const real_t* dx_z, const real_t* dmx_z, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total, const real_t Ct,
                               const real_t dt, const stencil::WallMask wall) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int kk = static_cast<int>(idx % i3);
  const int jj = static_cast<int>((idx / i3) % i2);
  const int ii = static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const std::size_t p = static_cast<std::size_t>(ii) * (i2 * i3) + jj * i3 + kk;
  stencil::build_cell(0, ii + h, jj + h, kk + h, ny_total, nz_total, nullptr,
                               T, U, V, W, conductivity, conductivity_derivative,
                               inverse_heat_capacity, inverse_heat_capacity_derivative,
                               dx_x, dmx_x, dx_y, dmx_y, dx_z, dmx_z, Ct, dt, true,
                               wall.lo[0] && ii == 0, wall.hi[0] && ii == i1 - 1,
                               wall.lo[1] && jj == 0, wall.hi[1] && jj == i2 - 1,
                               wall.lo[2] && kk == 0, wall.hi[2] && kk == i3 - 1, wall, A[p],
                               B[p], C[p], D[p]);
}

template <class KA, class DKA, class HA, class DHA>
__global__ void build_y_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const real_t* T, const real_t* U,
                               const real_t* V, const real_t* W,
                               const KA conductivity,
                               const DKA conductivity_derivative,
                               const HA inverse_heat_capacity,
                               const DHA inverse_heat_capacity_derivative,
                               const real_t* dx_x, const real_t* dmx_x,
                               const real_t* dx_y, const real_t* dmx_y,
                               const real_t* dx_z, const real_t* dmx_z, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total, const real_t Ct,
                               const real_t dt, const stencil::WallMask wall) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int kk = static_cast<int>(idx % i3);
  const int ii = static_cast<int>((idx / i3) % i1);
  const int jj = static_cast<int>(idx / (static_cast<long long>(i3) * i1));
  const std::size_t p = static_cast<std::size_t>(jj) * (i1 * i3) + ii * i3 + kk;
  stencil::build_cell(1, ii + h, jj + h, kk + h, ny_total, nz_total, source,
                               T, U, V, W, conductivity, conductivity_derivative,
                               inverse_heat_capacity, inverse_heat_capacity_derivative,
                               dx_x, dmx_x, dx_y, dmx_y, dx_z, dmx_z, Ct, dt, false,
                               wall.lo[0] && ii == 0, wall.hi[0] && ii == i1 - 1,
                               wall.lo[1] && jj == 0, wall.hi[1] && jj == i2 - 1,
                               wall.lo[2] && kk == 0, wall.hi[2] && kk == i3 - 1, wall, A[p],
                               B[p], C[p], D[p]);
}

template <class KA, class DKA, class HA, class DHA>
__global__ void build_z_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const real_t* T, const real_t* U,
                               const real_t* V, const real_t* W,
                               const KA conductivity,
                               const DKA conductivity_derivative,
                               const HA inverse_heat_capacity,
                               const DHA inverse_heat_capacity_derivative,
                               const real_t* dx_x, const real_t* dmx_x,
                               const real_t* dx_y, const real_t* dmx_y,
                               const real_t* dx_z, const real_t* dmx_z, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total, const real_t Ct,
                               const real_t dt, const stencil::WallMask wall) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int jj = static_cast<int>(idx % i2);
  const int ii = static_cast<int>((idx / i2) % i1);
  const int kk = static_cast<int>(idx / (static_cast<long long>(i2) * i1));
  const std::size_t p = static_cast<std::size_t>(kk) * (i1 * i2) + ii * i2 + jj;
  stencil::build_cell(2, ii + h, jj + h, kk + h, ny_total, nz_total, source,
                               T, U, V, W, conductivity, conductivity_derivative,
                               inverse_heat_capacity, inverse_heat_capacity_derivative,
                               dx_x, dmx_x, dx_y, dmx_y, dx_z, dmx_z, Ct, dt, false,
                               wall.lo[0] && ii == 0, wall.hi[0] && ii == i1 - 1,
                               wall.lo[1] && jj == 0, wall.hi[1] && jj == i2 - 1,
                               wall.lo[2] && kk == 0, wall.hi[2] && kk == i3 - 1, wall, A[p],
                               B[p], C[p], D[p]);
}

struct LaunchShape {
  int blocks, threads;
};

LaunchShape launch_shape(const parallel::Subdomain& sub) {
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  const int threads = 256;
  return {static_cast<int>((n + threads - 1) / threads), threads};
}

}  // namespace

#define MPMSTD_SCALAR_VARIABLE_ARGS                                                   \
  properties.conductivity.data(), properties.conductivity_derivative.data(),          \
      properties.inverse_heat_capacity.data(),                                        \
      properties.inverse_heat_capacity_derivative.data(),                             \
      grid.x.device_cell_width.data(), grid.x.device_center_distance.data(),          \
      grid.y.device_cell_width.data(), grid.y.device_center_distance.data(),          \
      grid.z.device_cell_width.data(), grid.z.device_center_distance.data(),          \
      halo_width, sub.x().n_local, sub.y().n_local, sub.z().n_local,                   \
      sub.y().n_total, sub.z().n_total, Ct, dt,                                        \
      wall

void build_x_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* T, const real_t* U, const real_t* V,
                             const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt) {
  const LaunchShape shape = launch_shape(sub);
  build_x_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, T, U, V, W,
                                                  MPMSTD_SCALAR_VARIABLE_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_y_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt) {
  const LaunchShape shape = launch_shape(sub);
  build_y_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, T, U, V, W,
                                                  MPMSTD_SCALAR_VARIABLE_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_z_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt) {
  const LaunchShape shape = launch_shape(sub);
  build_z_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, T, U, V, W,
                                                  MPMSTD_SCALAR_VARIABLE_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

#undef MPMSTD_SCALAR_VARIABLE_ARGS

// ── 상수물성 판 (plan §18-1): κ̂·1/(ρ̂Ĉp)≡1, 미분장≡0 ───────────────────────
#define MPMSTD_SCALAR_CONSTANT_ARGS                                                   \
  equation::UnitArray{}, equation::ZeroArray{}, equation::UnitArray{},                \
      equation::ZeroArray{},                                                          \
      grid.x.device_cell_width.data(), grid.x.device_center_distance.data(),          \
      grid.y.device_cell_width.data(), grid.y.device_center_distance.data(),          \
      grid.z.device_cell_width.data(), grid.z.device_center_distance.data(),          \
      halo_width, sub.x().n_local, sub.y().n_local, sub.z().n_local,                   \
      sub.y().n_total, sub.z().n_total, Ct, dt,                                        \
      wall

void build_x_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                      const real_t* T, const real_t* U,
                                      const real_t* V, const real_t* W,
                                      const grid::Grid& grid,
                                      const parallel::Subdomain& sub,
                                      const stencil::WallMask& wall, const real_t Ct,
                                      const real_t dt) {
  const LaunchShape shape = launch_shape(sub);
  build_x_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, T, U, V, W,
                                                  MPMSTD_SCALAR_CONSTANT_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_y_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                      const real_t* source, const real_t* T,
                                      const real_t* U, const real_t* V,
                                      const real_t* W, const grid::Grid& grid,
                                      const parallel::Subdomain& sub,
                                      const stencil::WallMask& wall, const real_t Ct,
                                      const real_t dt) {
  const LaunchShape shape = launch_shape(sub);
  build_y_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, T, U, V, W,
                                                  MPMSTD_SCALAR_CONSTANT_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_z_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                      const real_t* source, const real_t* T,
                                      const real_t* U, const real_t* V,
                                      const real_t* W, const grid::Grid& grid,
                                      const parallel::Subdomain& sub,
                                      const stencil::WallMask& wall, const real_t Ct,
                                      const real_t dt) {
  const LaunchShape shape = launch_shape(sub);
  build_z_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, T, U, V, W,
                                                  MPMSTD_SCALAR_CONSTANT_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

#undef MPMSTD_SCALAR_CONSTANT_ARGS

}  // namespace mpmstd::equation::scalar
