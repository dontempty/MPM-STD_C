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

// ── z 스윕 전용 커널 — 공유 메모리 (j,k) 타일 전치 (momentum/system/system.cu 의 build_kernel_z 와 같은 방법, 2026-10-17).
//    z 밴드 p = k·(i1·i2) + i·i2 + j (j 가 가장 빠름) 와 필드(k 가 가장 빠름)는 전치 관계. 예전 1차원 매핑(j 연속)은
//    build_cell 의 필드 로드가 전부 nz_total 보폭으로 흩어져 x·y 조립의 2.5–3.6 배(H200)·11 배(V100)였다 (실측).
//    ① 연속 스레드 = 연속 k 로 읽어 계수를 타일에 두고 ② 전치해 연속 j 로 밴드에 쓴다. 식은 build_cell 그대로 → 비트 동일.
constexpr int kTileZ = 32;
constexpr int kRowsZ = 8;

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
  __shared__ real_t sA[kTileZ][kTileZ + 1], sB[kTileZ][kTileZ + 1],
      sC[kTileZ][kTileZ + 1], sD[kTileZ][kTileZ + 1];
  const int ii = static_cast<int>(blockIdx.z);
  const int j0 = static_cast<int>(blockIdx.y) * kTileZ;
  const int k0 = static_cast<int>(blockIdx.x) * kTileZ;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);

  // ① 필드 순서(k 연속)로 읽어 계수 계산 → 타일
  for (int r = 0; r < kTileZ; r += kRowsZ) {
    const int kk = k0 + tx, jj = j0 + ty + r;
    if (kk < i3 && jj < i2) {
      stencil::build_cell(2, ii + h, jj + h, kk + h, ny_total, nz_total, source,
                                   T, U, V, W, conductivity, conductivity_derivative,
                                   inverse_heat_capacity, inverse_heat_capacity_derivative,
                                   dx_x, dmx_x, dx_y, dmx_y, dx_z, dmx_z, Ct, dt, false,
                                   wall.lo[0] && ii == 0, wall.hi[0] && ii == i1 - 1,
                                   wall.lo[1] && jj == 0, wall.hi[1] && jj == i2 - 1,
                                   wall.lo[2] && kk == 0, wall.hi[2] && kk == i3 - 1, wall,
                                   sA[ty + r][tx], sB[ty + r][tx], sC[ty + r][tx],
                                   sD[ty + r][tx]);
    }
  }
  __syncthreads();

  // ② 밴드 순서(j 연속)로 쓰기 — p = k·(i1·i2) + i·i2 + j
  for (int r = 0; r < kTileZ; r += kRowsZ) {
    const int jj = j0 + tx, kk = k0 + ty + r;
    if (kk < i3 && jj < i2) {
      const std::size_t p =
          static_cast<std::size_t>(kk) * (static_cast<std::size_t>(i1) * i2) + ii * i2 + jj;
      A[p] = sA[tx][ty + r];
      B[p] = sB[tx][ty + r];
      C[p] = sC[tx][ty + r];
      D[p] = sD[tx][ty + r];
    }
  }
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

// z 전용: (32×8) 스레드 블록, 그리드 (⌈i3/32⌉, ⌈i2/32⌉, i1)
struct LaunchShapeZ {
  dim3 blocks, threads;
};
LaunchShapeZ launch_shape_z(const parallel::Subdomain& sub) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  return {dim3((i3 + kTileZ - 1) / kTileZ, (i2 + kTileZ - 1) / kTileZ, i1), dim3(kTileZ, kRowsZ)};
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
  const LaunchShapeZ shape = launch_shape_z(sub);
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
  const LaunchShapeZ shape = launch_shape_z(sub);
  build_z_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, T, U, V, W,
                                                  MPMSTD_SCALAR_CONSTANT_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

#undef MPMSTD_SCALAR_CONSTANT_ARGS

}  // namespace mpmstd::equation::scalar
