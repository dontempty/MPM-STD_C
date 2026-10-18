#include "equation/scalar/system/system.hpp"

#include "equation/band_layout.hpp"

// [최적화] ① build_z_kernel 타일 전치(10-17) ② idx·idmx 격자 역수 전달(10-17) ③ build_y/z 의 source_layout 밴드 핑퐁(10-18)

#include <cstddef>

#include <mpi.h>

#include "common/cuda_check.hpp"
#include "equation/scalar/scalar_stencil.hpp"

// GPU 백엔드 — system.cpp 의 쌍둥이. 수식은 scalar_stencil.hpp.

namespace mpmstd::equation::scalar {

namespace {

// 2-D 블록 매핑 공통: (32×8) 스레드, 스레드가 k 를 걷고 j 방향 4행 (build_z 전치 커널과 x·y 조립 커널이 같이 쓴다)
constexpr int kTileZ = 32;
constexpr int kRowsZ = 8;

// sweep_axis 별 (셀 → 밴드 슬롯 p) 매핑만 커널 셋으로 나눈다 (CPU 루프와 동일).
template <class KA, class DKA, class HA, class DHA>
__global__ void build_x_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* T, const real_t* U, const real_t* V,
                               const real_t* W, const KA conductivity,
                               const DKA conductivity_derivative,
                               const HA inverse_heat_capacity,
                               const DHA inverse_heat_capacity_derivative,
                               const real_t* dx_x, const real_t* idx_x, const real_t* idmx_x,
                               const real_t* dx_y, const real_t* idx_y, const real_t* idmx_y,
                               const real_t* dx_z, const real_t* idx_z, const real_t* idmx_z, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total, const real_t Ct,
                               const real_t dt, const stencil::WallMask wall) {
  // [최적화 2026-10-18 실험] 1차원 스레드(64비트 나눗셈 3회) → build_z 와 같은 (i, j-타일, k-타일) 2-D 블록 매핑.
  //   밴드는 k 연속이라 전치 없음. 식·밴드 위치 동일 → 비트 동일.
  const int ii = static_cast<int>(blockIdx.z);
  const int j0 = static_cast<int>(blockIdx.y) * kTileZ, k0 = static_cast<int>(blockIdx.x) * kTileZ;
  const int kk = k0 + static_cast<int>(threadIdx.x);
  if (kk >= i3) return;
  for (int r = 0; r < kTileZ; r += kRowsZ) {
  const int jj = j0 + static_cast<int>(threadIdx.y) + r;
  if (jj >= i2) break;
  const std::size_t p = static_cast<std::size_t>(ii) * (i2 * i3) + jj * i3 + kk;
  stencil::build_cell(0, ii + h, jj + h, kk + h, ny_total, nz_total, nullptr, 0,
                               T, U, V, W, conductivity, conductivity_derivative,
                               inverse_heat_capacity, inverse_heat_capacity_derivative,
                               dx_x, idx_x, idmx_x, dx_y, idx_y, idmx_y, dx_z, idx_z, idmx_z, Ct, dt, true,
                               wall.lo[0] && ii == 0, wall.hi[0] && ii == i1 - 1,
                               wall.lo[1] && jj == 0, wall.hi[1] && jj == i2 - 1,
                               wall.lo[2] && kk == 0, wall.hi[2] && kk == i3 - 1, wall, A[p],
                               B[p], C[p], D[p]);
  }
}

template <class KA, class DKA, class HA, class DHA>
__global__ void build_y_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const SourceLayout source_layout, const real_t* T, const real_t* U,
                               const real_t* V, const real_t* W,
                               const KA conductivity,
                               const DKA conductivity_derivative,
                               const HA inverse_heat_capacity,
                               const DHA inverse_heat_capacity_derivative,
                               const real_t* dx_x, const real_t* idx_x, const real_t* idmx_x,
                               const real_t* dx_y, const real_t* idx_y, const real_t* idmx_y,
                               const real_t* dx_z, const real_t* idx_z, const real_t* idmx_z, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total, const real_t Ct,
                               const real_t dt, const stencil::WallMask wall) {
  // [최적화 2026-10-18 실험] build_x 와 같은 2-D 블록 매핑 (y 밴드 p = j·(i1·i3) + i·i3 + k, k 연속)
  const int ii = static_cast<int>(blockIdx.z);
  const int j0 = static_cast<int>(blockIdx.y) * kTileZ, k0 = static_cast<int>(blockIdx.x) * kTileZ;
  const int kk = k0 + static_cast<int>(threadIdx.x);
  if (kk >= i3) return;
  for (int r = 0; r < kTileZ; r += kRowsZ) {
  const int jj = j0 + static_cast<int>(threadIdx.y) + r;
  if (jj >= i2) break;
  const std::size_t p = static_cast<std::size_t>(jj) * (i1 * i3) + ii * i3 + kk;
  stencil::build_cell(1, ii + h, jj + h, kk + h, ny_total, nz_total, source,
                               source_index(source_layout, ii, jj, kk, i1, i2, i3, static_cast<std::size_t>(stencil::cell_index(ii + h, jj + h, kk + h, ny_total, nz_total))),
                               T, U, V, W, conductivity, conductivity_derivative,
                               inverse_heat_capacity, inverse_heat_capacity_derivative,
                               dx_x, idx_x, idmx_x, dx_y, idx_y, idmx_y, dx_z, idx_z, idmx_z, Ct, dt, false,
                               wall.lo[0] && ii == 0, wall.hi[0] && ii == i1 - 1,
                               wall.lo[1] && jj == 0, wall.hi[1] && jj == i2 - 1,
                               wall.lo[2] && kk == 0, wall.hi[2] && kk == i3 - 1, wall, A[p],
                               B[p], C[p], D[p]);
  }
}

// [최적화 2026-10-17] z 스윕 조립 32×32 타일 전치 (momentum build_kernel_z 와 같은 방법). 식은 build_cell 그대로(비트 동일). V100 ×4.

template <class KA, class DKA, class HA, class DHA>
__global__ void build_z_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const SourceLayout source_layout, const real_t* T, const real_t* U,
                               const real_t* V, const real_t* W,
                               const KA conductivity,
                               const DKA conductivity_derivative,
                               const HA inverse_heat_capacity,
                               const DHA inverse_heat_capacity_derivative,
                               const real_t* dx_x, const real_t* idx_x, const real_t* idmx_x,
                               const real_t* dx_y, const real_t* idx_y, const real_t* idmx_y,
                               const real_t* dx_z, const real_t* idx_z, const real_t* idmx_z, const int h,
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
                                   source_index(source_layout, ii, jj, kk, i1, i2, i3, static_cast<std::size_t>(stencil::cell_index(ii + h, jj + h, kk + h, ny_total, nz_total))),
                                   T, U, V, W, conductivity, conductivity_derivative,
                                   inverse_heat_capacity, inverse_heat_capacity_derivative,
                                   dx_x, idx_x, idmx_x, dx_y, idx_y, idmx_y, dx_z, idx_z, idmx_z, Ct, dt, false,
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

// x·y·z 공통: (32×8) 스레드 블록, 그리드 (⌈i3/32⌉, ⌈i2/32⌉, i1) — 2026-10-18 x·y 도 같은 매핑 (build_y 3.15 → 1.79 ms)
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
      grid.x.device_cell_width.data(), grid.x.device_inverse_cell_width.data(),       \
      grid.x.device_inverse_center_distance.data(),                                   \
      grid.y.device_cell_width.data(), grid.y.device_inverse_cell_width.data(),       \
      grid.y.device_inverse_center_distance.data(),                                   \
      grid.z.device_cell_width.data(), grid.z.device_inverse_cell_width.data(),       \
      grid.z.device_inverse_center_distance.data(),                                   \
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
  const LaunchShapeZ shape = launch_shape_z(sub);
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
                             const real_t dt, const SourceLayout source_layout) {
  const LaunchShapeZ shape = launch_shape_z(sub);
  build_y_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, source_layout, T, U, V, W,
                                                  MPMSTD_SCALAR_VARIABLE_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_z_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt, const SourceLayout source_layout) {
  const LaunchShapeZ shape = launch_shape_z(sub);
  build_z_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, source_layout, T, U, V, W,
                                                  MPMSTD_SCALAR_VARIABLE_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

#undef MPMSTD_SCALAR_VARIABLE_ARGS

// ── 상수물성 판 (plan §18-1): κ̂·1/(ρ̂Ĉp)≡1, 미분장≡0 ───────────────────────
#define MPMSTD_SCALAR_CONSTANT_ARGS                                                   \
  equation::UnitArray{}, equation::ZeroArray{}, equation::UnitArray{},                \
      equation::ZeroArray{},                                                          \
      grid.x.device_cell_width.data(), grid.x.device_inverse_cell_width.data(),       \
      grid.x.device_inverse_center_distance.data(),                                   \
      grid.y.device_cell_width.data(), grid.y.device_inverse_cell_width.data(),       \
      grid.y.device_inverse_center_distance.data(),                                   \
      grid.z.device_cell_width.data(), grid.z.device_inverse_cell_width.data(),       \
      grid.z.device_inverse_center_distance.data(),                                   \
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
  const LaunchShapeZ shape = launch_shape_z(sub);
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
                                      const real_t dt, const SourceLayout source_layout) {
  const LaunchShapeZ shape = launch_shape_z(sub);
  build_y_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, source_layout, T, U, V, W,
                                                  MPMSTD_SCALAR_CONSTANT_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_z_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                      const real_t* source, const real_t* T,
                                      const real_t* U, const real_t* V,
                                      const real_t* W, const grid::Grid& grid,
                                      const parallel::Subdomain& sub,
                                      const stencil::WallMask& wall, const real_t Ct,
                                      const real_t dt, const SourceLayout source_layout) {
  const LaunchShapeZ shape = launch_shape_z(sub);
  build_z_kernel<<<shape.blocks, shape.threads>>>(A, B, C, D, source, source_layout, T, U, V, W,
                                                  MPMSTD_SCALAR_CONSTANT_ARGS);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

#undef MPMSTD_SCALAR_CONSTANT_ARGS

}  // namespace mpmstd::equation::scalar
