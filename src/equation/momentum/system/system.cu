#include "equation/momentum/system/system.hpp"

#include <cstddef>

#include "common/cuda_check.hpp"
#include "equation/momentum/momentum_stencil.hpp"

// GPU 백엔드 — system.cpp 의 쌍둥이. 수식은 sweep_band 한 곳.

namespace mpmstd::equation::momentum {

namespace {

// ── x·y 스윕 공용 커널 — sweep_axis: 0/1. 밴드 슬롯 매핑과 이웃 선택만 다르다 (CPU 루프와 동일).
//    두 축 모두 밴드 sys 에 k 가 들어 있어 밴드 순서 = 필드 순서(k 가 가장 빠름) → 읽기·쓰기가 다 병합.
//    z 는 밴드 row 가 k 라 두 레이아웃이 전치 관계여서 별도 커널(build_kernel_z). 나중에 x·y 도
//    build_kernel_x / build_kernel_y 로 나눌 수 있게 런처(launch_xy / launch_z)도 축별로 분리해 둔다.
template <class MuArray, class RhoArray>
__global__ void build_kernel(const int sweep_axis, real_t* A, real_t* B, real_t* C,
                             real_t* D, const real_t* source, const real_t* q,
                             const real_t* advecting_velocity, const MuArray viscosity,
                             const RhoArray inverse_density, const real_t* dx_along,
                             const real_t* dmx_along, const real_t* dx_own,
                             const real_t* dmx_own, const long long own_stride,
                             const int own_axis, const int h, const int i1,
                             const int i2, const int i3, const int ny_total,
                             const int nz_total, const real_t Cmu, const real_t dt) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;

  int ii, jj, kk;
  std::size_t p;
  if (sweep_axis == 0) {
    kk = static_cast<int>(idx % i3);
    jj = static_cast<int>((idx / i3) % i2);
    ii = static_cast<int>(idx / (static_cast<long long>(i3) * i2));
    p = static_cast<std::size_t>(ii) * (i2 * i3) + jj * i3 + kk;
  } else {
    kk = static_cast<int>(idx % i3);
    ii = static_cast<int>((idx / i3) % i1);
    jj = static_cast<int>(idx / (static_cast<long long>(i3) * i1));
    p = static_cast<std::size_t>(jj) * (i1 * i3) + ii * i3 + kk;
  }
  const int i = ii + h, j = jj + h, k = kk + h;
  const long long gi = stencil::vindex(i, j, k, ny_total, nz_total);
  long long gm, gp;
  int a;
  if (sweep_axis == 0) {
    gm = stencil::vindex(i - 1, j, k, ny_total, nz_total);
    gp = stencil::vindex(i + 1, j, k, ny_total, nz_total);
    a = i;
  } else {
    gm = stencil::vindex(i, j - 1, k, ny_total, nz_total);
    gp = stencil::vindex(i, j + 1, k, ny_total, nz_total);
    a = j;
  }
  const int own_index = (own_axis == 0) ? i : (own_axis == 1) ? j : k;
  stencil::sweep_band(own_axis == sweep_axis, false, gi, gm, gp, a,
                               own_index, own_stride, q, advecting_velocity,
                               viscosity, inverse_density, dx_along, dmx_along,
                               dx_own, dmx_own, Cmu, dt, A[p], B[p], C[p]);
  D[p] = source[gi];
}

// ── z 스윕 전용 커널 — 공유 메모리 (j,k) 타일 전치 (2026-10-17, docs/Scalability/ranking_20261017.md §4b).
//    z 밴드는 p = k·(i1·i2) + i·i2 + j (j 가 가장 빠름), 필드는 k 가 가장 빠름 → i 를 고정하면 (j,k) 평면의
//    전치다. 예전 공용 커널은 밴드 순서로 스레드를 놓아 필드 로드 10여 개가 전부 nz_total 보폭으로 흩어졌고
//    (V100 에서 x·y 조립의 10 배, H200 2–4 배 — 실측), 여기서는
//      ① 연속 스레드(threadIdx.x) = 연속 k 로 필드를 읽어 계수를 공유 메모리 타일에 두고,
//      ② 타일을 전치해 연속 스레드 = 연속 j 로 밴드에 쓴다.
//    계수 식·피연산자는 sweep_band 그대로라 밴드 값은 비트 동일 (게이트: 스모크 FINAL 불변).
//    블록 = (kTileZ × kRowsZ) 스레드, 그리드 = (⌈i3/32⌉, ⌈i2/32⌉, i1).
constexpr int kTileZ = 32;
constexpr int kRowsZ = 8;

template <class MuArray, class RhoArray>
__global__ void build_kernel_z(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const real_t* q,
                               const real_t* advecting_velocity, const MuArray viscosity,
                               const RhoArray inverse_density, const real_t* dx_along,
                               const real_t* dmx_along, const real_t* dx_own,
                               const real_t* dmx_own, const long long own_stride,
                               const int own_axis, const int h, const int i1,
                               const int i2, const int i3, const int ny_total,
                               const int nz_total, const real_t Cmu, const real_t dt) {
  __shared__ real_t sA[kTileZ][kTileZ + 1], sB[kTileZ][kTileZ + 1],
      sC[kTileZ][kTileZ + 1], sD[kTileZ][kTileZ + 1];  // +1: 전치 읽기의 뱅크 충돌 방지
  const int ii = static_cast<int>(blockIdx.z);
  const int j0 = static_cast<int>(blockIdx.y) * kTileZ;
  const int k0 = static_cast<int>(blockIdx.x) * kTileZ;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  const int i = ii + h;

  // ① 필드 순서로 읽기 — tx 가 k 를 걷는다 (sweep_band 의 모든 로드가 병합)
  for (int r = 0; r < kTileZ; r += kRowsZ) {
    const int kk = k0 + tx, jj = j0 + ty + r;
    if (kk < i3 && jj < i2) {
      const int j = jj + h, k = kk + h;
      const long long gi = stencil::vindex(i, j, k, ny_total, nz_total);
      const long long gm = stencil::vindex(i, j, k - 1, ny_total, nz_total);
      const long long gp = stencil::vindex(i, j, k + 1, ny_total, nz_total);
      const int own_index = (own_axis == 0) ? i : (own_axis == 1) ? j : k;
      stencil::sweep_band(own_axis == 2, false, gi, gm, gp, k, own_index, own_stride, q,
                          advecting_velocity, viscosity, inverse_density, dx_along,
                          dmx_along, dx_own, dmx_own, Cmu, dt, sA[ty + r][tx],
                          sB[ty + r][tx], sC[ty + r][tx]);
      sD[ty + r][tx] = source[gi];
    }
  }
  __syncthreads();

  // ② 밴드 순서로 쓰기 — tx 가 j 를 걷는다. p = k·(i1·i2) + i·i2 + j
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

struct DeviceAxes {
  const real_t* dx;
  const real_t* dmx;
};

DeviceAxes device_axes(const int axis, const grid::Grid& grid) {
  if (axis == 0)
    return {grid.x.device_cell_width.data(), grid.x.device_center_distance.data()};
  if (axis == 1)
    return {grid.y.device_cell_width.data(), grid.y.device_center_distance.data()};
  return {grid.z.device_cell_width.data(), grid.z.device_center_distance.data()};
}

long long own_axis_stride(const int own_axis, const parallel::Subdomain& sub) {
  return (own_axis == 0) ? static_cast<long long>(sub.y().n_total) * sub.z().n_total
         : (own_axis == 1) ? static_cast<long long>(sub.z().n_total)
                           : 1;
}

// x·y 스윕 런처 (sweep_axis 0/1) — 1차원 그리드, 스레드 = 셀
template <class MuArray, class RhoArray>
void launch_xy(const int sweep_axis, real_t* A, real_t* B, real_t* C, real_t* D,
               const real_t* source, const real_t* q, const real_t* advecting_velocity,
               const MuArray viscosity, const RhoArray inverse_density,
               const grid::Grid& grid, const parallel::Subdomain& sub, const int own_axis,
               const real_t Cmu, const real_t dt) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  const DeviceAxes along = device_axes(sweep_axis, grid);
  const DeviceAxes own = device_axes(own_axis, grid);
  build_kernel<<<blocks, threads>>>(sweep_axis, A, B, C, D, source, q,
                                    advecting_velocity, viscosity, inverse_density,
                                    along.dx, along.dmx, own.dx, own.dmx,
                                    own_axis_stride(own_axis, sub), own_axis, halo_width,
                                    i1, i2, i3, sub.y().n_total, sub.z().n_total, Cmu, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

// z 스윕 런처 — (32×8) 스레드 블록, 그리드 (⌈i3/32⌉, ⌈i2/32⌉, i1)
template <class MuArray, class RhoArray>
void launch_z(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
              const real_t* q, const real_t* advecting_velocity, const MuArray viscosity,
              const RhoArray inverse_density, const grid::Grid& grid,
              const parallel::Subdomain& sub, const int own_axis, const real_t Cmu,
              const real_t dt) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const DeviceAxes along = device_axes(2, grid);
  const DeviceAxes own = device_axes(own_axis, grid);
  const dim3 threads(kTileZ, kRowsZ);
  const dim3 blocks((i3 + kTileZ - 1) / kTileZ, (i2 + kTileZ - 1) / kTileZ, i1);
  build_kernel_z<<<blocks, threads>>>(A, B, C, D, source, q, advecting_velocity, viscosity,
                                      inverse_density, along.dx, along.dmx, own.dx, own.dmx,
                                      own_axis_stride(own_axis, sub), own_axis, halo_width,
                                      i1, i2, i3, sub.y().n_total, sub.z().n_total, Cmu, dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace

void build_x_sweep_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  launch_xy(0, A, B, C, D, source, q, advecting_velocity, viscosity, inverse_density,
         grid, sub, own_axis, Cmu, dt);
}

void build_y_sweep_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  launch_xy(1, A, B, C, D, source, q, advecting_velocity, viscosity, inverse_density,
         grid, sub, own_axis, Cmu, dt);
}

void build_z_sweep_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  launch_z(A, B, C, D, source, q, advecting_velocity, viscosity, inverse_density,
         grid, sub, own_axis, Cmu, dt);
}

// ── 상수물성 판 (plan §18-1) — UnitArray 로 같은 커널을 돈다 ──────────────────
void build_x_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub,
                                     const int own_axis, const real_t viscosity,
                                     const real_t dt) {
  launch_xy(0, A, B, C, D, source, q, advecting_velocity, equation::UnitArray{},
         equation::UnitArray{}, grid, sub, own_axis, viscosity, dt);
}

void build_y_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub,
                                     const int own_axis, const real_t viscosity,
                                     const real_t dt) {
  launch_xy(1, A, B, C, D, source, q, advecting_velocity, equation::UnitArray{},
         equation::UnitArray{}, grid, sub, own_axis, viscosity, dt);
}

void build_z_sweep_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                                     const real_t* source, const real_t* q,
                                     const real_t* advecting_velocity,
                                     const grid::Grid& grid,
                                     const parallel::Subdomain& sub,
                                     const int own_axis, const real_t viscosity,
                                     const real_t dt) {
  launch_z(A, B, C, D, source, q, advecting_velocity, equation::UnitArray{},
         equation::UnitArray{}, grid, sub, own_axis, viscosity, dt);
}

}  // namespace mpmstd::equation::momentum
