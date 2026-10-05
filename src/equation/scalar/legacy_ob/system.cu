#include "equation/scalar/legacy_ob/system.hpp"

#include <cstddef>

#include <mpi.h>

#include "common/cuda_check.hpp"
#include "equation/scalar/legacy_ob/scalar_stencil.hpp"

// GPU 백엔드 — system.cpp 의 쌍둥이 (밴드 식은 scalar_stencil.hpp 공용).

namespace mpmstd::equation::scalar {

namespace {

enum : int { kThreads = 256 };

__device__ inline std::size_t at(const int i, const int j, const int k,
                                 const int ny_total, const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

// 스윕별 (row, sys) 배치 — system.hpp 표. sweep_axis: 0=x, 1=y, 2=z 로 밴드
// 메트릭·배치만 고른다 (내부 전용 커널 — 공개 API 는 방향별 함수다, D13).
__global__ void build_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* d_along,
                             const real_t* dm_along, const int sweep_axis,
                             const int h, const int i1, const int i2, const int i3,
                             const int ny_total, const int nz_total,
                             const real_t half_dt_alpha) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));

  int along = i, row = i - h;
  std::size_t p = static_cast<std::size_t>(i - h) * (i2 * i3) + (j - h) * i3 + (k - h);
  if (sweep_axis == 1) {
    along = j;
    row = j - h;
    p = static_cast<std::size_t>(row) * (i1 * i3) + (i - h) * i3 + (k - h);
  } else if (sweep_axis == 2) {
    along = k;
    row = k - h;
    p = static_cast<std::size_t>(row) * (i1 * i2) + (i - h) * i2 + (j - h);
  }
  const stencil::Band band = stencil::diffusion_band(half_dt_alpha, d_along[along],
                                                     dm_along[along],
                                                     dm_along[along + 1]);
  A[p] = band.a;
  B[p] = band.b;
  C[p] = band.c;
  D[p] = source[at(i, j, k, ny_total, nz_total)];
}

__global__ void fold_bottom_kernel(real_t* A, real_t* B, const int nsys) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < nsys) {
    B[s] -= A[s];
    A[s] = 0;
  }
}

__global__ void fold_top_kernel(real_t* B, real_t* C, const int nsys,
                                const long long top) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < nsys) {
    B[top + s] -= C[top + s];
    C[top + s] = 0;
  }
}

inline int blocks_cells(const parallel::Subdomain& sub) {
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

void launch_build(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                  const real_t* d_along, const real_t* dm_along, const int sweep_axis,
                  const parallel::Subdomain& sub, const real_t alpha, const real_t dt) {
  build_kernel<<<blocks_cells(sub), kThreads>>>(
      A, B, C, D, source, d_along, dm_along, sweep_axis, halo_width,
      sub.x().n_local, sub.y().n_local, sub.z().n_local, sub.y().n_total,
      sub.z().n_total, real_t(0.5) * dt * alpha);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace

void build_x_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t alpha, const real_t dt) {
  launch_build(A, B, C, D, source, grid.x.device_cell_width.data(),
               grid.x.device_center_distance.data(), 0, sub, alpha, dt);
}

void build_y_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t alpha, const real_t dt) {
  launch_build(A, B, C, D, source, grid.y.device_cell_width.data(),
               grid.y.device_center_distance.data(), 1, sub, alpha, dt);
}

void build_z_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t alpha, const real_t dt) {
  launch_build(A, B, C, D, source, grid.z.device_cell_width.data(),
               grid.z.device_center_distance.data(), 2, sub, alpha, dt);
}

void fold_z_wall_mirror(real_t* A, real_t* B, real_t* C,
                        const parallel::Subdomain& sub, const int nsys,
                        const int nrow) {
  const int blocks = (nsys + kThreads - 1) / kThreads;
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    fold_bottom_kernel<<<blocks, kThreads>>>(A, B, nsys);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    fold_top_kernel<<<blocks, kThreads>>>(B, C, nsys,
                                          static_cast<long long>(nrow - 1) * nsys);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
}

}  // namespace mpmstd::equation::scalar
