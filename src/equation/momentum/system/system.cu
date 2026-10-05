#include "equation/momentum/system/system.hpp"

#include <cstddef>

#include <mpi.h>

#include "common/cuda_check.hpp"
#include "equation/momentum/momentum_stencil.hpp"

// GPU 백엔드 — system.cpp 의 쌍둥이. 밴드 수식은 momentum_stencil.hpp(band_*)
// 하나를 같이 쓰고, 여기는 셀당 스레드 커널만 있다.

namespace mpmstd::equation::momentum {

namespace {

// 스윕별 (row, sys) 배치가 달라 커널을 방향별로 둔다 (D13 — 축 인자 없음).
enum : int { kThreads = 256 };

__device__ inline std::size_t at(const int i, const int j, const int k,
                                 const int ny_total, const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

__global__ void build_x_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const field::ConstView adv,
                               const real_t* dx, const real_t* dmx, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total,
                               const real_t coeff, const real_t cconv) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const stencil::Band band = stencil::band_x(adv, i, j, k, dx, dmx, coeff, cconv);
  const int nsys = i2 * i3;
  const std::size_t p =
      static_cast<std::size_t>(i - h) * nsys + (j - h) * i3 + (k - h);
  A[p] = band.a;
  B[p] = band.b;
  C[p] = band.c;
  D[p] = source[at(i, j, k, ny_total, nz_total)];
}

__global__ void build_y_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const field::ConstView adv,
                               const real_t* dy, const real_t* dmy, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total,
                               const real_t coeff, const real_t cconv) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const stencil::Band band = stencil::band_y(adv, i, j, k, dy, dmy, coeff, cconv);
  const int nsys = i1 * i3;
  const std::size_t p =
      static_cast<std::size_t>(j - h) * nsys + (i - h) * i3 + (k - h);
  A[p] = band.a;
  B[p] = band.b;
  C[p] = band.c;
  D[p] = source[at(i, j, k, ny_total, nz_total)];
}

__global__ void build_z_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                               const real_t* source, const field::ConstView adv,
                               const real_t* dz, const real_t* dmz, const int h,
                               const int i1, const int i2, const int i3,
                               const int ny_total, const int nz_total,
                               const real_t coeff, const real_t cconv,
                               const bool w_is_z_face) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(i1) * i2 * i3) return;
  const int k = h + static_cast<int>(idx % i3);
  const int j = h + static_cast<int>((idx / i3) % i2);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  const stencil::Band band = w_is_z_face
      ? stencil::band_z_face(adv, i, j, k, dz, dmz, coeff, cconv)
      : stencil::band_z_cell(adv, i, j, k, dz, dmz, coeff, cconv);
  const int nsys = i1 * i2;
  const std::size_t p =
      static_cast<std::size_t>(k - h) * nsys + (i - h) * i2 + (j - h);
  A[p] = band.a;
  B[p] = band.b;
  C[p] = band.c;
  D[p] = source[at(i, j, k, ny_total, nz_total)];
}

__global__ void fold_bottom_kernel(real_t* A, const int nsys) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < nsys) A[s] = 0;
}

__global__ void fold_top_kernel(real_t* C, const int nsys, const long long top) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < nsys) C[top + s] = 0;
}

__global__ void pin_kernel(real_t* A, real_t* B, real_t* C, real_t* D,
                           const int nsys) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s < nsys) {
    A[s] = 0;
    B[s] = 1;
    C[s] = 0;
    D[s] = 0;
  }
}

inline int blocks_for(const long long n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

}  // namespace

void build_x_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const field::Field& U_advect, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t viscosity,
                    const real_t dt) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  build_x_kernel<<<blocks_for(n), kThreads>>>(
      A, B, C, D, source, U_advect.view(), grid.x.device_cell_width.data(),
      grid.x.device_center_distance.data(), halo_width, i1, i2, i3,
      sub.y().n_total, sub.z().n_total, real_t(0.5) * dt * viscosity,
      real_t(0.25) * dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_y_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const field::Field& V_advect, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t viscosity,
                    const real_t dt) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  build_y_kernel<<<blocks_for(n), kThreads>>>(
      A, B, C, D, source, V_advect.view(), grid.y.device_cell_width.data(),
      grid.y.device_center_distance.data(), halo_width, i1, i2, i3,
      sub.y().n_total, sub.z().n_total, real_t(0.5) * dt * viscosity,
      real_t(0.25) * dt);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_z_system_cell(real_t* A, real_t* B, real_t* C, real_t* D,
                         const real_t* source, const field::Field& W_advect,
                         const grid::Grid& grid, const parallel::Subdomain& sub,
                         const real_t viscosity, const real_t dt) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  build_z_kernel<<<blocks_for(n), kThreads>>>(
      A, B, C, D, source, W_advect.view(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, i1, i2, i3,
      sub.y().n_total, sub.z().n_total, real_t(0.5) * dt * viscosity,
      real_t(0.25) * dt, false);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void build_z_system_z_face(real_t* A, real_t* B, real_t* C, real_t* D,
                           const real_t* source, const field::Field& W_advect,
                           const grid::Grid& grid, const parallel::Subdomain& sub,
                           const real_t viscosity, const real_t dt) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const long long n = static_cast<long long>(i1) * i2 * i3;
  build_z_kernel<<<blocks_for(n), kThreads>>>(
      A, B, C, D, source, W_advect.view(), grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), halo_width, i1, i2, i3,
      sub.y().n_total, sub.z().n_total, real_t(0.5) * dt * viscosity,
      real_t(0.25) * dt, true);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void fold_z_wall_value(real_t* A, real_t* B, real_t* C,
                       const parallel::Subdomain& sub, const int nsys,
                       const int nrow) {
  (void)B;  // ZeroGhost: 비대각만 제거 — B 유지 (system.hpp ⚠)
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    fold_bottom_kernel<<<blocks_for(nsys), kThreads>>>(A, nsys);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    fold_top_kernel<<<blocks_for(nsys), kThreads>>>(
        C, nsys, static_cast<long long>(nrow - 1) * nsys);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
}

void pin_z_bottom_wall_face_row(real_t* A, real_t* B, real_t* C, real_t* D,
                                const parallel::Subdomain& sub, const int nsys) {
  if (sub.topology().z().minus_neighbor != MPI_PROC_NULL) return;
  pin_kernel<<<blocks_for(nsys), kThreads>>>(A, B, C, D, nsys);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation::momentum
