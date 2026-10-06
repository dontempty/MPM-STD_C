#include "boundary/open.hpp"

#include <mpi.h>

#include <stdexcept>
#include <vector>

#include "common/cuda_check.hpp"

// GPU 백엔드 — open.cpp 의 쌍둥이 (식 동일). 유입 프로파일은 호스트 배열이라 디바이스에
// 복사해 쓴다; 유량 합은 atomicAdd (합의 비결정성은 1e-16 급 — byte 비교엔 주의).

namespace mpmstd::boundary {

namespace {

enum : int { kThreads = 256 };
inline int blocks_for(const long long n) { return static_cast<int>((n + kThreads - 1) / kThreads); }
bool owns_minus(const parallel::Subdomain& sub, int axis) {
  return sub.topology().axis(axis).minus_neighbor == MPI_PROC_NULL;
}
bool owns_plus(const parallel::Subdomain& sub, int axis) {
  return sub.topology().axis(axis).plus_neighbor == MPI_PROC_NULL;
}

__global__ void inflow_kernel(const field::View u, const field::View v, const field::View w,
                              const real_t* pu, const real_t* pv, const real_t* pw,
                              const int h, const int ny, const int nz, const bool mirror) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ny * nz) return;
  const int j = idx / nz, k = idx % nz;
  u(h, j, k) = pu[k];
  if (mirror) {
    v(0, j, k) = 2 * pv[k] - v(h, j, k);
    w(0, j, k) = 2 * pw[k] - w(h, j, k);
  } else {
    v(0, j, k) = pv[k];
    w(0, j, k) = pw[k];
  }
}

__global__ void outflow_kernel(const field::View u, const field::View v, const field::View w,
                               const int i_face, const int i_ghost, const int ny, const int nz,
                               const real_t cu, const real_t cc) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ny * nz) return;
  const int j = idx / nz, k = idx % nz;
  u(i_face, j, k) -= cu * (u(i_face, j, k) - u(i_face - 1, j, k));
  v(i_ghost, j, k) -= cc * (v(i_ghost, j, k) - v(i_ghost - 1, j, k));
  w(i_ghost, j, k) -= cc * (w(i_ghost, j, k) - w(i_ghost - 1, j, k));
}

__global__ void top_continuity_kernel(const field::View w, const field::ConstView u,
                                      const field::ConstView v, const real_t* dx,
                                      const real_t* dy, const real_t dz, const int h,
                                      const int nx, const int ny, const int k_face) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ni = nx - 2 * h, nj = ny - 2 * h;
  if (idx >= ni * nj) return;
  const int i = h + idx / nj, j = h + idx % nj;
  const int k_cell = k_face - 1;
  const real_t dudx = (u(i + 1, j, k_cell) - u(i, j, k_cell)) / dx[i];
  const real_t dvdy = (v(i, j + 1, k_cell) - v(i, j, k_cell)) / dy[j];
  w(i, j, k_face) = w(i, j, k_cell) - dz * (dudx + dvdy);
}

__global__ void top_profile_kernel(const field::View w, const real_t* w_top, const int nx,
                                   const int ny, const int k_face) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nx * ny) return;
  w(idx / ny, idx % ny, k_face) = w_top[idx / ny];
}

// 면 적분 (atomicAdd). which: 0 = x− 면 U, 1 = z+ 면 W, 2 = x+ 면 U (+ 면적을 [3] 에)
__global__ void flux_x_kernel(const field::ConstView u, const int i, const real_t* dy,
                              const real_t* dz, const int h, const int ny, const int nz,
                              real_t* sum, real_t* area) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int nj = ny - 2 * h, nk = nz - 2 * h;
  if (idx >= nj * nk) return;
  const int j = h + idx / nk, k = h + idx % nk;
  const real_t a = dy[j] * dz[k];
  atomicAdd(sum, u(i, j, k) * a);
  if (area) atomicAdd(area, a);
}
__global__ void flux_z_kernel(const field::ConstView w, const int k, const real_t* dx,
                              const real_t* dy, const int h, const int nx, const int ny,
                              real_t* sum) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int ni = nx - 2 * h, nj = ny - 2 * h;
  if (idx >= ni * nj) return;
  const int i = h + idx / nj, j = h + idx % nj;
  atomicAdd(sum, w(i, j, k) * dx[i] * dy[j]);
}
__global__ void add_face_kernel(const field::View u, const int i, const int h, const int ny,
                                const int nz, const real_t value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int nj = ny - 2 * h, nk = nz - 2 * h;
  if (idx >= nj * nk) return;
  u(i, h + idx / nk, h + idx % nk) += value;
}

}  // namespace

void set_inflow(field::Field& U, field::Field& V, field::Field& W, const InflowProfile& p,
                const VelocityGhost ghost, const parallel::Subdomain& sub) {
  if (!owns_minus(sub, 0)) return;
  const int h = halo_width;
  const int ny = sub.y().n_total, nz = sub.z().n_total;
  if (static_cast<int>(p.u.size()) != nz)
    throw std::invalid_argument("set_inflow: profile size must be nz_total");
  real_t *du = nullptr, *dv = nullptr, *dw = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&du), nz * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dv), nz * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dw), nz * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemcpy(du, p.u.data(), nz * sizeof(real_t), cudaMemcpyHostToDevice));
  MPMSTD_CUDA_CHECK(cudaMemcpy(dv, p.v.data(), nz * sizeof(real_t), cudaMemcpyHostToDevice));
  MPMSTD_CUDA_CHECK(cudaMemcpy(dw, p.w.data(), nz * sizeof(real_t), cudaMemcpyHostToDevice));
  inflow_kernel<<<blocks_for(ny * nz), kThreads>>>(U.view(), V.view(), W.view(), du, dv, dw, h,
                                                   ny, nz, ghost == VelocityGhost::Mirror);
  MPMSTD_CUDA_CHECK_LAUNCH();
  MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(du);
  cudaFree(dv);
  cudaFree(dw);
}

void update_outflow_convective(field::Field& U, field::Field& V, field::Field& W,
                               const real_t convective_velocity, const real_t dt,
                               const grid::Grid& grid, const parallel::Subdomain& sub) {
  if (!owns_plus(sub, 0)) return;
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int i_face = nx - h, i_ghost = nx - 1;
  const real_t cu = convective_velocity * dt / grid.x.cell_width[i_face - 1];
  const real_t cc = convective_velocity * dt / grid.x.center_distance[i_ghost];
  outflow_kernel<<<blocks_for(ny * nz), kThreads>>>(U.view(), V.view(), W.view(), i_face,
                                                    i_ghost, ny, nz, cu, cc);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void set_top_normal_from_continuity(field::Field& W, const field::Field& U,
                                    const field::Field& V, const grid::Grid& grid,
                                    const parallel::Subdomain& sub) {
  if (!owns_plus(sub, 2)) return;
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int k_face = nz - h;
  top_continuity_kernel<<<blocks_for((nx - 2 * h) * (ny - 2 * h)), kThreads>>>(
      W.view(), U.view(), V.view(), grid.x.device_cell_width.data(),
      grid.y.device_cell_width.data(), grid.z.cell_width[k_face - 1], h, nx, ny, k_face);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void set_top_normal_profile(field::Field& W, const std::vector<real_t>& w_top,
                            const parallel::Subdomain& sub) {
  if (!owns_plus(sub, 2)) return;
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  if (static_cast<int>(w_top.size()) != nx)
    throw std::invalid_argument("set_top_normal_profile: profile size must be nx_total");
  real_t* d = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d), nx * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemcpy(d, w_top.data(), nx * sizeof(real_t), cudaMemcpyHostToDevice));
  top_profile_kernel<<<blocks_for(nx * ny), kThreads>>>(W.view(), d, nx, ny, nz - h);
  MPMSTD_CUDA_CHECK_LAUNCH();
  MPMSTD_CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(d);
}

real_t correct_outflow_flux(field::Field& U, const field::Field& W, const grid::Grid& grid,
                            const parallel::Subdomain& sub) {
  const int h = halo_width;
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  real_t* d = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d), 4 * sizeof(real_t)));
  MPMSTD_CUDA_CHECK(cudaMemset(d, 0, 4 * sizeof(real_t)));
  const int n_x = (ny - 2 * h) * (nz - 2 * h), n_z = (nx - 2 * h) * (ny - 2 * h);
  if (owns_minus(sub, 0)) {
    flux_x_kernel<<<blocks_for(n_x), kThreads>>>(U.view(), h, grid.y.device_cell_width.data(),
                                                 grid.z.device_cell_width.data(), h, ny, nz,
                                                 d + 0, nullptr);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (owns_plus(sub, 2)) {
    flux_z_kernel<<<blocks_for(n_z), kThreads>>>(W.view(), nz - h, grid.x.device_cell_width.data(),
                                                 grid.y.device_cell_width.data(), h, nx, ny, d + 1);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (owns_plus(sub, 0)) {
    flux_x_kernel<<<blocks_for(n_x), kThreads>>>(U.view(), nx - h, grid.y.device_cell_width.data(),
                                                 grid.z.device_cell_width.data(), h, ny, nz,
                                                 d + 2, d + 3);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  real_t local[4];
  MPMSTD_CUDA_CHECK(cudaMemcpy(local, d, 4 * sizeof(real_t), cudaMemcpyDeviceToHost));
  cudaFree(d);
  real_t global[4];
  MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  const real_t correction = (global[0] - global[1] - global[2]) / global[3];
  if (owns_plus(sub, 0)) {
    add_face_kernel<<<blocks_for(n_x), kThreads>>>(U.view(), nx - h, h, ny, nz, correction);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  return correction;
}

}  // namespace mpmstd::boundary
