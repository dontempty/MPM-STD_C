#include "boundary/wall.hpp"

#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"

// GPU 백엔드 — wall.cpp 의 쌍둥이. ghost 공식은 한 줄짜리라 양쪽에 그대로 쓰고
// 상호 참조 주석으로 짝을 맞춘다 (wall.hpp 의 공식 표가 정본).

namespace mpmstd::boundary {

namespace {

enum : int { kThreads = 256 };

void check_dims(const char* who, const field::Field& f, const parallel::Subdomain& sub) {
  if (f.nx_total() != sub.x().n_total || f.ny_total() != sub.y().n_total
      || f.nz_total() != sub.z().n_total)
    throw std::invalid_argument(std::string(who) + ": field '" + f.name()
                                + "' does not match the subdomain");
}

__global__ void mirror_kernel(const field::View v, const int nx, const int ny,
                              const int ghost_k, const int interior_k,
                              const real_t value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nx * ny) return;
  const int i = idx / ny, j = idx % ny;
  v(i, j, ghost_k) = 2 * value - v(i, j, interior_k);  // = wall.cpp 의 mirror
}

__global__ void value_kernel(const field::View v, const int nx, const int ny,
                             const int ghost_k, const real_t value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nx * ny) return;
  v(idx / ny, idx % ny, ghost_k) = value;
}

__global__ void copy_kernel(const field::View v, const int nx, const int ny,
                            const int ghost_k, const int interior_k) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nx * ny) return;
  const int i = idx / ny, j = idx % ny;
  v(i, j, ghost_k) = v(i, j, interior_k);  // zero-gradient
}

inline int blocks_for(const int n) { return (n + kThreads - 1) / kThreads; }

}  // namespace

void fill_z_wall_mirror(field::Field& f, const parallel::Subdomain& sub,
                        real_t value_bottom, real_t value_top) {
  check_dims("fill_z_wall_mirror", f, sub);
  const int nx = f.nx_total(), ny = f.ny_total(), nz = f.nz_total();
  const int h = halo_width;
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    mirror_kernel<<<blocks_for(nx * ny), kThreads>>>(f.view(), nx, ny, 0, h,
                                                     value_bottom);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    mirror_kernel<<<blocks_for(nx * ny), kThreads>>>(f.view(), nx, ny, nz - 1,
                                                     nz - 1 - h, value_top);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
}

void fill_z_wall_value(field::Field& f, const parallel::Subdomain& sub,
                       real_t value_bottom, real_t value_top) {
  check_dims("fill_z_wall_value", f, sub);
  const int nx = f.nx_total(), ny = f.ny_total(), nz = f.nz_total();
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    value_kernel<<<blocks_for(nx * ny), kThreads>>>(f.view(), nx, ny, 0, value_bottom);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    value_kernel<<<blocks_for(nx * ny), kThreads>>>(f.view(), nx, ny, nz - 1,
                                                    value_top);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
}

void fill_z_wall_zero_gradient(field::Field& f, const parallel::Subdomain& sub) {
  check_dims("fill_z_wall_zero_gradient", f, sub);
  const int nx = f.nx_total(), ny = f.ny_total(), nz = f.nz_total();
  const int h = halo_width;
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    copy_kernel<<<blocks_for(nx * ny), kThreads>>>(f.view(), nx, ny, 0, h);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    copy_kernel<<<blocks_for(nx * ny), kThreads>>>(f.view(), nx, ny, nz - 1,
                                                   nz - 1 - h);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
}

void zero_z_wall_faces(field::Field& w, const parallel::Subdomain& sub) {
  check_dims("zero_z_wall_faces", w, sub);
  const int nx = w.nx_total(), ny = w.ny_total(), nz = w.nz_total();
  const int h = halo_width;
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    value_kernel<<<blocks_for(nx * ny), kThreads>>>(w.view(), nx, ny, h, 0);  // 내부 슬롯!
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    value_kernel<<<blocks_for(nx * ny), kThreads>>>(w.view(), nx, ny, nz - h, 0);  // ghost 슬롯
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
}

}  // namespace mpmstd::boundary
