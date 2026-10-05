#include "equation/momentum/wall_rows.hpp"

#include <mpi.h>

#include "common/cuda_check.hpp"

// GPU 백엔드 — wall_rows.cpp 의 쌍둥이.

namespace mpmstd::equation::momentum {

namespace {

enum : int { kThreads = 256 };
inline int blocks_for(const long long n) {
  return static_cast<int>((n + kThreads - 1) / kThreads);
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

}  // namespace

void fold_wall_value(real_t* A, real_t* B, real_t* C, const parallel::Subdomain& sub,
                     const int axis, const int nsys, const int nrow) {
  (void)B;  // ZeroGhost: 비대각만 제거 — B 유지 (system.hpp ⚠)
  if (sub.topology().axis(axis).minus_neighbor == MPI_PROC_NULL) {
    fold_bottom_kernel<<<blocks_for(nsys), kThreads>>>(A, nsys);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
  if (sub.topology().axis(axis).plus_neighbor == MPI_PROC_NULL) {
    fold_top_kernel<<<blocks_for(nsys), kThreads>>>(
        C, nsys, static_cast<long long>(nrow - 1) * nsys);
    MPMSTD_CUDA_CHECK_LAUNCH();
  }
}

void pin_minus_wall_face_row(real_t* A, real_t* B, real_t* C, real_t* D,
                             const parallel::Subdomain& sub, const int axis,
                             const int nsys) {
  if (sub.topology().axis(axis).minus_neighbor != MPI_PROC_NULL) return;
  pin_kernel<<<blocks_for(nsys), kThreads>>>(A, B, C, D, nsys);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void fold_z_wall_value(real_t* A, real_t* B, real_t* C,
                       const parallel::Subdomain& sub, const int nsys,
                       const int nrow) {
  fold_wall_value(A, B, C, sub, 2, nsys, nrow);
}

void pin_z_bottom_wall_face_row(real_t* A, real_t* B, real_t* C, real_t* D,
                                const parallel::Subdomain& sub, const int nsys) {
  pin_minus_wall_face_row(A, B, C, D, sub, 2, nsys);
}

}  // namespace mpmstd::equation::momentum
