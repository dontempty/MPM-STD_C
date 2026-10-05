#include "equation/momentum/wall_rows.hpp"

#include <cstddef>

#include <mpi.h>

// CPU 백엔드 — GPU 쌍둥이 wall_rows.cu. 수식 없음(행 마스킹만).

namespace mpmstd::equation::momentum {

void fold_wall_value(real_t* A, real_t* B, real_t* C, const parallel::Subdomain& sub,
                     const int axis, const int nsys, const int nrow) {
  (void)B;  // ZeroGhost: 비대각만 제거 — B 를 보상하면 벽 확산이 사라진다 (헤더 ⚠)
  if (sub.topology().axis(axis).minus_neighbor == MPI_PROC_NULL)
    for (int s = 0; s < nsys; ++s) A[s] = 0;
  if (sub.topology().axis(axis).plus_neighbor == MPI_PROC_NULL) {
    const std::size_t top = static_cast<std::size_t>(nrow - 1) * nsys;
    for (int s = 0; s < nsys; ++s) C[top + s] = 0;
  }
}

void pin_minus_wall_face_row(real_t* A, real_t* B, real_t* C, real_t* D,
                             const parallel::Subdomain& sub, const int axis,
                             const int nsys) {
  if (sub.topology().axis(axis).minus_neighbor != MPI_PROC_NULL) return;
  for (int s = 0; s < nsys; ++s) {
    A[s] = 0;
    B[s] = 1;
    C[s] = 0;
    D[s] = 0;
  }
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
