#include "equation/momentum/system/system.hpp"

#include <cstddef>

#include <mpi.h>

#include "equation/momentum/momentum_stencil.hpp"

// CPU 백엔드 — 밴드 수식은 momentum_stencil.hpp(band_*), 여기는 채우기 루프만.
// GPU 쌍둥이 system.cu 와 같은 "셀마다 계수 재계산" 모양을 유지한다 (메모리
// 바운드라 나눗셈 몇 번은 공짜).

namespace mpmstd::equation::momentum {

namespace {

inline std::size_t at(const int i, const int j, const int k, const int ny_total,
                      const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

}  // namespace

void build_x_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const field::Field& U_advect, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t viscosity,
                    const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const int nsys = i2 * i3;
  const real_t coeff = real_t(0.5) * dt * viscosity;
  const real_t cconv = real_t(0.25) * dt;
  const field::ConstView adv = U_advect.view();
  const real_t* dx = grid.x.cell_width.data();
  const real_t* dmx = grid.x.center_distance.data();

  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k) {
        const stencil::Band band = stencil::band_x(adv, i, j, k, dx, dmx, coeff, cconv);
        const std::size_t p = static_cast<std::size_t>(i - h) * nsys
                              + (j - h) * i3 + (k - h);
        A[p] = band.a;
        B[p] = band.b;
        C[p] = band.c;
        D[p] = source[at(i, j, k, ny_total, nz_total)];
      }
}

void build_y_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const field::Field& V_advect, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t viscosity,
                    const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const int nsys = i1 * i3;
  const real_t coeff = real_t(0.5) * dt * viscosity;
  const real_t cconv = real_t(0.25) * dt;
  const field::ConstView adv = V_advect.view();
  const real_t* dy = grid.y.cell_width.data();
  const real_t* dmy = grid.y.center_distance.data();

  for (int j = h; j < h + i2; ++j)
    for (int i = h; i < h + i1; ++i)
      for (int k = h; k < h + i3; ++k) {
        const stencil::Band band = stencil::band_y(adv, i, j, k, dy, dmy, coeff, cconv);
        const std::size_t p = static_cast<std::size_t>(j - h) * nsys
                              + (i - h) * i3 + (k - h);
        A[p] = band.a;
        B[p] = band.b;
        C[p] = band.c;
        D[p] = source[at(i, j, k, ny_total, nz_total)];
      }
}

void build_z_system_cell(real_t* A, real_t* B, real_t* C, real_t* D,
                         const real_t* source, const field::Field& W_advect,
                         const grid::Grid& grid, const parallel::Subdomain& sub,
                         const real_t viscosity, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const int nsys = i1 * i2;
  const real_t coeff = real_t(0.5) * dt * viscosity;
  const real_t cconv = real_t(0.25) * dt;
  const field::ConstView adv = W_advect.view();
  const real_t* dz = grid.z.cell_width.data();
  const real_t* dmz = grid.z.center_distance.data();

  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const stencil::Band band =
            stencil::band_z_cell(adv, i, j, k, dz, dmz, coeff, cconv);
        const std::size_t p = static_cast<std::size_t>(k - h) * nsys
                              + (i - h) * i2 + (j - h);
        A[p] = band.a;
        B[p] = band.b;
        C[p] = band.c;
        D[p] = source[at(i, j, k, ny_total, nz_total)];
      }
}

void build_z_system_z_face(real_t* A, real_t* B, real_t* C, real_t* D,
                           const real_t* source, const field::Field& W_advect,
                           const grid::Grid& grid, const parallel::Subdomain& sub,
                           const real_t viscosity, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const int nsys = i1 * i2;
  const real_t coeff = real_t(0.5) * dt * viscosity;
  const real_t cconv = real_t(0.25) * dt;
  const field::ConstView adv = W_advect.view();
  const real_t* dz = grid.z.cell_width.data();
  const real_t* dmz = grid.z.center_distance.data();

  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const stencil::Band band =
            stencil::band_z_face(adv, i, j, k, dz, dmz, coeff, cconv);
        const std::size_t p = static_cast<std::size_t>(k - h) * nsys
                              + (i - h) * i2 + (j - h);
        A[p] = band.a;
        B[p] = band.b;
        C[p] = band.c;
        D[p] = source[at(i, j, k, ny_total, nz_total)];
      }
}

void fold_z_wall_value(real_t* A, real_t* B, real_t* C,
                       const parallel::Subdomain& sub, const int nsys,
                       const int nrow) {
  (void)B;  // ZeroGhost: 비대각만 제거 — B 를 보상하면 벽 확산이 사라진다 (헤더 ⚠)
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL)
    for (int s = 0; s < nsys; ++s) A[s] = 0;
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    const std::size_t top = static_cast<std::size_t>(nrow - 1) * nsys;
    for (int s = 0; s < nsys; ++s) C[top + s] = 0;
  }
}

void pin_z_bottom_wall_face_row(real_t* A, real_t* B, real_t* C, real_t* D,
                                const parallel::Subdomain& sub, const int nsys) {
  if (sub.topology().z().minus_neighbor != MPI_PROC_NULL) return;
  for (int s = 0; s < nsys; ++s) {
    A[s] = 0;
    B[s] = 1;
    C[s] = 0;
    D[s] = 0;
  }
}

}  // namespace mpmstd::equation::momentum
