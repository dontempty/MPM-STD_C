#include "equation/scalar/legacy_ob/system.hpp"

#include <cstddef>

#include <mpi.h>

#include "equation/scalar/legacy_ob/scalar_stencil.hpp"

// CPU 백엔드 — 밴드 식은 scalar_stencil.hpp(diffusion_band), 여기는 채우기 루프만
// (GPU 쌍둥이 system.cu). 계수는 축 1차원이지만 셀당 재계산으로 커널과 모양 통일.

namespace mpmstd::equation::scalar {

namespace {

inline std::size_t at(const int i, const int j, const int k, const int ny_total,
                      const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

}  // namespace

void build_x_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t alpha, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i2 * i3;
  const real_t half_dt_alpha = real_t(0.5) * dt * alpha;
  const real_t* dx = grid.x.cell_width.data();
  const real_t* dmx = grid.x.center_distance.data();
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k) {
        const stencil::Band band =
            stencil::diffusion_band(half_dt_alpha, dx[i], dmx[i], dmx[i + 1]);
        const std::size_t p = static_cast<std::size_t>(i - h) * nsys
                              + (j - h) * i3 + (k - h);
        A[p] = band.a;
        B[p] = band.b;
        C[p] = band.c;
        D[p] = source[at(i, j, k, sub.y().n_total, sub.z().n_total)];
      }
}

void build_y_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t alpha, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i3;
  const real_t half_dt_alpha = real_t(0.5) * dt * alpha;
  const real_t* dy = grid.y.cell_width.data();
  const real_t* dmy = grid.y.center_distance.data();
  for (int j = h; j < h + i2; ++j)
    for (int i = h; i < h + i1; ++i)
      for (int k = h; k < h + i3; ++k) {
        const stencil::Band band =
            stencil::diffusion_band(half_dt_alpha, dy[j], dmy[j], dmy[j + 1]);
        const std::size_t p = static_cast<std::size_t>(j - h) * nsys
                              + (i - h) * i3 + (k - h);
        A[p] = band.a;
        B[p] = band.b;
        C[p] = band.c;
        D[p] = source[at(i, j, k, sub.y().n_total, sub.z().n_total)];
      }
}

void build_z_system(real_t* A, real_t* B, real_t* C, real_t* D, const real_t* source,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t alpha, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i2;
  const real_t half_dt_alpha = real_t(0.5) * dt * alpha;
  const real_t* dz = grid.z.cell_width.data();
  const real_t* dmz = grid.z.center_distance.data();
  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const stencil::Band band =
            stencil::diffusion_band(half_dt_alpha, dz[k], dmz[k], dmz[k + 1]);
        const std::size_t p = static_cast<std::size_t>(k - h) * nsys
                              + (i - h) * i2 + (j - h);
        A[p] = band.a;
        B[p] = band.b;
        C[p] = band.c;
        D[p] = source[at(i, j, k, sub.y().n_total, sub.z().n_total)];
      }
}

void fold_z_wall_mirror(real_t* A, real_t* B, real_t* C,
                        const parallel::Subdomain& sub, const int nsys,
                        const int nrow) {
  // 하단 벽: 행 0 의 A 가 ghost(δg = −δf₁)를 가리킨다 → B −= A 로 소거.
  if (sub.topology().z().minus_neighbor == MPI_PROC_NULL) {
    for (int s = 0; s < nsys; ++s) {
      B[s] -= A[s];
      A[s] = 0;
    }
  }
  // 상단 벽: 끝 행의 C 가 ghost 를 가리킨다 → B −= C.
  if (sub.topology().z().plus_neighbor == MPI_PROC_NULL) {
    const std::size_t top = static_cast<std::size_t>(nrow - 1) * nsys;
    for (int s = 0; s < nsys; ++s) {
      B[top + s] -= C[top + s];
      C[top + s] = 0;
    }
  }
}

}  // namespace mpmstd::equation::scalar
