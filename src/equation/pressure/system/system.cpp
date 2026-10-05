#include "equation/pressure/system/system.hpp"

#include <cmath>
#include <cstddef>

#include <mpi.h>

#include "equation/pressure/pressure_stencil.hpp"

namespace mpmstd::equation::pressure {

namespace {

const real_t two_pi = 2 * std::acos(real_t(-1));

// (1 − cos(2π·m/N)) · 2/d²  — 균일 간격 d 의 2차 중심차분 고유값.
// (MPM-STD_C 는 1/(dx·dmx_i)+1/(dx·dmx_{i+1}) 의 내부 평균을 쓰지만 균일격자에선
//  정확히 2/d² 와 같다. 균일성은 Solver 생성자가 검사한다.)
real_t eigenvalue(const int mode, const int n, const real_t spacing) {
  return (1 - std::cos(two_pi * mode / n)) * 2 / (spacing * spacing);
}

}  // namespace

std::vector<real_t> make_lambda_x(const grid::Grid& grid,
                                  const parallel::Subdomain& sub) {
  const int n = sub.x().n_global;
  const real_t dx = grid.x.cell_width[halo_width];
  std::vector<real_t> lambda(n / 2 + 1);  // r2c 절반 스펙트럼
  for (int m = 0; m < static_cast<int>(lambda.size()); ++m)
    lambda[m] = eigenvalue(m, n, dx);
  return lambda;
}

std::vector<real_t> make_lambda_y(const grid::Grid& grid,
                                  const parallel::Subdomain& sub) {
  const int n = sub.y().n_global;
  const real_t dy = grid.y.cell_width[halo_width];
  std::vector<real_t> lambda(n);  // c2c 전체 스펙트럼 (cos 이 대칭이라 부호 무관)
  for (int m = 0; m < n; ++m) lambda[m] = eigenvalue(m, n, dy);
  return lambda;
}

// CPU 백엔드 — 밴드 식은 pressure_stencil.hpp(z_mode_band), 여기는 루프만.
void build_z_modes_system(real_t* A, real_t* B, real_t* C, const grid::Grid& grid,
                          const parallel::Subdomain& sub, const real_t* lambda_x,
                          const int n_mode_x, const real_t* lambda_y,
                          const int n_mode_y) {
  const int h = halo_width;
  const int nz_local = sub.z().n_local;
  const int nsys = 2 * n_mode_x * n_mode_y;
  const bool owns_bottom = sub.topology().z().minus_neighbor == MPI_PROC_NULL;
  const bool owns_top = sub.topology().z().plus_neighbor == MPI_PROC_NULL;
  const real_t* dz = grid.z.cell_width.data();
  const real_t* dmz = grid.z.center_distance.data();

  for (int k = h; k < h + nz_local; ++k) {
    const int row = k - h;
    for (int mx = 0; mx < n_mode_x; ++mx) {
      for (int jy = 0; jy < n_mode_y; ++jy) {
        stencil::Band band = stencil::z_mode_band(
            dz[k], dmz[k], dmz[k + 1], lambda_x[mx] + lambda_y[jy],
            owns_bottom && row == 0, owns_top && row == nz_local - 1);
        // (0,0) 모드(λ=0, 순수 Neumann)는 특이 — 전역 z 첫 행을 항등행으로 핀.
        // 우변 0 은 solve 의 D 팩이, 상수 게이지는 평균 제거가 맡는다.
        if (owns_bottom && row == 0 && mx == 0 && jy == 0) band = {0, 1, 0};
        const std::size_t p = static_cast<std::size_t>(row) * nsys
                              + 2 * (static_cast<std::size_t>(mx) * n_mode_y + jy);
        A[p] = band.a;      // 실수부
        B[p] = band.b;
        C[p] = band.c;
        A[p + 1] = band.a;  // 허수부 — 같은 계수
        B[p + 1] = band.b;
        C[p + 1] = band.c;
      }
    }
  }
}

}  // namespace mpmstd::equation::pressure
