#include "equation/momentum/system/system_variable.hpp"

#include <cstddef>

#include "equation/momentum/momentum_variable_stencil.hpp"

// CPU 백엔드 — 수식은 momentum_variable_stencil.hpp 의 variable_sweep_band 한 곳
// (GPU 쌍둥이 system_variable.cu). 밴드 레이아웃·루프 순서는 OB system.cpp 동일.
//
// z_staggered 플래그는 참조 emit 에 있는 "z 스윕 × own≠z × 벽수직 성분" 분기인데
// 우리 성분 배치(U=x, V=y, W=z)에서는 그 조합이 생기지 않아 항상 false 다
// (W 의 z 스윕은 own 분기가 담당 — dx/dmx 메트릭이 자동으로 엇갈린격자 형태).

namespace mpmstd::equation::momentum {

namespace {

struct OwnSpec {
  long long stride;
  const real_t* dx;
  const real_t* dmx;
};

OwnSpec own_spec(const int own_axis, const grid::Grid& grid,
                 const parallel::Subdomain& sub) {
  if (own_axis == 0)
    return {static_cast<long long>(sub.y().n_total) * sub.z().n_total,
            grid.x.cell_width.data(), grid.x.center_distance.data()};
  if (own_axis == 1)
    return {static_cast<long long>(sub.z().n_total), grid.y.cell_width.data(),
            grid.y.center_distance.data()};
  return {1, grid.z.cell_width.data(), grid.z.center_distance.data()};
}

}  // namespace

void build_x_sweep_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const int nsys = i2 * i3;
  const OwnSpec own = own_spec(own_axis, grid, sub);
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(i - h) * nsys + (j - h) * i3 + (k - h);
        const long long gi = stencil::vindex(i, j, k, ny_total, nz_total);
        const long long gm = stencil::vindex(i - 1, j, k, ny_total, nz_total);
        const long long gp = stencil::vindex(i + 1, j, k, ny_total, nz_total);
        const int own_index = (own_axis == 0) ? i : (own_axis == 1) ? j : k;
        stencil::variable_sweep_band(own_axis == 0, false, gi, gm, gp, i, own_index,
                                     own.stride, q, advecting_velocity, viscosity,
                                     inverse_density, grid.x.cell_width.data(),
                                     grid.x.center_distance.data(), own.dx, own.dmx,
                                     Cmu, dt, A[p], B[p], C[p]);
        D[p] = source[gi];
      }
}

void build_y_sweep_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const int nsys = i1 * i3;
  const OwnSpec own = own_spec(own_axis, grid, sub);
  for (int j = h; j < h + i2; ++j)
    for (int i = h; i < h + i1; ++i)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(j - h) * nsys + (i - h) * i3 + (k - h);
        const long long gi = stencil::vindex(i, j, k, ny_total, nz_total);
        const long long gm = stencil::vindex(i, j - 1, k, ny_total, nz_total);
        const long long gp = stencil::vindex(i, j + 1, k, ny_total, nz_total);
        const int own_index = (own_axis == 0) ? i : (own_axis == 1) ? j : k;
        stencil::variable_sweep_band(own_axis == 1, false, gi, gm, gp, j, own_index,
                                     own.stride, q, advecting_velocity, viscosity,
                                     inverse_density, grid.y.cell_width.data(),
                                     grid.y.center_distance.data(), own.dx, own.dmx,
                                     Cmu, dt, A[p], B[p], C[p]);
        D[p] = source[gi];
      }
}

void build_z_sweep_variable(real_t* A, real_t* B, real_t* C, real_t* D,
                            const real_t* source, const real_t* q,
                            const real_t* advecting_velocity, const real_t* viscosity,
                            const real_t* inverse_density, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const int own_axis,
                            const real_t Cmu, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const int nsys = i1 * i2;
  const OwnSpec own = own_spec(own_axis, grid, sub);
  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const std::size_t p =
            static_cast<std::size_t>(k - h) * nsys + (i - h) * i2 + (j - h);
        const long long gi = stencil::vindex(i, j, k, ny_total, nz_total);
        const long long gm = stencil::vindex(i, j, k - 1, ny_total, nz_total);
        const long long gp = stencil::vindex(i, j, k + 1, ny_total, nz_total);
        const int own_index = (own_axis == 0) ? i : (own_axis == 1) ? j : k;
        stencil::variable_sweep_band(own_axis == 2, false, gi, gm, gp, k, own_index,
                                     own.stride, q, advecting_velocity, viscosity,
                                     inverse_density, grid.z.cell_width.data(),
                                     grid.z.center_distance.data(), own.dx, own.dmx,
                                     Cmu, dt, A[p], B[p], C[p]);
        D[p] = source[gi];
      }
}

}  // namespace mpmstd::equation::momentum
