#include "equation/scalar/system/system.hpp"

#include "equation/band_layout.hpp"

#include <cstddef>

#include <mpi.h>

#include "equation/scalar/scalar_stencil.hpp"

// CPU 백엔드 — 수식은 scalar_stencil.hpp 한 곳 (GPU 쌍둥이
// system.cu). 밴드 레이아웃·루프 순서는 OB system.cpp 와 동일.

namespace mpmstd::equation::scalar {

namespace {

}  // namespace

void build_x_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* T, const real_t* U, const real_t* V,
                             const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i2 * i3;
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(i - h) * nsys + (j - h) * i3 + (k - h);
        stencil::build_cell(
            0, i, j, k, sub.y().n_total, sub.z().n_total, nullptr, T, U, V, W,
            properties.conductivity.data(), properties.conductivity_derivative.data(),
            properties.inverse_heat_capacity.data(),
            properties.inverse_heat_capacity_derivative.data(),
            grid.x.cell_width.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(),
            grid.y.cell_width.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(),
            grid.z.cell_width.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), Ct, dt,
            /*is_first_sweep=*/true, wall.lo[0] && i == h, wall.hi[0] && i == h + i1 - 1,
            wall.lo[1] && j == h, wall.hi[1] && j == h + i2 - 1,
            wall.lo[2] && k == h, wall.hi[2] && k == h + i3 - 1, wall,
            A[p], B[p], C[p], D[p]);
      }
}

void build_y_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt, const SourceLayout source_layout) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i3;
  for (int j = h; j < h + i2; ++j)
    for (int i = h; i < h + i1; ++i)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(j - h) * nsys + (i - h) * i3 + (k - h);
        stencil::build_cell(
            1, i, j, k, sub.y().n_total, sub.z().n_total, source,
            source_index(source_layout, i - h, j - h, k - h, i1, i2, i3,
                         static_cast<std::size_t>(stencil::cell_index(i, j, k, sub.y().n_total, sub.z().n_total))),
            T, U, V, W,
            properties.conductivity.data(), properties.conductivity_derivative.data(),
            properties.inverse_heat_capacity.data(),
            properties.inverse_heat_capacity_derivative.data(),
            grid.x.cell_width.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(),
            grid.y.cell_width.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(),
            grid.z.cell_width.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lo[0] && i == h, wall.hi[0] && i == h + i1 - 1,
            wall.lo[1] && j == h, wall.hi[1] && j == h + i2 - 1,
            wall.lo[2] && k == h, wall.hi[2] && k == h + i3 - 1, wall,
            A[p], B[p], C[p], D[p]);
      }
}

void build_z_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt, const SourceLayout source_layout) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i2;
  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const std::size_t p =
            static_cast<std::size_t>(k - h) * nsys + (i - h) * i2 + (j - h);
        stencil::build_cell(
            2, i, j, k, sub.y().n_total, sub.z().n_total, source,
            source_index(source_layout, i - h, j - h, k - h, i1, i2, i3,
                         static_cast<std::size_t>(stencil::cell_index(i, j, k, sub.y().n_total, sub.z().n_total))),
            T, U, V, W,
            properties.conductivity.data(), properties.conductivity_derivative.data(),
            properties.inverse_heat_capacity.data(),
            properties.inverse_heat_capacity_derivative.data(),
            grid.x.cell_width.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(),
            grid.y.cell_width.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(),
            grid.z.cell_width.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lo[0] && i == h, wall.hi[0] && i == h + i1 - 1,
            wall.lo[1] && j == h, wall.hi[1] && j == h + i2 - 1,
            wall.lo[2] && k == h, wall.hi[2] && k == h + i3 - 1, wall,
            A[p], B[p], C[p], D[p]);
      }
}

// ── 상수물성 판 (plan §18-1): κ̂·1/(ρ̂Ĉp)≡1, 미분장≡0 ───────────────────────

void build_x_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* T, const real_t* U, const real_t* V,
                             const real_t* W,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i2 * i3;
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(i - h) * nsys + (j - h) * i3 + (k - h);
        stencil::build_cell(
            0, i, j, k, sub.y().n_total, sub.z().n_total, nullptr, T, U, V, W,
            equation::UnitArray{}, equation::ZeroArray{},
            equation::UnitArray{}, equation::ZeroArray{},
            grid.x.cell_width.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(),
            grid.y.cell_width.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(),
            grid.z.cell_width.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), Ct, dt,
            /*is_first_sweep=*/true, wall.lo[0] && i == h, wall.hi[0] && i == h + i1 - 1,
            wall.lo[1] && j == h, wall.hi[1] && j == h + i2 - 1,
            wall.lo[2] && k == h, wall.hi[2] && k == h + i3 - 1, wall,
            A[p], B[p], C[p], D[p]);
      }
}

void build_y_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt, const SourceLayout source_layout) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i3;
  for (int j = h; j < h + i2; ++j)
    for (int i = h; i < h + i1; ++i)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(j - h) * nsys + (i - h) * i3 + (k - h);
        stencil::build_cell(
            1, i, j, k, sub.y().n_total, sub.z().n_total, source,
            source_index(source_layout, i - h, j - h, k - h, i1, i2, i3,
                         static_cast<std::size_t>(stencil::cell_index(i, j, k, sub.y().n_total, sub.z().n_total))),
            T, U, V, W,
            equation::UnitArray{}, equation::ZeroArray{},
            equation::UnitArray{}, equation::ZeroArray{},
            grid.x.cell_width.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(),
            grid.y.cell_width.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(),
            grid.z.cell_width.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lo[0] && i == h, wall.hi[0] && i == h + i1 - 1,
            wall.lo[1] && j == h, wall.hi[1] && j == h + i2 - 1,
            wall.lo[2] && k == h, wall.hi[2] && k == h + i3 - 1, wall,
            A[p], B[p], C[p], D[p]);
      }
}

void build_z_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const stencil::WallMask& wall, const real_t Ct,
                             const real_t dt, const SourceLayout source_layout) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i2;
  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const std::size_t p =
            static_cast<std::size_t>(k - h) * nsys + (i - h) * i2 + (j - h);
        stencil::build_cell(
            2, i, j, k, sub.y().n_total, sub.z().n_total, source,
            source_index(source_layout, i - h, j - h, k - h, i1, i2, i3,
                         static_cast<std::size_t>(stencil::cell_index(i, j, k, sub.y().n_total, sub.z().n_total))),
            T, U, V, W,
            equation::UnitArray{}, equation::ZeroArray{},
            equation::UnitArray{}, equation::ZeroArray{},
            grid.x.cell_width.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(),
            grid.y.cell_width.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(),
            grid.z.cell_width.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lo[0] && i == h, wall.hi[0] && i == h + i1 - 1,
            wall.lo[1] && j == h, wall.hi[1] && j == h + i2 - 1,
            wall.lo[2] && k == h, wall.hi[2] && k == h + i3 - 1, wall,
            A[p], B[p], C[p], D[p]);
      }
}

}  // namespace mpmstd::equation::scalar
