#include "equation/scalar/system/system_variable.hpp"

#include <cstddef>

#include <mpi.h>

#include "equation/scalar/scalar_variable_stencil.hpp"

// CPU 백엔드 — 수식은 scalar_variable_stencil.hpp 한 곳 (GPU 쌍둥이
// system_variable.cu). 밴드 레이아웃·루프 순서는 OB system.cpp 와 동일.

namespace mpmstd::equation::scalar {

namespace {

struct WallOwnership {
  bool lower, upper;
};

WallOwnership z_wall_ownership(const parallel::Subdomain& sub) {
  return {sub.topology().z().minus_neighbor == MPI_PROC_NULL,
          sub.topology().z().plus_neighbor == MPI_PROC_NULL};
}

}  // namespace

void build_x_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* T, const real_t* U, const real_t* V,
                             const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t Ct, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i2 * i3;
  const WallOwnership wall = z_wall_ownership(sub);
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(i - h) * nsys + (j - h) * i3 + (k - h);
        stencil::build_variable_cell(
            0, i, j, k, sub.y().n_total, sub.z().n_total, nullptr, T, U, V, W,
            properties.conductivity.data(), properties.conductivity_derivative.data(),
            properties.inverse_heat_capacity.data(),
            properties.inverse_heat_capacity_derivative.data(),
            grid.x.cell_width.data(), grid.x.center_distance.data(),
            grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), Ct, dt,
            /*is_first_sweep=*/true, wall.lower && k == h, wall.upper && k == h + i3 - 1,
            A[p], B[p], C[p], D[p]);
      }
}

void build_y_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t Ct, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i3;
  const WallOwnership wall = z_wall_ownership(sub);
  for (int j = h; j < h + i2; ++j)
    for (int i = h; i < h + i1; ++i)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(j - h) * nsys + (i - h) * i3 + (k - h);
        stencil::build_variable_cell(
            1, i, j, k, sub.y().n_total, sub.z().n_total, source, T, U, V, W,
            properties.conductivity.data(), properties.conductivity_derivative.data(),
            properties.inverse_heat_capacity.data(),
            properties.inverse_heat_capacity_derivative.data(),
            grid.x.cell_width.data(), grid.x.center_distance.data(),
            grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lower && k == h, wall.upper && k == h + i3 - 1,
            A[p], B[p], C[p], D[p]);
      }
}

void build_z_system_variable_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const physics::PropertyFields& properties,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t Ct, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i2;
  const WallOwnership wall = z_wall_ownership(sub);
  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const std::size_t p =
            static_cast<std::size_t>(k - h) * nsys + (i - h) * i2 + (j - h);
        stencil::build_variable_cell(
            2, i, j, k, sub.y().n_total, sub.z().n_total, source, T, U, V, W,
            properties.conductivity.data(), properties.conductivity_derivative.data(),
            properties.inverse_heat_capacity.data(),
            properties.inverse_heat_capacity_derivative.data(),
            grid.x.cell_width.data(), grid.x.center_distance.data(),
            grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lower && k == h, wall.upper && k == h + i3 - 1,
            A[p], B[p], C[p], D[p]);
      }
}

// ── 상수물성 판 (plan §18-1): κ̂·1/(ρ̂Ĉp)≡1, 미분장≡0 ───────────────────────

void build_x_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* T, const real_t* U, const real_t* V,
                             const real_t* W,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t Ct, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i2 * i3;
  const WallOwnership wall = z_wall_ownership(sub);
  for (int i = h; i < h + i1; ++i)
    for (int j = h; j < h + i2; ++j)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(i - h) * nsys + (j - h) * i3 + (k - h);
        stencil::build_variable_cell(
            0, i, j, k, sub.y().n_total, sub.z().n_total, nullptr, T, U, V, W,
            equation::UnitArray{}, equation::ZeroArray{},
            equation::UnitArray{}, equation::ZeroArray{},
            grid.x.cell_width.data(), grid.x.center_distance.data(),
            grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), Ct, dt,
            /*is_first_sweep=*/true, wall.lower && k == h, wall.upper && k == h + i3 - 1,
            A[p], B[p], C[p], D[p]);
      }
}

void build_y_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t Ct, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i3;
  const WallOwnership wall = z_wall_ownership(sub);
  for (int j = h; j < h + i2; ++j)
    for (int i = h; i < h + i1; ++i)
      for (int k = h; k < h + i3; ++k) {
        const std::size_t p =
            static_cast<std::size_t>(j - h) * nsys + (i - h) * i3 + (k - h);
        stencil::build_variable_cell(
            1, i, j, k, sub.y().n_total, sub.z().n_total, source, T, U, V, W,
            equation::UnitArray{}, equation::ZeroArray{},
            equation::UnitArray{}, equation::ZeroArray{},
            grid.x.cell_width.data(), grid.x.center_distance.data(),
            grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lower && k == h, wall.upper && k == h + i3 - 1,
            A[p], B[p], C[p], D[p]);
      }
}

void build_z_system_constant_property(real_t* A, real_t* B, real_t* C, real_t* D,
                             const real_t* source, const real_t* T, const real_t* U,
                             const real_t* V, const real_t* W,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t Ct, const real_t dt) {
  const int h = halo_width;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const int nsys = i1 * i2;
  const WallOwnership wall = z_wall_ownership(sub);
  for (int k = h; k < h + i3; ++k)
    for (int i = h; i < h + i1; ++i)
      for (int j = h; j < h + i2; ++j) {
        const std::size_t p =
            static_cast<std::size_t>(k - h) * nsys + (i - h) * i2 + (j - h);
        stencil::build_variable_cell(
            2, i, j, k, sub.y().n_total, sub.z().n_total, source, T, U, V, W,
            equation::UnitArray{}, equation::ZeroArray{},
            equation::UnitArray{}, equation::ZeroArray{},
            grid.x.cell_width.data(), grid.x.center_distance.data(),
            grid.y.cell_width.data(), grid.y.center_distance.data(),
            grid.z.cell_width.data(), grid.z.center_distance.data(), Ct, dt,
            /*is_first_sweep=*/false, wall.lower && k == h, wall.upper && k == h + i3 - 1,
            A[p], B[p], C[p], D[p]);
      }
}

}  // namespace mpmstd::equation::scalar
