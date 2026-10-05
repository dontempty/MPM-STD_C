#include "equation/momentum/solve_kernels_variable.hpp"

#include "common/field_checks.hpp"
#include "equation/momentum/momentum_variable_stencil.hpp"

// CPU 백엔드 — 수식은 momentum_variable_stencil.hpp 한 곳 (GPU 쌍둥이
// solve_kernels_variable.cu).

namespace mpmstd::equation::momentum {

void couple_rhs_v_from_delta_u_variable(field::Field& rhs_v,
                                        const field::Field& delta_u,
                                        const field::Field& V,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  check_field_matches("couple_rhs_v_from_delta_u_variable", rhs_v, sub);
  check_field_matches("couple_rhs_v_from_delta_u_variable", delta_u, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  real_t* rhs = rhs_v.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        rhs[stencil::vindex(i, j, k, ny_total, nz_total)] -=
            stencil::couple_rhs_v_from_delta_u_cell(
                i, j, k, ny_total, nz_total, delta_u.data(), V.data(),
                viscosity.data(), inverse_density.data(), grid.x.cell_width.data(),
                grid.x.center_distance.data(), grid.y.cell_width.data(),
                grid.y.center_distance.data(), Cmu, dt);
}

void couple_rhs_w_from_delta_uv_variable(field::Field& rhs_w,
                                         const field::Field& delta_u,
                                         const field::Field& delta_v,
                                         const field::Field& W,
                                         const field::Field& viscosity,
                                         const field::Field& inverse_density,
                                         const grid::Grid& grid,
                                         const parallel::Subdomain& sub,
                                         const real_t Cmu, const real_t dt) {
  check_field_matches("couple_rhs_w_from_delta_uv_variable", rhs_w, sub);
  check_field_matches("couple_rhs_w_from_delta_uv_variable", delta_u, sub);
  check_field_matches("couple_rhs_w_from_delta_uv_variable", delta_v, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  real_t* rhs = rhs_w.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        rhs[stencil::vindex(i, j, k, ny_total, nz_total)] -=
            stencil::couple_rhs_w_from_delta_uv_cell(
                i, j, k, ny_total, nz_total, delta_u.data(), delta_v.data(), W.data(),
                viscosity.data(), inverse_density.data(), grid.x.cell_width.data(),
                grid.x.center_distance.data(), grid.y.cell_width.data(),
                grid.y.center_distance.data(), grid.z.cell_width.data(),
                grid.z.center_distance.data(), Cmu, dt);
}

void block_couple_delta_v_variable(field::Field& delta_v, const field::Field& delta_w,
                                   const field::Field& V,
                                   const field::Field& viscosity,
                                   const field::Field& inverse_density,
                                   const grid::Grid& grid,
                                   const parallel::Subdomain& sub, const real_t Cmu,
                                   const real_t dt) {
  check_field_matches("block_couple_delta_v_variable", delta_v, sub);
  check_field_matches("block_couple_delta_v_variable", delta_w, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  real_t* dv = delta_v.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        dv[stencil::vindex(i, j, k, ny_total, nz_total)] -=
            stencil::couple_delta_v_from_delta_w_cell(
                i, j, k, ny_total, nz_total, delta_w.data(), V.data(),
                viscosity.data(), inverse_density.data(), grid.y.cell_width.data(),
                grid.y.center_distance.data(), grid.z.cell_width.data(),
                grid.z.center_distance.data(), Cmu, dt);
}

void block_couple_delta_u_variable(field::Field& delta_u, const field::Field& delta_v,
                                   const field::Field& delta_w, const field::Field& U,
                                   const field::Field& viscosity,
                                   const field::Field& inverse_density,
                                   const grid::Grid& grid,
                                   const parallel::Subdomain& sub, const real_t Cmu,
                                   const real_t dt) {
  check_field_matches("block_couple_delta_u_variable", delta_u, sub);
  check_field_matches("block_couple_delta_u_variable", delta_v, sub);
  check_field_matches("block_couple_delta_u_variable", delta_w, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  real_t* du = delta_u.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        du[stencil::vindex(i, j, k, ny_total, nz_total)] -=
            stencil::couple_delta_u_from_delta_vw_cell(
                i, j, k, ny_total, nz_total, delta_v.data(), delta_w.data(), U.data(),
                viscosity.data(), inverse_density.data(), grid.x.cell_width.data(),
                grid.x.center_distance.data(), grid.y.cell_width.data(),
                grid.y.center_distance.data(), grid.z.cell_width.data(),
                grid.z.center_distance.data(), Cmu, dt);
}

}  // namespace mpmstd::equation::momentum
