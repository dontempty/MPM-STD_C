#include "equation/momentum/rhs/rhs_variable.hpp"

#include "common/field_checks.hpp"
#include "equation/momentum/momentum_variable_stencil.hpp"

// CPU 백엔드 — 수식은 momentum_variable_stencil.hpp 한 곳 (GPU 쌍둥이
// rhs_variable.cu). 세 성분이 같은 모양의 루프라 매크로 없이 그대로 쓴다.

namespace mpmstd::equation::momentum {

namespace {

template <real_t (*CellFunction)(int, int, int, int, int, const real_t*, const real_t*,
                                 const real_t*, const real_t*, const real_t*,
                                 const real_t*, const real_t*, const real_t*,
                                 const real_t*, const real_t*, const real_t*, real_t,
                                 real_t)>
void add_component(const char* who, field::Field& rhs_field, const field::Field& U,
                   const field::Field& V, const field::Field& W,
                   const field::Field& viscosity, const field::Field& inverse_density,
                   const grid::Grid& grid, const parallel::Subdomain& sub,
                   const real_t Cmu, const real_t dt) {
  check_field_matches(who, rhs_field, sub);
  check_field_matches(who, U, sub);
  check_field_matches(who, V, sub);
  check_field_matches(who, W, sub);
  check_field_matches(who, viscosity, sub);
  check_field_matches(who, inverse_density, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  real_t* rhs = rhs_field.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        rhs[stencil::vindex(i, j, k, ny_total, nz_total)] += CellFunction(
            i, j, k, ny_total, nz_total, U.data(), V.data(), W.data(),
            viscosity.data(), inverse_density.data(), grid.x.cell_width.data(),
            grid.x.center_distance.data(), grid.y.cell_width.data(),
            grid.y.center_distance.data(), grid.z.cell_width.data(),
            grid.z.center_distance.data(), Cmu, dt);
}

}  // namespace

void add_advection_diffusion_u_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_component<stencil::rhs_u_variable_cell>("add_advection_diffusion_u_variable",
                                              rhs, U, V, W, viscosity,
                                              inverse_density, grid, sub, Cmu, dt);
}

void add_advection_diffusion_v_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_component<stencil::rhs_v_variable_cell>("add_advection_diffusion_v_variable",
                                              rhs, U, V, W, viscosity,
                                              inverse_density, grid, sub, Cmu, dt);
}

void add_advection_diffusion_w_variable(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_component<stencil::rhs_w_variable_cell>("add_advection_diffusion_w_variable",
                                              rhs, U, V, W, viscosity,
                                              inverse_density, grid, sub, Cmu, dt);
}

}  // namespace mpmstd::equation::momentum
