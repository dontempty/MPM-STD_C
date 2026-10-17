#include "equation/momentum/rhs/rhs.hpp"

#include "common/field_checks.hpp"
#include "equation/momentum/momentum_stencil.hpp"
#include "equation/pressure/rhs/rhs.hpp"
#include "physics/buoyancy.hpp"

// CPU 백엔드 — 수식은 momentum_stencil.hpp 한 곳 (GPU 쌍둥이
// rhs.cu). 세 성분이 같은 모양의 루프라 매크로 없이 그대로 쓴다.

namespace mpmstd::equation::momentum {

namespace {

template <real_t (*CellFunction)(int, int, int, int, int, const real_t*, const real_t*,
                                 const real_t*, const real_t*, const real_t*,
                                 const real_t*, const real_t*, const real_t*,
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
            grid.z.center_distance.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), Cmu, dt);
}

template <real_t (*CellFunction)(int, int, int, int, int, const real_t*, const real_t*,
                                 const real_t*, const equation::UnitArray,
                                 const equation::UnitArray, const real_t*,
                                 const real_t*, const real_t*, const real_t*,
                                 const real_t*, const real_t*, const real_t*,
                                 const real_t*, const real_t*, const real_t*,
                                 const real_t*, const real_t*, real_t, real_t)>
void add_component_constant(const char* who, field::Field& rhs_field,
                            const field::Field& U, const field::Field& V,
                            const field::Field& W, const grid::Grid& grid,
                            const parallel::Subdomain& sub, const real_t viscosity,
                            const real_t dt) {
  check_field_matches(who, rhs_field, sub);
  check_field_matches(who, U, sub);
  check_field_matches(who, V, sub);
  check_field_matches(who, W, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  real_t* rhs = rhs_field.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        rhs[stencil::vindex(i, j, k, ny_total, nz_total)] += CellFunction(
            i, j, k, ny_total, nz_total, U.data(), V.data(), W.data(),
            equation::UnitArray{}, equation::UnitArray{}, grid.x.cell_width.data(),
            grid.x.center_distance.data(), grid.y.cell_width.data(),
            grid.y.center_distance.data(), grid.z.cell_width.data(),
            grid.z.center_distance.data(), grid.x.inverse_cell_width.data(),
            grid.x.inverse_center_distance.data(), grid.y.inverse_cell_width.data(),
            grid.y.inverse_center_distance.data(), grid.z.inverse_cell_width.data(),
            grid.z.inverse_center_distance.data(), viscosity, dt);
}

}  // namespace

void add_advection_diffusion_u_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_component<stencil::rhs_u_cell>("add_advection_diffusion_u_variable_property",
                                              rhs, U, V, W, viscosity,
                                              inverse_density, grid, sub, Cmu, dt);
}

void add_advection_diffusion_v_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_component<stencil::rhs_v_cell>("add_advection_diffusion_v_variable_property",
                                              rhs, U, V, W, viscosity,
                                              inverse_density, grid, sub, Cmu, dt);
}

void add_advection_diffusion_w_variable_property(field::Field& rhs, const field::Field& U,
                                        const field::Field& V, const field::Field& W,
                                        const field::Field& viscosity,
                                        const field::Field& inverse_density,
                                        const grid::Grid& grid,
                                        const parallel::Subdomain& sub,
                                        const real_t Cmu, const real_t dt) {
  add_component<stencil::rhs_w_cell>("add_advection_diffusion_w_variable_property",
                                              rhs, U, V, W, viscosity,
                                              inverse_density, grid, sub, Cmu, dt);
}

void add_advection_diffusion_u_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_component_constant<stencil::rhs_u_cell>(
      "add_advection_diffusion_u_constant_property", rhs, U, V, W, grid, sub,
      viscosity, dt);
}

void add_advection_diffusion_v_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_component_constant<stencil::rhs_v_cell>(
      "add_advection_diffusion_v_constant_property", rhs, U, V, W, grid, sub,
      viscosity, dt);
}

void add_advection_diffusion_w_constant_property(
    field::Field& rhs, const field::Field& U, const field::Field& V,
    const field::Field& W, const grid::Grid& grid, const parallel::Subdomain& sub,
    const real_t viscosity, const real_t dt) {
  add_component_constant<stencil::rhs_w_cell>(
      "add_advection_diffusion_w_constant_property", rhs, U, V, W, grid, sub,
      viscosity, dt);
}

// ── 전용 RHS 빌드 (CPU 판): GPU 판(rhs.cu)이 한 패스로 하는 것을 기존 호출 순서 그대로 (참조 구현) ──
void build_variable_property_rhs(field::Field& rhs_u, field::Field& rhs_v, field::Field& rhs_w,
                                 const field::Field& U, const field::Field& V,
                                 const field::Field& W, const field::Field& T,
                                 const field::Field& P, const field::Field& viscosity,
                                 const field::Field& inverse_density, const grid::Grid& grid,
                                 const parallel::Subdomain& sub, const real_t Cmu,
                                 const real_t density_quadratic_ratio,
                                 const real_t delta_temperature, const real_t dt) {
  rhs_u.fill(0);
  rhs_v.fill(0);
  rhs_w.fill(0);
  add_advection_diffusion_u_variable_property(rhs_u, U, V, W, viscosity, inverse_density, grid, sub, Cmu, dt);
  add_advection_diffusion_v_variable_property(rhs_v, U, V, W, viscosity, inverse_density, grid, sub, Cmu, dt);
  add_advection_diffusion_w_variable_property(rhs_w, U, V, W, viscosity, inverse_density, grid, sub, Cmu, dt);
  physics::add_buoyancy_z_variable_property(rhs_w, T, inverse_density, grid, sub,
                                            density_quadratic_ratio, delta_temperature, dt);
  pressure::add_gradient_variable_property(rhs_u, rhs_v, rhs_w, P, inverse_density, grid, sub, dt);
}

}  // namespace mpmstd::equation::momentum
