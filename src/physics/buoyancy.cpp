#include "physics/buoyancy.hpp"

#include "common/field_checks.hpp"
#include "physics/physics_stencil.hpp"

// CPU 백엔드 (GPU 쌍둥이 buoyancy.cu; 보간식은 physics_stencil.hpp 공용).

namespace mpmstd::physics {

void add_buoyancy_z_constant_property(field::Field& rhs_w_field, const field::Field& T_field,
                    const grid::Grid& grid, const parallel::Subdomain& sub,
                    const real_t dt) {
  check_field_matches("add_buoyancy_z_constant_property", rhs_w_field, sub);
  check_field_matches("add_buoyancy_z_constant_property", T_field, sub);
  const int h = halo_width;
  const field::View rhs_w = rhs_w_field.view();
  const field::ConstView T = T_field.view();
  const real_t* dz = grid.z.cell_width.data();
  const real_t* dmz = grid.z.center_distance.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        rhs_w(i, j, k) += dt * stencil::temperature_at_z_face(T, i, j, k, dz, dmz);
}

void add_buoyancy_z_variable_property(field::Field& rhs_w, const field::Field& T_field,
                             const field::Field& inverse_density,
                             const grid::Grid& grid, const parallel::Subdomain& sub,
                             const real_t density_quadratic_ratio,
                             const real_t delta_temperature, const real_t dt) {
  check_field_matches("add_buoyancy_z_variable_property", rhs_w, sub);
  check_field_matches("add_buoyancy_z_variable_property", T_field, sub);
  check_field_matches("add_buoyancy_z_variable_property", inverse_density, sub);
  const int h = halo_width;
  const field::View rhs = rhs_w.view();
  const field::ConstView T = T_field.view();
  const field::ConstView ir = inverse_density.view();
  const real_t* dz = grid.z.cell_width.data();
  const real_t* dmz = grid.z.center_distance.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) {
        const real_t T_face = stencil::temperature_at_z_face(T, i, j, k, dz, dmz);
        const real_t ir_face = stencil::temperature_at_z_face(ir, i, j, k, dz, dmz);
        rhs(i, j, k) += dt * ir_face
                        * (T_face
                           + density_quadratic_ratio * T_face * T_face
                                 * delta_temperature);
      }
}

}  // namespace mpmstd::physics
