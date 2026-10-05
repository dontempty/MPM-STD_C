#include "equation/scalar/rhs/rhs.hpp"

#include "common/field_checks.hpp"
#include "equation/transport_stencil.hpp"

// CPU 백엔드 — 수식은 transport_stencil.hpp 공용 (운동량과 같은 식, 계수만 α).
// GPU 쌍둥이 rhs.cu 가 같은 스텐실을 커널에서 부른다.

namespace mpmstd::equation::scalar {

void add_diffusion(field::Field& rhs_field, const field::Field& T_field,
                   const grid::Grid& grid, const parallel::Subdomain& sub,
                   const real_t alpha, const real_t dt) {
  check_field_matches("add_diffusion", rhs_field, sub);
  check_field_matches("add_diffusion", T_field, sub);
  const int h = halo_width;
  const real_t dt_alpha = dt * alpha;
  const field::View rhs = rhs_field.view();
  const field::ConstView T = T_field.view();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        rhs(i, j, k) += dt_alpha * equation::stencil::diffusion(
                            T, i, j, k, grid.x.cell_width.data(),
                            grid.x.center_distance.data(), grid.y.cell_width.data(),
                            grid.y.center_distance.data(), grid.z.cell_width.data(),
                            grid.z.center_distance.data(), /*w_is_z_face=*/false);
}

void add_convection(field::Field& rhs_field, const field::Field& T_field,
                    const field::Field& U_field, const field::Field& V_field,
                    const field::Field& W_field, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("add_convection", rhs_field, sub);
  check_field_matches("add_convection", T_field, sub);
  check_field_matches("add_convection", U_field, sub);
  check_field_matches("add_convection", V_field, sub);
  check_field_matches("add_convection", W_field, sub);
  const int h = halo_width;
  const field::View rhs = rhs_field.view();
  const field::ConstView T = T_field.view();
  const field::ConstView U = U_field.view();
  const field::ConstView V = V_field.view();
  const field::ConstView W = W_field.view();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        rhs(i, j, k) -= dt * equation::stencil::convection_divergence(
                            T, U, V, W, i, j, k, grid.x.cell_width.data(),
                            grid.y.cell_width.data(), grid.z.cell_width.data());
}

}  // namespace mpmstd::equation::scalar
