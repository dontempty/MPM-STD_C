#include "equation/momentum/rhs/rhs.hpp"

#include "common/field_checks.hpp"
#include "equation/transport_stencil.hpp"

// CPU 백엔드 — 수식은 momentum_stencil.hpp 가 정본이고 여기는 루프만 있다.
// GPU 쌍둥이 rhs.cu 가 같은 스텐실 함수를 커널에서 부른다 (빌드가 한쪽만 담는다).

namespace mpmstd::equation::momentum {

namespace {

void add_impl(const char* who, field::Field& rhs_field, const field::Field& q_field,
              const field::Field& U_field, const field::Field& V_field,
              const field::Field& W_field, const grid::Grid& grid,
              const parallel::Subdomain& sub, const real_t viscosity, const real_t dt,
              const bool w_is_z_face) {
  check_field_matches(who, rhs_field, sub);
  check_field_matches(who, q_field, sub);
  check_field_matches(who, U_field, sub);
  check_field_matches(who, V_field, sub);
  check_field_matches(who, W_field, sub);

  const int h = halo_width;
  const int nx = sub.x().n_total;
  const int ny = sub.y().n_total;
  const int nz = sub.z().n_total;
  const real_t nu_dt = viscosity * dt;
  const field::View rhs = rhs_field.view();
  const field::ConstView q = q_field.view();
  const field::ConstView U = U_field.view();
  const field::ConstView V = V_field.view();
  const field::ConstView W = W_field.view();

  for (int i = h; i < nx - h; ++i)
    for (int j = h; j < ny - h; ++j)
      for (int k = h; k < nz - h; ++k)
        rhs(i, j, k) += equation::stencil::advection_diffusion(
            q, U, V, W, i, j, k, grid.x.cell_width.data(),
            grid.x.center_distance.data(), grid.y.cell_width.data(),
            grid.y.center_distance.data(), grid.z.cell_width.data(),
            grid.z.center_distance.data(), nu_dt, dt, w_is_z_face);
}

}  // namespace

void add_advection_diffusion_cell(field::Field& rhs, const field::Field& q,
                                  const field::Field& U, const field::Field& V,
                                  const field::Field& W, const grid::Grid& grid,
                                  const parallel::Subdomain& sub,
                                  const real_t viscosity, const real_t dt) {
  add_impl("add_advection_diffusion_cell", rhs, q, U, V, W, grid, sub, viscosity, dt,
           /*w_is_z_face=*/false);
}

void add_advection_diffusion_z_face(field::Field& rhs, const field::Field& w,
                                    const field::Field& U, const field::Field& V,
                                    const field::Field& W, const grid::Grid& grid,
                                    const parallel::Subdomain& sub,
                                    const real_t viscosity, const real_t dt) {
  add_impl("add_advection_diffusion_z_face", rhs, w, U, V, W, grid, sub, viscosity, dt,
           /*w_is_z_face=*/true);
}

}  // namespace mpmstd::equation::momentum
