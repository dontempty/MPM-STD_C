#include "equation/pressure/rhs/rhs.hpp"

#include "common/field_checks.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// CPU 백엔드 — 수식은 pressure_stencil.hpp, 여기는 루프만 (GPU 쌍둥이 rhs.cu).

namespace mpmstd::equation::pressure {

void add_divergence(field::Field& rhs, const field::Field& U, const field::Field& V,
                    const field::Field& W, const grid::Grid& grid,
                    const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("add_divergence", rhs, sub);
  check_field_matches("add_divergence", U, sub);
  check_field_matches("add_divergence", V, sub);
  check_field_matches("add_divergence", W, sub);

  const int h = halo_width;
  const real_t inv_dt = 1 / dt;
  const field::View r = rhs.view();
  const field::ConstView u = U.view();
  const field::ConstView v = V.view();
  const field::ConstView w = W.view();

  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k)
        r(i, j, k) += inv_dt * stencil::divergence(u, v, w, i, j, k,
                                                   grid.x.cell_width[i],
                                                   grid.y.cell_width[j],
                                                   grid.z.cell_width[k]);
}

void add_gradient_constant_property(field::Field& rhs_u, field::Field& rhs_v, field::Field& rhs_w,
                  const field::Field& P, const grid::Grid& grid,
                  const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("add_gradient_constant_property", rhs_u, sub);
  check_field_matches("add_gradient_constant_property", rhs_v, sub);
  check_field_matches("add_gradient_constant_property", rhs_w, sub);
  check_field_matches("add_gradient_constant_property", P, sub);

  const int h = halo_width;
  const field::View ru = rhs_u.view();
  const field::View rv = rhs_v.view();
  const field::View rw = rhs_w.view();
  const field::ConstView p = P.view();

  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) {
        ru(i, j, k) -= dt * stencil::gradient_x(p, i, j, k, grid.x.center_distance[i]);
        rv(i, j, k) -= dt * stencil::gradient_y(p, i, j, k, grid.y.center_distance[j]);
        rw(i, j, k) -= dt * stencil::gradient_z(p, i, j, k, grid.z.center_distance[k]);
      }
}

void add_extra_term_variable_property(field::Field& rhs_field,
                             const field::Field& pressure_predictor,
                             const field::Field& inverse_density,
                             const grid::Grid& grid, const parallel::Subdomain& sub) {
  check_field_matches("add_extra_term_variable_property", rhs_field, sub);
  check_field_matches("add_extra_term_variable_property", pressure_predictor, sub);
  check_field_matches("add_extra_term_variable_property", inverse_density, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const auto at = [ny_total, nz_total](int i, int j, int k) {
    return (static_cast<long long>(i) * ny_total + j) * nz_total + k;
  };
  real_t* rhs = rhs_field.data();
  const real_t* ph = pressure_predictor.data();
  const real_t* ir = inverse_density.data();
  const real_t* dx1 = grid.x.cell_width.data();
  const real_t* dmx1 = grid.x.center_distance.data();
  const real_t* dx2 = grid.y.cell_width.data();
  const real_t* dmx2 = grid.y.center_distance.data();
  const real_t* dx3 = grid.z.cell_width.data();
  const real_t* dmx3 = grid.z.center_distance.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) {
        const long long c = at(i, j, k);
        const long long cxm = at(i - 1, j, k), cxp = at(i + 1, j, k);
        const long long cym = at(i, j - 1, k), cyp = at(i, j + 1, k);
        const long long czm = at(i, j, k - 1), czp = at(i, j, k + 1);
        // 면 평균 1/ρ̂ (체적가중) 과 면 ∇p̂
        const real_t ir_xlo = real_t(0.5) / dmx1[i] * (dx1[i - 1] * ir[c] + dx1[i] * ir[cxm]);
        const real_t ir_xhi = real_t(0.5) / dmx1[i + 1] * (dx1[i + 1] * ir[c] + dx1[i] * ir[cxp]);
        const real_t ir_ylo = real_t(0.5) / dmx2[j] * (dx2[j - 1] * ir[c] + dx2[j] * ir[cym]);
        const real_t ir_yhi = real_t(0.5) / dmx2[j + 1] * (dx2[j + 1] * ir[c] + dx2[j] * ir[cyp]);
        const real_t ir_zlo = real_t(0.5) / dmx3[k] * (dx3[k - 1] * ir[c] + dx3[k] * ir[czm]);
        const real_t ir_zhi = real_t(0.5) / dmx3[k + 1] * (dx3[k + 1] * ir[c] + dx3[k] * ir[czp]);
        const real_t gx_lo = (ph[c] - ph[cxm]) / dmx1[i];
        const real_t gx_hi = (ph[cxp] - ph[c]) / dmx1[i + 1];
        const real_t gy_lo = (ph[c] - ph[cym]) / dmx2[j];
        const real_t gy_hi = (ph[cyp] - ph[c]) / dmx2[j + 1];
        const real_t gz_lo = (ph[c] - ph[czm]) / dmx3[k];
        const real_t gz_hi = (ph[czp] - ph[c]) / dmx3[k + 1];
        rhs[c] += ((1 - ir_xhi) * gx_hi - (1 - ir_xlo) * gx_lo) / dx1[i]
                  + ((1 - ir_yhi) * gy_hi - (1 - ir_ylo) * gy_lo) / dx2[j]
                  + ((1 - ir_zhi) * gz_hi - (1 - ir_zlo) * gz_lo) / dx3[k];
      }
}

void add_gradient_variable_property(field::Field& rhs_u, field::Field& rhs_v,
                           field::Field& rhs_w, const field::Field& P,
                           const field::Field& inverse_density,
                           const grid::Grid& grid, const parallel::Subdomain& sub,
                           const real_t dt) {
  check_field_matches("add_gradient_variable_property", rhs_u, sub);
  check_field_matches("add_gradient_variable_property", P, sub);
  check_field_matches("add_gradient_variable_property", inverse_density, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const auto at = [ny_total, nz_total](int i, int j, int k) {
    return (static_cast<long long>(i) * ny_total + j) * nz_total + k;
  };
  real_t* du = rhs_u.data();
  real_t* dv = rhs_v.data();
  real_t* dw = rhs_w.data();
  const real_t* p = P.data();
  const real_t* ir = inverse_density.data();
  const real_t* dx1 = grid.x.cell_width.data();
  const real_t* dmx1 = grid.x.center_distance.data();
  const real_t* dx2 = grid.y.cell_width.data();
  const real_t* dmx2 = grid.y.center_distance.data();
  const real_t* dx3 = grid.z.cell_width.data();
  const real_t* dmx3 = grid.z.center_distance.data();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) {
        const long long c = at(i, j, k);
        const long long cxm = at(i - 1, j, k);
        const long long cym = at(i, j - 1, k);
        const long long czm = at(i, j, k - 1);
        const real_t irx = real_t(0.5) * (dx1[i - 1] * ir[c] + dx1[i] * ir[cxm]) / dmx1[i];
        const real_t iry = real_t(0.5) * (dx2[j - 1] * ir[c] + dx2[j] * ir[cym]) / dmx2[j];
        const real_t irz = real_t(0.5) * (dx3[k - 1] * ir[c] + dx3[k] * ir[czm]) / dmx3[k];
        du[c] -= dt * irx * (p[c] - p[cxm]) / dmx1[i];
        dv[c] -= dt * iry * (p[c] - p[cym]) / dmx2[j];
        dw[c] -= dt * irz * (p[c] - p[czm]) / dmx3[k];
      }
}

}  // namespace mpmstd::equation::pressure
