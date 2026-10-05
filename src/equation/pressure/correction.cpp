#include "equation/pressure/correction.hpp"

#include <stdexcept>

#include "common/field_checks.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// CPU 백엔드 — GPU 쌍둥이 correction.cu. 수식은 본문 주석 참조 (기울기 스텐실은 pressure_stencil.hpp).

namespace mpmstd::equation::pressure {


void correct_velocity_constant_property(field::Field& U, field::Field& V, field::Field& W,
                      const field::Field& phi, const grid::Grid& grid,
                      const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("correct_velocity_constant_property", U, sub);
  check_field_matches("correct_velocity_constant_property", V, sub);
  check_field_matches("correct_velocity_constant_property", W, sub);
  check_field_matches("correct_velocity_constant_property", phi, sub);

  const int h = halo_width;
  const field::View u = U.view();
  const field::View v = V.view();
  const field::View w = W.view();
  const field::ConstView p = phi.view();

  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) {
        u(i, j, k) -= dt * stencil::gradient_x(p, i, j, k, grid.x.center_distance[i]);
        v(i, j, k) -= dt * stencil::gradient_y(p, i, j, k, grid.y.center_distance[j]);
        w(i, j, k) -= dt * stencil::gradient_z(p, i, j, k, grid.z.center_distance[k]);
      }
}

void accumulate(field::Field& P, const field::Field& phi) {
  if (P.size() != phi.size())
    throw std::invalid_argument("pressure::accumulate: P and phi differ in shape");
  real_t* p = P.data();
  const real_t* d = phi.data();
  const long long n = static_cast<long long>(P.size());
  for (long long q = 0; q < n; ++q) p[q] += d[q];
}

void correct_velocity_variable_property(field::Field& U, field::Field& V, field::Field& W,
                               const field::Field& phi,
                               const field::Field& pressure_predictor,
                               const field::Field& inverse_density,
                               const grid::Grid& grid, const parallel::Subdomain& sub,
                               const real_t dt) {
  check_field_matches("correct_velocity_variable_property", U, sub);
  check_field_matches("correct_velocity_variable_property", phi, sub);
  check_field_matches("correct_velocity_variable_property", pressure_predictor, sub);
  check_field_matches("correct_velocity_variable_property", inverse_density, sub);
  const int h = halo_width;
  const int ny_total = sub.y().n_total, nz_total = sub.z().n_total;
  const auto at = [ny_total, nz_total](int i, int j, int k) {
    return (static_cast<long long>(i) * ny_total + j) * nz_total + k;
  };
  real_t* u = U.data();
  real_t* v = V.data();
  real_t* w = W.data();
  const real_t* p = phi.data();
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
        const long long cxm = at(i - 1, j, k);
        const long long cym = at(i, j - 1, k);
        const long long czm = at(i, j, k - 1);
        const real_t irc_x = real_t(0.5) / dmx1[i] * (dx1[i - 1] * ir[c] + dx1[i] * ir[cxm]);
        const real_t irc_y = real_t(0.5) / dmx2[j] * (dx2[j - 1] * ir[c] + dx2[j] * ir[cym]);
        const real_t irc_z = real_t(0.5) / dmx3[k] * (dx3[k - 1] * ir[c] + dx3[k] * ir[czm]);
        u[c] -= dt * ((p[c] - p[cxm]) / dmx1[i] + (irc_x - 1) * (ph[c] - ph[cxm]) / dmx1[i]);
        v[c] -= dt * ((p[c] - p[cym]) / dmx2[j] + (irc_y - 1) * (ph[c] - ph[cym]) / dmx2[j]);
        w[c] -= dt * ((p[c] - p[czm]) / dmx3[k] + (irc_z - 1) * (ph[c] - ph[czm]) / dmx3[k]);
      }
}

void advance_pressure_predictor(field::Field& pressure_predictor,
                                field::Field& previous_predictor,
                                const field::Field& phi) {
  if (pressure_predictor.size() != phi.size()
      || previous_predictor.size() != phi.size())
    throw std::invalid_argument(
        "advance_pressure_predictor: fields differ in shape");
  real_t* ph = pressure_predictor.data();
  real_t* p0 = previous_predictor.data();
  const real_t* dp = phi.data();
  const long long n = static_cast<long long>(phi.size());
  for (long long q = 0; q < n; ++q) {
    const real_t new_previous = real_t(0.5) * (ph[q] + p0[q]);
    p0[q] = new_previous;
    ph[q] = 2 * dp[q] - new_previous;
  }
}


}  // namespace mpmstd::equation::pressure
