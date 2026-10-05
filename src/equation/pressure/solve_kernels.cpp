#include "equation/pressure/solve_kernels.hpp"

#include <cstddef>

#include "common/field_checks.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// CPU 백엔드 — 팩·산포·축소 루프 (GPU 쌍둥이 solve_kernels.cu).

namespace mpmstd::equation::pressure {

void pack_rhs_to_lines(real_t* lines, const field::Field& rhs,
                       const parallel::Subdomain& sub, const int n_x, const int n_y,
                       const int nz_local) {
  (void)sub;  // GPU 쌍둥이와 시그니처 통일 (커널 발사 치수용)
  const int h = halo_width;
  const field::ConstView r = rhs.view();
  for (int j = h; j < h + n_y; ++j)
    for (int k = h; k < h + nz_local; ++k)
      for (int i = h; i < h + n_x; ++i)
        lines[(static_cast<std::size_t>(j - h) * nz_local + (k - h)) * n_x + (i - h)]
            = r(i, j, k);
}

void reorder_x_to_y(real_t* sxy, const real_t* sx, const int n_y, const int nz_local,
                    const int n_half_x) {
  for (int jy = 0; jy < n_y; ++jy)
    for (int kk = 0; kk < nz_local; ++kk)
      for (int mx = 0; mx < n_half_x; ++mx) {
        const std::size_t dst =
            2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
        const std::size_t src =
            2 * ((static_cast<std::size_t>(jy) * nz_local + kk) * n_half_x + mx);
        sxy[dst] = sx[src];
        sxy[dst + 1] = sx[src + 1];
      }
}

void reorder_y_to_x(real_t* sx, const real_t* sxy, const int n_y, const int nz_local,
                    const int n_half_x) {
  for (int jy = 0; jy < n_y; ++jy)
    for (int kk = 0; kk < nz_local; ++kk)
      for (int mx = 0; mx < n_half_x; ++mx) {
        const std::size_t dst =
            2 * ((static_cast<std::size_t>(jy) * nz_local + kk) * n_half_x + mx);
        const std::size_t src =
            2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
        sx[dst] = sxy[src];
        sx[dst + 1] = sxy[src + 1];
      }
}

void pack_spectrum_to_bands(real_t* D, const real_t* sxy, const int nz_local,
                            const int n_half_x, const int n_y,
                            const bool pin_mean_row) {
  const int nsys = 2 * n_half_x * n_y;
  for (int kk = 0; kk < nz_local; ++kk)
    for (int mx = 0; mx < n_half_x; ++mx)
      for (int jy = 0; jy < n_y; ++jy) {
        const std::size_t q =
            2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
        const std::size_t p = static_cast<std::size_t>(kk) * nsys
                              + 2 * (static_cast<std::size_t>(mx) * n_y + jy);
        const bool pinned = pin_mean_row && kk == 0 && mx == 0 && jy == 0;
        D[p] = pinned ? 0 : sxy[q];
        D[p + 1] = pinned ? 0 : sxy[q + 1];
      }
}

void unpack_bands_to_spectrum(real_t* sxy, const real_t* D, const int nz_local,
                              const int n_half_x, const int n_y) {
  const int nsys = 2 * n_half_x * n_y;
  for (int kk = 0; kk < nz_local; ++kk)
    for (int mx = 0; mx < n_half_x; ++mx)
      for (int jy = 0; jy < n_y; ++jy) {
        const std::size_t q =
            2 * ((static_cast<std::size_t>(kk) * n_half_x + mx) * n_y + jy);
        const std::size_t p = static_cast<std::size_t>(kk) * nsys
                              + 2 * (static_cast<std::size_t>(mx) * n_y + jy);
        sxy[q] = D[p];
        sxy[q + 1] = D[p + 1];
      }
}

void scatter_lines_to_field(field::Field& phi, const real_t* lines, const real_t scale,
                            const parallel::Subdomain& sub, const int n_x,
                            const int n_y, const int nz_local) {
  (void)sub;  // GPU 쌍둥이와 시그니처 통일
  const int h = halo_width;
  const field::View p = phi.view();
  for (int j = h; j < h + n_y; ++j)
    for (int k = h; k < h + nz_local; ++k)
      for (int i = h; i < h + n_x; ++i)
        p(i, j, k) =
            lines[(static_cast<std::size_t>(j - h) * nz_local + (k - h)) * n_x
                  + (i - h)]
            * scale;
}

real_t interior_sum(const field::Field& phi, const parallel::Subdomain& sub) {
  const int h = halo_width;
  const field::ConstView p = phi.view();
  real_t sum = 0;
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) sum += p(i, j, k);
  return sum;
}

void subtract_constant(field::Field& phi, const parallel::Subdomain& sub,
                       const real_t value) {
  const int h = halo_width;
  const field::View p = phi.view();
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) p(i, j, k) -= value;
}

// ── 공개: 속도 보정·압력 누적 (solve.hpp 선언) ──

void correct_velocity(field::Field& U, field::Field& V, field::Field& W,
                      const field::Field& phi, const grid::Grid& grid,
                      const parallel::Subdomain& sub, const real_t dt) {
  check_field_matches("correct_velocity", U, sub);
  check_field_matches("correct_velocity", V, sub);
  check_field_matches("correct_velocity", W, sub);
  check_field_matches("correct_velocity", phi, sub);

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
