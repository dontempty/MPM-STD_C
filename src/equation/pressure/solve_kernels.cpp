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

}  // namespace mpmstd::equation::pressure
