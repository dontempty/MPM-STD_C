// Unit test: stencil free functions on a uniform grid.
//
// Verifies second-order convergence of divergence_at_cell by computing the
// L_inf error of div(U) for an analytic field with known divergence at two
// grid resolutions.

#include "common/main.hpp"
#include "field/stencil/staggered.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <vector>

using namespace mpmstd;
using namespace mpmstd::field::stencil;

namespace {

// Build a uniform 1D metric: dx[i] = L/N for i in 0..N-1 (with halos: size N+2 total).
std::vector<real_t> make_uniform_dx(int n, real_t L) {
  return std::vector<real_t>(static_cast<std::size_t>(n) + 2, L / static_cast<real_t>(n));
}

// Cell-center x-coordinate for global index i (0-based interior).
inline real_t xc(int i, int n, real_t L) {
  const real_t dx = L / static_cast<real_t>(n);
  return (i + 0.5) * dx;
}

// Returns the L_inf error of divergence(U, V, W) on the interior, given an
// analytic field with prescribed divergence on a uniform grid.
real_t run_divergence_l_inf(int n) {
  const real_t L  = 1.0;
  const int    n1 = n + 2, n2 = n + 2, n3 = n + 2;
  const std::size_t total = static_cast<std::size_t>(n1) * n2 * n3;

  auto dx1 = make_uniform_dx(n, L);
  auto dx2 = make_uniform_dx(n, L);
  auto dx3 = make_uniform_dx(n, L);

  std::vector<real_t> U(total, 0.0), V(total, 0.0), W(total, 0.0);

  // Analytic field with periodic divergence:
  //   U(x, y, z) = sin(2 pi x)
  //   V = W = 0
  // => div(U) = 2 pi cos(2 pi x)
  //
  // U lives on the +x face at xf[i] = i * dx (global).  In our halo'd array
  // U[i, j, k] with i in [kHaloWidth, kHaloWidth + n] should equal
  //   sin(2 pi * (i - kHaloWidth) * dx).
  const real_t dx = L / static_cast<real_t>(n);
  for (int i = 0; i < n1; ++i) {
    for (int j = 0; j < n2; ++j) {
      for (int k = 0; k < n3; ++k) {
        const real_t x_face = (i - kHaloWidth) * dx;
        U[(i * n2 + j) * n3 + k] = std::sin(2.0 * M_PI * x_face);
      }
    }
  }

  real_t err_inf = 0.0;
  for (int i = kHaloWidth; i < kHaloWidth + n; ++i) {
    const real_t x_c = xc(i - kHaloWidth, n, L);
    const real_t expected_div = 2.0 * M_PI * std::cos(2.0 * M_PI * x_c);
    for (int j = kHaloWidth; j < kHaloWidth + n; ++j) {
      for (int k = kHaloWidth; k < kHaloWidth + n; ++k) {
        const real_t d = divergence_at_cell(U.data(), V.data(), W.data(),
                                              dx1.data(), dx2.data(), dx3.data(),
                                              i, j, k, n1, n2, n3);
        err_inf = std::max(err_inf, std::abs(d - expected_div));
      }
    }
  }
  return err_inf;
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  // Two resolutions; expect EOC = 2 from a 2nd-order central scheme.
  const real_t e16 = run_divergence_l_inf(16);
  const real_t e32 = run_divergence_l_inf(32);
  const real_t order = std::log(e16 / e32) / std::log(2.0);

  std::fprintf(stderr, "  divergence L_inf:  n=16 -> %g, n=32 -> %g, EOC = %g\n",
                double(e16), double(e32), double(order));

  // Allow some slack to absorb FP noise (we expect ~2.0).
  MPMSTD_TEST_CHECK(order > 1.9);
  MPMSTD_TEST_CHECK(order < 2.1);

  mpmstd_test_pass("divergence_second_order");
  std::cout << "test_stencil: ALL PASS\n";
  return 0;
}
