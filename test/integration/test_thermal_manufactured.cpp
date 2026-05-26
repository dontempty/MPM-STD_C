// Integration test: ScalarEquation diffusion (Phase 1, periodic).
//
// Analytic solution on a fully periodic cube of side L:
//
//     T(x, y, z, t) = cos(2*pi*x/L) * cos(2*pi*y/L) * cos(2*pi*z/L) * exp(-k*t)
//
// where  k = alpha * (2*pi/L)^2 * 3.
//
// We discretise the cube with N cells on each side (uniform grid), all axes
// periodic, and run M time steps of size dt.  We then measure the L_inf
// error at t = M*dt versus the analytic field, and verify that halving dt
// (with the spatial discretisation fixed) reduces the error by a factor of
// approximately 4 — the temporal EOC of the Douglas ADI scheme is 2.

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "boundary/main.hpp"
#include "grid/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "equation/scalar/main.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>

using namespace mpmstd;

namespace {

constexpr real_t kPi = static_cast<real_t>(M_PI);

real_t analytic(real_t x, real_t y, real_t z, real_t t,
                real_t L, real_t alpha) {
  const real_t wn   = 2.0 * kPi / L;
  const real_t k    = alpha * wn * wn * 3.0;
  return std::cos(wn * x) * std::cos(wn * y) * std::cos(wn * z) * std::exp(-k * t);
}

// Run M time steps of size dt on an N^3 uniform-periodic cube, return the
// L_inf interior error against the analytic solution at t = M*dt.
real_t run_manufactured(const parallel::mpi::MpiContext& mpi,
                        int N, int M, real_t dt) {
  const real_t L     = 1.0;
  const real_t alpha = 1.0;

  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, true, true});
  parallel::mpi::Subdomain   sub (topo, {N, N, N});

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a)
    axes[a] = grid::AxisConfig{ N, L, grid::StretchKind::Uniform, 0.0 };
  grid::Grid g(sub, axes);

  // Problem: every axis periodic ⇒ no global BC; halo exchange covers all.
  boundary::Problem problem;
  for (int a = 0; a < 3; ++a)
    problem.topology.axis[a] = boundary::AxisTopology::Periodic;
  for (auto* fb : { &problem.U, &problem.V, &problem.W, &problem.P, &problem.T })
    for (auto d : { Direction::X, Direction::Y, Direction::Z })
      for (auto s : { Side::Minus, Side::Plus })
        fb->face(d, s) = boundary::FaceBc::periodic();
  problem.validate();

  auto backend = parallel::make_default_backend();
  field::FieldRegistry fields(sub, *backend);
  auto& T = fields.add_scalar("T");

  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);
  boundary::BoundaryApplier bc(problem);

  equation::scalar::ScalarEquation
    thermal({ "T", alpha }, g, sub, fields, problem, problem.T, *tdma, bc);

  // Initial condition: analytic at t = 0.
  const auto& xc = g.xc(Direction::X);
  const auto& yc = g.xc(Direction::Y);
  const auto& zc = g.xc(Direction::Z);
  const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
  for (int i = 0; i < n1; ++i)
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k)
        T.host_at(i, j, k) = analytic(xc[i], yc[j], zc[k], 0.0, L, alpha);

  T.exchange_halo();
  // (apply_ghost is a no-op for all-periodic, but we keep the pattern.)
  bc.apply_ghost(T, problem.T);

  // Time march.
  for (int n = 0; n < M; ++n) {
    thermal.step(dt);
  }

  // L_inf error on interior cells.
  const real_t t_end = M * dt;
  real_t err = 0.0;
  for (int i = kHaloWidth; i < n1 - kHaloWidth; ++i)
    for (int j = kHaloWidth; j < n2 - kHaloWidth; ++j)
      for (int k = kHaloWidth; k < n3 - kHaloWidth; ++k) {
        const real_t ref = analytic(xc[i], yc[j], zc[k], t_end, L, alpha);
        const real_t e   = std::abs(T.host_at(i, j, k) - ref);
        if (e > err) err = e;
      }
  return err;
}

} // anonymous namespace


int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  // -------------------------------------------------------------------------
  // Spatial EOC test
  //
  // Keep dt very small so that the temporal truncation (CN/Douglas ADI is
  // O(dt^2)) is dwarfed by the spatial truncation (2nd-order central FD is
  // O(dx^2)). Then refine N and observe a ~4x reduction in L_inf error
  // ⇒ spatial EOC ≈ 2.
  // -------------------------------------------------------------------------
  const int    N_coarse = 16;
  const int    N_fine   = 32;
  const int    M        = 10;        // few short steps suffice
  const real_t dt       = 1.0e-4;    // dt^2 ~ 1e-8  <<  dx_coarse^2 ~ 4e-3
  const real_t t_end    = M * dt;

  const real_t e_coarse = run_manufactured(mpi, N_coarse, M, dt);
  const real_t e_fine   = run_manufactured(mpi, N_fine,   M, dt);
  const real_t eoc      = std::log(e_coarse / e_fine) / std::log(2.0);

  if (mpi.is_root()) {
    std::fprintf(stderr,
      "  M = %d,  dt = %g,  t_end = %g\n"
      "    N = %d -> L_inf = %.3e\n"
      "    N = %d -> L_inf = %.3e\n"
      "    spatial EOC ≈ %.3f\n",
      M, double(dt), double(t_end),
      N_coarse, double(e_coarse), N_fine, double(e_fine), double(eoc));
  }

  // 2nd-order central FD ⇒ spatial EOC = 2. Allow some slack for the modest
  // resolutions used here.
  MPMSTD_TEST_CHECK(eoc      > 1.7);
  MPMSTD_TEST_CHECK(e_fine   < 1e-3);

  if (mpi.is_root()) {
    mpmstd_test_pass("thermal_manufactured_spatial_eoc");
    std::cout << "test_thermal_manufactured: ALL PASS\n";
  }
  return 0;
}
