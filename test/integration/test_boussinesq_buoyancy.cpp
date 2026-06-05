// Integration test: Boussinesq buoyancy in the W-momentum equation.
//
// Problem setup (z-periodic, no convection, no viscous decay):
//   T(x,y,z) = sin(2π z / L)   (frozen, acts as body force)
//   W = 0 initially
//   ν → 0 (very small viscosity)
//
// One step of the W-equation with source f_z = T:
//   δ ≈ dt · T     (viscous contribution is O(ν·dt) ≈ 0)
//   W^1 = W^0 + δ = dt · T
//
// After M steps (all with frozen T, all-periodic, no convection):
//   W^M = M·dt · T  (exactly, because the source is linear in T)
//
// We verify:
//   max|W(t_end) - t_end · sin(2π z / L)| < 1e-10

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "boundary/main.hpp"
#include "grid/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "equation/momentum/main.hpp"
#include "physics/boussinesq.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>

using namespace mpmstd;

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  const int N = 32;
  const real_t L = 1.0;

  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, true, true});
  parallel::mpi::Subdomain   sub (topo, {N, N, N});

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a)
    axes[a] = grid::AxisConfig{ N, L, grid::StretchKind::Uniform, 0.0 };
  grid::Grid g(sub, axes);

  boundary::Problem problem;   // all-Periodic
  problem.validate();

  auto backend = parallel::make_default_backend();
  field::FieldRegistry fields(sub, *backend);

  auto& W   = fields.add_scalar("W");
  auto& T   = fields.add_scalar("T");
  auto& src = fields.add_scalar("W_src");   // buoyancy source for W

  // Frozen advecting velocity (zero → no convection term needed).
  auto& U = fields.add_scalar("U");
  auto& V = fields.add_scalar("V");
  U.fill_host(0.0);  V.fill_host(0.0);  W.fill_host(0.0);
  U.exchange_halo(); V.exchange_halo(); W.exchange_halo();

  // Very small viscosity → viscous term negligible.
  const real_t nu = 1e-8;
  const real_t dt = 1e-3;
  const int    M  = 10;

  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);
  boundary::BoundaryApplier bc(problem);

  // W-momentum equation with buoyancy source ("W_src") and no convection.
  equation::momentum::MomentumEquation mom_W(
      { "W", nu, /*with_convection=*/false, /*source_name=*/"W_src" },
      g, sub, fields, problem, problem.W, *tdma, bc);

  // Frozen temperature: T = sin(2π z / L)
  const auto& zc = g.xc(Direction::Z);
  const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
  constexpr real_t kPi = static_cast<real_t>(M_PI);

  for (int i = 0; i < n1; ++i)
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k)
        T.host_at(i, j, k) = std::sin(2.0 * kPi * zc[k] / L);

  T.exchange_halo();

  // Time march: update buoyancy source from T then advance W.
  for (int n = 0; n < M; ++n) {
    physics::compute_z_buoyancy(src, T);
    mom_W.step(dt);
  }

  // Analytic solution: W(t) = t_end * T  (because source is constant and ν≈0)
  const real_t t_end = M * dt;
  real_t err = 0.0;
  for (int i = kHaloWidth; i < n1 - kHaloWidth; ++i)
    for (int j = kHaloWidth; j < n2 - kHaloWidth; ++j)
      for (int k = kHaloWidth; k < n3 - kHaloWidth; ++k) {
        const real_t ref = t_end * std::sin(2.0 * kPi * zc[k] / L);
        const real_t e   = std::abs(W.host_at(i, j, k) - ref);
        if (e > err) err = e;
      }

  if (mpi.is_root()) {
    std::fprintf(stderr,
      "  N = %d, M = %d, dt = %g, t_end = %g\n"
      "  L_inf |W - t_end*sin(2πz/L)| = %.3e\n",
      N, M, double(dt), double(t_end), double(err));
  }

  // Viscous contribution ≈ O(ν·t_end) = O(1e-10) → allow 1e-8 margin.
  MPMSTD_TEST_CHECK(err < 1e-8);

  if (mpi.is_root()) {
    mpmstd_test_pass("boussinesq_linear_buoyancy");
    std::cout << "test_boussinesq_buoyancy: ALL PASS\n";
  }
  return 0;
}
