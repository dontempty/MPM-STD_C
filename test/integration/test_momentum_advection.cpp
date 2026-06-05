// Integration test: MomentumEquation with convection on triple-periodic cube.
//
// Analytic solution (frozen uniform advecting velocity + viscous decay):
//
//   q(x,y,z,t) = cos(k(x - u_x t)) * cos(k(y - u_y t)) * cos(k(z - u_z t))
//                * exp(-ν k² · 3 · t)
//
// where k = 2π/L.  This satisfies  ∂q/∂t + (u_x,u_y,u_z)·∇q = ν ∇²q.
//
// The advecting velocity (U,V,W) is frozen (constant in space and time).
// We refine the spatial grid (fixed small dt) and observe L_inf error
// reducing by ~4× per N-doubling ⇒ spatial EOC ≈ 2.

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "boundary/main.hpp"
#include "grid/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "equation/momentum/main.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>

using namespace mpmstd;

namespace {

constexpr real_t kPi = static_cast<real_t>(M_PI);

struct Setup {
  real_t L   = 1.0;
  real_t nu  = 0.01;     // kinematic viscosity
  real_t u_x = 1.0;
  real_t u_y = 0.5;
  real_t u_z = 0.25;
};

real_t analytic(const Setup& s, real_t x, real_t y, real_t z, real_t t) {
  const real_t k     = 2.0 * kPi / s.L;
  const real_t decay = s.nu * k * k * 3.0;
  return std::cos(k * (x - s.u_x * t))
       * std::cos(k * (y - s.u_y * t))
       * std::cos(k * (z - s.u_z * t))
       * std::exp(-decay * t);
}

real_t run_momentum(const parallel::mpi::MpiContext& mpi,
                     const Setup& s,
                     int N, int M, real_t dt) {
  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, true, true});
  parallel::mpi::Subdomain   sub (topo, {N, N, N});

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a)
    axes[a] = grid::AxisConfig{ N, s.L, grid::StretchKind::Uniform, 0.0 };
  grid::Grid g(sub, axes);

  boundary::Problem problem;   // all-Periodic (neutral default)
  problem.validate();

  auto backend = parallel::make_default_backend();
  field::FieldRegistry fields(sub, *backend);

  // The momentum component being tested.
  auto& Q = fields.add_scalar("Q");
  // Frozen advecting velocity fields.
  auto& U = fields.add_scalar("U");
  auto& V = fields.add_scalar("V");
  auto& W = fields.add_scalar("W");

  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);
  boundary::BoundaryApplier bc(problem);

  // MomentumEquation uses "Q" as the component name, advected by U/V/W.
  // We rename the velocity registry keys to match the equation's lookup.
  // (Normally this would be problem.U / problem.V / problem.W.)
  equation::momentum::MomentumEquation mom_eq(
      { "Q", s.nu, /*with_convection=*/true, /*source_name=*/"" },
      g, sub, fields, problem, problem.U,   // problem.U is all-Periodic
      *tdma, bc);

  // Frozen uniform velocity.
  U.fill_host(s.u_x);
  V.fill_host(s.u_y);
  W.fill_host(s.u_z);
  U.exchange_halo();
  V.exchange_halo();
  W.exchange_halo();

  // Initial condition.
  const auto& xc = g.xc(Direction::X);
  const auto& yc = g.xc(Direction::Y);
  const auto& zc = g.xc(Direction::Z);
  const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];

  for (int i = 0; i < n1; ++i)
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k)
        Q.host_at(i, j, k) = analytic(s, xc[i], yc[j], zc[k], 0.0);

  Q.exchange_halo();
  bc.apply_ghost(Q, problem.U);

  // Time march.
  for (int n = 0; n < M; ++n) mom_eq.step(dt);

  // L_inf error on interior cells.
  const real_t t_end = M * dt;
  real_t err = 0.0;
  for (int i = kHaloWidth; i < n1 - kHaloWidth; ++i)
    for (int j = kHaloWidth; j < n2 - kHaloWidth; ++j)
      for (int k = kHaloWidth; k < n3 - kHaloWidth; ++k) {
        const real_t ref = analytic(s, xc[i], yc[j], zc[k], t_end);
        const real_t e   = std::abs(Q.host_at(i, j, k) - ref);
        if (e > err) err = e;
      }
  return err;
}

} // anonymous namespace


int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  Setup s;
  const int    M  = 20;
  const real_t dt = 5e-4;   // dt² ≈ 2.5e-7 << dx² at N=16 → spatial error dominates

  const real_t e16  = run_momentum(mpi, s, 16, M, dt);
  const real_t e32  = run_momentum(mpi, s, 32, M, dt);
  const real_t eoc  = std::log(e16 / e32) / std::log(2.0);

  if (mpi.is_root()) {
    std::fprintf(stderr,
      "  M = %d, dt = %g, t_end = %g\n"
      "    N = 16 -> L_inf = %.3e\n"
      "    N = 32 -> L_inf = %.3e\n"
      "    spatial EOC ≈ %.3f\n",
      M, double(dt), double(M*dt), double(e16), double(e32), double(eoc));
  }

  // CN + central convection → 2nd-order spatial.
  MPMSTD_TEST_CHECK(eoc > 1.7);
  MPMSTD_TEST_CHECK(e32 < 1e-2);

  if (mpi.is_root()) {
    mpmstd_test_pass("momentum_advection_traveling_wave");
    std::cout << "test_momentum_advection: ALL PASS\n";
  }
  return 0;
}
