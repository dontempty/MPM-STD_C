// Integration test: ScalarEquation diffusion with Dirichlet wall on z.
//
// Topology: x periodic, y periodic, z non-periodic (wall).
// BCs on z: T_minus = -0.5, T_plus = +0.5 (cell-centered Dirichlet — the
//            ghost layer is set directly to the wall value).
//
// We start from T = 0 everywhere and integrate forward in time.  The
// numerical steady state of the discrete diffusion operator with
// ghost-value-equals-wall-value Dirichlet BCs is the linear interpolation
// between the two ghost cells:
//
//     T_i = T_minus + (T_plus - T_minus) * i / (n_tot - 1),  i = 0..n_tot-1
//
// (Here `i` indexes the local-with-halo cell, so i=0 is the lower ghost and
// i=n_tot-1 is the upper ghost — both fixed by apply_ghost.)

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


int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  // Single rank — keeps the test focused on Phase 2's wall row modification.
  const int    N_xy   = 4;     // periodic axes: any size works
  const int    N_z    = 16;    // wall axis
  const real_t L      = 1.0;
  const real_t alpha  = 1.0;
  const real_t T_lo   = -0.5;
  const real_t T_hi   =  0.5;
  const int    M      = 200;   // enough steps to converge
  const real_t dt     = 5e-3;  // CFL-safe for α=1, dx=1/16

  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, true, false});
  parallel::mpi::Subdomain   sub (topo, {N_xy, N_xy, N_z});

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a)
    axes[a] = grid::AxisConfig{ (a == 2 ? N_z : N_xy),
                                 L, grid::StretchKind::Uniform, 0.0 };
  grid::Grid g(sub, axes);

  // Problem: x,y periodic; z non-periodic.  For consistency with the
  // topology, every field face on z must be a non-Periodic kind, so we
  // set the velocity & pressure z-faces to plausible defaults too
  // (Dirichlet 0 / Neumann 0 — the test only uses the thermal field, but
  // Problem::validate() requires all 5 fields to agree with the topology).
  boundary::Problem problem;
  problem.topology.axis[to_int(Direction::Z)] = boundary::AxisTopology::NonPeriodic;
  for (auto* fb : { &problem.U, &problem.V, &problem.W }) {
    fb->face(Direction::Z, Side::Minus) = boundary::FaceBc::dirichlet(0.0);
    fb->face(Direction::Z, Side::Plus ) = boundary::FaceBc::dirichlet(0.0);
  }
  problem.P.face(Direction::Z, Side::Minus) = boundary::FaceBc::neumann(0.0);
  problem.P.face(Direction::Z, Side::Plus ) = boundary::FaceBc::neumann(0.0);
  problem.T.face(Direction::Z, Side::Minus) = boundary::FaceBc::dirichlet(T_lo);
  problem.T.face(Direction::Z, Side::Plus ) = boundary::FaceBc::dirichlet(T_hi);
  problem.validate();

  auto backend = parallel::make_default_backend();
  field::FieldRegistry fields(sub, *backend);
  auto& T = fields.add_scalar("T");

  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);
  boundary::BoundaryApplier bc(problem);

  equation::scalar::ScalarEquation
    thermal({ "T", alpha }, g, sub, fields, problem, problem.T, *tdma, bc);

  // ----- Initial condition: T = 0 (interior). Wall ghosts are set by apply_ghost. -----
  T.fill_host(0.0);
  T.exchange_halo();
  bc.apply_ghost(T, problem.T);

  // ----- Time march toward steady state. -----
  for (int n = 0; n < M; ++n) thermal.step(dt);

  // ----- Compare against the numerical steady state. -----
  const int n3 = sub.n_total()[2];
  const int im = sub.n_total()[0] / 2;
  const int jm = sub.n_total()[1] / 2;
  real_t err = 0.0;
  for (int k = 0; k < n3; ++k) {
    const real_t expected = T_lo + (T_hi - T_lo) * static_cast<real_t>(k)
                                                 / static_cast<real_t>(n3 - 1);
    const real_t got = T.host_at(im, jm, k);
    err = std::max(err, std::abs(got - expected));
  }

  if (mpi.is_root()) {
    std::fprintf(stderr,
      "  N_xy = %d, N_z = %d,  M = %d,  dt = %g\n"
      "    L_inf error vs analytical steady state = %.3e\n",
      N_xy, N_z, M, double(dt), double(err));

    // Print a 1-D profile (sanity).
    std::fprintf(stderr, "    T(x_mid, y_mid, z):");
    for (int k = 0; k < n3; ++k) std::fprintf(stderr, " %+.3f", double(T.host_at(im, jm, k)));
    std::fprintf(stderr, "\n");
  }

  // After many steps the diffusion has fully relaxed.  Accept any small
  // residual from finite time integration.
  MPMSTD_TEST_CHECK(err < 1e-5);

  if (mpi.is_root()) {
    mpmstd_test_pass("thermal_wall_z_dirichlet_steady");
    std::cout << "test_thermal_wall: ALL PASS\n";
  }
  return 0;
}
