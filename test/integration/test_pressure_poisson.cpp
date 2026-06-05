// Integration test: PressureSolver projects div(U*) to machine-epsilon zero.
//
// Setup:
//   Domain  [0,1]^3,  uniform 16×16×16
//   X, Y    periodic   (required for FFT-based pressure solver)
//   Z       wall       (Neumann dP/dz=0 at both ends)
//
// After solve(dt, U, V, W, P):
//   L_inf(div(U^{n+1})) < 1e-10
//
// Theory: discrete pressure correction is exact (same FD stencil in RHS and
//         gradient projection), so residual is purely floating-point.

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "boundary/main.hpp"
#include "grid/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "equation/pressure/main.hpp"
#include "test_helpers.hpp"

#include <cmath>
#include <iostream>

using namespace mpmstd;

namespace {

// Returns (L_inf, max_pos, max_div) of div(U,V,W) at interior cells.
// Uses the SAME forward-difference stencil as compute_divergence_rhs_.
real_t compute_l_inf_divergence(
    const field::ScalarField& U,
    const field::ScalarField& V,
    const field::ScalarField& W,
    const grid::Grid& g,
    int n1_tot, int n2_tot, int n3_tot,
    int n1_int, int n2_int, int n3_int) {

  const int h   = kHaloWidth;
  const real_t* u   = U.host_ptr();
  const real_t* v   = V.host_ptr();
  const real_t* w   = W.host_ptr();
  const real_t* dx1 = g.dx_ptr(Direction::X);
  const real_t* dx2 = g.dx_ptr(Direction::Y);
  const real_t* dx3 = g.dx_ptr(Direction::Z);

  real_t linf = 0.0;
  for (int ii = 0; ii < n1_int; ++ii) {
    const int i  = ii + h, ip = i + 1;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int j  = jj + h, jp = j + 1;
      for (int kk = 0; kk < n3_int; ++kk) {
        const int k  = kk + h, kp = k + 1;

        auto at = [&](const real_t* f, int a, int b, int c) {
          return f[(a * n2_tot + b) * n3_tot + c];
        };
        const real_t div =
            (at(u, ip, j,  k ) - at(u, i, j, k)) / dx1[i]
          + (at(v, i,  jp, k ) - at(v, i, j, k)) / dx2[j]
          + (at(w, i,  j,  kp) - at(w, i, j, k)) / dx3[k];
        if (std::abs(div) > linf) linf = std::abs(div);
      }
    }
  }
  return linf;
}

} // anonymous namespace


int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  const int    N  = 16;
  const real_t L  = 1.0;
  const real_t dt = 0.1;

  // Topology: periodic X, Y; wall Z.
  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, true, false});
  parallel::mpi::Subdomain   sub (topo, {N, N, N});

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a)
    axes[a] = grid::AxisConfig{ N, L, grid::StretchKind::Uniform, 0.0 };
  grid::Grid g(sub, axes);

  // Problem: x, y periodic; z non-periodic (Neumann pressure, no-slip velocity).
  boundary::Problem problem;
  problem.topology.axis[to_int(Direction::Z)] = boundary::AxisTopology::NonPeriodic;
  for (auto* fb : { &problem.U, &problem.V, &problem.W }) {
    fb->face(Direction::Z, Side::Minus) = boundary::FaceBc::dirichlet(0.0);
    fb->face(Direction::Z, Side::Plus ) = boundary::FaceBc::dirichlet(0.0);
  }
  problem.P.face(Direction::Z, Side::Minus) = boundary::FaceBc::neumann(0.0);
  problem.P.face(Direction::Z, Side::Plus ) = boundary::FaceBc::neumann(0.0);
  problem.T.face(Direction::Z, Side::Minus) = boundary::FaceBc::dirichlet(0.0);
  problem.T.face(Direction::Z, Side::Plus ) = boundary::FaceBc::dirichlet(0.0);
  problem.validate();

  auto backend = parallel::make_default_backend();
  field::FieldRegistry fields(sub, *backend);
  auto& U = fields.add_scalar("U");
  auto& V = fields.add_scalar("V");
  auto& W = fields.add_scalar("W");
  auto& P = fields.add_scalar("P");

  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);
  boundary::BoundaryApplier bc(problem);

  const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
  const int n1_int = sub.n_interior()[0];
  const int n2_int = sub.n_interior()[1];
  const int n3_int = sub.n_interior()[2];

  const std::vector<real_t>& xc = g.xc(Direction::X);
  const std::vector<real_t>& yc = g.xc(Direction::Y);
  // W is face-centred on the Z-face: W[k] lives at face k (z = xf[k]).
  // Using face positions guarantees W = 0 at the wall faces (k=h and k=h+N),
  // so the discrete compatibility condition sum(div) = 0 holds exactly.
  const std::vector<real_t>& zf = g.xf(Direction::Z);
  constexpr real_t kPi = static_cast<real_t>(M_PI);

  // --- Initialize divergent velocity field (pure W divergence) ---
  const real_t A_u = 0.0, A_v = 0.0, A_w = 1.0;

  for (int i = 0; i < n1; ++i)
    for (int j = 0; j < n2; ++j)
      for (int k = 0; k < n3; ++k) {
        U.host_at(i, j, k) = A_u * std::sin(2.0 * kPi * xc[i] / L);
        V.host_at(i, j, k) = A_v * std::cos(2.0 * kPi * yc[j] / L);
        W.host_at(i, j, k) = A_w * std::sin(kPi * zf[k] / L);
        P.host_at(i, j, k) = 0.0;
      }

  U.exchange_halo(); bc.apply_ghost(U, problem.U);
  V.exchange_halo(); bc.apply_ghost(V, problem.V);
  W.exchange_halo(); bc.apply_ghost(W, problem.W);
  P.exchange_halo(); bc.apply_ghost(P, problem.P);

  const real_t div_before = compute_l_inf_divergence(U, V, W, g, n1, n2, n3, n1_int, n2_int, n3_int);

  // --- Pressure solve + projection ---
  equation::pressure::PressureSolver ps(g, sub, fields, problem, *tdma, bc);
  ps.solve(dt, U, V, W, P);

  const real_t div_after = compute_l_inf_divergence(U, V, W, g, n1, n2, n3, n1_int, n2_int, n3_int);

  if (mpi.is_root()) {
    std::fprintf(stderr,
      "  N = %d×%d×%d, dt = %g\n"
      "  L_inf(div U*) before  = %.6e\n"
      "  L_inf(div U ) after   = %.6e\n",
      N, N, N, double(dt), double(div_before), double(div_after));
  }

  MPMSTD_TEST_CHECK(div_before > 1e-3);
  MPMSTD_TEST_CHECK(div_after  < 1e-10);

  if (mpi.is_root()) {
    mpmstd_test_pass("pressure_poisson_projection");
    std::cout << "test_pressure_poisson: ALL PASS\n";
  }
  return 0;
}
