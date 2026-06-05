// Integration test: PressureSolver projects div(U*) to near-machine-epsilon
// with arbitrary MPI decomposition (np1 x np2 x np3).
//
// Run with e.g.  mpirun -np 4 ./test_pressure_poisson_mpi
// Supported world sizes: 1, 2, 4, 8
//
// Grid: 16x16x16 (uniform), periodic X Y, wall Z.
// Each dimension is split as evenly as possible over the world size.

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
#include <array>

using namespace mpmstd;

namespace {

real_t compute_l_inf_divergence(
    const field::ScalarField& U,
    const field::ScalarField& V,
    const field::ScalarField& W,
    const grid::Grid& g,
    int n1_tot, int n2_tot, int n3_tot,
    int n1_int, int n2_int, int n3_int)
{
  const int h   = kHaloWidth;
  const real_t* u   = U.host_ptr();
  const real_t* v   = V.host_ptr();
  const real_t* w   = W.host_ptr();
  const real_t* dx1 = g.dx_ptr(Direction::X);
  const real_t* dx2 = g.dx_ptr(Direction::Y);
  const real_t* dx3 = g.dx_ptr(Direction::Z);

  real_t linf = 0.0;
  for (int ii = 0; ii < n1_int; ++ii) {
    const int i = ii + h, ip = i + 1;
    for (int jj = 0; jj < n2_int; ++jj) {
      const int j = jj + h, jp = j + 1;
      for (int kk = 0; kk < n3_int; ++kk) {
        const int k = kk + h, kp = k + 1;
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

int main(int argc, char** argv)
{
  parallel::mpi::MpiContext mpi(&argc, &argv);

  const int world = mpi.world_size();

  // Choose np1, np2, np3 based on world size.
  // For simplicity: split entirely in Z (np3 = world, np1=np2=1)
  // but if np3 doesn't divide 16 cleanly, fall back to np1 decomposition.
  // For world=4: np1=1, np2=1, np3=4 → nz_loc=4, n3_I_=4/1=4  ✓
  // For world=8: np1=2, np2=1, np3=4 → nz_loc=4, n3_I_=4/2=2  ✓
  int np1 = 1, np2 = 1, np3 = 1;
  int N = 16;
  if      (world == 1)  { np1=1; np2=1; np3=1; N=16; }
  // np1=2: nz_loc=16/1=16, n3_I=16/2=8 ≥ 3 ✓
  else if (world == 2)  { np1=2; np2=1; np3=1; N=16; }
  // np1=1, np3=4: nz_loc=16/4=4, n3_I=4/1=4 ≥ 3 ✓
  else if (world == 4)  { np1=1; np2=1; np3=4; N=16; }
  // np1=2, np3=4: nz_loc=32/4=8, n3_I=8/2=4 ≥ 3 ✓
  else if (world == 8)  { np1=2; np2=1; np3=4; N=32; }
  else {
    if (mpi.is_root())
      std::fprintf(stderr, "test_pressure_poisson_mpi: unsupported world size %d\n", world);
    return 1;
  }

  if (mpi.is_root())
    std::fprintf(stderr, "  MPI decomposition: np1=%d np2=%d np3=%d  N=%d\n", np1, np2, np3, N);

  parallel::mpi::MpiTopology topo(mpi, {np1, np2, np3}, {true, true, false});
  parallel::mpi::Subdomain   sub (topo, {N, N, N});

  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a)
    axes[a] = grid::AxisConfig{ N, 1.0, grid::StretchKind::Uniform, 0.0 };
  grid::Grid g(sub, axes);

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

  const int n1t = sub.n_total()[0], n2t = sub.n_total()[1], n3t = sub.n_total()[2];
  const int n1i = sub.n_interior()[0];
  const int n2i = sub.n_interior()[1];
  const int n3i = sub.n_interior()[2];

  // Initialise a divergent velocity field.
  const std::vector<real_t>& zf = g.xf(Direction::Z);
  constexpr real_t kPi = static_cast<real_t>(M_PI);

  for (int i = 0; i < n1t; ++i)
    for (int j = 0; j < n2t; ++j)
      for (int k = 0; k < n3t; ++k) {
        U.host_at(i, j, k) = real_t{0};
        V.host_at(i, j, k) = real_t{0};
        W.host_at(i, j, k) = std::sin(kPi * zf[k] / real_t{1.0});
        P.host_at(i, j, k) = real_t{0};
      }

  U.exchange_halo(); bc.apply_ghost(U, problem.U);
  V.exchange_halo(); bc.apply_ghost(V, problem.V);
  W.exchange_halo(); bc.apply_ghost(W, problem.W);
  P.exchange_halo(); bc.apply_ghost(P, problem.P);

  const real_t div_before = compute_l_inf_divergence(U, V, W, g, n1t, n2t, n3t, n1i, n2i, n3i);

  // Reduce to global max
  real_t g_div_before = div_before;
  MPI_Allreduce(&div_before,  &g_div_before,  1,
                std::is_same_v<real_t,double> ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX,
                topo.cart_comm());

  equation::pressure::PressureSolver ps(g, sub, fields, problem, *tdma, bc);
  ps.solve(real_t{0.1}, U, V, W, P);

  const real_t div_after = compute_l_inf_divergence(U, V, W, g, n1t, n2t, n3t, n1i, n2i, n3i);
  real_t g_div_after = div_after;
  MPI_Allreduce(&div_after,  &g_div_after,  1,
                std::is_same_v<real_t,double> ? MPI_DOUBLE : MPI_FLOAT, MPI_MAX,
                topo.cart_comm());

  if (mpi.is_root()) {
    std::fprintf(stderr,
      "  L_inf(div U*) before = %.6e\n"
      "  L_inf(div U ) after  = %.6e\n",
      double(g_div_before), double(g_div_after));
  }

  MPMSTD_TEST_CHECK(g_div_before > 1e-3);
  MPMSTD_TEST_CHECK(g_div_after  < 1e-10);

  if (mpi.is_root()) {
    mpmstd_test_pass("pressure_poisson_mpi_projection");
    std::cout << "test_pressure_poisson_mpi: ALL PASS\n";
  }
  return 0;
}
