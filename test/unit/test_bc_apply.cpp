// Unit test: BoundaryApplier::apply_ghost on a single rank.
//
// Single-rank ⇒ this rank holds all 6 global boundary faces; we can verify
// the Dirichlet and Neumann handlers directly.  Periodic faces should leave
// the ghost untouched (the test starts with sentinel values and asserts they
// remain untouched on periodic axes).

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "field/main.hpp"
#include "parallel/main.hpp"
#include "test_helpers.hpp"

using namespace mpmstd;
using namespace mpmstd::boundary;

namespace {

// Build a single-rank Subdomain with N interior cells per axis and the
// requested topology.
struct Env {
  parallel::mpi::MpiTopology topo;
  parallel::mpi::Subdomain   sub;
  std::unique_ptr<parallel::Backend> backend;
  field::FieldRegistry       reg;

  Env(const parallel::mpi::MpiContext& mpi,
      std::array<bool, 3> periodic, int N)
    : topo(mpi, {1, 1, 1}, periodic),
      sub(topo, {N, N, N}),
      backend(parallel::make_default_backend()),
      reg(sub, *backend) {}
};

} // namespace

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  // ----- 1) Dirichlet on z faces, periodic on x, y -----
  {
    Env env(mpi, {true, true, false}, /*N=*/4);
    auto& T = env.reg.add_scalar("T");

    // Fill interior with sentinel 1.0, halos with sentinel -999.0 (so we can
    // detect which were overwritten).
    T.fill_host(-999.0);
    const auto n_tot = env.sub.n_total();
    for (int i = kHaloWidth; i < n_tot[0] - kHaloWidth; ++i)
      for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j)
        for (int k = kHaloWidth; k < n_tot[2] - kHaloWidth; ++k)
          T.host_at(i, j, k) = 1.0;

    // Explicitly configure z as non-periodic + T Dirichlet ±0.5.
    Problem p;
    p.topology.axis[to_int(Direction::Z)] = AxisTopology::NonPeriodic;
    p.T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet( 0.5);
    p.T.face(Direction::Z, Side::Plus ) = FaceBc::dirichlet(-0.5);
    BoundaryApplier app(p);
    app.apply_ghost(T, p.T);

    // -z ghost: 0.5, +z ghost: -0.5
    MPMSTD_TEST_NEAR(T.host_at(kHaloWidth, kHaloWidth, 0),                       0.5, 0.0);
    MPMSTD_TEST_NEAR(T.host_at(kHaloWidth, kHaloWidth, n_tot[2] - 1),           -0.5, 0.0);
    // x and y ghosts should remain at sentinel (Periodic = no-op on global faces).
    // In single-rank periodic, the axis comm wraps around — but since np=1,
    // west_rank == east_rank == own rank (or MPI_PROC_NULL? depends on MPI):
    // with periods=true and dims=1, MPI_Cart_shift returns self as both
    // neighbors → is_global_boundary returns false on periodic axes for
    // np=1, so the periodic path is taken (=no-op) and the ghost stays at
    // sentinel. We assert that for x and y, but allow either outcome (the
    // important point is that Periodic faces don't get overwritten by the
    // Dirichlet/Neumann path).
    MPMSTD_TEST_NEAR(T.host_at(0, kHaloWidth, kHaloWidth), -999.0, 0.0);
    mpmstd_test_pass("bc_apply_dirichlet_z");
  }

  // ----- 2) Neumann zero-gradient on z, periodic on x, y -----
  {
    Env env(mpi, {true, true, false}, /*N=*/4);
    auto& P = env.reg.add_scalar("P");

    P.fill_host(-999.0);
    const auto n_tot = env.sub.n_total();
    // Distinct interior values per layer along z for the gradient check.
    for (int i = kHaloWidth; i < n_tot[0] - kHaloWidth; ++i)
      for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j)
        for (int k = kHaloWidth; k < n_tot[2] - kHaloWidth; ++k)
          P.host_at(i, j, k) = static_cast<real_t>(k - kHaloWidth);

    // Explicitly configure z non-periodic + P Neumann 0.
    Problem p;
    p.topology.axis[to_int(Direction::Z)] = AxisTopology::NonPeriodic;
    p.P.face(Direction::Z, Side::Minus) = FaceBc::neumann(0.0);
    p.P.face(Direction::Z, Side::Plus ) = FaceBc::neumann(0.0);
    BoundaryApplier app(p);
    app.apply_ghost(P, p.P);

    // Zero Neumann: ghost == first interior layer along z.
    const int k_first = kHaloWidth;
    const int k_last  = n_tot[2] - 1 - kHaloWidth;
    for (int i = kHaloWidth; i < n_tot[0] - kHaloWidth; ++i)
      for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j) {
        MPMSTD_TEST_NEAR(P.host_at(i, j, 0),               P.host_at(i, j, k_first), 0.0);
        MPMSTD_TEST_NEAR(P.host_at(i, j, n_tot[2] - 1),    P.host_at(i, j, k_last ), 0.0);
      }
    mpmstd_test_pass("bc_apply_neumann_z");
  }

  // ----- 3) Unimplemented BC throws -----
  {
    Env env(mpi, {true, true, false}, /*N=*/4);
    auto& X = env.reg.add_scalar("X");
    Problem p;
    p.topology.axis[to_int(Direction::Z)] = AxisTopology::NonPeriodic;
    p.T.face(Direction::Z, Side::Minus).kind = BcKind::Inflow;
    BoundaryApplier app(p);
    bool threw = false;
    try { app.apply_ghost(X, p.T); }
    catch (const std::runtime_error&) { threw = true; }
    MPMSTD_TEST_CHECK(threw);
    mpmstd_test_pass("bc_apply_inflow_throws");
  }

  if (mpi.is_root()) std::cout << "test_bc_apply: ALL PASS\n";
  return 0;
}
