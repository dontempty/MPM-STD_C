// Multi-rank unit test: BoundaryApplier only touches ghost cells on ranks
// that own a global boundary face.
//
// We split the Z axis (the wall-normal direction in the RBC default) across
// multiple ranks.  Only the bottom-rank (z_rank == 0) should fill the -z
// ghost, and only the top-rank (z_rank == nproc_z - 1) should fill the +z
// ghost.  All interior-rank z faces are interface halos and must remain at
// the sentinel value chosen below.
//
// Run with: mpirun -np 2 test_bc_apply_multi  (also np=4 with {1,1,4})

#include "common/main.hpp"
#include "parallel/main.hpp"
#include "field/main.hpp"
#include "boundary/main.hpp"
#include "test_helpers.hpp"

#include <array>

using namespace mpmstd;
using namespace mpmstd::boundary;

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);
  const int np = mpi.world_size();

  // Z = wall-normal, distributed across all ranks.
  std::array<int, 3> dims     = {1, 1, np};
  std::array<bool, 3> periodic = {true, true, false};

  parallel::mpi::MpiTopology topo(mpi, dims, periodic);
  parallel::mpi::Subdomain   sub (topo, {4, 4, 4 * np});   // 4*np cells along z

  auto backend = parallel::make_default_backend();
  field::FieldRegistry reg(sub, *backend);
  auto& T = reg.add_scalar("T");

  // Sentinel value lets us detect which ghost slabs got overwritten.
  constexpr real_t SENTINEL = -999.0;
  T.fill_host(SENTINEL);

  // Interior gets a distinct value so we can also check Neumann behaviour
  // (currently we only verify the Dirichlet T case).  Use 1.0 here.
  const auto n_tot = sub.n_total();
  for (int i = kHaloWidth; i < n_tot[0] - kHaloWidth; ++i)
    for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j)
      for (int k = kHaloWidth; k < n_tot[2] - kHaloWidth; ++k)
        T.host_at(i, j, k) = 1.0;

  // Explicitly configure: z non-periodic with Dirichlet ±0.5 on T.
  Problem p;
  p.topology.axis[to_int(Direction::Z)] = AxisTopology::NonPeriodic;
  p.T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet( 0.5);
  p.T.face(Direction::Z, Side::Plus ) = FaceBc::dirichlet(-0.5);
  BoundaryApplier app(p);
  app.apply_ghost(T, p.T);

  // ---- Verify ----
  const int my_z = topo.axis(Direction::Z).rank;
  const bool own_minus = (my_z == 0);
  const bool own_plus  = (my_z == np - 1);

  // Lower-z ghost plane:
  for (int i = kHaloWidth; i < n_tot[0] - kHaloWidth; ++i)
    for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j) {
      const real_t got = T.host_at(i, j, 0);
      if (own_minus) {
        MPMSTD_TEST_NEAR(got, 0.5, 0.0);                  // Dirichlet hot wall
      } else {
        MPMSTD_TEST_NEAR(got, SENTINEL, 0.0);             // interior interface; untouched
      }
    }
  // Upper-z ghost plane:
  for (int i = kHaloWidth; i < n_tot[0] - kHaloWidth; ++i)
    for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j) {
      const real_t got = T.host_at(i, j, n_tot[2] - 1);
      if (own_plus) {
        MPMSTD_TEST_NEAR(got, -0.5, 0.0);                 // Dirichlet cold wall
      } else {
        MPMSTD_TEST_NEAR(got, SENTINEL, 0.0);
      }
    }

  // x and y face ghosts: Periodic → BoundaryApplier should not touch them,
  // and on np_x=np_y=1 with periods=true the helper recognises wrap-around
  // (axis comm neighbours are valid, not MPI_PROC_NULL) so it skips them.
  // Verify the lower-x plane stayed at SENTINEL on every rank.
  for (int j = kHaloWidth; j < n_tot[1] - kHaloWidth; ++j)
    for (int k = kHaloWidth; k < n_tot[2] - kHaloWidth; ++k) {
      MPMSTD_TEST_NEAR(T.host_at(0, j, k), SENTINEL, 0.0);
    }

  if (mpi.is_root()) {
    mpmstd_test_pass("bc_apply_multi_only_wall_ranks_touch_z_ghost");
    std::cout << "test_bc_apply_multi: ALL PASS\n";
  }
  return 0;
}
