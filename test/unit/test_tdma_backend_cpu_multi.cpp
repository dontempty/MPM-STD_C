// Multi-rank unit test: PaScaLTDMACpuBackend correctly couples sub-systems
// across MPI ranks.
//
// We arrange a Laplacian-like tridiagonal system distributed along the Z axis
// (topology {1, 1, world_size}), so the TDMA wraps onto PaScaL_TDMA's
// alltoall transpose path.  The right-hand side is set so the analytic
// solution is a constant vector (verifiable on every rank).
//
// Run with: mpirun -np 4 test_tdma_backend_cpu_multi   (also np=2, 8)

#include "common/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "parallel/main.hpp"
#include "test_helpers.hpp"

#include <vector>

using namespace mpmstd;
using namespace mpmstd::linear_solver::tdma;

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  // Distribute along Z, total nprocs ranks. x,y left as 1.
  const int np = mpi.world_size();
  parallel::mpi::MpiTopology topo(mpi, {1, 1, np}, {true, true, true});

  auto reg = TdmaRegistry::make_default(topo);
  TdmaSolver& tdma = reg->get(Direction::Z);

  // ----- Test 1: non-cyclic Laplacian distributed over Z -----
  // Global system size N_glb = 8 * np.  Each rank owns 8 consecutive rows.
  // System:  B = 2, A = C = -1, with Dirichlet endpoints (A[0]=0, C[N-1]=0
  // on the global side).  RHS = [1, 0, 0, ..., 0, 1] yields x ≡ 1.
  //
  // NOTE: PaScaL_TDMA requires n_sys >= nprocs so every rank gets at least
  // one of the independent systems after the alltoall transpose. We pick a
  // batch size that scales with the world size.
  {
    const int n_local = 8;                    // rows on each rank
    const int n_sys   = (np >= 4) ? np : 4;   // independent systems for batching

    std::vector<real_t> A(n_sys * n_local, -1.0);
    std::vector<real_t> B(n_sys * n_local,  2.0);
    std::vector<real_t> C(n_sys * n_local, -1.0);
    std::vector<real_t> D(n_sys * n_local,  0.0);

    // RHS = 1 on the very first global row and the very last global row,
    // zero elsewhere.
    const int my_z_rank = topo.axis(Direction::Z).rank;
    if (my_z_rank == 0) {
      // Drop the A coefficient on global row 0 (Dirichlet-like first row).
      for (int i = 0; i < n_sys; ++i) {
        A[0 * n_sys + i] = 0.0;
        D[0 * n_sys + i] = 1.0;
      }
    }
    if (my_z_rank == np - 1) {
      for (int i = 0; i < n_sys; ++i) {
        C[(n_local - 1) * n_sys + i] = 0.0;
        D[(n_local - 1) * n_sys + i] = 1.0;
      }
    }

    tdma.solve_many(A.data(), B.data(), C.data(), D.data(), n_sys, n_local);

    // Every entry of D should be ~1.0 on every rank.
    for (int j = 0; j < n_local; ++j) {
      for (int i = 0; i < n_sys; ++i) {
        MPMSTD_TEST_NEAR(D[j * n_sys + i], 1.0, 1e-10);
      }
    }
    if (mpi.is_root()) mpmstd_test_pass("tdma_multi_laplacian_distributed");
  }

  // ----- Test 2: cyclic identity distributed -----
  // Diagonal-only system (A=C=0, B=1, D=arbitrary) using the cyclic path.
  // Output should equal input on every rank.
  {
    const int n_local = 5;
    const int n_sys   = (np >= 4) ? np : 4;
    std::vector<real_t> A(n_sys * n_local, 0.0);
    std::vector<real_t> B(n_sys * n_local, 1.0);
    std::vector<real_t> C(n_sys * n_local, 0.0);
    std::vector<real_t> D(n_sys * n_local);

    const int my_z_rank = topo.axis(Direction::Z).rank;
    for (int j = 0; j < n_local; ++j)
      for (int i = 0; i < n_sys; ++i)
        D[j * n_sys + i] = static_cast<real_t>(my_z_rank * 1000 + j * 10 + i);

    auto expected = D;
    tdma.solve_many_cyclic(A.data(), B.data(), C.data(), D.data(), n_sys, n_local);

    for (std::size_t k = 0; k < D.size(); ++k) {
      MPMSTD_TEST_NEAR(D[k], expected[k], 1e-12);
    }
    if (mpi.is_root()) mpmstd_test_pass("tdma_multi_cyclic_identity_distributed");
  }

  if (mpi.is_root()) std::cout << "test_tdma_backend_cpu_multi: ALL PASS\n";
  return 0;
}
