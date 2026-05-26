// Unit test: PaScaLTDMACpuBackend wraps PaScaL_TDMA_C correctly.
//
// Setup: solve a known small tridiagonal system on 1 rank and compare against
// the analytic answer.  Both non-cyclic and cyclic variants.

#include "common/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "parallel/main.hpp"
#include "test_helpers.hpp"

#include <vector>

using namespace mpmstd;
using namespace mpmstd::linear_solver::tdma;

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);
  parallel::mpi::MpiTopology topo(mpi, {1, 1, 1}, {true, true, true});

  // ---- Non-cyclic: identity-diagonal system ----
  // A simple diagonal-only system [1] · x = b, so x = b.
  {
    auto reg = TdmaRegistry::make_default(topo);
    TdmaSolver& tdma = reg->get(Direction::Z);

    const int n_sys = 4, n_row = 5;
    std::vector<real_t> A(n_sys * n_row, 0.0);  // sub-diagonal: 0
    std::vector<real_t> B(n_sys * n_row, 1.0);  // diagonal: 1
    std::vector<real_t> C(n_sys * n_row, 0.0);  // super-diagonal: 0
    std::vector<real_t> D(n_sys * n_row);
    for (int j = 0; j < n_row; ++j)
      for (int i = 0; i < n_sys; ++i)
        D[j * n_sys + i] = static_cast<real_t>(j * 100 + i);

    auto D_expected = D;
    tdma.solve_many(A.data(), B.data(), C.data(), D.data(), n_sys, n_row);

    for (std::size_t k = 0; k < D.size(); ++k)
      MPMSTD_TEST_NEAR(D[k], D_expected[k], 1e-14);

    mpmstd_test_pass("tdma_solve_many_identity");
  }

  // ---- Non-cyclic: known tridiagonal solution ----
  // Solve (B = 2, A = C = -1) on a 5-row system with RHS [1, 0, 0, 0, 1]:
  // analytic solution (Dirichlet-like) for this Laplacian-style row.
  {
    auto reg = TdmaRegistry::make_default(topo);
    TdmaSolver& tdma = reg->get(Direction::Z);

    const int n_sys = 1, n_row = 5;
    std::vector<real_t> A(n_sys * n_row, -1.0);
    std::vector<real_t> B(n_sys * n_row,  2.0);
    std::vector<real_t> C(n_sys * n_row, -1.0);
    std::vector<real_t> D(n_sys * n_row);
    D = {1.0, 0.0, 0.0, 0.0, 1.0};

    // First-row stencil uses A[0]; last-row uses C[n_row-1]. For a "Dirichlet"
    // treatment we set those to 0:
    A[0]                    = 0.0;
    C[(n_row - 1) * n_sys]  = 0.0;
    tdma.solve_many(A.data(), B.data(), C.data(), D.data(), n_sys, n_row);

    // Analytic: linear system  [2 -1 0 0 0; -1 2 -1 0 0; 0 -1 2 -1 0;
    //                           0 0 -1 2 -1; 0 0 0 -1 2] x = [1;0;0;0;1]
    // Solution: x = [1; 1; 1; 1; 1]
    for (int j = 0; j < n_row; ++j) {
      MPMSTD_TEST_NEAR(D[j * n_sys + 0], 1.0, 1e-12);
    }
    mpmstd_test_pass("tdma_solve_many_laplacian_dirichlet");
  }

  // ---- Cyclic: simple identity ----
  {
    auto reg = TdmaRegistry::make_default(topo);
    TdmaSolver& tdma = reg->get(Direction::Z);

    const int n_sys = 2, n_row = 4;
    std::vector<real_t> A(n_sys * n_row, 0.0);
    std::vector<real_t> B(n_sys * n_row, 1.0);
    std::vector<real_t> C(n_sys * n_row, 0.0);
    std::vector<real_t> D{ 1.0, 2.0,  3.0, 4.0,  5.0, 6.0,  7.0, 8.0 };

    auto expected = D;
    tdma.solve_many_cyclic(A.data(), B.data(), C.data(), D.data(), n_sys, n_row);

    for (std::size_t k = 0; k < D.size(); ++k)
      MPMSTD_TEST_NEAR(D[k], expected[k], 1e-14);

    mpmstd_test_pass("tdma_solve_many_cyclic_identity");
  }

  // ---- Plan caching: solving twice with different n_sys should not crash ----
  {
    auto reg = TdmaRegistry::make_default(topo);
    TdmaSolver& tdma = reg->get(Direction::Z);

    for (int n_sys : {3, 5, 3}) {  // first call creates a plan; third reuses cached plan for n_sys=3
      const int n_row = 4;
      std::vector<real_t> A(n_sys * n_row, 0.0);
      std::vector<real_t> B(n_sys * n_row, 1.0);
      std::vector<real_t> C(n_sys * n_row, 0.0);
      std::vector<real_t> D(n_sys * n_row, 7.0);
      tdma.solve_many(A.data(), B.data(), C.data(), D.data(), n_sys, n_row);
      for (auto v : D) MPMSTD_TEST_NEAR(v, 7.0, 1e-14);
    }
    mpmstd_test_pass("tdma_plan_cache");
  }

  if (mpi.is_root()) std::cout << "test_tdma_backend_cpu: ALL PASS\n";
  return 0;
}
