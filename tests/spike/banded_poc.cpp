// P-0.5 spike (recommended) — Bands / ScalarSystem + solve_banded_cpu.
//
//   ./banded_poc        (single rank is enough)
//
// Validates the assemble -> solve data path + signatures by solving a 1D
// Poisson problem  -u'' = f  on [0,1] with u(0)=u(1)=0 and a manufactured
// solution u = sin(pi x), f = pi^2 sin(pi x). Interior nodes i=1..N, h=1/(N+1):
//   -u_{i-1} + 2 u_i - u_{i+1} = h^2 f_i      (Dirichlet boundaries folded as 0)
// A second-order scheme, so max|error| ~ O(h^2). We batch a few identical
// systems to exercise the [n_row x n_sys] layout.

#include "core/bands.hpp"
#include "core/system.hpp"
#include "solve/banded_solver.hpp"

#include "parallel/mpi/mpi_context.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace mpmstd;

int main(int argc, char** argv) {
  parallel::mpi::MpiContext mpi(&argc, &argv);

  const int    N  = 128;
  const double pi = 3.14159265358979323846;
  const double h  = 1.0 / (N + 1);

  core::ScalarSystem sys;                       // exercise System -> Bands path
  core::Bands&       b = sys.along(Direction::Z);
  const int n_sys = 3;                          // batch a few identical systems
  b.allocate(n_sys, N, /*bandwidth=*/1);

  for (int s = 0; s < n_sys; ++s)
    for (int r = 0; r < N; ++r) {
      const double x = (r + 1) * h;
      b.lo1 [b.idx(s, r)] = (r == 0)     ? 0.0 : -1.0;
      b.up1 [b.idx(s, r)] = (r == N - 1) ? 0.0 : -1.0;
      b.diag[b.idx(s, r)] = 2.0;
      b.rhs [b.idx(s, r)] = h * h * pi * pi * std::sin(pi * x);
    }

  solve::solve_banded_cpu(b);

  double max_err = 0.0;
  for (int s = 0; s < n_sys; ++s)
    for (int r = 0; r < N; ++r) {
      const double x = (r + 1) * h;
      max_err = std::max(max_err, std::abs(b.rhs[b.idx(s, r)] - std::sin(pi * x)));
    }

  const bool ok = (max_err < 1e-3);   // 2nd-order: ~2.5e-5 at N=128
  if (mpi.is_root()) {
    if (ok) std::printf("[PASS] banded_poc: Bands/ScalarSystem + solve_banded_cpu — 1D Poisson max|err|=%.3e (< 1e-3)\n", max_err);
    else    std::printf("[FAIL] banded_poc: max|err|=%.3e\n", max_err);
  }
  return ok ? 0 : 1;
}
