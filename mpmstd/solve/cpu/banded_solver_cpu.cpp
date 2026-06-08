#include "solve/banded_solver.hpp"

#include <vector>

namespace mpmstd::solve {

// Self-contained Thomas algorithm over a batched tridiagonal core::Bands.
// Row-major [n_row x n_sys]; solves each of the n_sys systems independently,
// overwriting rhs with the solution. (Spike body; P1 swaps in PaScaL_TDMA.)
void solve_banded_cpu(core::Bands& b) {
  const int ns = b.n_sys;
  const int nr = b.n_row;
  if (nr <= 0 || ns <= 0) return;

  std::vector<double> cprime(static_cast<std::size_t>(nr));

  for (int s = 0; s < ns; ++s) {
    // forward elimination
    double beta = b.diag[b.idx(s, 0)];
    b.rhs[b.idx(s, 0)] = b.rhs[b.idx(s, 0)] / beta;
    cprime[0] = b.up1[b.idx(s, 0)] / beta;
    for (int r = 1; r < nr; ++r) {
      const double lo = b.lo1[b.idx(s, r)];
      const double m  = b.diag[b.idx(s, r)] - lo * cprime[r - 1];
      cprime[r] = b.up1[b.idx(s, r)] / m;
      b.rhs[b.idx(s, r)] =
          (b.rhs[b.idx(s, r)] - lo * b.rhs[b.idx(s, r - 1)]) / m;
    }
    // back substitution
    for (int r = nr - 2; r >= 0; --r) {
      b.rhs[b.idx(s, r)] -= cprime[r] * b.rhs[b.idx(s, r + 1)];
    }
  }
}

} // namespace mpmstd::solve
