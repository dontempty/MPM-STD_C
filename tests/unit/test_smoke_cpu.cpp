// P0 smoke test — proves a test binary links the refactored libmpmstd and calls
// a real library symbol (solve::solve_banded_cpu). Not a numerics test (that's
// the spike/banded_poc + P1 regressions); just build+link plumbing.

#include "core/bands.hpp"
#include "solve/banded_solver.hpp"

#include <cmath>
#include <cstdio>

int main() {
  mpmstd::core::Bands b;
  b.allocate(/*n_sys=*/1, /*n_row=*/3, /*bandwidth=*/1);
  for (int r = 0; r < 3; ++r) {            // identity system: diag=1 → solution == rhs
    b.diag[b.idx(0, r)] = 1.0;
    b.rhs [b.idx(0, r)] = r + 1.0;
  }
  mpmstd::solve::solve_banded_cpu(b);

  const bool ok = std::fabs(b.rhs[b.idx(0, 0)] - 1.0) < 1e-12 &&
                  std::fabs(b.rhs[b.idx(0, 1)] - 2.0) < 1e-12 &&
                  std::fabs(b.rhs[b.idx(0, 2)] - 3.0) < 1e-12;
  std::printf("%s test_smoke_cpu: solve_banded_cpu via libmpmstd\n", ok ? "[PASS]" : "[FAIL]");
  return ok ? 0 : 1;
}
