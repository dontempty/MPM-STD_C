#pragma once

#include "common/types.hpp"   // real_t

#include <cstddef>
#include <vector>

namespace mpmstd::core {

// =============================================================================
// Bands  — banded coefficient storage for a BATCH of independent 1D systems
//          along one ADI sweep direction (the "A" of A x = b).
// -----------------------------------------------------------------------------
// Memory layout matches PaScaL_TDMA_C: row-major [n_row x n_sys], i.e. system
// `s`, row `r` lives at offset r*n_sys + s (see idx()). `n_sys` independent
// tridiagonal systems each of length `n_row` are solved simultaneously.
//
//   bandwidth = 1 : tridiagonal (lo1, diag, up1)            <- current
//   bandwidth = 2 : pentadiagonal (lo2, lo1, diag, up1, up2) <- future / high-order
//
// rhs is OVERWRITTEN with the solution in place (PaScaL_TDMA convention).
//
// For the CPU spike this struct OWNS host storage (std::vector). The GPU twin
// will hold device pointers with the identical logical layout; the assemble_*
// (fills) and solve_* (consumes) free functions are what differ by backend, not
// this layout.
// =============================================================================
struct Bands {
  int bandwidth = 1;   // 1 = tridiagonal, 2 = pentadiagonal
  int n_sys     = 0;   // number of independent systems (batch size)
  int n_row     = 0;   // rows per system (points along the sweep axis)

  std::vector<real_t> lo2, lo1, diag, up1, up2;  // diagonals
  std::vector<real_t> rhs;                        // RHS -> solution (in place)

  void allocate(int n_sys_, int n_row_, int bandwidth_ = 1) {
    bandwidth = bandwidth_;
    n_sys     = n_sys_;
    n_row     = n_row_;
    const std::size_t n = size();
    lo1.assign(n, real_t{0});
    diag.assign(n, real_t{0});
    up1.assign(n, real_t{0});
    rhs.assign(n, real_t{0});
    if (bandwidth >= 2) {
      lo2.assign(n, real_t{0});
      up2.assign(n, real_t{0});
    }
  }

  std::size_t size() const {
    return static_cast<std::size_t>(n_sys) * static_cast<std::size_t>(n_row);
  }

  // flat index of system `s` (0..n_sys-1), row `r` (0..n_row-1)
  int idx(int s, int r) const { return r * n_sys + s; }
};

} // namespace mpmstd::core
