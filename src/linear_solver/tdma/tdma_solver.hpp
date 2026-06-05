#pragma once

#include "common/types.hpp"

namespace mpmstd::linear_solver::tdma {

// Abstract TDMA backend.
//
// Memory layout (matches PaScaL_TDMA_C): row-major [n_row × n_sys], i.e. row j
// starts at offset j*n_sys.  `n_sys` independent tridiagonal systems each of
// length `n_row` are solved simultaneously; the distribution across MPI ranks
// is handled by the backend's PaScaL plan (created at construction time).
//
// A,B,C : sub-, main-, super-diagonal coefficients (each [n_row × n_sys])
// D     : right-hand side; overwritten with solution on return
//
// `solve_many`        : non-cyclic (e.g. Dirichlet/Neumann on the wall axis)
// `solve_many_cyclic` : cyclic / periodic
//
// Pointers passed to solve_*() must point to the buffer matching this
// backend's residency:
//   CPU backend  : host pointer
//   CUDA backend : device pointer

class TdmaSolver {
public:
  virtual ~TdmaSolver() = default;

  virtual void solve_many(real_t* A, real_t* B, real_t* C, real_t* D,
                           int n_sys, int n_row) = 0;

  virtual void solve_many_cyclic(real_t* A, real_t* B, real_t* C, real_t* D,
                                  int n_sys, int n_row) = 0;

  // Optional: set spectral-radius coefficients before z-direction solve.
  // Called by the ADI driver before solve_many on the wall-normal axis.
  // Default no-op for backends that don't use filtering.
  virtual void set_rho(const real_t* A, const real_t* B, const real_t* C,
                        int n_sys) {}
  virtual void set_eps_constant(double /*eps*/) {}
};

} // namespace mpmstd::linear_solver::tdma
