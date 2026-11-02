#ifndef TDMA_LOCAL_HPP
#define TDMA_LOCAL_HPP

#include <cstddef>

/// Local (single-rank) Thomas algorithm for multiple RHS systems.
/// Layout: row-major [n_row × n_sys], i.e. row j starts at offset j*n_sys.

void ftdma_many(double* __restrict A,
               double* __restrict B,
               double* __restrict C,
               double* __restrict D,
               int n_sys, int n_row);

/// Cyclic (periodic) Thomas via Sherman-Morrison.
/// `E` is a caller-owned [n_row × n_sys] workspace; it is fully initialised
/// here, so the caller does not need to clear it between calls. Passing the
/// buffer in (rather than allocating locally) mirrors the GPU entry point
/// `ftdma_cyclic_many_cuda(..., d_E, ...)` and keeps the per-solve heap
/// allocation out of the hot path.
void ftdma_cyclic_many(double* __restrict A,
                    double* __restrict B,
                    double* __restrict C,
                    double* __restrict D,
                    double* __restrict E,
                    int n_sys, int n_row);

#endif // TDMA_LOCAL_HPP
