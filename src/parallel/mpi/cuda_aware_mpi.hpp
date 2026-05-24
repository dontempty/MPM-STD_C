#pragma once

#include "common/macros.hpp"
#include "common/types.hpp"

namespace mpmstd::parallel::mpi {

// Thin helpers around MPI calls that may be device-aware.
//
// CPU build              : just call MPI directly with the host pointer.
// CUDA + CUDA-aware MPI  : pass device pointer directly.
// CUDA without CUDA-aware: caller is expected to stage into a host buffer first
//                          (added in M5'; for M0..M4 we assume CUDA-aware MPI).
//
// At M0 this is a stub: Subdomain::exchange_halo already calls MPI directly
// with the buffer pointer, which is fine for both CPU and CUDA-aware MPI.

// Returns true if the build assumes CUDA-aware MPI is available.
constexpr bool is_cuda_aware_mpi_enabled() {
#if defined(MPMSTD_BACKEND_CUDA) && !defined(MPMSTD_NO_CUDA_AWARE_MPI)
  return true;
#else
  return false;
#endif
}

} // namespace mpmstd::parallel::mpi
