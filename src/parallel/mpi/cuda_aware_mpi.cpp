#include "parallel/mpi/cuda_aware_mpi.hpp"

// Placeholder for the host-staging fallback path. Will be filled in M5' if
// the cluster's MPI implementation turns out NOT to be CUDA-aware.
//
// Until then, this translation unit exists only to keep the build system from
// complaining about missing object files.

namespace mpmstd::parallel::mpi {
// (nothing yet)
} // namespace mpmstd::parallel::mpi
