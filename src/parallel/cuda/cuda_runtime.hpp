#pragma once

// CUDA runtime initialization and device management.
// CPU build : every function is a safe no-op.
// CUDA build: actual cudaSetDevice / device-info queries (implemented in M5').

namespace mpmstd::parallel::cuda_helpers {

// Initializes CUDA runtime and binds the current MPI rank to a GPU.
// rank_in_node: typically MPI_Comm_rank on a node-local communicator.
// Returns the bound device id (or -1 on CPU build).
int initialize_device(int rank_in_node);

// Synchronizes the current device (no-op on CPU build).
void synchronize_device();

// Returns the number of CUDA-capable devices visible (0 on CPU build).
int device_count();

} // namespace mpmstd::parallel::cuda_helpers
