#pragma once

// =============================================================================
// core/gpu_binding.hpp
// -----------------------------------------------------------------------------
// rank <-> GPU binding for "1 MPI rank == 1 GPU". parallel::mpi::MpiContext
// exposes node_rank()/node_comm() (split by MPI_COMM_TYPE_SHARED, built for
// per-GPU binding); this header declares the cudaSetDevice() wrapper as a free
// function with the usual _cpu/_gpu suffixes (the _cpu variant is a no-op).
//
// The three parallel/mpi includes below are also re-provided here so callers
// that reached MpiContext/MpiTopology/Subdomain through this header keep
// compiling; those callers qualify the types as parallel::mpi::* (namespace rule).
// =============================================================================

#include "parallel/mpi/mpi_context.hpp"
#include "parallel/mpi/mpi_topology.hpp"
#include "parallel/mpi/subdomain.hpp"

namespace mpmstd::core {

// rev.2 C2 — bind this rank to its node-local GPU (cudaSetDevice(node_rank %
// device_count)). MUST be called once after MpiContext construction in GPU
// apps. ⚠ Guarantees the past "multiple ranks on one GPU" mistake cannot recur.
//   _cpu : no-op (so backend-parameterized code can call the matching suffix).
//   _gpu : cudaSetDevice(...). Defined in core/gpu/halo_gpu.cu.
void bind_gpu_to_local_rank_cpu(const parallel::mpi::MpiContext& ctx);
void bind_gpu_to_local_rank_gpu(const parallel::mpi::MpiContext& ctx);

} // namespace mpmstd::core
