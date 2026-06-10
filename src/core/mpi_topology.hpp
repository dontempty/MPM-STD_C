#pragma once

// =============================================================================
// core/mpi_topology.hpp
// -----------------------------------------------------------------------------
// Host metadata (MPI topology / subdomain / context) is NOT duplicated per
// backend (rev.2 C1): there is ONE host-single type for each. This header
// REUSES the existing, validated implementations verbatim and re-exports them
// into the `core` namespace so callers write core::MpiTopology / core::Subdomain
// / core::MpiContext uniformly across the new tree.
//
// It also adds the rev.2 C2 piece that was missing: rank <-> GPU binding for
// "1 MPI rank == 1 GPU". MpiContext already exposes node_rank()/node_comm()
// (split by MPI_COMM_TYPE_SHARED, built precisely for per-GPU binding), so the
// only new thing is the cudaSetDevice() call, exposed as an explicit free
// function with the usual _cpu/_gpu suffixes (the _cpu variant is a no-op).
// =============================================================================

#include "parallel/mpi/mpi_context.hpp"
#include "parallel/mpi/mpi_topology.hpp"
#include "parallel/mpi/subdomain.hpp"

namespace mpmstd::core {

using MpiContext  = parallel::mpi::MpiContext;
using MpiTopology = parallel::mpi::MpiTopology;
using Subdomain   = parallel::mpi::Subdomain;

// rev.2 C2 — bind this rank to its node-local GPU (cudaSetDevice(node_rank %
// device_count)). MUST be called once after MpiContext construction in GPU
// apps. ⚠ Guarantees the past "multiple ranks on one GPU" mistake cannot recur.
//   _cpu : no-op (so backend-parameterized code can call the matching suffix).
//   _gpu : cudaSetDevice(...). Defined in core/gpu/halo_gpu.cu.
void bind_gpu_to_local_rank_cpu(const MpiContext& ctx);
void bind_gpu_to_local_rank_gpu(const MpiContext& ctx);

} // namespace mpmstd::core
