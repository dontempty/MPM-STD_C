#pragma once

#include "core/grid.hpp"          // Grid
#include "core/gpu_binding.hpp"  // MpiTopology, Subdomain
#include "linear_solver/tdma/tdma_registry.hpp"

namespace mpmstd::core {

// =============================================================================
// Domain — geometry + parallel preprocessing context (rev.2 structural redesign).
// -----------------------------------------------------------------------------
// Bundles the "where/how" the MPI/topology/subdomain/grid preprocessing produces,
// plus the distributed TDMA registry. Host-single, app-agnostic, fixed for the
// whole run; carries NO mutable physics state. A lightweight ref-view: the caller
// owns grid/topology/subdomain/tdma (built once in main) and they outlive Domain.
// Equation/physics/post free functions take `const Domain&` as their single
// context argument, replacing the scattered grid/sub/topo/tdma args.
//
// `tdma` is a reference (the unique_ptr from make_default is unwrapped ONCE in
// main: `TdmaRegistry& tdma = *owner;`) so call sites never see a `*`.
struct Domain {
  const grid::Grid&                        grid;
  const parallel::mpi::MpiTopology&                 topo;
  const parallel::mpi::Subdomain&                   sub;
  linear_solver::tdma::TdmaRegistry& tdma;
  // Device-resident grid arrays (GpuGridArrays*) for CUDA builds; null in CPU builds.
  const void*                        gpu_grid = nullptr;
};

} // namespace mpmstd::core
