#pragma once

#include "core/grid.hpp"          // Grid
#include "core/mpi_topology.hpp"  // MpiTopology, Subdomain

namespace mpmstd::core {

// =============================================================================
// Domain — geometry + parallel preprocessing context (rev.2 structural redesign).
// -----------------------------------------------------------------------------
// Bundles the "where/how" the MPI/topology/subdomain/grid preprocessing produces.
// Host-single, app-agnostic, fixed for the whole run, and carries NO mutable
// physics state (it is NOT a god-object). A lightweight ref-view: the caller
// owns the grid/topology/subdomain (built once in main) and they outlive Domain.
// Equation/physics/post free functions take `const Domain&` as their single
// "context" argument, replacing the scattered grid/sub/topo/tdma args.
//
// (TdmaRegistry is held separately — it is mutable solver machinery, not pure
//  geometry — and passed where a distributed solve needs it.)
struct Domain {
  const Grid&        grid;
  const MpiTopology& topo;
  const Subdomain&   sub;
};

} // namespace mpmstd::core
