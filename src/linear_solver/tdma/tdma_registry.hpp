#pragma once

#include "common/direction.hpp"
#include "linear_solver/tdma/tdma_solver.hpp"
#include "parallel/mpi/mpi_topology.hpp"

#include <array>
#include <memory>

namespace mpmstd::linear_solver::tdma {

// TdmaRegistry holds one TdmaSolver instance per axis.  Equation modules pick
// the right backend by direction; the factory function decides at build time
// whether to use the CPU or CUDA backend, but the registry interface is the
// same either way.
//
// Plans inside each backend are cached by n_sys (built lazily), so the same
// registry can service the entire time-loop without re-creating MPI
// alltoall datatypes.

class TdmaRegistry {
public:
  // Build a registry that uses the build's default backend for every axis.
  // CPU build → PaScaLTDMACpuBackend on each axis_comm.
  // CUDA build → PaScaLTDMACudaBackend on each axis_comm.
  static std::unique_ptr<TdmaRegistry>
  make_default(const parallel::mpi::MpiTopology& topo);

  TdmaSolver& get(Direction d) { return *backends_[to_int(d)]; }

private:
  std::array<std::unique_ptr<TdmaSolver>, 3> backends_;
};

} // namespace mpmstd::linear_solver::tdma
