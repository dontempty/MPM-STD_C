#pragma once

#include "common/direction.hpp"
#include "parallel/mpi/mpi_topology.hpp"
#ifdef MPMSTD_BACKEND_CUDA
#include "linear_solver/tdma/pascal_tdma_cuda_backend.hpp"
#else
#include "linear_solver/tdma/pascal_tdma_cpu_backend.hpp"
#endif

#include <array>
#include <memory>

namespace mpmstd::linear_solver::tdma {

// TdmaBackend is the build's concrete TDMA backend.  There is exactly one
// backend per build (selected at compile time), so the registry stores it by
// concrete type rather than through a virtual base:
//   CPU build  → PaScaLTDMACpuBackend
//   CUDA build → PaScaLTDMACudaBackend  (also exposes solve_many_device for the
//                GPU-resident path — a member a shared virtual base could not
//                carry, which is why the GPU kernels had to static_cast).
#ifdef MPMSTD_BACKEND_CUDA
using TdmaBackend = PaScaLTDMACudaBackend;
#else
using TdmaBackend = PaScaLTDMACpuBackend;
#endif

// TdmaRegistry holds one TdmaBackend instance per axis.  Equation modules pick
// the right backend by direction.
//
// Plans inside each backend are cached by n_sys (built lazily), so the same
// registry can service the entire time-loop without re-creating MPI
// alltoall datatypes.

class TdmaRegistry {
public:
  // Build a registry that uses the build's default backend for every axis.
  static std::unique_ptr<TdmaRegistry>
  make_default(const parallel::mpi::MpiTopology& topo);

  TdmaBackend& get(Direction d) { return *backends_[to_int(d)]; }

private:
  std::array<std::unique_ptr<TdmaBackend>, 3> backends_;
};

} // namespace mpmstd::linear_solver::tdma
