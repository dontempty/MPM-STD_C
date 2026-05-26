#pragma once

#include "linear_solver/tdma/tdma_solver.hpp"
#include "parallel/mpi/mpi_topology.hpp"

#include <memory>
#include <unordered_map>

namespace mpmstd::linear_solver::tdma {

// CPU backend that wraps PaScaLTDMAMany (the C++/MPI library).
// Plans are cached by n_sys; the same backend object can be re-used across
// many time steps and ADI stages that share the same per-direction n_sys.

class PaScaLTDMACpuBackend final : public TdmaSolver {
public:
  // axis_comm carries the per-direction CartComm1D from MpiTopology.
  explicit PaScaLTDMACpuBackend(const parallel::mpi::CartComm1D& axis_comm);
  ~PaScaLTDMACpuBackend() override;

  PaScaLTDMACpuBackend(const PaScaLTDMACpuBackend&) = delete;
  PaScaLTDMACpuBackend& operator=(const PaScaLTDMACpuBackend&) = delete;

  void solve_many(real_t* A, real_t* B, real_t* C, real_t* D,
                   int n_sys, int n_row) override;

  void solve_many_cyclic(real_t* A, real_t* B, real_t* C, real_t* D,
                          int n_sys, int n_row) override;

private:
  // Forward declaration to avoid leaking PaScaL_TDMA's headers from this hpp.
  struct Plan;
  Plan& get_or_create_plan(int n_sys);

  parallel::mpi::CartComm1D axis_;
  std::unordered_map<int, std::unique_ptr<Plan>> plans_;
};

} // namespace mpmstd::linear_solver::tdma
