#pragma once

#include "linear_solver/tdma/tdma_solver.hpp"
#include "parallel/mpi/mpi_topology.hpp"

#include <memory>
#include <unordered_map>

namespace mpmstd::linear_solver::tdma {

// CUDA backend that wraps PaScaLTDMAManyCUDA.
//
// Phase-1 (M1) provides only the class shell. The constructor throws on a CPU
// build so callers cannot accidentally instantiate it; the solve methods
// throw "not implemented" stubs that will be filled in M5'.

class PaScaLTDMACudaBackend final : public TdmaSolver {
public:
  explicit PaScaLTDMACudaBackend(const parallel::mpi::CartComm1D& axis_comm);
  ~PaScaLTDMACudaBackend() override;

  PaScaLTDMACudaBackend(const PaScaLTDMACudaBackend&) = delete;
  PaScaLTDMACudaBackend& operator=(const PaScaLTDMACudaBackend&) = delete;

  void solve_many(real_t* A, real_t* B, real_t* C, real_t* D,
                   int n_sys, int n_row) override;

  void solve_many_cyclic(real_t* A, real_t* B, real_t* C, real_t* D,
                          int n_sys, int n_row) override;

private:
  struct Plan;
  Plan& get_or_create_plan(int n_sys);

  parallel::mpi::CartComm1D axis_;
  std::unordered_map<int, std::unique_ptr<Plan>> plans_;
};

} // namespace mpmstd::linear_solver::tdma
