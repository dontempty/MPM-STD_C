#pragma once

#include "common/types.hpp"
#include "parallel/mpi/mpi_topology.hpp"

#include <memory>
#include <unordered_map>

namespace mpmstd::linear_solver::tdma {

// CUDA backend that wraps PaScaLTDMAManyCUDA.
//
// Phase-1 (M1) provides only the class shell. The constructor throws on a CPU
// build so callers cannot accidentally instantiate it; the solve methods
// throw "not implemented" stubs that will be filled in M5'.

class PaScaLTDMACudaBackend final {
public:
  explicit PaScaLTDMACudaBackend(const parallel::mpi::CartComm1D& axis_comm);
  ~PaScaLTDMACudaBackend();

  PaScaLTDMACudaBackend(const PaScaLTDMACudaBackend&) = delete;
  PaScaLTDMACudaBackend& operator=(const PaScaLTDMACudaBackend&) = delete;

  void solve_many(real_t* A, real_t* B, real_t* C, real_t* D,
                   int n_sys, int n_row);

  void solve_many_cyclic(real_t* A, real_t* B, real_t* C, real_t* D,
                          int n_sys, int n_row);

  // Device-pointer variants: A/B/C/D are already on device — no H↔D transfer.
  void solve_many_device(double* d_A, double* d_B, double* d_C, double* d_D,
                          int n_sys, int n_row);
  void solve_many_cyclic_device(double* d_A, double* d_B, double* d_C, double* d_D,
                                 int n_sys, int n_row);

private:
  struct Plan;
  Plan& get_or_create_plan(int n_sys);

  parallel::mpi::CartComm1D axis_;
  std::unordered_map<int, std::unique_ptr<Plan>> plans_;
  // Flat device staging buffer for H→D→H TDMA transfer.
  // Holds 4 contiguous arrays [n_sys × n_row]: [A|B|C|D].
  double* d_buf_ = nullptr;   // device, 4 × n_buf_ doubles
  std::size_t n_buf_ = 0;     // capacity in doubles (0 = unallocated)
};

} // namespace mpmstd::linear_solver::tdma
