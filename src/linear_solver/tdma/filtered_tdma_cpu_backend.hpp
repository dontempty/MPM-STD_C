#pragma once

#include "linear_solver/tdma/tdma_solver.hpp"
#include "parallel/mpi/mpi_topology.hpp"

#include <memory>

namespace mpmstd::linear_solver::tdma {

// CPU backend wrapping FilteredTDMA (DistD2 algorithm).
// Used for the wall-normal (Z) axis where the near-wall tanh stretching
// pushes the spectral radius rho = a/b close to 0.5, destabilising the
// standard Thomas algorithm.  FilteredTDMA limits back-substitution to J
// rows from each boundary so that high-frequency error modes do not grow.
//
// Call set_rho(A,B,C,n_sys) BEFORE each solve_many(), using the same bands
// that will be passed to solve_many().  set_eps_constant(1e-12) once at
// construction; the default 1e-12 matches Filtered_TDMA/channel production.

class FilteredTDMACpuBackend final : public TdmaSolver {
public:
  explicit FilteredTDMACpuBackend(const parallel::mpi::CartComm1D& axis_comm,
                                   double eps_constant = 1.0e-12);
  ~FilteredTDMACpuBackend() override;

  FilteredTDMACpuBackend(const FilteredTDMACpuBackend&)            = delete;
  FilteredTDMACpuBackend& operator=(const FilteredTDMACpuBackend&) = delete;

  void solve_many(real_t* A, real_t* B, real_t* C, real_t* D,
                   int n_sys, int n_row) override;

  // Cyclic fallback: FilteredTDMA only supports non-cyclic.
  // Periodic directions should continue to use PaScaLTDMACpuBackend.
  void solve_many_cyclic(real_t* A, real_t* B, real_t* C, real_t* D,
                          int n_sys, int n_row) override;

  void set_rho(const real_t* A, const real_t* B, const real_t* C,
                int n_sys) override;

  void set_eps_constant(double eps) override;

private:
  struct Plan;
  Plan& get_or_create_plan(int n_sys, int n_row);

  parallel::mpi::CartComm1D axis_;
  double eps_constant_;
  std::unordered_map<std::size_t, std::unique_ptr<Plan>> plans_;
};

} // namespace mpmstd::linear_solver::tdma
