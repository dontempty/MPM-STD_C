#include "linear_solver/tdma/pascal_tdma_cpu_backend.hpp"

#include "pascal_tdma_many.hpp"   // from PaScaL_TDMA_C

#include <stdexcept>
#include <type_traits>

namespace mpmstd::linear_solver::tdma {

// Internal plan: owns one PaScaLTDMAMany solver instance per (n_sys) key.
struct PaScaLTDMACpuBackend::Plan {
  std::unique_ptr<PaScaLTDMAMany> solver;
};

PaScaLTDMACpuBackend::PaScaLTDMACpuBackend(const parallel::mpi::CartComm1D& axis)
  : axis_(axis) {}

PaScaLTDMACpuBackend::~PaScaLTDMACpuBackend() = default;

PaScaLTDMACpuBackend::Plan&
PaScaLTDMACpuBackend::get_or_create_plan(int n_sys) {
  auto it = plans_.find(n_sys);
  if (it != plans_.end()) return *it->second;

  // PaScaL_TDMA distributes `n_sys` independent tridiagonal systems across
  // the axis communicator's nprocs ranks (via para_range). Each rank must
  // end up with at least one system; otherwise the internal subarray DDTs
  // become invalid and MPI_Type_create_subarray aborts with an
  // unhelpful runtime error.
  if (n_sys < axis_.nprocs) {
    throw std::runtime_error(
      "PaScaLTDMACpuBackend: n_sys (" + std::to_string(n_sys) +
      ") must be >= nprocs on the TDMA axis (" + std::to_string(axis_.nprocs) +
      "). Batch more independent systems per call, or reduce process count "
      "along this axis.");
  }

  auto plan = std::make_unique<Plan>();
  plan->solver = std::make_unique<PaScaLTDMAMany>(
      n_sys, axis_.rank, axis_.nprocs, axis_.comm);
  auto& ref = *plan;
  plans_.emplace(n_sys, std::move(plan));
  return ref;
}

// PaScaL_TDMA_C operates on double*. Our real_t is double by default; if the
// project switches to single precision we will need a conversion path. For
// now we assert at compile time so the situation is caught early.
static_assert(std::is_same_v<real_t, double>,
              "PaScaLTDMACpuBackend currently requires real_t == double "
              "(PaScaL_TDMA_C library uses double precision).");

void PaScaLTDMACpuBackend::solve_many(real_t* A, real_t* B, real_t* C, real_t* D,
                                        int n_sys, int n_row) {
  auto& plan = get_or_create_plan(n_sys);
  plan.solver->solve(A, B, C, D, n_sys, n_row);
}

void PaScaLTDMACpuBackend::solve_many_cyclic(real_t* A, real_t* B, real_t* C, real_t* D,
                                               int n_sys, int n_row) {
  auto& plan = get_or_create_plan(n_sys);
  plan.solver->solve_cyclic(A, B, C, D, n_sys, n_row);
}

} // namespace mpmstd::linear_solver::tdma
