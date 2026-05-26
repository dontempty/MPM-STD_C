#include "linear_solver/tdma/pascal_tdma_cuda_backend.hpp"
#include "common/macros.hpp"

#include <stdexcept>

#ifdef MPMSTD_BACKEND_CUDA
  #include "pascal_tdma_many_cuda.hpp"
#endif

namespace mpmstd::linear_solver::tdma {

#ifdef MPMSTD_BACKEND_CUDA
struct PaScaLTDMACudaBackend::Plan {
  std::unique_ptr<PaScaLTDMAManyCUDA> solver;
};
#else
struct PaScaLTDMACudaBackend::Plan {
  // Empty in CPU build — never instantiated.
};
#endif

PaScaLTDMACudaBackend::PaScaLTDMACudaBackend(const parallel::mpi::CartComm1D& axis)
  : axis_(axis) {
#ifndef MPMSTD_BACKEND_CUDA
  throw std::runtime_error(
    "PaScaLTDMACudaBackend instantiated in a CPU build. "
    "Rebuild with BACKEND=cuda, or use PaScaLTDMACpuBackend.");
#else
  // Real plans are built lazily by get_or_create_plan().
#endif
}

PaScaLTDMACudaBackend::~PaScaLTDMACudaBackend() = default;

PaScaLTDMACudaBackend::Plan&
PaScaLTDMACudaBackend::get_or_create_plan(int n_sys) {
#ifdef MPMSTD_BACKEND_CUDA
  auto it = plans_.find(n_sys);
  if (it != plans_.end()) return *it->second;

  auto plan = std::make_unique<Plan>();
  plan->solver = std::make_unique<PaScaLTDMAManyCUDA>(
      n_sys, axis_.rank, axis_.nprocs, axis_.comm);
  auto& ref = *plan;
  plans_.emplace(n_sys, std::move(plan));
  return ref;
#else
  MPMSTD_UNUSED(n_sys);
  throw std::runtime_error("PaScaLTDMACudaBackend: CPU build cannot create plans");
#endif
}

void PaScaLTDMACudaBackend::solve_many(real_t* /*A*/, real_t* /*B*/,
                                         real_t* /*C*/, real_t* /*D*/,
                                         int /*n_sys*/, int /*n_row*/) {
#ifdef MPMSTD_BACKEND_CUDA
  throw std::runtime_error(
    "PaScaLTDMACudaBackend::solve_many: implementation deferred to M5'");
#else
  throw std::runtime_error(
    "PaScaLTDMACudaBackend::solve_many: not available in CPU build");
#endif
}

void PaScaLTDMACudaBackend::solve_many_cyclic(real_t* /*A*/, real_t* /*B*/,
                                                real_t* /*C*/, real_t* /*D*/,
                                                int /*n_sys*/, int /*n_row*/) {
#ifdef MPMSTD_BACKEND_CUDA
  throw std::runtime_error(
    "PaScaLTDMACudaBackend::solve_many_cyclic: implementation deferred to M5'");
#else
  throw std::runtime_error(
    "PaScaLTDMACudaBackend::solve_many_cyclic: not available in CPU build");
#endif
}

} // namespace mpmstd::linear_solver::tdma
