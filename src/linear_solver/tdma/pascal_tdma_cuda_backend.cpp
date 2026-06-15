#include "linear_solver/tdma/pascal_tdma_cuda_backend.hpp"
#include "common/macros.hpp"

#include <stdexcept>

#ifdef MPMSTD_BACKEND_CUDA
  #include <cuda_runtime.h>
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
  static_assert(sizeof(real_t) == sizeof(double),
    "PaScaLTDMACudaBackend: real_t must be double (build with PRECISION=double)");
#endif
}

PaScaLTDMACudaBackend::~PaScaLTDMACudaBackend() {
#ifdef MPMSTD_BACKEND_CUDA
  if (d_buf_) cudaFree(d_buf_);
#endif
}

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

// Ensure device staging buffer holds 4 × n_sys × n_row doubles.
// The 4 regions are: [A|B|C|D], each n_sys*n_row contiguous doubles.
static inline std::size_t buf_count(int n_sys, int n_row) {
  return static_cast<std::size_t>(4) * n_sys * n_row;
}

void PaScaLTDMACudaBackend::solve_many(real_t* A, real_t* B, real_t* C, real_t* D,
                                         int n_sys, int n_row) {
#ifdef MPMSTD_BACKEND_CUDA
  auto& plan = get_or_create_plan(n_sys);
  const std::size_t need = buf_count(n_sys, n_row);
  if (need > n_buf_) {
    if (d_buf_) cudaFree(d_buf_);
    cudaMalloc(reinterpret_cast<void**>(&d_buf_), need * sizeof(double));
    n_buf_ = need;
  }
  const std::size_t stripe = static_cast<std::size_t>(n_sys) * n_row;
  double* dA = d_buf_;
  double* dB = dA + stripe;
  double* dC = dB + stripe;
  double* dD = dC + stripe;
  cudaMemcpy(dA, A, stripe * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, stripe * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, stripe * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dD, D, stripe * sizeof(double), cudaMemcpyHostToDevice);
  plan.solver->solve(dA, dB, dC, dD, n_sys, n_row);
  cudaMemcpy(D, dD, stripe * sizeof(double), cudaMemcpyDeviceToHost);
#else
  MPMSTD_UNUSED(A); MPMSTD_UNUSED(B); MPMSTD_UNUSED(C); MPMSTD_UNUSED(D);
  MPMSTD_UNUSED(n_sys); MPMSTD_UNUSED(n_row);
  throw std::runtime_error(
    "PaScaLTDMACudaBackend::solve_many: not available in CPU build");
#endif
}

void PaScaLTDMACudaBackend::solve_many_cyclic(real_t* A, real_t* B, real_t* C, real_t* D,
                                                int n_sys, int n_row) {
#ifdef MPMSTD_BACKEND_CUDA
  auto& plan = get_or_create_plan(n_sys);
  const std::size_t need = buf_count(n_sys, n_row);
  if (need > n_buf_) {
    if (d_buf_) cudaFree(d_buf_);
    cudaMalloc(reinterpret_cast<void**>(&d_buf_), need * sizeof(double));
    n_buf_ = need;
  }
  const std::size_t stripe = static_cast<std::size_t>(n_sys) * n_row;
  double* dA = d_buf_;
  double* dB = dA + stripe;
  double* dC = dB + stripe;
  double* dD = dC + stripe;
  cudaMemcpy(dA, A, stripe * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, stripe * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, stripe * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dD, D, stripe * sizeof(double), cudaMemcpyHostToDevice);
  plan.solver->solve_cyclic(dA, dB, dC, dD, n_sys, n_row);
  cudaMemcpy(D, dD, stripe * sizeof(double), cudaMemcpyDeviceToHost);
#else
  MPMSTD_UNUSED(A); MPMSTD_UNUSED(B); MPMSTD_UNUSED(C); MPMSTD_UNUSED(D);
  MPMSTD_UNUSED(n_sys); MPMSTD_UNUSED(n_row);
  throw std::runtime_error(
    "PaScaLTDMACudaBackend::solve_many_cyclic: not available in CPU build");
#endif
}

void PaScaLTDMACudaBackend::solve_many_device(double* d_A, double* d_B, double* d_C, double* d_D,
                                               int n_sys, int n_row) {
#ifdef MPMSTD_BACKEND_CUDA
  auto& plan = get_or_create_plan(n_sys);
  plan.solver->solve(d_A, d_B, d_C, d_D, n_sys, n_row);
#else
  MPMSTD_UNUSED(d_A); MPMSTD_UNUSED(d_B); MPMSTD_UNUSED(d_C); MPMSTD_UNUSED(d_D);
  MPMSTD_UNUSED(n_sys); MPMSTD_UNUSED(n_row);
  throw std::runtime_error("PaScaLTDMACudaBackend::solve_many_device: CPU build");
#endif
}

void PaScaLTDMACudaBackend::solve_many_cyclic_device(double* d_A, double* d_B, double* d_C, double* d_D,
                                                      int n_sys, int n_row) {
#ifdef MPMSTD_BACKEND_CUDA
  auto& plan = get_or_create_plan(n_sys);
  plan.solver->solve_cyclic(d_A, d_B, d_C, d_D, n_sys, n_row);
#else
  MPMSTD_UNUSED(d_A); MPMSTD_UNUSED(d_B); MPMSTD_UNUSED(d_C); MPMSTD_UNUSED(d_D);
  MPMSTD_UNUSED(n_sys); MPMSTD_UNUSED(n_row);
  throw std::runtime_error("PaScaLTDMACudaBackend::solve_many_cyclic_device: CPU build");
#endif
}

} // namespace mpmstd::linear_solver::tdma
