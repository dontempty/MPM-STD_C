#pragma once

#include "equation/pressure/pressure_base.hpp"
#include "linear_solver/tdma/pascal_tdma_cpu_backend.hpp"
#include "parallel/mpi/mpi_topology.hpp"

#include <complex>
#include <memory>
#include <vector>

struct fftw_plan_s;

namespace mpmstd::equation::pressure {

// PressureSolver — pencil-decomposition FFT-based pressure Poisson solver.
//
// Supports arbitrary np1 × np2 × np3 MPI decompositions (previously np1=np2=1 only).
//
// Algorithm:
//   C→I  (Alltoall on axis(X).comm)   → I-pencil [n1m × ny_loc × n3_I]
//   FFT x  (batched r2c)              → hat_I [Nxh × ny_loc × n3_I]
//   I→Y  (Alltoallv on axis(Y).comm)  → Y-pencil [h1p_Y_me × n2m × n3_I]
//   FFT y  (batched c2c)
//   TDMA z (distributed over comm_xz)
//   IFFT y, Y→I, IFFT x
//   scale, unpack, transpose I→C
//   apply Neumann ghost + halo exchange + project
class PressureSolver : public PressureSolverBase {
public:
  PressureSolver(const grid::Grid&                  grid,
                  const parallel::mpi::Subdomain&    sub,
                  const boundary::Problem&           problem,
                  linear_solver::tdma::TdmaRegistry& tdma);
  ~PressureSolver();

  PressureSolver(const PressureSolver&)            = delete;
  PressureSolver& operator=(const PressureSolver&) = delete;

  void solve(real_t dt,
             core::CpuField& U,
             core::CpuField& V,
             core::CpuField& W,
             core::CpuField& P) override;

private:
  void compute_divergence_rhs_pencil_();
  void transpose_C_to_I_();
  void transpose_I_to_C_();
  void transpose_I_to_Y_();
  void transpose_Y_to_I_();
  void fft_x_forward_();
  void fft_x_backward_();
  void fft_y_forward_();
  void fft_y_backward_();
  void solve_tdma_z_();

  // ---- cached pointers to current solve() arguments (avoid passing everywhere)
  const core::CpuField* cur_U_ = nullptr;
  const core::CpuField* cur_V_ = nullptr;
  const core::CpuField* cur_W_ = nullptr;
  real_t cur_dt_ = 0.0;

  // ---- I-pencil split factor
  int nz_loc_ = 0, ny_loc_ = 0, nx_loc_ = 0;
  int n3_I_   = 0;   // nz_loc_ / np1  (must divide evenly)
  int Nxh_    = 0;   // n1m / 2 + 1

  // ---- Y-pencil x-freq split
  std::vector<int> h1p_Y_, ix_start_Y_;
  int h1p_Y_me_ = 0, ix_start_Y_me_ = 0, n_sys_Y_ = 0;

  // ---- Wavenumbers
  std::vector<double> lambda_x_;   // [Nxh_]
  std::vector<double> lambda_y_;   // [n2m_]  full spectrum (c2c y-FFT)

  // ---- Global z-metrics (1-indexed: index k_g+1 = global z cell k_g)
  std::vector<double> dz_g_, dmz_g_;

  // ---- Pencil buffers
  std::vector<double>               rhs_C_;    // C-layout, no halo
  std::vector<double>               rhs_I_;    // I-pencil layout
  std::vector<std::complex<double>> hat_I_;    // after r2c
  std::vector<std::complex<double>> hat_Y_;    // Y-pencil layout
  std::vector<double>               dp_I_;     // inverse transform result
  std::vector<real_t>               dp_C_;     // unpacked dP, C-layout, for projection

  // ---- Transpose buffers (pre-allocated)
  std::vector<double> tx_sbuf_C_, tx_rbuf_C_;
  std::vector<double> tx_sbuf_Y_, tx_rbuf_Y_;

  // ---- FFTW plans (1D batched)
  fftw_plan_s* plan_fwd_x_ = nullptr;
  fftw_plan_s* plan_bwd_x_ = nullptr;
  fftw_plan_s* plan_fwd_y_ = nullptr;
  fftw_plan_s* plan_bwd_y_ = nullptr;

  // ---- TDMA z (distributed over comm_xz)
  std::vector<double> tdma_A_r_, tdma_B_r_, tdma_C_r_, tdma_D_r_;
  std::vector<double> tdma_A_c_, tdma_B_c_, tdma_C_c_, tdma_D_c_;
  std::unique_ptr<linear_solver::tdma::TdmaSolver> tdma_z_;
  parallel::mpi::CartComm1D comm_xz_1d_;
};

} // namespace mpmstd::equation::pressure
