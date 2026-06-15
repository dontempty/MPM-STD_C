// GPU pressure Poisson solver — cuFFT + host-staged MPI + CPU TDMA.
//
// Design: PressureSolverGpu mirrors PressureSolver (the CPU pencil-FFT engine)
// exactly but replaces FFTW with cuFFT. MPI transposes are host-staged (pinned
// buffers, same pack/unpack logic as CPU). TDMA-z remains on CPU. The divergence
// and projection steps download/upload GPU fields to CPU, compute on CPU, then
// re-upload — this keeps the engine self-contained without needing device-resident
// metric arrays in the pressure layer.
//
// Memory model:
//   - Pencil buffers are cudaMallocHost (pinned) — same names as CPU engine.
//   - Device buffers d_rhs_I_, d_hat_I_, d_hat_Y_, d_dp_I_ hold cuFFT operands.
//   - hat_I_ and hat_Y_ pinned as double2 (= cufftDoubleComplex) for H↔D copy.

#pragma once

#include "equation/pressure/pressure_engine.hpp"   // includes: PressureSolverBase, Grid, Subdomain, Problem, TdmaRegistry
#include "core/boundary_ops.hpp"                    // sync_field_gpu
#include "core/field_gpu.hpp"                       // GpuField
#include "common/macros.hpp"                        // kHaloWidth
#include "linear_solver/tdma/pascal_tdma_cpu_backend.hpp"
#include "linear_solver/tdma/pascal_tdma_cuda_backend.hpp"

#include <cufft.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace mpmstd::equation::pressure {

// ── Error-check macros ───────────────────────────────────────────────────────

#define CUDA_CHK(expr) do {                                              \
  cudaError_t _e = (expr);                                               \
  if (_e != cudaSuccess) {                                               \
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                 cudaGetErrorName(_e), __FILE__, __LINE__,               \
                 cudaGetErrorString(_e));                                 \
    std::abort();                                                         \
  }                                                                       \
} while(0)

#define CUFFT_CHK(expr) do {                                             \
  cufftResult _r = (expr);                                               \
  if (_r != CUFFT_SUCCESS) {                                             \
    std::fprintf(stderr, "cuFFT error %d at %s:%d\n",                   \
                 (int)_r, __FILE__, __LINE__);                           \
    std::abort();                                                         \
  }                                                                       \
} while(0)


// ── Layout index helpers ─────────────────────────────────────────────────────

namespace {

// C-layout (x-fastest): idx = i + nx*(j + ny*k)
inline std::size_t idx_C_gpu(int i, int j, int k, int nx, int ny)
{ return (std::size_t)i + (std::size_t)nx * ((std::size_t)j + (std::size_t)ny * k); }

// I-layout (x-fastest): idx = i + n1m*(j + ny*k)
inline std::size_t idx_I_gpu(int i, int j, int k, int n1m, int ny)
{ return (std::size_t)i + (std::size_t)n1m * ((std::size_t)j + (std::size_t)ny * k); }

// Y-layout (ix-fastest): idx = ix + h1pY*(j + n2m*k)
inline std::size_t idx_Y_gpu(int ix, int j, int k, int h1pY, int n2m)
{ return (std::size_t)ix + (std::size_t)h1pY * ((std::size_t)j + (std::size_t)n2m * k); }

// ── CUDA kernel: scale buffer ─────────────────────────────────────────────────
__global__ void k_gpu_scale(double* buf, double scale, std::size_t n)
{
  const std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) buf[idx] *= scale;
}

// ── GPU TDMA split/merge kernels ─────────────────────────────────────────────
// hat_Y_ layout: (s + n_sys*k) where s=ixl+h1p_Y*j — SAME as TDMA k*n_sys+s index.
// So split/merge are plain real↔complex scatter with stride-1 inner loop.

__global__ void k_tdma_split_hat_Y(const double2* hat_Y, double* Dr, double* Dc,
                                    std::size_t n)
{
  const std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { Dr[idx] = hat_Y[idx].x;  Dc[idx] = hat_Y[idx].y; }
}

__global__ void k_tdma_merge_hat_Y(const double* Dr, const double* Dc, double2* hat_Y,
                                    std::size_t n)
{
  const std::size_t idx = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { hat_Y[idx] = {Dr[idx], Dc[idx]}; }
}

// ── GPU pressure kernels ──────────────────────────────────────────────────────
// Thread layout: blockDim.x=k (Z, fastest in ZF field), blockDim.y=j, blockIdx.z=i
// rhs_C layout (XF): ii + nx*(jj + ny*kk)  ← X-fastest for C→I transpose compat.

__global__ void k_divergence_rhs_pres(
    const double* __restrict__ U, const double* __restrict__ V, const double* __restrict__ W,
    double* __restrict__ rhs_C,
    const double* __restrict__ dx1, const double* __restrict__ dx2, const double* __restrict__ dx3,
    int n2t, int n3t, int nx, int ny, int nz, int h, double inv_dt)
{
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int ii = blockIdx.z;
  if (kk >= nz || jj >= ny) return;
  const int i = ii+h, j = jj+h, k = kk+h;
  const double div =
      (U[((i+1)*n2t+j)*n3t+k] - U[(i*n2t+j)*n3t+k]) / dx1[i]
    + (V[(i*n2t+j+1)*n3t+k]  - V[(i*n2t+j)*n3t+k])  / dx2[j]
    + (W[(i*n2t+j)*n3t+k+1]  - W[(i*n2t+j)*n3t+k])  / dx3[k];
  rhs_C[ii + nx*(jj + ny*kk)] = div * inv_dt;
}

__global__ void k_extra_term_nob_pres(
    double* __restrict__ rhs_C,
    const double* __restrict__ inv_rho, const double* __restrict__ dPhat,
    const double* __restrict__ dx1, const double* __restrict__ dmx1,
    const double* __restrict__ dx2, const double* __restrict__ dmx2,
    const double* __restrict__ dx3, const double* __restrict__ dmx3,
    int n2t, int n3t, int nx, int ny, int nz, int h)
{
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int ii = blockIdx.z;
  if (kk >= nz || jj >= ny) return;
  const int i = ii+h, j = jj+h, k = kk+h;
  const double irc = inv_rho[(i*n2t+j)*n3t+k];
  const double ir_xlo = 0.5/dmx1[i]   * (dx1[i-1]*irc + dx1[i]*inv_rho[((i-1)*n2t+j)*n3t+k]);
  const double ir_xhi = 0.5/dmx1[i+1] * (dx1[i+1]*irc + dx1[i]*inv_rho[((i+1)*n2t+j)*n3t+k]);
  const double ir_ylo = 0.5/dmx2[j]   * (dx2[j-1]*irc + dx2[j]*inv_rho[(i*n2t+j-1)*n3t+k]);
  const double ir_yhi = 0.5/dmx2[j+1] * (dx2[j+1]*irc + dx2[j]*inv_rho[(i*n2t+j+1)*n3t+k]);
  const double ir_zlo = 0.5/dmx3[k]   * (dx3[k-1]*irc + dx3[k]*inv_rho[(i*n2t+j)*n3t+k-1]);
  const double ir_zhi = 0.5/dmx3[k+1] * (dx3[k+1]*irc + dx3[k]*inv_rho[(i*n2t+j)*n3t+k+1]);
  const double ph  = dPhat[(i*n2t+j)*n3t+k];
  const double gx_lo = (ph - dPhat[((i-1)*n2t+j)*n3t+k]) / dmx1[i];
  const double gx_hi = (dPhat[((i+1)*n2t+j)*n3t+k] - ph) / dmx1[i+1];
  const double gy_lo = (ph - dPhat[(i*n2t+j-1)*n3t+k]) / dmx2[j];
  const double gy_hi = (dPhat[(i*n2t+j+1)*n3t+k] - ph) / dmx2[j+1];
  const double gz_lo = (ph - dPhat[(i*n2t+j)*n3t+k-1]) / dmx3[k];
  const double gz_hi = (dPhat[(i*n2t+j)*n3t+k+1] - ph) / dmx3[k+1];
  const double ex = ((1.0-ir_xhi)*gx_hi - (1.0-ir_xlo)*gx_lo) / dx1[i];
  const double ey = ((1.0-ir_yhi)*gy_hi - (1.0-ir_ylo)*gy_lo) / dx2[j];
  const double ez = ((1.0-ir_zhi)*gz_hi - (1.0-ir_zlo)*gz_lo) / dx3[k];
  rhs_C[ii + nx*(jj + ny*kk)] += ex + ey + ez;
}

__global__ void k_project_ob_pres(
    double* __restrict__ U, double* __restrict__ V, double* __restrict__ W,
    const double* __restrict__ P,
    const double* __restrict__ dmx1, const double* __restrict__ dmx2, const double* __restrict__ dmx3,
    int n2t, int n3t, int nx, int ny, int nz, int h, double dt)
{
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int ii = blockIdx.z;
  if (kk >= nz || jj >= ny) return;
  const int i = ii+h, j = jj+h, k = kk+h;
  const int c   = (i*n2t+j)*n3t+k;
  const double Pc = P[c];
  U[c] -= dt * (Pc - P[((i-1)*n2t+j)*n3t+k]) / dmx1[i];
  V[c] -= dt * (Pc - P[(i*n2t+j-1)*n3t+k])   / dmx2[j];
  W[c] -= dt * (Pc - P[(i*n2t+j)*n3t+k-1])   / dmx3[k];
}

__global__ void k_project_nob_pres(
    double* __restrict__ U, double* __restrict__ V, double* __restrict__ W,
    const double* __restrict__ P, const double* __restrict__ dPhat,
    const double* __restrict__ inv_rho,
    const double* __restrict__ dx1, const double* __restrict__ dmx1,
    const double* __restrict__ dx2, const double* __restrict__ dmx2,
    const double* __restrict__ dx3, const double* __restrict__ dmx3,
    int n2t, int n3t, int nx, int ny, int nz, int h, double dt)
{
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int ii = blockIdx.z;
  if (kk >= nz || jj >= ny) return;
  const int i = ii+h, j = jj+h, k = kk+h;
  const int c   = (i*n2t+j)*n3t+k;
  const int cxm = ((i-1)*n2t+j)*n3t+k;
  const int cym = (i*n2t+j-1)*n3t+k;
  const int czm = (i*n2t+j)*n3t+k-1;
  const double irc  = inv_rho[c];
  const double irc_x = 0.5/dmx1[i] * (dx1[i-1]*irc + dx1[i]*inv_rho[cxm]);
  const double irc_y = 0.5/dmx2[j] * (dx2[j-1]*irc + dx2[j]*inv_rho[cym]);
  const double irc_z = 0.5/dmx3[k] * (dx3[k-1]*irc + dx3[k]*inv_rho[czm]);
  const double Pc   = P[c];
  const double ph   = dPhat[c];
  U[c] -= dt * ((Pc - P[cxm])/dmx1[i] + (irc_x - 1.0)*(ph - dPhat[cxm])/dmx1[i]);
  V[c] -= dt * ((Pc - P[cym])/dmx2[j] + (irc_y - 1.0)*(ph - dPhat[cym])/dmx2[j]);
  W[c] -= dt * ((Pc - P[czm])/dmx3[k] + (irc_z - 1.0)*(ph - dPhat[czm])/dmx3[k]);
}

// ── GPU pressure fast-path kernels ───────────────────────────────────────────
// Used when np1==1 && np2==1: all pencil buffers stay on device.

// Parallel sum reduction via shared memory + per-block atomicAdd.
__global__ void k_pres_local_sum(const double* data, double* result, int n)
{
  extern __shared__ double s_[];
  int tid = threadIdx.x;
  double val = 0.0;
  for (int i = (int)((std::size_t)blockIdx.x*blockDim.x + tid); i < n;
       i += (int)((std::size_t)gridDim.x*blockDim.x))
    val += data[i];
  s_[tid] = val;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
    if (tid < stride) s_[tid] += s_[tid + stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(result, s_[0]);
}

// Transpose C-layout (x-fastest) dp → ZF-layout P, subtract mean, fill Neumann ghost.
__global__ void k_unpack_dp_C_to_P(
    const double* __restrict__ dp_C,
    double* __restrict__ P,
    double mean_dp,
    int nx, int ny, int nz,
    int n2t, int n3t, int h,
    bool lower_wall, bool upper_wall)
{
  int kk = (int)((std::size_t)blockIdx.x*blockDim.x + threadIdx.x);
  int jj = (int)((std::size_t)blockIdx.y*blockDim.y + threadIdx.y);
  int ii = blockIdx.z;
  if (kk >= nz || jj >= ny) return;
  int i = ii+h, j = jj+h, k = kk+h;
  double val = dp_C[ii + nx*(jj + (std::size_t)ny*kk)] - mean_dp;
  P[(std::size_t)(i*n2t+j)*n3t+k] = val;
  if (lower_wall && kk == 0)     P[(std::size_t)(i*n2t+j)*n3t+(h-1)] = val;
  if (upper_wall && kk == nz-1)  P[(std::size_t)(i*n2t+j)*n3t+(h+nz)] = val;
}

} // anonymous namespace


// ── PressureSolverGpu ────────────────────────────────────────────────────────

class PressureSolverGpu {
public:
  PressureSolverGpu(const grid::Grid&                  grid,
                    const parallel::mpi::Subdomain&    sub,
                    const boundary::Problem&           problem,
                    linear_solver::tdma::TdmaRegistry& tdma);
  ~PressureSolverGpu();

  PressureSolverGpu(const PressureSolverGpu&)            = delete;
  PressureSolverGpu& operator=(const PressureSolverGpu&) = delete;

  // OB pressure solve (constant density)
  void solve(real_t dt, core::GpuFields& gpu_fields,
             const boundary::Problem& problem,
             const parallel::mpi::Subdomain& sub);

  // NOB pressure solve (variable density): adds ExtraTerm, NOB projection, updates dPhat
  void solve_nob(real_t dt, core::GpuFields& gpu_fields,
                 const boundary::Problem& problem,
                 const parallel::mpi::Subdomain& sub,
                 const core::GpuField& inv_rho,
                 core::GpuField& dPhat);

private:
  void transpose_C_to_I_();
  void transpose_I_to_C_();
  void transpose_I_to_Y_();
  void transpose_Y_to_I_();
  void solve_tdma_z_();
  void solve_tdma_z_gpu_();       // GPU-resident TDMA (np2==1 only)
  void run_forward_pipeline_();   // C→I, FFT-x, I→Y, FFT-y  (uses pinned buffers)
  void run_backward_pipeline_();  // IFFT-y, Y→I, IFFT-x, scale, I→C
  void unpack_P_to_gpu_(core::GpuField& P_gpu, const parallel::mpi::Subdomain& sub);
  void unpack_P_fast_(core::GpuField& P_gpu, const parallel::mpi::Subdomain& sub);

  // GPU divergence/projection helpers (eliminate per-step D2H/H2D of full fields)
  void divergence_rhs_gpu_(const double* d_U, const double* d_V, const double* d_W, double dt);
  void extra_term_nob_gpu_(const double* d_ir, const double* d_ph);
  void project_ob_gpu_(double* d_U, double* d_V, double* d_W, const double* d_P, double dt);
  void project_nob_gpu_(double* d_U, double* d_V, double* d_W,
                         const double* d_P, const double* d_dPhat,
                         const double* d_ir, double dt);

  // ── shared geometry / decomposition state ────────────────────────────────
  const grid::Grid&                    grid_;
  const parallel::mpi::Subdomain&      sub_;
  const boundary::Problem&             problem_;
  linear_solver::tdma::TdmaRegistry&   tdma_;

  int n1_tot_, n2_tot_, n3_tot_;
  int n1_int_, n2_int_, n3_int_;

  int nx_loc_, ny_loc_, nz_loc_;
  int n3_I_;
  int Nxh_;

  std::vector<int>    h1p_Y_, ix_start_Y_;
  int h1p_Y_me_, ix_start_Y_me_, n_sys_Y_;

  std::vector<double> lambda_x_, lambda_y_;
  std::vector<double> dz_g_, dmz_g_;

  // ── pinned host pencil buffers ────────────────────────────────────────────
  double*    rhs_C_  = nullptr;
  double*    rhs_I_  = nullptr;
  double2*   hat_I_  = nullptr;   // cufftDoubleComplex = double2
  double2*   hat_Y_  = nullptr;
  double*    dp_I_   = nullptr;
  double*    dp_C_   = nullptr;

  // ── pinned transpose buffers ──────────────────────────────────────────────
  double* tx_sbuf_C_ = nullptr;
  double* tx_rbuf_C_ = nullptr;
  double* tx_sbuf_Y_ = nullptr;
  double* tx_rbuf_Y_ = nullptr;

  // ── device buffers for cuFFT ──────────────────────────────────────────────
  double*              d_rhs_I_ = nullptr;
  cufftDoubleComplex*  d_hat_I_ = nullptr;
  cufftDoubleComplex*  d_hat_Y_ = nullptr;
  double*              d_dp_I_  = nullptr;

  // ── device metric arrays (n*_tot_ elements each) ──────────────────────────
  double* d_dx1_  = nullptr;  double* d_dmx1_ = nullptr;
  double* d_dx2_  = nullptr;  double* d_dmx2_ = nullptr;
  double* d_dx3_  = nullptr;  double* d_dmx3_ = nullptr;
  // ── device RHS buffer (sz_C elements, XF layout — same as pinned rhs_C_) ──
  double* d_rhs_C_dev_ = nullptr;
  // ── host staging buffer for unpack_P_to_gpu_ (pre-allocated, size n*_tot_³) ──
  std::vector<double> p_host_unpack_;

  // ── GPU fast path (np1==1 && np2==1) ─────────────────────────────────────
  // When true: rhs stays on GPU (skip D2H+C→I+H2D), hat_I_==hat_Y_ alias,
  // backward skips D2H dp_I_+I→C, unpack uses GPU kernel.
  bool   gpu_fast_path_    = false;
  bool   hat_Y_is_alias_   = false;   // d_hat_Y_ borrowed from d_hat_I_
  double* d_local_sum_     = nullptr; // 1-element reduction buffer

  // ── GPU TDMA z buffers (active when np2==1) ───────────────────────────────
  // A0/B0/C0: precomputed constant matrices (n_sys_Y_ × n3_I_)
  // Aw/Bw/Cw: working copies overwritten in-place by Thomas algorithm
  // Dr/Dc:    RHS real / imaginary parts extracted from d_hat_Y_
  bool   gpu_tdma_enabled_ = false;
  double* d_tdma_A0_ = nullptr;  double* d_tdma_Aw_ = nullptr;
  double* d_tdma_B0_ = nullptr;  double* d_tdma_Bw_ = nullptr;
  double* d_tdma_C0_ = nullptr;  double* d_tdma_Cw_ = nullptr;
  double* d_tdma_Dr_ = nullptr;
  double* d_tdma_Dc_ = nullptr;
  std::unique_ptr<linear_solver::tdma::PaScaLTDMACudaBackend> tdma_z_cuda_;

  // ── cuFFT plans ───────────────────────────────────────────────────────────
  cufftHandle plan_fwd_x_ = 0;
  cufftHandle plan_bwd_x_ = 0;
  cufftHandle plan_y_     = 0;

  // ── CPU TDMA z ────────────────────────────────────────────────────────────
  std::vector<double> tdma_A_r_, tdma_B_r_, tdma_C_r_, tdma_D_r_;
  std::vector<double> tdma_A_c_, tdma_B_c_, tdma_C_c_, tdma_D_c_;
  std::unique_ptr<linear_solver::tdma::TdmaSolver> tdma_z_;
  parallel::mpi::CartComm1D comm_xz_1d_;
};


// ── constructor ───────────────────────────────────────────────────────────────

PressureSolverGpu::PressureSolverGpu(const grid::Grid&                  grid,
                                      const parallel::mpi::Subdomain&    sub,
                                      const boundary::Problem&           problem,
                                      linear_solver::tdma::TdmaRegistry& tdma)
  : grid_(grid), sub_(sub), problem_(problem), tdma_(tdma)
{
  if (!problem_.topology.is_periodic(Direction::X) ||
      !problem_.topology.is_periodic(Direction::Y)) {
    throw std::runtime_error(
      "PressureSolverGpu: X and Y must be periodic.");
  }

  n1_tot_ = sub_.n_total()[0];    n1_int_ = sub_.n_interior()[0];
  n2_tot_ = sub_.n_total()[1];    n2_int_ = sub_.n_interior()[1];
  n3_tot_ = sub_.n_total()[2];    n3_int_ = sub_.n_interior()[2];
  nx_loc_ = n1_int_;
  ny_loc_ = n2_int_;
  nz_loc_ = n3_int_;

  const int N1g = sub_.n_global()[0];
  const int N2g = sub_.n_global()[1];
  const int N3g = sub_.n_global()[2];
  Nxh_ = N1g / 2 + 1;

  const int np1 = sub_.topology().dims()[0];
  const int np2 = sub_.topology().dims()[1];
  const int np3 = sub_.topology().dims()[2];
  (void)np3;

  if (nz_loc_ % np1 != 0)
    throw std::runtime_error("PressureSolverGpu: nz_loc not divisible by np1.");
  n3_I_ = nz_loc_ / np1;

  const int size_xz_check = sub_.topology().size_xz();
  if (size_xz_check > 1 && n3_I_ < 3)
    throw std::runtime_error("PressureSolverGpu: n3_I must be >= 3 when comm_xz > 1 rank.");

  // Y-pencil split
  h1p_Y_.resize(np2);
  ix_start_Y_.resize(np2);
  {
    int base = Nxh_ / np2, extra = Nxh_ % np2;
    for (int r = 0; r < np2; ++r) {
      h1p_Y_[r]      = base + (r < extra ? 1 : 0);
      ix_start_Y_[r] = r * base + std::min(r, extra);
    }
  }
  const int ry      = sub_.topology().axis(Direction::Y).rank;
  h1p_Y_me_      = h1p_Y_[ry];
  ix_start_Y_me_ = ix_start_Y_[ry];
  n_sys_Y_       = h1p_Y_me_ * N2g;

  // Wavenumbers
  {
    const int h = kHaloWidth;
    const real_t* dx1  = grid_.dx_ptr(Direction::X);
    const real_t* dmx1 = grid_.dmx_ptr(Direction::X);
    const real_t* dx2  = grid_.dx_ptr(Direction::Y);
    const real_t* dmx2 = grid_.dmx_ptr(Direction::Y);

    double sum_x = 0.0;
    for (int ii = 0; ii < n1_int_; ++ii) {
      const int i = ii + h;
      sum_x += 1.0/(dx1[i]*dmx1[i]) + 1.0/(dx1[i]*dmx1[i+1]);
    }
    const double K_x = sum_x / n1_int_;

    double sum_y = 0.0;
    for (int jj = 0; jj < n2_int_; ++jj) {
      const int j = jj + h;
      sum_y += 1.0/(dx2[j]*dmx2[j]) + 1.0/(dx2[j]*dmx2[j+1]);
    }
    const double K_y = sum_y / n2_int_;

    lambda_x_.resize(Nxh_);
    for (int mx = 0; mx < Nxh_; ++mx) {
      const double arg = 2.0 * M_PI * mx / N1g;
      lambda_x_[mx] = (1.0 - std::cos(arg)) * K_x;
    }
    lambda_y_.resize(N2g);
    for (int j = 0; j < N2g; ++j) {
      const int    jj  = (j <= N2g / 2) ? j : j - N2g;
      const double arg = 2.0 * M_PI * jj / N2g;
      lambda_y_[j] = (1.0 - std::cos(arg)) * K_y;
    }
  }

  // Global z-metrics
  {
    const int h = kHaloWidth;
    const real_t* dx3_local  = grid_.dx_ptr(Direction::Z);
    const real_t* dmx3_local = grid_.dmx_ptr(Direction::Z);
    dz_g_ .assign(N3g + 2, 0.0);
    dmz_g_.assign(N3g + 2, 0.0);
    MPI_Allgather(dx3_local  + h, nz_loc_, MPI_DOUBLE,
                  dz_g_.data()  + 1, nz_loc_, MPI_DOUBLE,
                  sub_.topology().axis(Direction::Z).comm);
    MPI_Allgather(dmx3_local + h, nz_loc_, MPI_DOUBLE,
                  dmz_g_.data() + 1, nz_loc_, MPI_DOUBLE,
                  sub_.topology().axis(Direction::Z).comm);
    double dmz_last = static_cast<double>(dmx3_local[h + nz_loc_]);
    MPI_Bcast(&dmz_last, 1, MPI_DOUBLE,
              sub_.topology().axis(Direction::Z).nprocs - 1,
              sub_.topology().axis(Direction::Z).comm);
    dmz_g_[N3g + 1] = dmz_last;
  }

  // Allocate pinned host pencil buffers
  const std::size_t sz_C   = (std::size_t)nx_loc_ * ny_loc_ * nz_loc_;
  const std::size_t sz_I   = (std::size_t)N1g     * ny_loc_ * n3_I_;
  const std::size_t sz_hI  = (std::size_t)Nxh_    * ny_loc_ * n3_I_;
  const std::size_t sz_hY  = (std::size_t)std::max(h1p_Y_me_, 1) * N2g * std::max(n3_I_, 1);

  CUDA_CHK(cudaMallocHost(&rhs_C_,  sz_C   * sizeof(double)));
  CUDA_CHK(cudaMallocHost(&rhs_I_,  sz_I   * sizeof(double)));
  CUDA_CHK(cudaMallocHost(&hat_I_,  sz_hI  * sizeof(double2)));
  CUDA_CHK(cudaMallocHost(&hat_Y_,  sz_hY  * sizeof(double2)));
  CUDA_CHK(cudaMallocHost(&dp_I_,   sz_I   * sizeof(double)));
  CUDA_CHK(cudaMallocHost(&dp_C_,   sz_C   * sizeof(double)));

  // Allocate pinned transpose buffers
  {
    const std::size_t blk_C = (std::size_t)nx_loc_ * ny_loc_ * n3_I_;
    CUDA_CHK(cudaMallocHost(&tx_sbuf_C_, np1 * blk_C * sizeof(double)));
    CUDA_CHK(cudaMallocHost(&tx_rbuf_C_, np1 * blk_C * sizeof(double)));

    const std::size_t yn3   = (std::size_t)ny_loc_ * n3_I_;
    const std::size_t nfwd  = (std::size_t)Nxh_ * yn3 * 2;
    const std::size_t nbwd  = (std::size_t)np2 * (std::size_t)std::max(h1p_Y_me_, 1) * yn3 * 2;
    CUDA_CHK(cudaMallocHost(&tx_sbuf_Y_, std::max(nfwd, nbwd) * sizeof(double)));
    CUDA_CHK(cudaMallocHost(&tx_rbuf_Y_, std::max(nfwd, nbwd) * sizeof(double)));
  }

  // Allocate device buffers
  // Fast path: np1==1 && np2==1 — divergence stays on GPU, hat_I_==hat_Y_ alias
  gpu_fast_path_ = (np1 == 1 && np2 == 1);

  CUDA_CHK(cudaMalloc(&d_rhs_I_, sz_I   * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_hat_I_, sz_hI  * sizeof(cufftDoubleComplex)));
  if (np2 == 1 && sz_hI == sz_hY) {
    // hat_Y_ and hat_I_ have identical layout for np2==1; alias to avoid extra allocation
    d_hat_Y_       = d_hat_I_;
    hat_Y_is_alias_ = true;
  } else {
    CUDA_CHK(cudaMalloc(&d_hat_Y_, sz_hY * sizeof(cufftDoubleComplex)));
    hat_Y_is_alias_ = false;
  }
  CUDA_CHK(cudaMalloc(&d_dp_I_,  sz_I   * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_local_sum_, sizeof(double)));

  // Allocate and copy device metric arrays
  CUDA_CHK(cudaMalloc(&d_rhs_C_dev_, sz_C * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dx1_,  (std::size_t)n1_tot_ * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dmx1_, (std::size_t)n1_tot_ * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dx2_,  (std::size_t)n2_tot_ * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dmx2_, (std::size_t)n2_tot_ * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dx3_,  (std::size_t)n3_tot_ * sizeof(double)));
  CUDA_CHK(cudaMalloc(&d_dmx3_, (std::size_t)n3_tot_ * sizeof(double)));
  {
    const real_t* dx1  = grid_.dx_ptr(Direction::X);
    const real_t* dmx1 = grid_.dmx_ptr(Direction::X);
    const real_t* dx2  = grid_.dx_ptr(Direction::Y);
    const real_t* dmx2 = grid_.dmx_ptr(Direction::Y);
    const real_t* dx3  = grid_.dx_ptr(Direction::Z);
    const real_t* dmx3 = grid_.dmx_ptr(Direction::Z);
    auto copy_to_dev = [&](double* d, const real_t* h, int n) {
      if constexpr (sizeof(real_t) == sizeof(double)) {
        CUDA_CHK(cudaMemcpy(d, h, (std::size_t)n * sizeof(double), cudaMemcpyHostToDevice));
      } else {
        std::vector<double> tmp(n);
        for (int k = 0; k < n; ++k) tmp[k] = static_cast<double>(h[k]);
        CUDA_CHK(cudaMemcpy(d, tmp.data(), (std::size_t)n * sizeof(double), cudaMemcpyHostToDevice));
      }
    };
    copy_to_dev(d_dx1_,  dx1,  n1_tot_);
    copy_to_dev(d_dmx1_, dmx1, n1_tot_);
    copy_to_dev(d_dx2_,  dx2,  n2_tot_);
    copy_to_dev(d_dmx2_, dmx2, n2_tot_);
    copy_to_dev(d_dx3_,  dx3,  n3_tot_);
    copy_to_dev(d_dmx3_, dmx3, n3_tot_);
  }

  // cuFFT plans
  {
    const int howmany_x = ny_loc_ * n3_I_;
    int n1g_arr[1] = { N1g };
    CUFFT_CHK(cufftPlanMany(&plan_fwd_x_, 1, n1g_arr,
                             n1g_arr, 1, N1g,
                             &Nxh_,   1, Nxh_,
                             CUFFT_D2Z, howmany_x));
    CUFFT_CHK(cufftPlanMany(&plan_bwd_x_, 1, n1g_arr,
                             &Nxh_,   1, Nxh_,
                             n1g_arr, 1, N1g,
                             CUFFT_Z2D, howmany_x));
  }
  if (h1p_Y_me_ > 0 && n3_I_ > 0) {
    int n2g_arr[1] = { N2g };
    // stride = h1p_Y_me_ (columns), dist = 1 (rows pack tightly) — matches FFTW plan
    CUFFT_CHK(cufftPlanMany(&plan_y_, 1, n2g_arr,
                             n2g_arr, h1p_Y_me_, 1,
                             n2g_arr, h1p_Y_me_, 1,
                             CUFFT_Z2Z, h1p_Y_me_));
  }

  // Host staging buffer for unpack_P_to_gpu_ (zero-initialized once)
  p_host_unpack_.assign((std::size_t)n1_tot_ * n2_tot_ * n3_tot_, 0.0);

  // CPU TDMA z (fallback for np2>1 or multi-rank Y)
  {
    const int rank_xz = sub_.topology().rank_xz();
    const int size_xz = sub_.topology().size_xz();
    comm_xz_1d_.comm      = sub_.topology().comm_xz();
    comm_xz_1d_.rank      = rank_xz;
    comm_xz_1d_.nprocs    = size_xz;
    comm_xz_1d_.west_rank = (rank_xz > 0)           ? rank_xz - 1 : MPI_PROC_NULL;
    comm_xz_1d_.east_rank = (rank_xz < size_xz - 1) ? rank_xz + 1 : MPI_PROC_NULL;

    tdma_z_ = std::make_unique<linear_solver::tdma::PaScaLTDMACpuBackend>(comm_xz_1d_);

    const std::size_t tsz = (std::size_t)n3_I_ * std::max(n_sys_Y_, 1);
    tdma_A_r_.resize(tsz);  tdma_B_r_.resize(tsz);
    tdma_C_r_.resize(tsz);  tdma_D_r_.resize(tsz);
    tdma_A_c_.resize(tsz);  tdma_B_c_.resize(tsz);
    tdma_C_c_.resize(tsz);  tdma_D_c_.resize(tsz);
  }

  // GPU TDMA z: precompute A/B/C on CPU, upload, init CUDA backend (np2==1 only)
  {
    const int np2     = sub_.topology().dims()[1];
    const int N2g     = sub_.n_global()[1];
    const int rank_xz = sub_.topology().rank_xz();
    const int size_xz = sub_.topology().size_xz();

    if (np2 == 1 && n_sys_Y_ > 0 && n3_I_ > 0) {
      const std::size_t tsz = (std::size_t)n_sys_Y_ * n3_I_;

      std::vector<double> A0(tsz), B0(tsz), C0(tsz);
      for (int k = 0; k < n3_I_; ++k) {
        int k_g = rank_xz * n3_I_ + k;
        double dz   = dz_g_ [k_g + 1];
        double dmz  = dmz_g_[k_g + 1];
        double dmzp = dmz_g_[k_g + 2];
        bool lower_wall = (rank_xz == 0            && k == 0);
        bool upper_wall = (rank_xz == size_xz - 1  && k == n3_I_ - 1);
        double a_val = lower_wall ? 0.0 : 1.0 / (dz * dmz);
        double c_val = upper_wall ? 0.0 : 1.0 / (dz * dmzp);
        for (int s = 0; s < n_sys_Y_; ++s) {
          int ixl = s % h1p_Y_me_;
          int j   = s / h1p_Y_me_;
          double lam = lambda_x_[ix_start_Y_me_ + ixl] + lambda_y_[j];
          std::size_t idx = (std::size_t)k * n_sys_Y_ + s;
          A0[idx] = a_val;
          B0[idx] = -(a_val + c_val) - lam;
          C0[idx] = c_val;
        }
      }
      // Zero-mode fix: set pressure to 0 for wavenumber (0,0)
      if (rank_xz == 0 && ix_start_Y_me_ == 0) {
        A0[0] = 0.0; B0[0] = 1.0; C0[0] = 0.0;
      }

      CUDA_CHK(cudaMalloc(&d_tdma_A0_, tsz * sizeof(double)));
      CUDA_CHK(cudaMalloc(&d_tdma_B0_, tsz * sizeof(double)));
      CUDA_CHK(cudaMalloc(&d_tdma_C0_, tsz * sizeof(double)));
      CUDA_CHK(cudaMalloc(&d_tdma_Aw_, tsz * sizeof(double)));
      CUDA_CHK(cudaMalloc(&d_tdma_Bw_, tsz * sizeof(double)));
      CUDA_CHK(cudaMalloc(&d_tdma_Cw_, tsz * sizeof(double)));
      CUDA_CHK(cudaMalloc(&d_tdma_Dr_, tsz * sizeof(double)));
      CUDA_CHK(cudaMalloc(&d_tdma_Dc_, tsz * sizeof(double)));

      CUDA_CHK(cudaMemcpy(d_tdma_A0_, A0.data(), tsz*sizeof(double), cudaMemcpyHostToDevice));
      CUDA_CHK(cudaMemcpy(d_tdma_B0_, B0.data(), tsz*sizeof(double), cudaMemcpyHostToDevice));
      CUDA_CHK(cudaMemcpy(d_tdma_C0_, C0.data(), tsz*sizeof(double), cudaMemcpyHostToDevice));

      tdma_z_cuda_ = std::make_unique<linear_solver::tdma::PaScaLTDMACudaBackend>(comm_xz_1d_);
      gpu_tdma_enabled_ = true;
    }
  }
}


// ── destructor ────────────────────────────────────────────────────────────────

PressureSolverGpu::~PressureSolverGpu()
{
  if (plan_fwd_x_) cufftDestroy(plan_fwd_x_);
  if (plan_bwd_x_) cufftDestroy(plan_bwd_x_);
  if (plan_y_    ) cufftDestroy(plan_y_);

  cudaFreeHost(rhs_C_);   cudaFreeHost(rhs_I_);
  cudaFreeHost(hat_I_);   cudaFreeHost(hat_Y_);
  cudaFreeHost(dp_I_);    cudaFreeHost(dp_C_);
  cudaFreeHost(tx_sbuf_C_); cudaFreeHost(tx_rbuf_C_);
  cudaFreeHost(tx_sbuf_Y_); cudaFreeHost(tx_rbuf_Y_);

  cudaFree(d_rhs_I_); cudaFree(d_hat_I_);
  if (!hat_Y_is_alias_) cudaFree(d_hat_Y_);
  cudaFree(d_dp_I_);
  cudaFree(d_local_sum_);
  cudaFree(d_rhs_C_dev_);
  cudaFree(d_dx1_);  cudaFree(d_dmx1_);
  cudaFree(d_dx2_);  cudaFree(d_dmx2_);
  cudaFree(d_dx3_);  cudaFree(d_dmx3_);
  cudaFree(d_tdma_A0_); cudaFree(d_tdma_B0_); cudaFree(d_tdma_C0_);
  cudaFree(d_tdma_Aw_); cudaFree(d_tdma_Bw_); cudaFree(d_tdma_Cw_);
  cudaFree(d_tdma_Dr_); cudaFree(d_tdma_Dc_);
}


// ── GPU divergence/projection helpers ────────────────────────────────────────

void PressureSolverGpu::divergence_rhs_gpu_(const double* d_U, const double* d_V, const double* d_W, double dt)
{
  const dim3 blk(64, 4, 1);
  const dim3 grd((nz_loc_+63)/64, (ny_loc_+3)/4, nx_loc_);
  k_divergence_rhs_pres<<<grd, blk>>>(
      d_U, d_V, d_W, d_rhs_C_dev_,
      d_dx1_, d_dx2_, d_dx3_,
      n2_tot_, n3_tot_, nx_loc_, ny_loc_, nz_loc_, kHaloWidth, 1.0/dt);
  CUDA_CHK(cudaGetLastError());
}

void PressureSolverGpu::extra_term_nob_gpu_(const double* d_ir, const double* d_ph)
{
  const dim3 blk(64, 4, 1);
  const dim3 grd((nz_loc_+63)/64, (ny_loc_+3)/4, nx_loc_);
  k_extra_term_nob_pres<<<grd, blk>>>(
      d_rhs_C_dev_, d_ir, d_ph,
      d_dx1_, d_dmx1_, d_dx2_, d_dmx2_, d_dx3_, d_dmx3_,
      n2_tot_, n3_tot_, nx_loc_, ny_loc_, nz_loc_, kHaloWidth);
  CUDA_CHK(cudaGetLastError());
}

void PressureSolverGpu::project_ob_gpu_(double* d_U, double* d_V, double* d_W, const double* d_P, double dt)
{
  const dim3 blk(64, 4, 1);
  const dim3 grd((nz_loc_+63)/64, (ny_loc_+3)/4, nx_loc_);
  k_project_ob_pres<<<grd, blk>>>(
      d_U, d_V, d_W, d_P,
      d_dmx1_, d_dmx2_, d_dmx3_,
      n2_tot_, n3_tot_, nx_loc_, ny_loc_, nz_loc_, kHaloWidth, dt);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());
}

void PressureSolverGpu::project_nob_gpu_(double* d_U, double* d_V, double* d_W,
                                          const double* d_P, const double* d_dPhat,
                                          const double* d_ir, double dt)
{
  const dim3 blk(64, 4, 1);
  const dim3 grd((nz_loc_+63)/64, (ny_loc_+3)/4, nx_loc_);
  k_project_nob_pres<<<grd, blk>>>(
      d_U, d_V, d_W, d_P, d_dPhat, d_ir,
      d_dx1_, d_dmx1_, d_dx2_, d_dmx2_, d_dx3_, d_dmx3_,
      n2_tot_, n3_tot_, nx_loc_, ny_loc_, nz_loc_, kHaloWidth, dt);
  CUDA_CHK(cudaGetLastError());
  CUDA_CHK(cudaDeviceSynchronize());
}

// ── transpose_C_to_I_ ─────────────────────────────────────────────────────────
// For np1=1, C-layout == I-layout (N1g==nx_loc_): plain memcpy.

void PressureSolverGpu::transpose_C_to_I_()
{
  const int np1 = sub_.topology().dims()[0];
  const int blk = nx_loc_ * ny_loc_ * n3_I_;
  const int N1g = sub_.n_global()[0];

  if (np1 == 1) {
    std::memcpy(rhs_I_, rhs_C_, (std::size_t)blk * sizeof(double));
    return;
  }

  for (int s = 0; s < np1; ++s) {
    double* sb = tx_sbuf_C_ + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          sb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k] =
              rhs_C_[idx_C_gpu(i, j, s * n3_I_ + k, nx_loc_, ny_loc_)];
  }
  MPI_Alltoall(tx_sbuf_C_, blk, MPI_DOUBLE,
               tx_rbuf_C_, blk, MPI_DOUBLE,
               sub_.topology().axis(Direction::X).comm);
  for (int s = 0; s < np1; ++s) {
    const double* rb = tx_rbuf_C_ + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          rhs_I_[idx_I_gpu(s * nx_loc_ + i, j, k, N1g, ny_loc_)] =
              rb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k];
  }
}

void PressureSolverGpu::transpose_I_to_C_()
{
  const int np1 = sub_.topology().dims()[0];
  const int blk = nx_loc_ * ny_loc_ * n3_I_;
  const int N1g = sub_.n_global()[0];

  if (np1 == 1) {
    std::memcpy(dp_C_, dp_I_, (std::size_t)blk * sizeof(double));
    return;
  }

  for (int s = 0; s < np1; ++s) {
    double* sb = tx_sbuf_C_ + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          sb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k] =
              dp_I_[idx_I_gpu(s * nx_loc_ + i, j, k, N1g, ny_loc_)];
  }
  MPI_Alltoall(tx_sbuf_C_, blk, MPI_DOUBLE,
               tx_rbuf_C_, blk, MPI_DOUBLE,
               sub_.topology().axis(Direction::X).comm);
  for (int s = 0; s < np1; ++s) {
    const double* rb = tx_rbuf_C_ + (std::size_t)s * blk;
    for (int i = 0; i < nx_loc_; ++i)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k)
          dp_C_[idx_C_gpu(i, j, s * n3_I_ + k, nx_loc_, ny_loc_)] =
              rb[(std::size_t)i * ny_loc_ * n3_I_ + j * n3_I_ + k];
  }
}

void PressureSolverGpu::transpose_I_to_Y_()
{
  const int np2 = sub_.topology().dims()[1];
  const int yn3 = ny_loc_ * n3_I_;
  const int N2g = sub_.n_global()[1];

  // For np2=1: h1p_Y_me_==Nxh_, N2g==ny_loc_, j_base==0 → hat_Y_ and hat_I_ share layout.
  if (np2 == 1) {
    std::memcpy(hat_Y_, hat_I_, (std::size_t)Nxh_ * ny_loc_ * n3_I_ * sizeof(double2));
    return;
  }

  std::vector<int> scnts(np2), sdsp(np2), rcnts(np2), rdsp(np2);
  {
    int off = 0;
    for (int r = 0; r < np2; ++r) {
      scnts[r] = h1p_Y_[r] * yn3 * 2;
      sdsp[r]  = off;
      off     += scnts[r];
    }
  }
  const int rcnt_each = h1p_Y_me_ * yn3 * 2;
  for (int r = 0; r < np2; ++r) {
    rcnts[r] = rcnt_each;
    rdsp[r]  = r * rcnt_each;
  }

  {
    int off = 0;
    for (int r = 0; r < np2; ++r) {
      double* sb  = tx_sbuf_Y_ + off;
      const int ix0 = ix_start_Y_[r], nxr = h1p_Y_[r];
      for (int ixl = 0; ixl < nxr; ++ixl)
        for (int j = 0; j < ny_loc_; ++j)
          for (int k = 0; k < n3_I_; ++k) {
            const double2 c = hat_I_[idx_I_gpu(ix0 + ixl, j, k, Nxh_, ny_loc_)];
            int p = (int)(((std::size_t)ixl * yn3 + j * n3_I_ + k) * 2);
            sb[p] = c.x;  sb[p+1] = c.y;
          }
      off += scnts[r];
    }
  }
  MPI_Alltoallv(tx_sbuf_Y_, scnts.data(), sdsp.data(), MPI_DOUBLE,
                tx_rbuf_Y_, rcnts.data(), rdsp.data(), MPI_DOUBLE,
                sub_.topology().axis(Direction::Y).comm);
  for (int r = 0; r < np2; ++r) {
    const double* rb    = tx_rbuf_Y_ + (std::size_t)r * rcnt_each;
    const int     j_base = r * ny_loc_;
    for (int ixl = 0; ixl < h1p_Y_me_; ++ixl)
      for (int jl = 0; jl < ny_loc_; ++jl)
        for (int k = 0; k < n3_I_; ++k) {
          int p = (int)(((std::size_t)ixl * yn3 + jl * n3_I_ + k) * 2);
          hat_Y_[idx_Y_gpu(ixl, j_base + jl, k, h1p_Y_me_, N2g)] = {rb[p], rb[p + 1]};
        }
  }
}

void PressureSolverGpu::transpose_Y_to_I_()
{
  const int np2 = sub_.topology().dims()[1];
  const int yn3 = ny_loc_ * n3_I_;
  const int N2g = sub_.n_global()[1];

  // For np2=1: same layout as transpose_I_to_Y_ → reverse direction memcpy.
  if (np2 == 1) {
    std::memcpy(hat_I_, hat_Y_, (std::size_t)Nxh_ * ny_loc_ * n3_I_ * sizeof(double2));
    return;
  }

  const int scnt_each = h1p_Y_me_ * yn3 * 2;
  std::vector<int> scnts(np2), sdsp(np2), rcnts(np2), rdsp(np2);
  for (int r = 0; r < np2; ++r) {
    scnts[r] = scnt_each;
    sdsp[r]  = r * scnt_each;
    rcnts[r] = h1p_Y_[r] * yn3 * 2;
  }
  {
    int off = 0;
    for (int r = 0; r < np2; ++r) { rdsp[r] = off; off += rcnts[r]; }
  }

  for (int r = 0; r < np2; ++r) {
    double*   sb     = tx_sbuf_Y_ + (std::size_t)r * scnt_each;
    const int j_base = r * ny_loc_;
    for (int ixl = 0; ixl < h1p_Y_me_; ++ixl)
      for (int jl = 0; jl < ny_loc_; ++jl)
        for (int k = 0; k < n3_I_; ++k) {
          const double2 c = hat_Y_[idx_Y_gpu(ixl, j_base + jl, k, h1p_Y_me_, N2g)];
          int p = (int)(((std::size_t)ixl * yn3 + jl * n3_I_ + k) * 2);
          sb[p] = c.x;  sb[p+1] = c.y;
        }
  }
  MPI_Alltoallv(tx_sbuf_Y_, scnts.data(), sdsp.data(), MPI_DOUBLE,
                tx_rbuf_Y_, rcnts.data(), rdsp.data(), MPI_DOUBLE,
                sub_.topology().axis(Direction::Y).comm);
  int off = 0;
  for (int r = 0; r < np2; ++r) {
    const double* rb  = tx_rbuf_Y_ + off;
    int           ix0 = ix_start_Y_[r], nxr = h1p_Y_[r];
    for (int ixl = 0; ixl < nxr; ++ixl)
      for (int j = 0; j < ny_loc_; ++j)
        for (int k = 0; k < n3_I_; ++k) {
          int p = (int)(((std::size_t)ixl * yn3 + j * n3_I_ + k) * 2);
          hat_I_[idx_I_gpu(ix0 + ixl, j, k, Nxh_, ny_loc_)] = {rb[p], rb[p+1]};
        }
    off += rcnts[r];
  }
}


// ── solve_tdma_z_ — verbatim from CPU engine ──────────────────────────────────

void PressureSolverGpu::solve_tdma_z_()
{
  if (!tdma_z_ || n_sys_Y_ == 0 || n3_I_ == 0) return;

  const int N2g     = sub_.n_global()[1];
  const int rank_xz = sub_.topology().rank_xz();
  const int size_xz = sub_.topology().size_xz();

  for (int k = 0; k < n3_I_; ++k) {
    const int    k_g  = rank_xz * n3_I_ + k;
    const double dz   = dz_g_ [k_g + 1];
    const double dmz  = dmz_g_[k_g + 1];
    const double dmzp = dmz_g_[k_g + 2];

    const bool lower_wall = (rank_xz == 0            && k == 0);
    const bool upper_wall = (rank_xz == size_xz - 1  && k == n3_I_ - 1);

    const double a_val = lower_wall ? 0.0 : 1.0 / (dz * dmz );
    const double c_val = upper_wall ? 0.0 : 1.0 / (dz * dmzp);

    for (int j = 0; j < N2g; ++j)
      for (int ixl = 0; ixl < h1p_Y_me_; ++ixl) {
        const int         s   = ixl + h1p_Y_me_ * j;
        const std::size_t idx = (std::size_t)k * n_sys_Y_ + s;
        const double lam      = lambda_x_[ix_start_Y_me_ + ixl] + lambda_y_[j];

        tdma_A_r_[idx] = tdma_A_c_[idx] = a_val;
        tdma_C_r_[idx] = tdma_C_c_[idx] = c_val;
        tdma_B_r_[idx] = tdma_B_c_[idx] = -(a_val + c_val) - lam;

        const double2 cv = hat_Y_[idx_Y_gpu(ixl, j, k, h1p_Y_me_, N2g)];
        tdma_D_r_[idx] = cv.x;
        tdma_D_c_[idx] = cv.y;
      }
  }

  if (rank_xz == 0 && ix_start_Y_me_ == 0 && h1p_Y_me_ > 0) {
    tdma_A_r_[0] = tdma_A_c_[0] = 0.0;
    tdma_C_r_[0] = tdma_C_c_[0] = 0.0;
    tdma_B_r_[0] = tdma_B_c_[0] = 1.0;
    tdma_D_r_[0] = tdma_D_c_[0] = 0.0;
  }

  tdma_z_->solve_many(tdma_A_r_.data(), tdma_B_r_.data(), tdma_C_r_.data(), tdma_D_r_.data(),
                      n_sys_Y_, n3_I_);
  tdma_z_->solve_many(tdma_A_c_.data(), tdma_B_c_.data(), tdma_C_c_.data(), tdma_D_c_.data(),
                      n_sys_Y_, n3_I_);

  for (int k = 0; k < n3_I_; ++k)
    for (int j = 0; j < N2g; ++j)
      for (int ixl = 0; ixl < h1p_Y_me_; ++ixl) {
        const int         s   = ixl + h1p_Y_me_ * j;
        const std::size_t idx = (std::size_t)k * n_sys_Y_ + s;
        hat_Y_[idx_Y_gpu(ixl, j, k, h1p_Y_me_, N2g)] = {tdma_D_r_[idx], tdma_D_c_[idx]};
      }
}


// ── GPU TDMA z (np2==1 path — hat_Y_ stays on device throughout) ─────────────

void PressureSolverGpu::solve_tdma_z_gpu_()
{
  if (!gpu_tdma_enabled_) { solve_tdma_z_(); return; }
  if (n_sys_Y_ == 0 || n3_I_ == 0) return;

  const std::size_t tsz = (std::size_t)n_sys_Y_ * n3_I_;
  const int blk = 256;
  const int grd = (int)((tsz + blk - 1) / blk);

  // Split d_hat_Y_ (double2) → d_tdma_Dr_, d_tdma_Dc_ (double each)
  k_tdma_split_hat_Y<<<grd, blk>>>(d_hat_Y_, d_tdma_Dr_, d_tdma_Dc_, tsz);
  CUDA_CHK(cudaGetLastError());

  // Real solve: restore A/B/C (Thomas modifies in-place), then solve
  CUDA_CHK(cudaMemcpy(d_tdma_Aw_, d_tdma_A0_, tsz*sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CHK(cudaMemcpy(d_tdma_Bw_, d_tdma_B0_, tsz*sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CHK(cudaMemcpy(d_tdma_Cw_, d_tdma_C0_, tsz*sizeof(double), cudaMemcpyDeviceToDevice));
  tdma_z_cuda_->solve_many_device(d_tdma_Aw_, d_tdma_Bw_, d_tdma_Cw_, d_tdma_Dr_, n_sys_Y_, n3_I_);

  // Imaginary solve
  CUDA_CHK(cudaMemcpy(d_tdma_Aw_, d_tdma_A0_, tsz*sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CHK(cudaMemcpy(d_tdma_Bw_, d_tdma_B0_, tsz*sizeof(double), cudaMemcpyDeviceToDevice));
  CUDA_CHK(cudaMemcpy(d_tdma_Cw_, d_tdma_C0_, tsz*sizeof(double), cudaMemcpyDeviceToDevice));
  tdma_z_cuda_->solve_many_device(d_tdma_Aw_, d_tdma_Bw_, d_tdma_Cw_, d_tdma_Dc_, n_sys_Y_, n3_I_);

  // Merge results back into d_hat_Y_
  k_tdma_merge_hat_Y<<<grd, blk>>>(d_tdma_Dr_, d_tdma_Dc_, d_hat_Y_, tsz);
  CUDA_CHK(cudaGetLastError());
}


// ── Forward pipeline: C→I, FFT-x, I→Y, FFT-y ────────────────────────────────

void PressureSolverGpu::run_forward_pipeline_()
{
  using hrc = std::chrono::high_resolution_clock;
  auto ms_ = [](hrc::time_point a, hrc::time_point b){
    return std::chrono::duration<double,std::milli>(b-a).count(); };
  static int _fwd_n = 0; ++_fwd_n;
  int _cr = 0; MPI_Comm_rank(sub_.topology().cart_comm(), &_cr);
  const bool _pr = (_fwd_n <= 5 && _cr == 0);

  const int N1g = sub_.n_global()[0];
  const int N2g = sub_.n_global()[1];

  if (gpu_fast_path_) {
    // np1==1, np2==1: d_rhs_C_dev_ has same layout as rhs_I_, d_hat_Y_==d_hat_I_ alias.
    // Entire pipeline stays on device — no H↔D transfers.
    const std::size_t slab = (std::size_t)h1p_Y_me_ * N2g;
    CUFFT_CHK(cufftExecD2Z(plan_fwd_x_, d_rhs_C_dev_, d_hat_I_));
    if (h1p_Y_me_ > 0) {
      for (int k = 0; k < n3_I_; ++k)
        CUFFT_CHK(cufftExecZ2Z(plan_y_,
                               d_hat_I_ + k * slab,
                               d_hat_I_ + k * slab,
                               CUFFT_FORWARD));
    }
    CUDA_CHK(cudaDeviceSynchronize());
    return;
  }

  auto F0 = hrc::now();
  // C→I
  transpose_C_to_I_();
  auto F1 = hrc::now();

  // H2D rhs_I, cuFFT r2c → d_hat_I_, D2H hat_I_
  CUDA_CHK(cudaMemcpy(d_rhs_I_, rhs_I_,
                      (std::size_t)N1g * ny_loc_ * n3_I_ * sizeof(double),
                      cudaMemcpyHostToDevice));
  auto F2 = hrc::now();
  CUFFT_CHK(cufftExecD2Z(plan_fwd_x_, d_rhs_I_, d_hat_I_));
  CUDA_CHK(cudaDeviceSynchronize());
  auto F3 = hrc::now();
  CUDA_CHK(cudaMemcpy(hat_I_, d_hat_I_,
                      (std::size_t)Nxh_ * ny_loc_ * n3_I_ * sizeof(double2),
                      cudaMemcpyDeviceToHost));
  auto F4 = hrc::now();

  // I→Y
  transpose_I_to_Y_();
  auto F5 = hrc::now();

  // H2D hat_Y_, cuFFT forward y (one slab per k)
  // GPU-TDMA path: skip final D2H — d_hat_Y_ stays on device for solve_tdma_z_gpu_()
  {
    const std::size_t slab = (std::size_t)h1p_Y_me_ * N2g;
    CUDA_CHK(cudaMemcpy(d_hat_Y_, hat_Y_,
                        slab * n3_I_ * sizeof(double2),
                        cudaMemcpyHostToDevice));
    auto F6 = hrc::now();
    if (h1p_Y_me_ > 0) {
      for (int k = 0; k < n3_I_; ++k)
        CUFFT_CHK(cufftExecZ2Z(plan_y_,
                               d_hat_Y_ + k * slab,
                               d_hat_Y_ + k * slab,
                               CUFFT_FORWARD));
    }
    CUDA_CHK(cudaDeviceSynchronize());
    auto F7 = hrc::now();
    if (!gpu_tdma_enabled_)
      CUDA_CHK(cudaMemcpy(hat_Y_, d_hat_Y_,
                          slab * n3_I_ * sizeof(double2),
                          cudaMemcpyDeviceToHost));
    if (_pr)
      std::printf("[FWD #%d] C2I=%.2f H2D_rhs=%.2f fftx=%.2f D2H_hat=%.2f I2Y=%.2f H2D_hatY=%.2f ffty=%.2f ms  n3_I=%d h1p=%d\n",
                  _fwd_n, ms_(F0,F1), ms_(F1,F2), ms_(F2,F3), ms_(F3,F4),
                  ms_(F4,F5), ms_(F5,F6), ms_(F6,F7), n3_I_, h1p_Y_me_);
  }
}


// ── Backward pipeline: IFFT-y, Y→I, IFFT-x, scale, I→C ──────────────────────

void PressureSolverGpu::run_backward_pipeline_()
{
  const int N1g = sub_.n_global()[0];
  const int N2g = sub_.n_global()[1];
  const std::size_t slab = (std::size_t)h1p_Y_me_ * N2g;
  const std::size_t sz   = (std::size_t)N1g * ny_loc_ * n3_I_;

  if (gpu_fast_path_) {
    // np1==1, np2==1: d_hat_Y_==d_hat_I_ alias, stay fully on device.
    if (h1p_Y_me_ > 0) {
      for (int k = 0; k < n3_I_; ++k)
        CUFFT_CHK(cufftExecZ2Z(plan_y_,
                               d_hat_I_ + k * slab,
                               d_hat_I_ + k * slab,
                               CUFFT_INVERSE));
    }
    // c2r directly from d_hat_I_ (np2=1: hat_I_ layout == hat_Y_ layout)
    CUFFT_CHK(cufftExecZ2D(plan_bwd_x_, d_hat_I_, d_dp_I_));
    const double scale = 1.0 / (double(N1g) * double(sub_.n_global()[1]));
    k_gpu_scale<<<(int)((sz+255)/256), 256>>>(d_dp_I_, scale, sz);
    CUDA_CHK(cudaDeviceSynchronize());
    // d_dp_I_ now holds the scaled pressure increment in I-layout (=C-layout for np1=1)
    // Caller must call unpack_P_fast_() instead of unpack_P_to_gpu_()
    return;
  }

  // GPU-TDMA path: d_hat_Y_ is already on device with TDMA result.
  // CPU-TDMA path: hat_Y_ is on CPU, upload first.
  if (!gpu_tdma_enabled_)
    CUDA_CHK(cudaMemcpy(d_hat_Y_, hat_Y_,
                        slab * n3_I_ * sizeof(double2),
                        cudaMemcpyHostToDevice));

  // cuFFT inverse y (in-place on d_hat_Y_)
  if (h1p_Y_me_ > 0) {
    for (int k = 0; k < n3_I_; ++k)
      CUFFT_CHK(cufftExecZ2Z(plan_y_,
                             d_hat_Y_ + k * slab,
                             d_hat_Y_ + k * slab,
                             CUFFT_INVERSE));
  }
  CUDA_CHK(cudaDeviceSynchronize());

  if (gpu_tdma_enabled_) {
    // GPU path: D2D hat_Y→hat_I (same layout for np2=1), skip CPU transposes.
    CUDA_CHK(cudaMemcpy(d_hat_I_, d_hat_Y_,
                        slab * n3_I_ * sizeof(double2),
                        cudaMemcpyDeviceToDevice));
  } else {
    // CPU path: D2H hat_Y_, CPU transpose Y→I, H2D hat_I_
    CUDA_CHK(cudaMemcpy(hat_Y_, d_hat_Y_,
                        slab * n3_I_ * sizeof(double2),
                        cudaMemcpyDeviceToHost));
    transpose_Y_to_I_();
    CUDA_CHK(cudaMemcpy(d_hat_I_, hat_I_,
                        (std::size_t)Nxh_ * ny_loc_ * n3_I_ * sizeof(double2),
                        cudaMemcpyHostToDevice));
  }

  // cuFFT c2r → d_dp_I_, scale, D2H dp_I_
  // (d_hat_I_ already populated — either by D2D above or H2D in CPU path)
  CUFFT_CHK(cufftExecZ2D(plan_bwd_x_, d_hat_I_, d_dp_I_));
  CUDA_CHK(cudaDeviceSynchronize());

  {
    const double scale = 1.0 / (static_cast<double>(N1g) * static_cast<double>(sub_.n_global()[1]));
    const int blk = 256;
    const int grd = (int)((sz + blk - 1) / blk);
    k_gpu_scale<<<grd, blk>>>(d_dp_I_, scale, sz);
    CUDA_CHK(cudaDeviceSynchronize());
  }

  CUDA_CHK(cudaMemcpy(dp_I_, d_dp_I_,
                      (std::size_t)N1g * ny_loc_ * n3_I_ * sizeof(double),
                      cudaMemcpyDeviceToHost));

  // I→C
  transpose_I_to_C_();
}


// ── unpack dp_C_ → GPU P field ────────────────────────────────────────────────

void PressureSolverGpu::unpack_P_to_gpu_(core::GpuField& P_gpu,
                                          const parallel::mpi::Subdomain& sub)
{
  const int h   = kHaloWidth;
  const int n2t = n2_tot_, n3t = n3_tot_;

  // Compute and remove global mean
  double local_sum = 0.0;
  const std::size_t n_loc = (std::size_t)nx_loc_ * ny_loc_ * nz_loc_;
  for (std::size_t q = 0; q < n_loc; ++q)
    local_sum += dp_C_[q];
  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                sub.topology().cart_comm());
  const int    n_int_global = sub_.n_global()[0] * sub_.n_global()[1] * sub_.n_global()[2];
  const double mean_dp = global_sum / n_int_global;

  // Fill pre-allocated host buffer (inner cells only; ghost cells stay 0 except walls)
  double* p_host = p_host_unpack_.data();

  for (int kk = 0; kk < nz_loc_; ++kk)
    for (int jj = 0; jj < ny_loc_; ++jj)
      for (int ii = 0; ii < nx_loc_; ++ii) {
        const int i = ii + h, j = jj + h, k = kk + h;
        p_host[(std::size_t)(i * n2t + j) * n3t + k] =
            dp_C_[idx_C_gpu(ii, jj, kk, nx_loc_, ny_loc_)] - mean_dp;
      }

  // Neumann ghost at z-walls
  const int rz      = sub.topology().axis(Direction::Z).rank;
  const int np3_val = sub.topology().axis(Direction::Z).nprocs;
  if (rz == 0) {
    for (int jj = 0; jj < ny_loc_; ++jj)
      for (int ii = 0; ii < nx_loc_; ++ii) {
        const int i = ii + h, j = jj + h;
        p_host[(std::size_t)(i * n2t + j) * n3t + (h - 1)] =
            p_host[(std::size_t)(i * n2t + j) * n3t + h];
      }
  }
  if (rz == np3_val - 1) {
    for (int jj = 0; jj < ny_loc_; ++jj)
      for (int ii = 0; ii < nx_loc_; ++ii) {
        const int i = ii + h, j = jj + h;
        p_host[(std::size_t)(i * n2t + j) * n3t + (h + nz_loc_)] =
            p_host[(std::size_t)(i * n2t + j) * n3t + (h + nz_loc_ - 1)];
      }
  }

  P_gpu.to_device(p_host);
}


// ── GPU fast-path unpack (np1==1, np2==1): no CPU loop, no pageable H2D ──────
void PressureSolverGpu::unpack_P_fast_(core::GpuField& P_gpu,
                                        const parallel::mpi::Subdomain& sub)
{
  // d_dp_I_ holds scaled pressure in I-layout (= C-layout for np1=1):
  //   dp[ii + nx_loc_*(jj + ny_loc_*kk)]  — x-fastest, no halos

  // GPU sum reduction for local mean
  const int n_loc = nx_loc_ * ny_loc_ * nz_loc_;
  CUDA_CHK(cudaMemset(d_local_sum_, 0, sizeof(double)));
  const int blk_r = 256;
  const int grd_r = std::min((n_loc + blk_r - 1) / blk_r, 512);
  k_pres_local_sum<<<grd_r, blk_r, (std::size_t)blk_r * sizeof(double)>>>(
      d_dp_I_, d_local_sum_, n_loc);
  CUDA_CHK(cudaDeviceSynchronize());

  double local_sum = 0.0;
  CUDA_CHK(cudaMemcpy(&local_sum, d_local_sum_, sizeof(double), cudaMemcpyDeviceToHost));

  double global_sum = 0.0;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
                sub.topology().cart_comm());
  const double mean_dp = global_sum / (sub_.n_global()[0] * sub_.n_global()[1] * sub_.n_global()[2]);

  // GPU kernel: transpose C→ZF layout, subtract mean, fill Neumann ghost
  const int rz      = sub.topology().axis(Direction::Z).rank;
  const int np3_val = sub.topology().axis(Direction::Z).nprocs;
  const bool lower_wall = (rz == 0);
  const bool upper_wall = (rz == np3_val - 1);

  const dim3 blk3(64, 4, 1);
  const dim3 grd3((nz_loc_+63)/64, (ny_loc_+3)/4, nx_loc_);
  k_unpack_dp_C_to_P<<<grd3, blk3>>>(d_dp_I_, P_gpu.data(), mean_dp,
                                      nx_loc_, ny_loc_, nz_loc_,
                                      n2_tot_, n3_tot_, kHaloWidth,
                                      lower_wall, upper_wall);
  CUDA_CHK(cudaGetLastError());
}


// ── OB solve ──────────────────────────────────────────────────────────────────

void PressureSolverGpu::solve(real_t dt, core::GpuFields& gpu_fields,
                               const boundary::Problem& problem,
                               const parallel::mpi::Subdomain& sub)
{
  const std::size_t sz_C = (std::size_t)nx_loc_ * ny_loc_ * nz_loc_;
  double* d_U = gpu_fields[core::Var::U].data();
  double* d_V = gpu_fields[core::Var::V].data();
  double* d_W = gpu_fields[core::Var::W].data();

  // 1. GPU divergence → d_rhs_C_dev_
  divergence_rhs_gpu_(d_U, d_V, d_W, static_cast<double>(dt));
  CUDA_CHK(cudaDeviceSynchronize());

  // 2. D2H only needed for non-fast-path (C→I on CPU)
  if (!gpu_fast_path_)
    CUDA_CHK(cudaMemcpy(rhs_C_, d_rhs_C_dev_, sz_C * sizeof(double), cudaMemcpyDeviceToHost));

  // 3–5. Forward pipeline, TDMA, backward pipeline
  run_forward_pipeline_();
  solve_tdma_z_gpu_();
  run_backward_pipeline_();

  // 6. Unpack P, sync P on GPU
  if (gpu_fast_path_) unpack_P_fast_(gpu_fields[core::Var::P], sub);
  else                unpack_P_to_gpu_(gpu_fields[core::Var::P], sub);
  core::sync_field_gpu(gpu_fields[core::Var::P], problem.P, sub);

  // 7. GPU OB projection — no D2H P, no H2D U,V,W
  project_ob_gpu_(d_U, d_V, d_W, gpu_fields[core::Var::P].data(), static_cast<double>(dt));

  core::sync_field_gpu(gpu_fields[core::Var::U], problem.U, sub);
  core::sync_field_gpu(gpu_fields[core::Var::V], problem.V, sub);
  core::sync_field_gpu(gpu_fields[core::Var::W], problem.W, sub);
}


// ── NOB solve ─────────────────────────────────────────────────────────────────

void PressureSolverGpu::solve_nob(real_t dt, core::GpuFields& gpu_fields,
                                   const boundary::Problem& problem,
                                   const parallel::mpi::Subdomain& sub,
                                   const core::GpuField& inv_rho,
                                   core::GpuField& dPhat)
{
  using hrc = std::chrono::high_resolution_clock;
  auto ms_ = [](hrc::time_point t0){
    return std::chrono::duration<double,std::milli>(hrc::now()-t0).count(); };
  static int _nob_cnt = 0; ++_nob_cnt;
  int _cart_rank = 0;
  MPI_Comm_rank(sub_.topology().cart_comm(), &_cart_rank);
  const bool _pr = (_nob_cnt <= 10 && _cart_rank == 0);

  const std::size_t sz_C = (std::size_t)nx_loc_ * ny_loc_ * nz_loc_;
  const std::size_t nf   = (std::size_t)n1_tot_ * n2_tot_ * n3_tot_;

  double* d_U  = gpu_fields[core::Var::U].data();
  double* d_V  = gpu_fields[core::Var::V].data();
  double* d_W  = gpu_fields[core::Var::W].data();
  const double* d_ir = inv_rho.data();
  const double* d_ph = dPhat.data();

  auto T0 = hrc::now();
  // 1. GPU divergence → d_rhs_C_dev_
  divergence_rhs_gpu_(d_U, d_V, d_W, static_cast<double>(dt));
  // 2. GPU NOB extra term → d_rhs_C_dev_ +=
  extra_term_nob_gpu_(d_ir, d_ph);
  CUDA_CHK(cudaDeviceSynchronize());
  auto T1 = hrc::now();
  // 3. D2H only needed for non-fast-path
  if (!gpu_fast_path_)
    CUDA_CHK(cudaMemcpy(rhs_C_, d_rhs_C_dev_, sz_C * sizeof(double), cudaMemcpyDeviceToHost));
  auto T2 = hrc::now();
  // 4–6. Forward pipeline, TDMA, backward pipeline
  run_forward_pipeline_();
  auto T3 = hrc::now();
  solve_tdma_z_gpu_();
  auto T4 = hrc::now();
  run_backward_pipeline_();
  auto T5 = hrc::now();
  // 7. Unpack P, sync P
  if (gpu_fast_path_) unpack_P_fast_(gpu_fields[core::Var::P], sub);
  else                unpack_P_to_gpu_(gpu_fields[core::Var::P], sub);
  core::sync_field_gpu(gpu_fields[core::Var::P], problem.P, sub);
  auto T6 = hrc::now();
  // 8. GPU NOB projection
  project_nob_gpu_(d_U, d_V, d_W, gpu_fields[core::Var::P].data(), d_ph, d_ir,
                   static_cast<double>(dt));
  auto T7 = hrc::now();
  // 9. dPhat D2D
  CUDA_CHK(cudaMemcpy(dPhat.data(), gpu_fields[core::Var::P].data(),
                      nf * sizeof(double), cudaMemcpyDeviceToDevice));
  core::sync_field_gpu(gpu_fields[core::Var::U], problem.U, sub);
  core::sync_field_gpu(gpu_fields[core::Var::V], problem.V, sub);
  core::sync_field_gpu(gpu_fields[core::Var::W], problem.W, sub);
  cudaDeviceSynchronize();
  auto T8 = hrc::now();

  if (_pr) {
    auto seg = [](hrc::time_point a, hrc::time_point b){
      return std::chrono::duration<double,std::milli>(b-a).count(); };
    std::printf("[NOB_PRES #%d] fast=%d gpu_k=%.2f D2H=%.2f fwd=%.2f tdma=%.2f bwd=%.2f unpack=%.2f proj=%.2f sync=%.2f ms\n",
                _nob_cnt, (int)gpu_fast_path_,
                seg(T0,T1), seg(T1,T2), seg(T2,T3), seg(T3,T4),
                seg(T4,T5), seg(T5,T6), seg(T6,T7), seg(T7,T8));
  }
}

} // namespace mpmstd::equation::pressure
