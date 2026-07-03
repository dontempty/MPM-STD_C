#include "equation/scalar/assemble.hpp"
#include "equation/scalar/solve.hpp"
#include "equation/scalar/kernels.hpp"
#include "equation/scalar/scalar_var_stencil.hpp"
#include "core/boundary_ops.hpp"
#include "linear_solver/tdma/pascal_tdma_cuda_backend.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"
#include "core/gpu/gpu_grid.hpp"

#ifdef MPMSTD_BACKEND_CUDA
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>

namespace {

#define CUDA_CHK(expr)  do {                                             \
  cudaError_t _e = (expr);                                               \
  if (_e != cudaSuccess) {                                               \
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                 cudaGetErrorName(_e), __FILE__, __LINE__,               \
                 cudaGetErrorString(_e));                                 \
    std::abort();                                                         \
  }                                                                       \
} while(0)

// ---------------------------------------------------------------------------
//  Device buffer cache (one per MPI rank / per process)
// ---------------------------------------------------------------------------
static double* d_rhs_    = nullptr;
static double* d_stage_  = nullptr;
static double* d_A_      = nullptr;
static double* d_B_      = nullptr;
static double* d_C_      = nullptr;
static double* d_D_      = nullptr;
static std::size_t d_field_cap_ = 0;
static std::size_t d_band_cap_  = 0;

// OB (const-diff) property arrays: kappa=Ct, dkappa=0, inv_rhocp=1, dinv_rhocp=0
static double* d_ob_kappa_     = nullptr;  // filled with alpha_T
static double* d_ob_dkappa_    = nullptr;  // filled with 0
static double* d_ob_inv_rhocp_ = nullptr;  // filled with 1
static double* d_ob_dinv_rcp_  = nullptr;  // filled with 0
static std::size_t d_ob_cap_   = 0;
static double      d_ob_Ct_    = -1.0;     // sentinel: != alpha_T means needs refill

__global__ void k_fill_const(double* buf, double val, std::size_t n) {
  const std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = val;
}

void ensure_scalar_device_bufs(std::size_t n_field, std::size_t n_band) {
  if (n_field > d_field_cap_) {
    CUDA_CHK(cudaFree(d_rhs_));   CUDA_CHK(cudaMalloc(&d_rhs_,   n_field * sizeof(double)));
    CUDA_CHK(cudaFree(d_stage_)); CUDA_CHK(cudaMalloc(&d_stage_, n_field * sizeof(double)));
    d_field_cap_ = n_field;
  }
  if (n_band > d_band_cap_) {
    CUDA_CHK(cudaFree(d_A_)); CUDA_CHK(cudaMalloc(&d_A_, n_band * sizeof(double)));
    CUDA_CHK(cudaFree(d_B_)); CUDA_CHK(cudaMalloc(&d_B_, n_band * sizeof(double)));
    CUDA_CHK(cudaFree(d_C_)); CUDA_CHK(cudaMalloc(&d_C_, n_band * sizeof(double)));
    CUDA_CHK(cudaFree(d_D_)); CUDA_CHK(cudaMalloc(&d_D_, n_band * sizeof(double)));
    d_band_cap_ = n_band;
  }
}

void ensure_ob_property_bufs(std::size_t n_field, double Ct) {
  if (n_field > d_ob_cap_) {
    CUDA_CHK(cudaFree(d_ob_kappa_));     CUDA_CHK(cudaMalloc(&d_ob_kappa_,     n_field * sizeof(double)));
    CUDA_CHK(cudaFree(d_ob_dkappa_));    CUDA_CHK(cudaMalloc(&d_ob_dkappa_,    n_field * sizeof(double)));
    CUDA_CHK(cudaFree(d_ob_inv_rhocp_)); CUDA_CHK(cudaMalloc(&d_ob_inv_rhocp_, n_field * sizeof(double)));
    CUDA_CHK(cudaFree(d_ob_dinv_rcp_));  CUDA_CHK(cudaMalloc(&d_ob_dinv_rcp_,  n_field * sizeof(double)));
    d_ob_cap_ = n_field;
    d_ob_Ct_  = -1.0;  // force refill
  }
  if (Ct != d_ob_Ct_) {
    const int blk = 256;
    const int grd = (int)((n_field + blk - 1) / blk);
    k_fill_const<<<grd, blk>>>(d_ob_kappa_,     1.0, n_field);  // κ̂=1 (OB: κ/κ_ref=1); Ct=α_T is in irc_h
    k_fill_const<<<grd, blk>>>(d_ob_dkappa_,    0.0, n_field);
    k_fill_const<<<grd, blk>>>(d_ob_inv_rhocp_, 1.0, n_field);
    k_fill_const<<<grd, blk>>>(d_ob_dinv_rcp_,  0.0, n_field);
    CUDA_CHK(cudaDeviceSynchronize());
    d_ob_Ct_ = Ct;
  }
}

// ---------------------------------------------------------------------------
//  GPU kernel: explicit energy residual (one thread per interior cell)
// ---------------------------------------------------------------------------
__global__ void k_energy_residual_var(
    double* rhs, const double* T,
    const double* U, const double* V, const double* W,
    const double* kappa, const double* dkappa,
    const double* inv_rhocp, const double* dinv_rhocp,
    const double* dx1, const double* dmx1,
    const double* dx2, const double* dmx2,
    const double* dx3, const double* dmx3,
    int n1_int, int n2_int, int n3_int,
    int n1_tot, int n2_tot, int n3_tot,
    double Ct) {
  using namespace mpmstd::equation::scalar::kernels::stencil;
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int kk = blockIdx.z * blockDim.z + threadIdx.z;
  if (ii >= n1_int || jj >= n2_int || kk >= n3_int) return;
  const int i = ii + kH, j = jj + kH, k = kk + kH;
  const int c = idx3(i, j, k, n2_tot, n3_tot);
  rhs[c] = energy_rhs_cell(i, j, k, n2_tot, n3_tot,
                            T, U, V, W, kappa, dkappa, inv_rhocp, dinv_rhocp,
                            dx1, dmx1, dx2, dmx2, dx3, dmx3, Ct);
}

// ---------------------------------------------------------------------------
//  GPU kernel: build ADI bands for one sweep direction
//  Grid layout: (n_row × n_sys) threads — each emits one TDMA entry.
// ---------------------------------------------------------------------------
__global__ void k_build_adi_var(
    mpmstd::Direction d,
    double* A, double* B, double* C, double* DD,
    const double* src, const double* T,
    const double* U, const double* V, const double* W,
    const double* kappa, const double* dkappa,
    const double* inv_rhocp, const double* dinv_rhocp,
    const double* dx1, const double* dmx1,
    const double* dx2, const double* dmx2,
    const double* dx3, const double* dmx3,
    int n1_int, int n2_int, int n3_int,
    int n1_tot, int n2_tot, int n3_tot,
    double Ct, double dt, bool is_stage1) {
  using namespace mpmstd::equation::scalar::kernels::stencil;

  // sys in threadIdx.x so warp has adjacent sys values → coalesced writes to A/B/C/D[row*n_sys+sys]
  const int sys = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  int i, j, k, n_row_max, n_sys_max, p;
  if (d == mpmstd::Direction::X) {
    n_row_max = n1_int; n_sys_max = n2_int * n3_int;
    if (row >= n_row_max || sys >= n_sys_max) return;
    i = row + kH;
    j = (sys / n3_int) + kH;
    k = (sys % n3_int) + kH;
    p = row * n_sys_max + sys;
  } else if (d == mpmstd::Direction::Y) {
    n_row_max = n2_int; n_sys_max = n1_int * n3_int;
    if (row >= n_row_max || sys >= n_sys_max) return;
    j = row + kH;
    i = (sys / n3_int) + kH;
    k = (sys % n3_int) + kH;
    p = row * n_sys_max + sys;
  } else {
    n_row_max = n3_int; n_sys_max = n1_int * n2_int;
    if (row >= n_row_max || sys >= n_sys_max) return;
    k = row + kH;
    i = (sys / n2_int) + kH;
    j = (sys % n2_int) + kH;
    p = row * n_sys_max + sys;
  }

  // PaScaL_TCS jem/jep mask: zero dkPP at wall-adjacent Z faces (all directions)
  const int kk = k - kH;
  const bool z_wall_lo = (kk == 0);
  const bool z_wall_hi = (kk == n3_int - 1);

  double am, ac, ap, Dv;
  build_cell_var(d, i, j, k, n1_tot, n2_tot, n3_tot,
                 src, T, U, V, W, kappa, dkappa, inv_rhocp, dinv_rhocp,
                 dx1, dmx1, dx2, dmx2, dx3, dmx3,
                 Ct, dt, is_stage1, z_wall_lo, z_wall_hi, am, ac, ap, Dv);
  A[p] = am; B[p] = ac; C[p] = ap; DD[p] = Dv;
}

// ---------------------------------------------------------------------------
//  GPU kernel: scatter TDMA solution from band layout back to 3D field
// ---------------------------------------------------------------------------
__global__ void k_scatter_from_tdma(
    mpmstd::Direction d,
    double* dst, const double* D_sol,
    int n1_int, int n2_int, int n3_int,
    int n1_tot, int n2_tot, int n3_tot) {
  using namespace mpmstd::equation::scalar::kernels::stencil;
  const int sys = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;

  int i, j, k, n_row_max, n_sys_max;
  if (d == mpmstd::Direction::X) {
    n_row_max = n1_int; n_sys_max = n2_int * n3_int;
    if (row >= n_row_max || sys >= n_sys_max) return;
    i = row + kH; j = (sys / n3_int) + kH; k = (sys % n3_int) + kH;
  } else if (d == mpmstd::Direction::Y) {
    n_row_max = n2_int; n_sys_max = n1_int * n3_int;
    if (row >= n_row_max || sys >= n_sys_max) return;
    j = row + kH; i = (sys / n3_int) + kH; k = (sys % n3_int) + kH;
  } else {
    n_row_max = n3_int; n_sys_max = n1_int * n2_int;
    if (row >= n_row_max || sys >= n_sys_max) return;
    k = row + kH; i = (sys / n2_int) + kH; j = (sys % n2_int) + kH;
  }
  dst[idx3(i, j, k, n2_tot, n3_tot)] = D_sol[row * n_sys_max + sys];
}

// ---------------------------------------------------------------------------
//  GPU kernel: add increment delta to T (interior cells only — halo undefined)
// ---------------------------------------------------------------------------
__global__ void k_add_increment(double* T, const double* delta,
                                 int n1i, int n2i, int n3i,
                                 int n2t, int n3t) {
  using namespace mpmstd::equation::scalar::kernels::stencil;
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int kk = blockIdx.z * blockDim.z + threadIdx.z;
  if (ii >= n1i || jj >= n2i || kk >= n3i) return;
  const int c = idx3(ii + kH, jj + kH, kk + kH, n2t, n3t);
  T[c] += delta[c];
}

// ---------------------------------------------------------------------------
//  GPU kernel: copy interior values from src to dst (same shape)
// ---------------------------------------------------------------------------
__global__ void k_copy_field(double* dst, const double* src, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) dst[idx] = src[idx];
}

// ---------------------------------------------------------------------------
//  GPU kernel: replace D2H + modify_tdma_row_cpu + H2D for non-periodic dirs.
//  Operates on device band arrays [n_row × n_sys] (row-major).
//  lkind/ukind: 0=ZeroGhost (zero off-diag only), 1=Antisymmetric (B-=off),
//               2=Neumann (B+=off).
// ---------------------------------------------------------------------------
__global__ void k_modify_tdma_row_gpu(double* A, double* B, double* C,
                                       int n_sys, int n_row,
                                       bool owns_lower, bool owns_upper,
                                       int lkind, int ukind) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_sys) return;
    if (owns_lower) {
        const int p = s;
        if (lkind == 1) B[p] -= A[p];
        else if (lkind == 2) B[p] += A[p];
        A[p] = 0.0;
    }
    if (owns_upper) {
        const int p = (n_row - 1) * n_sys + s;
        if (ukind == 1) B[p] -= C[p];
        else if (ukind == 2) B[p] += C[p];
        C[p] = 0.0;
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  Helper: map FaceBc to integer kind for k_modify_tdma_row_gpu
// ---------------------------------------------------------------------------
static int modify_ghost_kind(const mpmstd::boundary::FaceBc& face) {
    if (face.kind == mpmstd::boundary::BcKind::Neumann) return 2;
    if (face.ghost_policy == mpmstd::boundary::GhostPolicy::Antisymmetric) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
//  Helper: get PaScaLTDMACudaBackend from registry (CUDA builds only)
// ---------------------------------------------------------------------------
static mpmstd::linear_solver::tdma::PaScaLTDMACudaBackend&
cuda_tdma(mpmstd::linear_solver::tdma::TdmaRegistry& tdma,
          mpmstd::Direction d) {
  return static_cast<mpmstd::linear_solver::tdma::PaScaLTDMACudaBackend&>(
      tdma.get(d));
}

#endif // MPMSTD_BACKEND_CUDA

// ============================================================================
//  Public GPU functions (assemble + solve + update for scalar, variable diff)
// ============================================================================

namespace mpmstd::equation {

// ###########################################################################
// ## OB CONSTANT-DIFFUSIVITY SCALAR — DEAD FOR DHVC                         ##
// ## assemble_scalar_const_diff_gpu / solve_scalar_gpu use the OB energy    ##
// ## scheme. DHVC (dhvc_gpu) routes through assemble_scalar_var_diff_gpu +  ##
// ## solve_scalar_var_gpu with κ≡1, dκ≡0 (the NOB-faithful 0.5-skew semi-   ##
// ## implicit + −M·Tⁿ residual scheme). These OB scalar fns are LIVE ONLY   ##
// ## for the rbc_gpu_ob binary. DO NOT call from DHVC.                      ##
// ###########################################################################
void assemble_scalar_const_diff_gpu(const core::Domain& domain, core::GpuFields& fields,
                                     core::GpuScalarSystem& sys, real_t dt) {
#ifdef MPMSTD_BACKEND_CUDA
  const auto nt = domain.sub.n_total();
  const std::size_t n_field = (std::size_t)nt[0] * nt[1] * nt[2];
  const double Ct = static_cast<double>(fields.constant(core::Const::alpha_T));
  ensure_scalar_device_bufs(n_field, (std::size_t)domain.sub.n_interior()[0] *
                                     domain.sub.n_interior()[1] * domain.sub.n_interior()[2]);
  ensure_ob_property_bufs(n_field, Ct);

  const auto* dg = static_cast<const core::gpu::GpuGridArrays*>(domain.gpu_grid);
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
  const auto ni = domain.sub.n_interior();
  const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
  const dim3 blk(8, 8, 8);
  const dim3 grd((n1i + blk.x-1)/blk.x, (n2i + blk.y-1)/blk.y, (n3i + blk.z-1)/blk.z);
  k_energy_residual_var<<<grd, blk>>>(
      d_rhs_, fields[core::Var::T].data(),
      fields[core::Var::U].data(), fields[core::Var::V].data(), fields[core::Var::W].data(),
      d_ob_kappa_, d_ob_dkappa_, d_ob_inv_rhocp_, d_ob_dinv_rcp_,
      dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
      n1i, n2i, n3i, n1t, n2t, n3t, Ct);
  CUDA_CHK(cudaDeviceSynchronize());
#endif
}

void solve_scalar_gpu(const core::Domain& domain, const core::BoundaryCondition& bc,
                       core::GpuFields& fields, core::GpuScalarSystem& sys, real_t dt) {
#ifdef MPMSTD_BACKEND_CUDA
  const auto nt = domain.sub.n_total();
  const std::size_t n_field = (std::size_t)nt[0] * nt[1] * nt[2];
  const double Ct = static_cast<double>(fields.constant(core::Const::alpha_T));
  ensure_ob_property_bufs(n_field, Ct);

  const auto* dg = static_cast<const core::gpu::GpuGridArrays*>(domain.gpu_grid);
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
  const auto ni = domain.sub.n_interior();
  const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
  const std::size_t n_band = (std::size_t)n1i * n2i * n3i;
  ensure_scalar_device_bufs(n_field, n_band);

  double* in   = d_rhs_;
  double* outb = d_stage_;
  bool    first_sweep = true;

  for (Direction dir : bc.topology.sweep_order()) {
    int n_row, n_sys;
    switch (dir) {
      case Direction::X: n_row = n1i; n_sys = n2i * n3i; break;
      case Direction::Y: n_row = n2i; n_sys = n1i * n3i; break;
      default:           n_row = n3i; n_sys = n1i * n2i; break;
    }
    const bool is_stage1 = first_sweep;
    first_sweep = false;

    // sys in x-thread → coalesced writes to band arrays
    const dim3 blk(32, 16);
    const dim3 grd((n_sys + blk.x-1)/blk.x, (n_row + blk.y-1)/blk.y);
    k_build_adi_var<<<grd, blk>>>(
        dir, d_A_, d_B_, d_C_, d_D_, in, fields[core::Var::T].data(),
        fields[core::Var::U].data(), fields[core::Var::V].data(), fields[core::Var::W].data(),
        d_ob_kappa_, d_ob_dkappa_, d_ob_inv_rhocp_, d_ob_dinv_rcp_,
        dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
        n1i, n2i, n3i, n1t, n2t, n3t,
        Ct, static_cast<double>(dt), is_stage1);

    if (!bc.topology.is_periodic(dir)) {
      // GPU kernel replaces D2H + modify_tdma_row_cpu + H2D (eliminates ~138 MB pageable transfer)
      const auto& axis_dir = domain.sub.topology().axis(dir);
      const bool owns_lower = (axis_dir.west_rank == MPI_PROC_NULL);
      const bool owns_upper = (axis_dir.east_rank == MPI_PROC_NULL);
      const int lkind = modify_ghost_kind(bc.T.face(dir, mpmstd::Side::Minus));
      const int ukind = modify_ghost_kind(bc.T.face(dir, mpmstd::Side::Plus));
      {
        const int blk1d = 256;
        const int grd1d = (n_sys + blk1d - 1) / blk1d;
        k_modify_tdma_row_gpu<<<grd1d, blk1d>>>(d_A_, d_B_, d_C_, n_sys, n_row,
                                                  owns_lower, owns_upper, lkind, ukind);
      }
      cuda_tdma(domain.tdma, dir).solve_many_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);
    } else {
      cuda_tdma(domain.tdma, dir).solve_many_cyclic_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);
    }

    const dim3 blk2(32, 16);
    const dim3 grd2((n_sys + blk2.x-1)/blk2.x, (n_row + blk2.y-1)/blk2.y);
    k_scatter_from_tdma<<<grd2, blk2>>>(dir, outb, d_D_,
                                         n1i, n2i, n3i, n1t, n2t, n3t);
    std::swap(in, outb);
  }

  {
    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i + blk.x-1)/blk.x, (n2i + blk.y-1)/blk.y, (n3i + blk.z-1)/blk.z);
    k_add_increment<<<grd, blk>>>(fields[core::Var::T].data(), in,
                                   n1i, n2i, n3i, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());
  }
#endif
}

void assemble_scalar_var_diff_gpu(
    const core::Domain& domain, core::GpuFields& fields,
    const core::GpuField& kappa,   const core::GpuField& dkappa,
    const core::GpuField& inv_rcp, const core::GpuField& dinv_rcp,
    core::GpuScalarSystem& /*sys*/, real_t /*dt*/) {
#ifdef MPMSTD_BACKEND_CUDA
  const auto nt = domain.sub.n_total();
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
  const auto ni = domain.sub.n_interior();
  const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
  const std::size_t n_field = (std::size_t)n1t * n2t * n3t;
  const std::size_t n_band  = (std::size_t)n1i * n2i * n3i;
  ensure_scalar_device_bufs(n_field, n_band);

  const auto* dg = static_cast<const core::gpu::GpuGridArrays*>(domain.gpu_grid);
  const real_t Ct = fields.constant(core::Const::alpha_T);
  const double* d_T  = fields[core::Var::T].data();
  const double* d_U  = fields[core::Var::U].data();
  const double* d_V  = fields[core::Var::V].data();
  const double* d_W  = fields[core::Var::W].data();
  const double* d_kp = kappa.data(), *d_dkp = dkappa.data();
  const double* d_ir = inv_rcp.data(), *d_dir = dinv_rcp.data();

  // Explicit residual → d_rhs_
  const dim3 blk(8, 8, 8);
  const dim3 grd((n1i + blk.x-1)/blk.x, (n2i + blk.y-1)/blk.y, (n3i + blk.z-1)/blk.z);
  k_energy_residual_var<<<grd, blk>>>(
      d_rhs_, d_T, d_U, d_V, d_W, d_kp, d_dkp, d_ir, d_dir,
      dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
      n1i, n2i, n3i, n1t, n2t, n3t, static_cast<double>(Ct));
  CUDA_CHK(cudaDeviceSynchronize());

  // d_rhs_ now holds the explicit residual; solve_scalar_var_gpu reads it directly.
#endif
}

void solve_scalar_var_gpu(
    const core::Domain& domain, const core::BoundaryCondition& bc,
    core::GpuFields& fields,
    const core::GpuField& kappa,   const core::GpuField& dkappa,
    const core::GpuField& inv_rcp, const core::GpuField& dinv_rcp,
    core::GpuScalarSystem& sys, real_t dt) {
#ifdef MPMSTD_BACKEND_CUDA
  const auto nt = domain.sub.n_total();
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
  const auto ni = domain.sub.n_interior();
  const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
  const std::size_t n_field = (std::size_t)n1t * n2t * n3t;

  const std::size_t n_band = (std::size_t)n1i * n2i * n3i;
  ensure_scalar_device_bufs(n_field, n_band);

  const auto* dg = static_cast<const core::gpu::GpuGridArrays*>(domain.gpu_grid);
  const real_t Ct = fields.constant(core::Const::alpha_T);
  const double* d_T  = fields[core::Var::T].data();
  const double* d_U  = fields[core::Var::U].data();
  const double* d_V  = fields[core::Var::V].data();
  const double* d_W  = fields[core::Var::W].data();
  const double* d_kp = kappa.data(), *d_dkp = dkappa.data();
  const double* d_ir = inv_rcp.data(), *d_dir = dinv_rcp.data();

  // d_rhs_ holds the explicit residual written by assemble_scalar_var_diff_gpu.
  using hrc = std::chrono::high_resolution_clock;
  using ms_t = std::chrono::duration<double, std::milli>;
  static int s_sca_n = 0;
  ++s_sca_n;
  // per-sweep: [0]=X [1]=Y [2]=Z
  double tBld[3]={}, tTdma[3]={}, tSct[3]={};

  double* in   = d_rhs_;
  double* outb = d_stage_;
  bool    first_sweep = true;
  int     sw = 0;

  for (Direction dir : bc.topology.sweep_order()) {
    int n_row, n_sys;
    switch (dir) {
      case Direction::X: n_row = n1i; n_sys = n2i * n3i; break;
      case Direction::Y: n_row = n2i; n_sys = n1i * n3i; break;
      default:           n_row = n3i; n_sys = n1i * n2i; break;
    }
    const bool is_stage1 = first_sweep;
    first_sweep = false;

    auto Ta = hrc::now();
    const dim3 blk(32, 16);
    const dim3 grd((n_sys + blk.x-1)/blk.x, (n_row + blk.y-1)/blk.y);
    k_build_adi_var<<<grd, blk>>>(
        dir, d_A_, d_B_, d_C_, d_D_, in, d_T,
        d_U, d_V, d_W, d_kp, d_dkp, d_ir, d_dir,
        dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
        n1i, n2i, n3i, n1t, n2t, n3t,
        static_cast<double>(Ct), static_cast<double>(dt), is_stage1);
    CUDA_CHK(cudaDeviceSynchronize());
    tBld[sw] = ms_t(hrc::now()-Ta).count();

    if (!bc.topology.is_periodic(dir)) {
      // GPU kernel replaces D2H + modify_tdma_row_cpu + H2D (eliminates ~86 ms pageable transfer)
      const auto& axis_dir = domain.sub.topology().axis(dir);
      const bool owns_lower = (axis_dir.west_rank == MPI_PROC_NULL);
      const bool owns_upper = (axis_dir.east_rank == MPI_PROC_NULL);
      const int lkind = modify_ghost_kind(bc.T.face(dir, mpmstd::Side::Minus));
      const int ukind = modify_ghost_kind(bc.T.face(dir, mpmstd::Side::Plus));
      auto Tt0 = hrc::now();
      {
        const int blk1d = 256;
        const int grd1d = (n_sys + blk1d - 1) / blk1d;
        k_modify_tdma_row_gpu<<<grd1d, blk1d>>>(d_A_, d_B_, d_C_, n_sys, n_row,
                                                  owns_lower, owns_upper, lkind, ukind);
      }
      cuda_tdma(domain.tdma, dir).solve_many_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);
      CUDA_CHK(cudaDeviceSynchronize());
      tTdma[sw] = ms_t(hrc::now()-Tt0).count();
    } else {
      auto Tt0 = hrc::now();
      cuda_tdma(domain.tdma, dir).solve_many_cyclic_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);
      CUDA_CHK(cudaDeviceSynchronize());
      tTdma[sw] = ms_t(hrc::now()-Tt0).count();
    }

    auto Ts0 = hrc::now();
    const dim3 blk2(32, 16);
    const dim3 grd2((n_sys + blk2.x-1)/blk2.x, (n_row + blk2.y-1)/blk2.y);
    k_scatter_from_tdma<<<grd2, blk2>>>(dir, outb, d_D_,
                                         n1i, n2i, n3i, n1t, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());
    tSct[sw] = ms_t(hrc::now()-Ts0).count();
    std::swap(in, outb);
    ++sw;
  }

  if (s_sca_n <= 20) {
    std::printf("[SCA #%d] X(bld=%.2f tdma=%.2f sct=%.2f) Y(bld=%.2f tdma=%.2f sct=%.2f) Z(bld=%.2f tdma=%.2f sct=%.2f) ms\n",
                s_sca_n,
                tBld[0], tTdma[0], tSct[0],
                tBld[1], tTdma[1], tSct[1],
                tBld[2], tTdma[2], tSct[2]);
  }

  // in now points to the final increment δT; add to T (interior only — halo cells not valid)
  {
    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i + blk.x-1)/blk.x, (n2i + blk.y-1)/blk.y, (n3i + blk.z-1)/blk.z);
    k_add_increment<<<grd, blk>>>(fields[core::Var::T].data(), in,
                                   n1i, n2i, n3i, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());
  }

  // Copy updated T back to sys state (host rhs.data() not needed; just sync halo via MPI)
  // Halo sync is done by caller via sync_field_cpu → but we're using GpuField here.
  // TODO: add sync_field_gpu.
#endif
}

} // namespace mpmstd::equation
