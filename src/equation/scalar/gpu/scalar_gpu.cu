#include "equation/scalar/solve.hpp"
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

// OB (const-diff) property arrays: KPP=Ct, dKPP=0, invRhoCp=1, dinvRhoCp=0
static double* d_ob_KPP_     = nullptr;  // filled with alpha_T
static double* d_ob_dKPP_    = nullptr;  // filled with 0
static double* d_ob_invRhoCp_ = nullptr;  // filled with 1
static double* d_ob_dinvRhoCp_  = nullptr;  // filled with 0
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
    CUDA_CHK(cudaFree(d_ob_KPP_));     CUDA_CHK(cudaMalloc(&d_ob_KPP_,     n_field * sizeof(double)));
    CUDA_CHK(cudaFree(d_ob_dKPP_));    CUDA_CHK(cudaMalloc(&d_ob_dKPP_,    n_field * sizeof(double)));
    CUDA_CHK(cudaFree(d_ob_invRhoCp_)); CUDA_CHK(cudaMalloc(&d_ob_invRhoCp_, n_field * sizeof(double)));
    CUDA_CHK(cudaFree(d_ob_dinvRhoCp_));  CUDA_CHK(cudaMalloc(&d_ob_dinvRhoCp_,  n_field * sizeof(double)));
    d_ob_cap_ = n_field;
    d_ob_Ct_  = -1.0;  // force refill
  }
  if (Ct != d_ob_Ct_) {
    const int blk = 256;
    const int grd = (int)((n_field + blk - 1) / blk);
    k_fill_const<<<grd, blk>>>(d_ob_KPP_,     1.0, n_field);  // κ̂=1 (OB: κ/κ_ref=1); Ct=α_T is in invRhoCpCt_half
    k_fill_const<<<grd, blk>>>(d_ob_dKPP_,    0.0, n_field);
    k_fill_const<<<grd, blk>>>(d_ob_invRhoCp_, 1.0, n_field);
    k_fill_const<<<grd, blk>>>(d_ob_dinvRhoCp_,  0.0, n_field);
    CUDA_CHK(cudaDeviceSynchronize());
    d_ob_Ct_ = Ct;
  }
}

// ---------------------------------------------------------------------------
//  GPU kernel: build ADI bands for one sweep direction
//  Grid layout: (n_row × n_sys) threads — each emits one TDMA entry.
// ---------------------------------------------------------------------------
__global__ void k_build_A_rhs_nob(
    mpmstd::Direction d,
    double* A, double* B, double* C, double* DD,
    const double* src, const double* T,
    const double* U, const double* V, const double* W,
    const double* KPP, const double* dKPP,
    const double* invRhoCp, const double* dinvRhoCp,
    const double* dx1, const double* dmx1,
    const double* dx2, const double* dmx2,
    const double* dx3, const double* dmx3,
    int n1_int, int n2_int, int n3_int,
    int n1_tot, int n2_tot, int n3_tot,
    double Ct, double dt, bool is_stage1,
    bool z_owns_lower, bool z_owns_upper) {
  using namespace mpmstd::equation::scalar::kernels;   // idx3, kH 등 → equation/scalar/scalar_var_stencil.hpp

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

  // PaScaL_TCS jem/jep mask: zero dkPP at wall-adjacent Z faces (all directions).
  // Gated on physical-wall ownership: without the guard, np3>1 fired these
  // masks at interior rank seams (mid-height for np3=2, exactly the theta_c
  // plane) — audit D5, Report/RBC.
  const int kk = k - kH;
  const bool z_wall_lo = z_owns_lower && (kk == 0);
  const bool z_wall_hi = z_owns_upper && (kk == n3_int - 1);

  double am, ac, ap, Dv;
  // → equation/scalar/scalar_var_stencil.hpp : build_A_rhs_cell — 셀 하나의
  //   A 밴드 + RHS 수치 전체가 여기 있음 (CPU 경로와 공유하는 단일 구현)
  build_A_rhs_cell(d, i, j, k, n1_tot, n2_tot, n3_tot,
                 src, T, U, V, W, KPP, dKPP, invRhoCp, dinvRhoCp,
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
  using namespace mpmstd::equation::scalar::kernels;   // idx3, kH 등 → equation/scalar/scalar_var_stencil.hpp
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
  using namespace mpmstd::equation::scalar::kernels;   // idx3, kH 등 → equation/scalar/scalar_var_stencil.hpp
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
//  tdma.get(d) → linear_solver/tdma/tdma_registry.hpp
//  PaScaLTDMACudaBackend(solve_many_device 등) → linear_solver/tdma/pascal_tdma_cuda_backend.hpp
// ---------------------------------------------------------------------------
static mpmstd::linear_solver::tdma::PaScaLTDMACudaBackend&
cuda_tdma(mpmstd::linear_solver::tdma::TdmaRegistry& tdma,
          mpmstd::Direction d) {
  return tdma.get(d);
}

#endif // MPMSTD_BACKEND_CUDA

// ============================================================================
//  Public GPU functions — solve only: the A bands AND the delta-form RHS are
//  built together per sweep by k_build_A_rhs_nob (fused build_A_RHS; the
//  stage-1 explicit residual R is computed inline, no separate coef pass).
// ============================================================================

namespace mpmstd::equation::scalar {

// ###########################################################################
// ## OB CONSTANT-DIFFUSIVITY SCALAR — DEAD FOR DHVC                        ##
// ## solve_scalar_constant_property_gpu = the nob kernels driven with unit properties     ##
// ## (κ≡1, dκ≡0). DHVC (dhvc_gpu) routes through solve_scalar_variable_property_gpu     ##
// ## with κ≡1, dκ≡0 (the NOB-faithful 0.5-skew semi-implicit + −M·Tⁿ      ##
// ## residual scheme). This OB wrapper is LIVE ONLY for the rbc_gpu_ob    ##
// ## binary. DO NOT call from DHVC.                                       ##
// ###########################################################################
// 선언부: equation/scalar/solve.hpp
void solve_scalar_constant_property_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                       core::GpuFields& fields, equation::scalar::GpuLinearSystem& sys, real_t dt) {
#ifdef MPMSTD_BACKEND_CUDA
  const auto nt = domain.sub.n_total();
  const std::size_t n_field = (std::size_t)nt[0] * nt[1] * nt[2];
  const double Ct = static_cast<double>(fields.constant(core::Const::alpha_T));
  ensure_ob_property_bufs(n_field, Ct);

  const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);   // → core/gpu/gpu_grid.hpp
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
  const auto ni = domain.sub.n_interior();
  const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
  const std::size_t n_band = (std::size_t)n1i * n2i * n3i;
  ensure_scalar_device_bufs(n_field, n_band);

  // Z physical-wall ownership for the jem/jep dkPP masks (audit D5): the mask
  // must never fire at interior np3 rank seams.
  const auto& z_axis_own = domain.sub.topology().axis(Direction::Z);
  const bool z_wall_owns_lower = !bc.topology.is_periodic(Direction::Z)
      && (z_axis_own.west_rank == MPI_PROC_NULL);
  const bool z_wall_owns_upper = !bc.topology.is_periodic(Direction::Z)
      && (z_axis_own.east_rank == MPI_PROC_NULL);

  // ping-pong scratch (stage 1 ignores `in` — R computed inline in the kernel)
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
    k_build_A_rhs_nob<<<grd, blk>>>(
        dir, d_A_, d_B_, d_C_, d_D_, in, fields[core::Var::T].data(),
        fields[core::Var::U].data(), fields[core::Var::V].data(), fields[core::Var::W].data(),
        d_ob_KPP_, d_ob_dKPP_, d_ob_invRhoCp_, d_ob_dinvRhoCp_,
        dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
        n1i, n2i, n3i, n1t, n2t, n3t,
        Ct, static_cast<double>(dt), is_stage1,
        z_wall_owns_lower, z_wall_owns_upper);

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
      cuda_tdma(domain.tdma, dir).solve_many_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);   // → pascal_tdma_cuda_backend.hpp
    } else {
      cuda_tdma(domain.tdma, dir).solve_many_cyclic_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);   // → pascal_tdma_cuda_backend.hpp
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

// NOB 에너지 solve — TCS mpi_thermal_solver 전체에 대응하는 단일 진입점.
// 3-factor 델타-폼 ADI: sweep(x→y→z)마다 융합 커널 k_build_A_rhs_nob가
// A 밴드 + RHS(D)를 동시에 조립(stage-1은 명시 잔차 R을 인라인 계산)하고,
// 분산 TDMA로 δ를 푼 뒤 다음 sweep으로 넘긴다. 마지막에 T += δT.
//   KPP/dKPP (κ̂, dκ̂/dΔθ)          — TCS KPP/dKPP
//   invRhoCp/dinvRhoCp (1/(ρ̂Ĉp), 도함수) — TCS invRhoCp/dinvRhoCp
//   sys : CPU 경로용 밴드 저장소 — GPU 경로는 디바이스 캐시(d_A_ 등)를 쓰므로 미사용
// 선언부: equation/scalar/solve.hpp
void solve_scalar_variable_property_gpu(
    const core::Domain& domain, const boundary::BoundaryCondition& bc,
    core::GpuFields& fields,
    const core::GpuField& KPP,   const core::GpuField& dKPP,
    const core::GpuField& invRhoCp, const core::GpuField& dinvRhoCp,
    equation::scalar::GpuLinearSystem& sys, real_t dt) {
#ifdef MPMSTD_BACKEND_CUDA
  // 로컬 배열 크기: n?t = 할로 포함 전체(total), n?i = 내부(interior)만
  const auto nt = domain.sub.n_total();
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
  const auto ni = domain.sub.n_interior();
  const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
  const std::size_t n_field = (std::size_t)n1t * n2t * n3t;   // 필드 배열 총 길이

  const std::size_t n_band = (std::size_t)n1i * n2i * n3i;    // 내부 셀 수 = 밴드 길이
  ensure_scalar_device_bufs(n_field, n_band);   // d_A_~d_D_, d_rhs_/d_stage_ lazy 할당

  const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);  // 격자 메트릭(dx,dmx) 디바이스 복사본 → core/gpu/gpu_grid.hpp
  const real_t Ct = fields.constant(core::Const::alpha_T);    // Ct = 1/√(Ra·Pr) — TCS의 Ct
  const double* d_T  = fields[core::Var::T].data();           // θⁿ (디바이스 포인터)
  const double* d_U  = fields[core::Var::U].data();           // 대류 속도 U,V,W (MAC 면 값)
  const double* d_V  = fields[core::Var::V].data();
  const double* d_W  = fields[core::Var::W].data();
  const double* d_KPP = KPP.data(), *d_dKPP = dKPP.data();      // κ̂, dκ̂/dΔθ
  const double* d_invRhoCp = invRhoCp.data(), *d_dinvRhoCp = dinvRhoCp.data();  // 1/(ρ̂Ĉp), 그 도함수

  // sweep별 구간 타이밍(처음 20스텝만 출력): [0]=X [1]=Y [2]=Z
  using hrc = std::chrono::high_resolution_clock;
  using ms_t = std::chrono::duration<double, std::milli>;
  static int s_sca_n = 0;
  ++s_sca_n;
  double tBld[3]={}, tTdma[3]={}, tSct[3]={};

  // z-벽 dkPP 마스크(jem/jep)는 물리적 벽을 소유한 랭크에서만 발동해야 함 —
  // np3>1일 때 내부 랭크 경계(seam)에서 발동하면 안 됨 (audit D5).
  // west/east_rank == MPI_PROC_NULL ⇔ 이 랭크가 그쪽 물리 벽을 소유.
  const auto& z_axis_own = domain.sub.topology().axis(Direction::Z);
  const bool z_wall_owns_lower = !bc.topology.is_periodic(Direction::Z) && (z_axis_own.west_rank == MPI_PROC_NULL);
  const bool z_wall_owns_upper = !bc.topology.is_periodic(Direction::Z) && (z_axis_own.east_rank == MPI_PROC_NULL);

  // sweep 간 증분을 주고받는 핑퐁 버퍼: in = 이번 sweep의 입력(직전 sweep의 δ),
  // outb = 이번 sweep의 출력. stage-1은 `in`을 읽지 않음(융합 커널이 R을 인라인 계산).
  double* in   = d_rhs_;
  double* outb = d_stage_;
  bool    first_sweep = true;
  int     sw = 0;

  // ── 3-sweep ADI 루프: (I+dt·M_x)(I+dt·M_y)(I+dt·M_z)·δθ = dt·RHS ──────────
  // sweep_order()는 주기 방향(x,y) 먼저, 벽 방향(z)을 마지막에 배치.
  for (Direction dir : bc.topology.sweep_order()) {
    // TDMA 배치 레이아웃: n_row = sweep 축 길이(삼중대각 행 수),
    // n_sys = 나머지 두 축의 곱(독립 시스템 개수)
    int n_row, n_sys;
    switch (dir) {
      case Direction::X: n_row = n1i; n_sys = n2i * n3i; break;
      case Direction::Y: n_row = n2i; n_sys = n1i * n3i; break;
      default:           n_row = n3i; n_sys = n1i * n2i; break;
    }
    const bool is_stage1 = first_sweep;   // 첫 sweep만 RHS(R − M·θⁿ) 팩 — build_A_rhs_cell 참조
    first_sweep = false;

    // [1] A 밴드 + RHS 동시 조립 (융합 build_A_RHS): 스레드 (sys,row) 하나가
    //     셀 하나의 (A,B,C,D)[row·n_sys+sys]를 씀 (sys가 x-스레드 → 코얼레스드)
    auto Ta = hrc::now();
    const dim3 blk(32, 16);
    const dim3 grd((n_sys + blk.x-1)/blk.x, (n_row + blk.y-1)/blk.y);
    k_build_A_rhs_nob<<<grd, blk>>>(
        dir, d_A_, d_B_, d_C_, d_D_, in, d_T,
        d_U, d_V, d_W, d_KPP, d_dKPP, d_invRhoCp, d_dinvRhoCp,
        dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
        n1i, n2i, n3i, n1t, n2t, n3t,
        static_cast<double>(Ct), static_cast<double>(dt), is_stage1,
        z_wall_owns_lower, z_wall_owns_upper);
    CUDA_CHK(cudaDeviceSynchronize());
    tBld[sw] = ms_t(hrc::now()-Ta).count();

    if (!bc.topology.is_periodic(dir)) {
      // [2a] 벽 방향(z): 벽 BC를 행렬에 접고(fold) 일반 TDMA로 풀이.
      //      k_modify_tdma_row_gpu가 CPU 왕복(D2H + modify_tdma_row_cpu + H2D,
      //      ~86 ms pageable 전송) 없이 디바이스에서 첫/끝 행을 수정.
      const auto& axis_dir = domain.sub.topology().axis(dir);
      const bool owns_lower = (axis_dir.west_rank == MPI_PROC_NULL);   // 물리 벽 소유 랭크만
      const bool owns_upper = (axis_dir.east_rank == MPI_PROC_NULL);
      const int lkind = modify_ghost_kind(bc.T.face(dir, mpmstd::Side::Minus));  // 고스트 정책 → fold 방식
      const int ukind = modify_ghost_kind(bc.T.face(dir, mpmstd::Side::Plus));
      auto Tt0 = hrc::now();
      {
        const int blk1d = 256;
        const int grd1d = (n_sys + blk1d - 1) / blk1d;
        k_modify_tdma_row_gpu<<<grd1d, blk1d>>>(d_A_, d_B_, d_C_, n_sys, n_row,
                                                  owns_lower, owns_upper, lkind, ukind);
      }
      // 분산 PaScaL_TDMA (디바이스 상주, CUDA-aware MPI) — 여기가 Ax=b 풀이
      cuda_tdma(domain.tdma, dir).solve_many_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);   // → pascal_tdma_cuda_backend.hpp
      CUDA_CHK(cudaDeviceSynchronize());
      tTdma[sw] = ms_t(hrc::now()-Tt0).count();
    } else {
      // [2b] 주기 방향(x,y): 순환(cyclic) TDMA
      auto Tt0 = hrc::now();
      cuda_tdma(domain.tdma, dir).solve_many_cyclic_device(d_A_, d_B_, d_C_, d_D_, n_sys, n_row);   // → pascal_tdma_cuda_backend.hpp
      CUDA_CHK(cudaDeviceSynchronize());
      tTdma[sw] = ms_t(hrc::now()-Tt0).count();
    }

    // [3] 해(밴드 레이아웃의 D)를 3D 필드 배열로 산란(scatter) → 다음 sweep 입력
    auto Ts0 = hrc::now();
    const dim3 blk2(32, 16);
    const dim3 grd2((n_sys + blk2.x-1)/blk2.x, (n_row + blk2.y-1)/blk2.y);
    k_scatter_from_tdma<<<grd2, blk2>>>(dir, outb, d_D_,
                                         n1i, n2i, n3i, n1t, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());
    tSct[sw] = ms_t(hrc::now()-Ts0).count();
    std::swap(in, outb);   // 이번 출력이 다음 sweep의 입력이 되도록 핑퐁 교대
    ++sw;
  }

  if (s_sca_n <= 20) {
    std::printf("[SCA #%d] X(bld=%.2f tdma=%.2f sct=%.2f) Y(bld=%.2f tdma=%.2f sct=%.2f) Z(bld=%.2f tdma=%.2f sct=%.2f) ms\n",
                s_sca_n,
                tBld[0], tTdma[0], tSct[0],
                tBld[1], tTdma[1], tSct[1],
                tBld[2], tTdma[2], tSct[2]);
  }

  // [4] 3 sweep을 통과한 `in` = 최종 증분 δθ → Tⁿ⁺¹ = Tⁿ + δθ
  //     (내부 셀만 — δ의 할로는 무효; T 할로 교환/BC 재적용은 호출자의 sync_field_gpu가 담당)
  {
    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i + blk.x-1)/blk.x, (n2i + blk.y-1)/blk.y, (n3i + blk.z-1)/blk.z);
    k_add_increment<<<grd, blk>>>(fields[core::Var::T].data(), in,
                                   n1i, n2i, n3i, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());
  }
#endif
}

} // namespace mpmstd::equation::scalar
