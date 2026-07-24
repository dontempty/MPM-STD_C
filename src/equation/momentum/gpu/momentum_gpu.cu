// =============================================================================
//  momentum_gpu.cu  —  GPU implementation of MPM-STD BW-ADI momentum solver
//
//  Ports the five stub functions in the original skeleton to full CUDA kernels.
//  Pattern follows scalar_gpu.cu exactly: static device buffers, ensure_bufs(),
//  per-direction ADI loop, host round-trip for non-periodic BC modification.
//
//  OB  (const-visc):  assemble_momentum_const_visc_gpu + solve_momentum_gpu
//  NOB (var-visc):    assemble_momentum_var_visc_gpu   + solve_momentum_var_gpu
//  Both: update_velocity_gpu
// =============================================================================

#include "equation/momentum/assemble.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/momentum/kernels.hpp"
#include "core/boundary_ops.hpp"
#include "linear_solver/tdma/pascal_tdma_cuda_backend.hpp"
#include "linear_solver/tdma/tdma_registry.hpp"
#include "core/gpu/gpu_grid.hpp"

#ifdef MPMSTD_BACKEND_CUDA
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace {

// ---------------------------------------------------------------------------
//  Error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHK(expr)  do {                                              \
  cudaError_t _e = (expr);                                                \
  if (_e != cudaSuccess) {                                                \
    std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                 \
                 cudaGetErrorName(_e), __FILE__, __LINE__,                \
                 cudaGetErrorString(_e));                                  \
    std::abort();                                                          \
  }                                                                        \
} while(0)

// ---------------------------------------------------------------------------
//  Device buffer cache  (one set per process)
//  RHS for each component is staged in mom.dU/dV/dW.data() directly,
//  matching the CPU convention (rhs_u/v/w live in MomentumSystem).
// ---------------------------------------------------------------------------
static double* d_m_stage_ = nullptr;   // ping-pong scratch for one component
static double* d_m_A_     = nullptr;
static double* d_m_B_     = nullptr;
static double* d_m_C_     = nullptr;
static double* d_m_D_     = nullptr;
// OB constant-property buffers (mu = nu, inv_rho = 1) — same shape as field
static double* d_m_ob_mu_   = nullptr;
static double* d_m_ob_irho_ = nullptr;
static std::size_t d_m_nf_  = 0;   // current field capacity (elements)
static std::size_t d_m_nb_  = 0;   // current band capacity
static std::size_t d_m_ob_cap_ = 0;
static double      d_m_ob_nu_  = -1.0;  // sentinel: re-fill when nu changes

// ---------------------------------------------------------------------------
//  Halo-width constant for kernels
// ---------------------------------------------------------------------------
static constexpr int kH = 1;


// ---------------------------------------------------------------------------
//  Utilities used inside kernels
// ---------------------------------------------------------------------------
__device__ static int idx3m(int i, int j, int k, int n2, int n3)
{
    return (i * n2 + j) * n3 + k;
}

// Volume-weighted 2-pt face interp (mirrors CPU mface / MPMSTD_HD mface)
__device__ static double mfce(double dxA, double fA, double dxB, double fB, double dmx)
{
    return 0.5 * (dxA * fA + dxB * fB) / dmx;
}

// ---------------------------------------------------------------------------
//  Generic fill kernel
// ---------------------------------------------------------------------------
__global__ void k_m_fill(double* buf, double val, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] = val;
}

// ---------------------------------------------------------------------------
//  ensure_bufs(): grow device allocations if needed
// ---------------------------------------------------------------------------
void ensure_mom_bufs(std::size_t nf, std::size_t nb)
{
    if (nf > d_m_nf_) {
        CUDA_CHK(cudaFree(d_m_stage_)); CUDA_CHK(cudaMalloc(&d_m_stage_,  nf * sizeof(double)));
        d_m_nf_ = nf;
    }
    if (nb > d_m_nb_) {
        CUDA_CHK(cudaFree(d_m_A_)); CUDA_CHK(cudaMalloc(&d_m_A_, nb * sizeof(double)));
        CUDA_CHK(cudaFree(d_m_B_)); CUDA_CHK(cudaMalloc(&d_m_B_, nb * sizeof(double)));
        CUDA_CHK(cudaFree(d_m_C_)); CUDA_CHK(cudaMalloc(&d_m_C_, nb * sizeof(double)));
        CUDA_CHK(cudaFree(d_m_D_)); CUDA_CHK(cudaMalloc(&d_m_D_, nb * sizeof(double)));
        d_m_nb_ = nb;
    }
}

void ensure_ob_property_bufs_mom(std::size_t nf, double nu)
{
    if (nf > d_m_ob_cap_) {
        CUDA_CHK(cudaFree(d_m_ob_mu_));   CUDA_CHK(cudaMalloc(&d_m_ob_mu_,   nf * sizeof(double)));
        CUDA_CHK(cudaFree(d_m_ob_irho_)); CUDA_CHK(cudaMalloc(&d_m_ob_irho_, nf * sizeof(double)));
        d_m_ob_cap_ = nf;
        d_m_ob_nu_  = -1.0;  // force refill
    }
    if (nu != d_m_ob_nu_) {
        const int blk = 256;
        const int grd = (int)((nf + blk - 1) / blk);
        k_m_fill<<<grd, blk>>>(d_m_ob_mu_,   nu,  (int)nf);
        k_m_fill<<<grd, blk>>>(d_m_ob_irho_, 1.0, (int)nf);
        CUDA_CHK(cudaDeviceSynchronize());
        d_m_ob_nu_ = nu;
    }
}

// ###########################################################################
// ## OB CONSTANT-PROPERTY MOMENTUM KERNELS — DEAD FOR DHVC                  ##
// ## k_mom_rhs_ob / k_mom_adi_ob / k_block_couple_d{U,V}_ob and their       ##
// ## wrappers (assemble_momentum_const_visc_gpu, solve_momentum_gpu) use    ##
// ## the divergent OB scheme: conservative-flux explicit convection         ##
// ## (conv_factor=1), pure Laplacian (no own×2 / no transposed cross-       ##
// ## stress), and NO −M·qⁿ delta-form residual — the root of the Report 18  ##
// ## explicit-convection dt collapse. DHVC (dhvc_gpu, apps/dhvc/main_gpu.cu)##
// ## does NOT use these: it routes through the _var (NOB-faithful) kernels  ##
// ## with unit properties (μ≡1, 1/ρ≡1). These OB kernels are LIVE ONLY for  ##
// ## the rbc_gpu_ob binary (apps/rbc/main_gpu_ob.cu). DO NOT call from DHVC. ##
// ###########################################################################
// ===========================================================================
//  KERNEL 1: OB explicit momentum RHS  (compute_mpmstd_rhs, one thread / cell)
//  rhs[c] = dt * nu * (Lx + Ly + Lz)(q)[c]
//           - dt * div(u*q)[c]   (conv_factor = 1, matching CPU assemble)
//
//  z_staggered=true for W (face-centred in z).
// ===========================================================================
__global__ void k_mom_rhs_ob(
        double*       rhs,
        const double* q,
        const double* U, const double* V, const double* W,
        const double* dx1, const double* dmx1,
        const double* dx2, const double* dmx2,
        const double* dx3, const double* dmx3,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double nu, double dt,
        bool z_staggered)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int c   = idx3m(i,   j,   k,   n2t, n3t);
    const int xm  = idx3m(i-1, j,   k,   n2t, n3t);
    const int xp  = idx3m(i+1, j,   k,   n2t, n3t);
    const int ym  = idx3m(i,   j-1, k,   n2t, n3t);
    const int yp  = idx3m(i,   j+1, k,   n2t, n3t);
    const int zm  = idx3m(i,   j,   k-1, n2t, n3t);
    const int zp  = idx3m(i,   j,   k+1, n2t, n3t);

    // Viscous Laplacian
    const double Lx = ((q[xp] - q[c]) / dmx1[i+1]
                     - (q[c]  - q[xm]) / dmx1[i  ]) / dx1[i];
    const double Ly = ((q[yp] - q[c]) / dmx2[j+1]
                     - (q[c]  - q[ym]) / dmx2[j  ]) / dx2[j];
    const double Lz = z_staggered
        ? ((q[zp] - q[c]) / dx3[k  ]
         - (q[c]  - q[zm]) / dx3[k-1]) / dmx3[k]
        : ((q[zp] - q[c]) / dmx3[k+1]
         - (q[c]  - q[zm]) / dmx3[k  ]) / dx3[k];

    // Convection: conservative flux form, full coefficient (conv_factor=1)
    const double fx_p = U[xp] * 0.5 * (q[c]  + q[xp]);
    const double fx_m = U[c ] * 0.5 * (q[xm] + q[c ]);
    const double fy_p = V[yp] * 0.5 * (q[c]  + q[yp]);
    const double fy_m = V[c ] * 0.5 * (q[ym] + q[c ]);
    const double fz_p = W[zp] * 0.5 * (q[c]  + q[zp]);
    const double fz_m = W[c ] * 0.5 * (q[zm] + q[c ]);

    const double div_uq = (fx_p - fx_m) / dx1[i]
                        + (fy_p - fy_m) / dx2[j]
                        + (fz_p - fz_m) / dx3[k];

    // conv_factor = 1.0 (as called from assemble_momentum_const_visc_cpu with conv=1)
    rhs[c] = nu * dt * (Lx + Ly + Lz) - 1.0 * dt * div_uq;
}

// ===========================================================================
//  KERNEL 2: OB ADI band assembly
//  Builds (A,B,C,D) for one sweep direction.  visc_factor = 0.5 always for
//  momentum OB (all three directions Crank-Nicolson).
//
//  Layout: p = row * n_sys + sys, row = swept-dir index, sys = remaining.
// ===========================================================================
__global__ void k_mom_adi_ob(
        mpmstd::Direction d,
        double* A, double* B, double* C, double* D,
        const double* src,
        const double* q,       // own velocity component
        const double* q_adv,   // advecting velocity along d
        const double* dx_a,    // swept-axis dx
        const double* dmx_a,   // swept-axis dmx
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double nu, double dt,
        bool z_staggered)
{
    // sys in threadIdx.x → warp has adjacent sys → coalesced writes to A/B/C/D[row*n_sys+sys]
    const int sys = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    int i, j, k, n_row_max, n_sys_max;
    if (d == mpmstd::Direction::X) {
        n_row_max = n1i; n_sys_max = n2i * n3i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        i = row + kH; j = (sys / n3i) + kH; k = (sys % n3i) + kH;
    } else if (d == mpmstd::Direction::Y) {
        n_row_max = n2i; n_sys_max = n1i * n3i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        j = row + kH; i = (sys / n3i) + kH; k = (sys % n3i) + kH;
    } else {
        n_row_max = n3i; n_sys_max = n1i * n2i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        k = row + kH; i = (sys / n2i) + kH; j = (sys % n2i) + kH;
    }
    const int p = row * n_sys_max + sys;

    // Determine the axis index 'a' for swept-axis metrics
    const int a = (d == mpmstd::Direction::X) ? i
                : (d == mpmstd::Direction::Y) ? j : k;

    const double coeff = 0.5 * dt * nu;  // visc_factor = 0.5 always for momentum OB
    const double cconv = 0.25 * dt;

    double av, bv, cv;
    double ac, bc, cc;

    if (d == mpmstd::Direction::X) {
        av = -coeff / (dx_a[a] * dmx_a[a  ]);
        cv = -coeff / (dx_a[a] * dmx_a[a+1]);
        bv = 1.0 - av - cv;
        const int gi  = idx3m(i,   j, k, n2t, n3t);
        const int gim = idx3m(i-1, j, k, n2t, n3t);
        const int gip = idx3m(i+1, j, k, n2t, n3t);
        const double u5 = 0.5 * (q_adv[gim] + q_adv[gi]);
        const double u6 = 0.5 * (q_adv[gi]  + q_adv[gip]);
        ac = -cconv * u5 / dmx_a[a  ];
        cc =  cconv * u6 / dmx_a[a+1];
        bc = -cconv * (-u6 / dmx_a[a+1] + u5 / dmx_a[a]);
    } else if (d == mpmstd::Direction::Y) {
        av = -coeff / (dx_a[a] * dmx_a[a  ]);
        cv = -coeff / (dx_a[a] * dmx_a[a+1]);
        bv = 1.0 - av - cv;
        const int gi  = idx3m(i, j,   k, n2t, n3t);
        const int gjm = idx3m(i, j-1, k, n2t, n3t);
        const int gjp = idx3m(i, j+1, k, n2t, n3t);
        const double v5 = 0.5 * (q_adv[gjm] + q_adv[gi]);
        const double v6 = 0.5 * (q_adv[gi]  + q_adv[gjp]);
        ac = -cconv * v5 / dmx_a[a  ];
        cc =  cconv * v6 / dmx_a[a+1];
        bc = -cconv * (-v6 / dmx_a[a+1] + v5 / dmx_a[a]);
    } else {  // Direction::Z
        if (z_staggered) {
            av = -coeff / (dmx_a[k] * dx_a[k-1]);
            cv = -coeff / (dmx_a[k] * dx_a[k  ]);
        } else {
            av = -coeff / (dx_a[k] * dmx_a[k  ]);
            cv = -coeff / (dx_a[k] * dmx_a[k+1]);
        }
        bv = 1.0 - av - cv;
        const int gi  = idx3m(i, j, k,   n2t, n3t);
        const int gkm = idx3m(i, j, k-1, n2t, n3t);
        const int gkp = idx3m(i, j, k+1, n2t, n3t);
        const double w5 = 0.5 * (q_adv[gkm] + q_adv[gi]);
        const double w6 = 0.5 * (q_adv[gi]  + q_adv[gkp]);
        if (z_staggered) {
            ac = -cconv * w5 / dx_a[k-1];
            cc =  cconv * w6 / dx_a[k  ];
            bc = -cconv * (-w6 / dx_a[k] + w5 / dx_a[k-1]);
        } else {
            ac = -cconv * w5 / dmx_a[k  ];
            cc =  cconv * w6 / dmx_a[k+1];
            bc = -cconv * (-w6 / dmx_a[k+1] + w5 / dmx_a[k]);
        }
    }

    const int gi_src = idx3m(i, j, k, n2t, n3t);
    A[p] = av + ac;
    B[p] = bv + bc;
    C[p] = cv + cc;
    D[p] = src[gi_src];
}

// ===========================================================================
//  KERNEL 3: OB block coupling  — dV -= dt * 0.25 * (dW * dV/dz)
//  Mirrors CPU block_couple_dV.
// ===========================================================================
__global__ void k_block_couple_dV_ob(
        double* dV,
        const double* dW,
        const double* V,
        const double* dx2, const double* dmx2,
        const double* dx3, const double* dmx3,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double dt)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int jm = j-1, km = k-1, kp = k+1;
    const int c   = idx3m(i, j,  k,  n2t, n3t);
    const int cjm = idx3m(i, jm, k,  n2t, n3t);
    const int ckm = idx3m(i, j,  km, n2t, n3t);
    const int ckp = idx3m(i, j,  kp, n2t, n3t);

    // dW interpolated to the V-location's lower/upper z-faces (y-stagger)
    const double dwm5 = 0.5 * (dx2[jm] * dW[c]   + dx2[j] * dW[cjm]) / dmx2[j];
    const double dwm6 = 0.5 * (dx2[jm] * dW[ckp] + dx2[j] * dW[idx3m(i, jm, kp, n2t, n3t)]) / dmx2[j];

    const double dvdz5 = (V[c]   - V[ckm]) / dmx3[k];
    const double dvdz6 = (V[ckp] - V[c])   / dmx3[kp];

    const double M23dW = 0.25 * (dwm5 * dvdz5 + dwm6 * dvdz6);
    dV[c] -= dt * M23dW;
}

// ===========================================================================
//  KERNEL 4: OB block coupling — dU -= dt * 0.25 * (dV * dU/dy + dW * dU/dz)
//  Mirrors CPU block_couple_dU.
// ===========================================================================
__global__ void k_block_couple_dU_ob(
        double* dU,
        const double* dV,
        const double* dW,
        const double* U,
        const double* dx1, const double* dmx1,
        const double* dmx2,
        const double* dmx3,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double dt)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int im = i-1, jm = j-1, jp = j+1, km = k-1, kp = k+1;
    const int c    = idx3m(i,  j,  k,  n2t, n3t);
    const int cim  = idx3m(im, j,  k,  n2t, n3t);
    const int cjm  = idx3m(i,  jm, k,  n2t, n3t);
    const int cjp  = idx3m(i,  jp, k,  n2t, n3t);
    const int ckm  = idx3m(i,  j,  km, n2t, n3t);
    const int ckp  = idx3m(i,  j,  kp, n2t, n3t);

    // dV interpolated to U-location's lower/upper y-faces (x-stagger)
    const double dvm3 = 0.5 * (dx1[im] * dV[c]   + dx1[i] * dV[cim]) / dmx1[i];
    const double dvm4 = 0.5 * (dx1[im] * dV[cjp] + dx1[i] * dV[idx3m(im, jp, k, n2t, n3t)]) / dmx1[i];
    const double dudy3 = (U[c]   - U[cjm]) / dmx2[j];
    const double dudy4 = (U[cjp] - U[c])   / dmx2[jp];

    // dW interpolated to U-location's lower/upper z-faces (x-stagger)
    const double dwm5 = 0.5 * (dx1[im] * dW[c]   + dx1[i] * dW[cim]) / dmx1[i];
    const double dwm6 = 0.5 * (dx1[im] * dW[ckp] + dx1[i] * dW[idx3m(im, j, kp, n2t, n3t)]) / dmx1[i];
    const double dudz5 = (U[c]   - U[ckm]) / dmx3[k];
    const double dudz6 = (U[ckp] - U[c])   / dmx3[kp];

    const double M12dV = 0.25 * (dvm3 * dudy3 + dvm4 * dudy4);
    const double M13dW = 0.25 * (dwm5 * dudz5 + dwm6 * dudz6);
    dU[c] -= dt * (M12dV + M13dW);
}

// ===========================================================================
//  KERNEL 5: scatter TDMA solution from band layout back to 3D field
//  (identical to scalar version — reused for all components)
// ===========================================================================
__global__ void k_m_scatter(
        mpmstd::Direction d,
        double* dst, const double* D_sol,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t)
{
    const int sys = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    int i, j, k, n_row_max, n_sys_max;
    if (d == mpmstd::Direction::X) {
        n_row_max = n1i; n_sys_max = n2i * n3i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        i = row + kH; j = (sys / n3i) + kH; k = (sys % n3i) + kH;
    } else if (d == mpmstd::Direction::Y) {
        n_row_max = n2i; n_sys_max = n1i * n3i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        j = row + kH; i = (sys / n3i) + kH; k = (sys % n3i) + kH;
    } else {
        n_row_max = n3i; n_sys_max = n1i * n2i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        k = row + kH; i = (sys / n2i) + kH; j = (sys % n2i) + kH;
    }
    dst[idx3m(i, j, k, n2t, n3t)] = D_sol[row * n_sys_max + sys];
}

// ===========================================================================
//  KERNEL 6: NOB explicit RHS  (compute_mpmstd_rhs_var, own=X/Y/Z)
//  All three components share this kernel; own_dir selects the branch.
// ===========================================================================
__global__ void k_mom_rhs_var(
        double*       rhs,
        const double* q,
        const double* U, const double* V, const double* W,
        const double* mu, const double* invrho,
        const double* dx1, const double* dmx1,
        const double* dx2, const double* dmx2,
        const double* dx3, const double* dmx3,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double Cmu, double dt,
        mpmstd::Direction own_dir)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int im=i-1, ip=i+1, jm=j-1, jp=j+1, km=k-1, kp=k+1;
    const int c   = idx3m(i,  j,  k,  n2t, n3t);
    const int cim = idx3m(im, j,  k,  n2t, n3t);
    const int cip = idx3m(ip, j,  k,  n2t, n3t);
    const int cjm = idx3m(i,  jm, k,  n2t, n3t);
    const int cjp = idx3m(i,  jp, k,  n2t, n3t);
    const int ckm = idx3m(i,  j,  km, n2t, n3t);
    const int ckp = idx3m(i,  j,  kp, n2t, n3t);

    constexpr double kHalf = 0.5;
    constexpr double kQ    = 0.25;
    constexpr double kTwo  = 2.0;
    double R = 0.0;

    // ---- own=X (dU component) -----------------------------------------------
    if (own_dir == mpmstd::Direction::X) {
        const double u1 = kHalf*(U[cim] + U[c]);
        const double u2 = kHalf*(U[c]   + U[cip]);
        const double w5 = mfce(dx1[im], W[c],   dx1[i], W[cim],                        dmx1[i]);
        const double w6 = mfce(dx1[im], W[ckp], dx1[i], W[idx3m(im,j,kp,n2t,n3t)],    dmx1[i]);
        const double v3 = mfce(dx1[im], V[c],   dx1[i], V[cim],                        dmx1[i]);
        const double v4 = mfce(dx1[im], V[cjp], dx1[i], V[idx3m(im,jp,k,n2t,n3t)],    dmx1[i]);

        const double dudx1 = (U[c]   - U[cim])/dx1[im];
        const double dudx2 = (U[cip] - U[c]  )/dx1[i];
        const double dudz5 = (U[c]   - U[ckm])/dmx3[k];
        const double dudz6 = (U[ckp] - U[c]  )/dmx3[kp];
        const double dudy3 = (U[c]   - U[cjm])/dmx2[j];
        const double dudy4 = (U[cjp] - U[c]  )/dmx2[jp];
        const double dwdx5 = (W[c]   - W[cim])/dmx1[i];
        const double dwdx6 = (W[ckp] - W[idx3m(im,j,kp,n2t,n3t)])/dmx1[i];
        const double dvdx3 = (V[c]   - V[cim])/dmx1[i];
        const double dvdx4 = (V[cjp] - V[idx3m(im,jp,k,n2t,n3t)])/dmx1[i];

        const double za = mfce(dx1[im], mu[ckm], dx1[i], mu[idx3m(im,j,km,n2t,n3t)],  dmx1[i]);
        const double zc = mfce(dx1[im], mu[c],   dx1[i], mu[cim],                      dmx1[i]);
        const double zb = mfce(dx1[im], mu[ckp], dx1[i], mu[idx3m(im,j,kp,n2t,n3t)],  dmx1[i]);
        const double muz5 = kHalf*(dx3[km]*zc + dx3[k]*za)/dmx3[k];
        const double muz6 = kHalf*(dx3[kp]*zc + dx3[k]*zb)/dmx3[kp];
        const double ya = mfce(dx1[im], mu[cjm], dx1[i], mu[idx3m(im,jm,k,n2t,n3t)],  dmx1[i]);
        const double yb = mfce(dx1[im], mu[cjp], dx1[i], mu[idx3m(im,jp,k,n2t,n3t)],  dmx1[i]);
        const double muy3 = kHalf*(dx2[jm]*zc + dx2[j]*ya)/dmx2[j];
        const double muy4 = kHalf*(dx2[jp]*zc + dx2[j]*yb)/dmx2[jp];

        const double irc = mfce(dx1[im], invrho[c], dx1[i], invrho[cim], dmx1[i]);
        const double inh = kHalf*Cmu*irc;

        const double vu1  = (mu[c]*dudx2 - mu[cim]*dudx1)/dmx1[i];
        const double vuZ  = (muz6*dudz6 - muz5*dudz5)/dx3[k];
        const double vuY  = (muy4*dudy4 - muy3*dudy3)/dx2[j];
        const double vuZc = (muz6*dwdx6 - muz5*dwdx5)/dx3[k];
        const double vuYc = (muy4*dvdx4 - muy3*dvdx3)/dx2[j];
        R = inh*(kTwo*vu1 + vuZ + vuY + vuZc + vuYc);

        // Report/34-class fix: own-X band used mu[c] (cell) on BOTH faces
        // (ported PaScaL bug, module_solve_momentum.f90:371,375 pre-fix) --
        // matches own-Y/own-Z below, which already differ mu[c] vs neighbor.
        const double mACI  =  inh/dmx1[i]*(mu[c]/dx1[i] + mu[cim]/dx1[im])*kTwo
                           + kQ*(kHalf*dudx2 + kHalf*dudx1 - u2/dx1[i] + u1/dx1[im]);
        const double mAPI  = -inh/dmx1[i]*mu[c]/dx1[i]*kTwo    + kQ*( u2/dx1[i]  + kHalf*dudx2);
        const double mAMI  = -inh/dmx1[i]*mu[cim]/dx1[im]*kTwo + kQ*(-u1/dx1[im] + kHalf*dudx1);
        const double mACK  =  inh/dx3[k]*(muz6/dmx3[kp] + muz5/dmx3[k]) + kQ*(-w6/dmx3[kp] + w5/dmx3[k]);
        const double mAPK  = -inh/dx3[k]*muz6/dmx3[kp] + kQ*( w6/dmx3[kp]);
        const double mAMK  = -inh/dx3[k]*muz5/dmx3[k]  + kQ*(-w5/dmx3[k]);
        const double mACJ  =  inh/dx2[j]*(muy4/dmx2[jp] + muy3/dmx2[j]) + kQ*(-v4/dmx2[jp] + v3/dmx2[j]);
        const double mAPJ  = -inh/dx2[j]*muy4/dmx2[jp] + kQ*( v4/dmx2[jp]);
        const double mAMJ  = -inh/dx2[j]*muy3/dmx2[j]  + kQ*(-v3/dmx2[j]);

        R -= ( mAPI*U[cip] + mACI*U[c] + mAMI*U[cim]
             + mAPJ*U[cjp] + mACJ*U[c] + mAMJ*U[cjm]
             + mAPK*U[ckp] + mACK*U[c] + mAMK*U[ckm] );
        R -= ( kQ*(w6*dudz6 + w5*dudz5) - inh/dx3[k]*(muz6*dwdx6 - muz5*dwdx5) );
        R -= ( kQ*(v4*dudy4 + v3*dudy3) - inh/dx2[j]*(muy4*dvdx4 - muy3*dvdx3) );
    }
    // ---- own=Y (dV component) -----------------------------------------------
    else if (own_dir == mpmstd::Direction::Y) {
        const double v5 = kHalf*(V[c] + V[cjm]);
        const double v6 = kHalf*(V[c] + V[cjp]);
        const double u1 = mfce(dx2[jm], U[c],   dx2[j], U[cjm],                        dmx2[j]);
        const double u2 = mfce(dx2[jm], U[cip], dx2[j], U[idx3m(ip,jm,k,n2t,n3t)],    dmx2[j]);
        const double w3 = mfce(dx2[jm], W[c],   dx2[j], W[cjm],                        dmx2[j]);
        const double w4 = mfce(dx2[jm], W[ckp], dx2[j], W[idx3m(i,jm,kp,n2t,n3t)],    dmx2[j]);

        const double dvdx1 = (V[c]   - V[cim])/dmx1[i];
        const double dvdx2 = (V[cip] - V[c]  )/dmx1[ip];
        const double dvdz3 = (V[c]   - V[ckm])/dmx3[k];
        const double dvdz4 = (V[ckp] - V[c]  )/dmx3[kp];
        const double dvdy5 = (V[c]   - V[cjm])/dx2[jm];
        const double dvdy6 = (V[cjp] - V[c]  )/dx2[j];
        const double dudy1 = (U[c]   - U[cjm])/dmx2[j];
        const double dudy2 = (U[cip] - U[idx3m(ip,jm,k,n2t,n3t)])/dmx2[j];
        const double dwdy3 = (W[c]   - W[cjm])/dmx2[j];
        const double dwdy4 = (W[ckp] - W[idx3m(i,jm,kp,n2t,n3t)])/dmx2[j];

        const double xa = mfce(dx2[jm], mu[cim], dx2[j], mu[idx3m(im,jm,k,n2t,n3t)],  dmx2[j]);
        const double xc = mfce(dx2[jm], mu[c],   dx2[j], mu[cjm],                      dmx2[j]);
        const double xb = mfce(dx2[jm], mu[cip], dx2[j], mu[idx3m(ip,jm,k,n2t,n3t)],  dmx2[j]);
        const double mu1x = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];
        const double mu2x = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];
        const double za = mfce(dx2[jm], mu[ckm], dx2[j], mu[idx3m(i,jm,km,n2t,n3t)],  dmx2[j]);
        const double zb = mfce(dx2[jm], mu[ckp], dx2[j], mu[idx3m(i,jm,kp,n2t,n3t)],  dmx2[j]);
        const double muz3 = kHalf*(dx3[km]*xc + dx3[k]*za)/dmx3[k];
        const double muz4 = kHalf*(dx3[kp]*xc + dx3[k]*zb)/dmx3[kp];

        const double irc = mfce(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]);
        const double inh = kHalf*Cmu*irc;

        const double vX   = (mu2x*dvdx2 - mu1x*dvdx1)/dx1[i];
        const double vZ   = (muz4*dvdz4 - muz3*dvdz3)/dx3[k];
        const double vYo  = (mu[c]*dvdy6 - mu[cjm]*dvdy5)/dmx2[j];
        const double vXc  = (mu2x*dudy2 - mu1x*dudy1)/dx1[i];
        const double vZc  = (muz4*dwdy4 - muz3*dwdy3)/dx3[k];
        R = inh*(vX + vZ + kTwo*vYo + vXc + vZc);

        const double mACI =  inh/dx1[i]*(mu2x/dmx1[ip] + mu1x/dmx1[i]) + kQ*(-u2/dmx1[ip] + u1/dmx1[i]);
        const double mAPI = -inh/dx1[i]*mu2x/dmx1[ip] + kQ*( u2/dmx1[ip]);
        const double mAMI = -inh/dx1[i]*mu1x/dmx1[i]  + kQ*(-u1/dmx1[i]);
        const double mACK =  inh/dx3[k]*(muz4/dmx3[kp] + muz3/dmx3[k]) + kQ*(-w4/dmx3[kp] + w3/dmx3[k]);
        const double mAPK = -inh/dx3[k]*muz4/dmx3[kp] + kQ*( w4/dmx3[kp]);
        const double mAMK = -inh/dx3[k]*muz3/dmx3[k]  + kQ*(-w3/dmx3[k]);
        const double mACJ =  inh/dmx2[j]*(mu[c]/dx2[j] + mu[cjm]/dx2[jm])*kTwo
                           + kQ*(kHalf*dvdy6 + kHalf*dvdy5 - v6/dx2[j] + v5/dx2[jm]);
        const double mAPJ = -inh/dmx2[j]*mu[c]/dx2[j]*kTwo   + kQ*( v6/dx2[j]  + kHalf*dvdy6);
        const double mAMJ = -inh/dmx2[j]*mu[cjm]/dx2[jm]*kTwo+ kQ*(-v5/dx2[jm] + kHalf*dvdy5);

        R -= ( mAPI*V[cip] + mACI*V[c] + mAMI*V[cim]
             + mAPJ*V[cjp] + mACJ*V[c] + mAMJ*V[cjm]
             + mAPK*V[ckp] + mACK*V[c] + mAMK*V[ckm] );
        R -= ( kQ*(u2*dvdx2 + u1*dvdx1) - inh/dx1[i]*(mu2x*dudy2 - mu1x*dudy1) );
        R -= ( kQ*(w3*dvdz3 + w4*dvdz4) - inh/dx3[k]*(muz4*dwdy4 - muz3*dwdy3) );
    }
    // ---- own=Z (dW component, wall-staggered) --------------------------------
    else {
        const double w5 = kHalf*(W[ckm] + W[c]);
        const double w6 = kHalf*(W[c]   + W[ckp]);
        const double u1 = mfce(dx3[km], U[c],   dx3[k], U[ckm],                        dmx3[k]);
        const double u2 = mfce(dx3[km], U[cip], dx3[k], U[idx3m(ip,j,km,n2t,n3t)],    dmx3[k]);
        const double v3 = mfce(dx3[km], V[c],   dx3[k], V[ckm],                        dmx3[k]);
        const double v4 = mfce(dx3[km], V[cjp], dx3[k], V[idx3m(i,jp,km,n2t,n3t)],    dmx3[k]);

        const double dwdx1 = (W[c]   - W[cim])/dmx1[i];
        const double dwdx2 = (W[cip] - W[c]  )/dmx1[ip];
        const double dwdz5 = (W[c]   - W[ckm])/dx3[km];
        const double dwdz6 = (W[ckp] - W[c]  )/dx3[k];
        const double dwdy3 = (W[c]   - W[cjm])/dmx2[j];
        const double dwdy4 = (W[cjp] - W[c]  )/dmx2[jp];
        const double dudz1 = (U[c]   - U[ckm])/dmx3[k];
        const double dudz2 = (U[cip] - U[idx3m(ip,j,km,n2t,n3t)])/dmx3[k];
        const double dvdz3 = (V[c]   - V[ckm])/dmx3[k];
        const double dvdz4 = (V[cjp] - V[idx3m(i,jp,km,n2t,n3t)])/dmx3[k];

        const double xa = mfce(dx3[km], mu[cim], dx3[k], mu[idx3m(im,j,km,n2t,n3t)],  dmx3[k]);
        const double xc = mfce(dx3[km], mu[c],   dx3[k], mu[ckm],                      dmx3[k]);
        const double xb = mfce(dx3[km], mu[cip], dx3[k], mu[idx3m(ip,j,km,n2t,n3t)],  dmx3[k]);
        const double mu1z = kHalf*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];
        const double mu2z = kHalf*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];
        const double ya = mfce(dx3[km], mu[cjm], dx3[k], mu[idx3m(i,jm,km,n2t,n3t)],  dmx3[k]);
        const double yb = mfce(dx3[km], mu[cjp], dx3[k], mu[idx3m(i,jp,km,n2t,n3t)],  dmx3[k]);
        const double muy5 = kHalf*(dx2[jm]*xc + dx2[j]*ya)/dmx2[j];
        const double muy6 = kHalf*(dx2[jp]*xc + dx2[j]*yb)/dmx2[jp];

        const double irc = mfce(dx3[km], invrho[c], dx3[k], invrho[ckm], dmx3[k]);
        const double inh = kHalf*Cmu*irc;

        const double vX   = (mu2z*dwdx2 - mu1z*dwdx1)/dx1[i];
        const double vZo  = (mu[c]*dwdz6 - mu[ckm]*dwdz5)/dmx3[k];
        const double vY   = (muy6*dwdy4 - muy5*dwdy3)/dx2[j];
        const double vXc  = (mu2z*dudz2 - mu1z*dudz1)/dx1[i];
        const double vYc  = (muy6*dvdz4 - muy5*dvdz3)/dx2[j];
        R = inh*(vX + kTwo*vZo + vY + vXc + vYc);

        const double mACI =  inh/dx1[i]*(mu2z/dmx1[ip] + mu1z/dmx1[i]) + kQ*(-u2/dmx1[ip] + u1/dmx1[i]);
        const double mAPI = -inh/dx1[i]*mu2z/dmx1[ip] + kQ*( u2/dmx1[ip]);
        const double mAMI = -inh/dx1[i]*mu1z/dmx1[i]  + kQ*(-u1/dmx1[i]);
        const double mACK =  inh/dmx3[k]*(mu[c]/dx3[k] + mu[ckm]/dx3[km])*kTwo
                           + kQ*(kHalf*dwdz6 + kHalf*dwdz5 - w6/dx3[k] + w5/dx3[km]);
        const double mAPK = -inh/dmx3[k]*mu[c]/dx3[k]*kTwo   + kQ*( w6/dx3[k]  + kHalf*dwdz6);
        const double mAMK = -inh/dmx3[k]*mu[ckm]/dx3[km]*kTwo+ kQ*(-w5/dx3[km] + kHalf*dwdz5);
        const double mACJ =  inh/dx2[j]*(muy6/dmx2[jp] + muy5/dmx2[j]) + kQ*(-v4/dmx2[jp] + v3/dmx2[j]);
        const double mAPJ = -inh/dx2[j]*muy6/dmx2[jp] + kQ*( v4/dmx2[jp]);
        const double mAMJ = -inh/dx2[j]*muy5/dmx2[j]  + kQ*(-v3/dmx2[j]);

        R -= ( mAPI*W[cip] + mACI*W[c] + mAMI*W[cim]
             + mAPJ*W[cjp] + mACJ*W[c] + mAMJ*W[cjm]
             + mAPK*W[ckp] + mACK*W[c] + mAMK*W[ckm] );
        R -= ( kQ*(u2*dwdx2 + u1*dwdx1) - inh/dx1[i]*(mu2z*dudz2 - mu1z*dudz1) );
        R -= ( kQ*(v4*dwdy4 + v3*dwdy3) - inh/dx2[j]*(muy6*dvdz4 - muy5*dvdz3) );
    }

    rhs[c] = R * dt;
}

// ===========================================================================
//  KERNEL 7: NOB ADI band assembly
//  Mirrors CPU build_adi_bands_var (emit lambda).  own_dir selects which
//  component (determines stencil type: own vs cross, which metrics to use).
// ===========================================================================
__global__ void k_mom_adi_var(
        mpmstd::Direction d, mpmstd::Direction own_dir,
        double* A, double* B, double* C, double* D,
        const double* src,
        const double* q,       // own velocity (for own-dir conv ∂q terms)
        const double* q_adv,   // advecting velocity along d
        const double* mu, const double* invrho,
        const double* dxa, const double* dmxa,   // swept-axis
        const double* dxo, const double* dmxo,   // own-axis
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double Cmu, double dt,
        bool z_staggered)
{
    const int sys = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    int i, j, k, n_row_max, n_sys_max;
    if (d == mpmstd::Direction::X) {
        n_row_max = n1i; n_sys_max = n2i * n3i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        i = row + kH; j = (sys / n3i) + kH; k = (sys % n3i) + kH;
    } else if (d == mpmstd::Direction::Y) {
        n_row_max = n2i; n_sys_max = n1i * n3i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        j = row + kH; i = (sys / n3i) + kH; k = (sys % n3i) + kH;
    } else {
        n_row_max = n3i; n_sys_max = n1i * n2i;
        if (row >= n_row_max || sys >= n_sys_max) return;
        k = row + kH; i = (sys / n2i) + kH; j = (sys % n2i) + kH;
    }
    const int p   = row * n_sys_max + sys;

    // swept-axis index 'a' and own-axis index 'bo'
    const int a  = (d == mpmstd::Direction::X) ? i : (d == mpmstd::Direction::Y) ? j : k;
    const int bo = (own_dir == mpmstd::Direction::X) ? i : (own_dir == mpmstd::Direction::Y) ? j : k;

    const int gi = idx3m(i, j, k, n2t, n3t);

    // own-direction stride (for volw_o)
    const int ostride = (own_dir == mpmstd::Direction::X) ? n2t*n3t
                      : (own_dir == mpmstd::Direction::Y) ? n3t : 1;

    // own-face volume-weighted interp helper (lambda → inline here)
    // volw_o(f, cell, b) = 0.5*(dxo[b-1]*f[cell] + dxo[b]*f[cell-ostride])/dmxo[b]
    const int gm_own = gi - ostride;  // own-minus neighbour

    const double ir_raw = 0.5*(dxo[bo-1]*invrho[gi] + dxo[bo]*invrho[gm_own])/dmxo[bo];

    const bool own = (d == own_dir);
    const double inh = 0.5 * Cmu * dt;  // × ir per cell
    const double cq  = 0.25 * dt;

    double ad, cd;
    if (own) {
        const double cmu = mu[gi];
        // Report/34-class fix: own-X previously fell through to mu[gi] here
        // too (ported PaScaL bug) -- own-minus face always uses the own-axis
        // minus-neighbour's mu, for all three own directions (X/Y/Z-staggered).
        const double mmu = mu[gi - ostride];
        ad = -inh * ir_raw / dmxa[a] * mmu / dxa[a-1] * 2.0;
        cd = -inh * ir_raw / dmxa[a] * cmu / dxa[a]   * 2.0;
    } else {
        // Doubly volume-weighted cross-face μ
        const double muc = 0.5*(dxo[bo-1]*mu[gi]        + dxo[bo]*mu[gm_own])/dmxo[bo];
        // swept-axis minus/plus neighbours
        const int gsm = (d == mpmstd::Direction::X) ? idx3m(i-1,j,k,n2t,n3t)
                      : (d == mpmstd::Direction::Y) ? idx3m(i,j-1,k,n2t,n3t)
                                                    : idx3m(i,j,k-1,n2t,n3t);
        const int gsp = (d == mpmstd::Direction::X) ? idx3m(i+1,j,k,n2t,n3t)
                      : (d == mpmstd::Direction::Y) ? idx3m(i,j+1,k,n2t,n3t)
                                                    : idx3m(i,j,k+1,n2t,n3t);
        const int gsm_own = gsm - ostride;
        const int gsp_own = gsp - ostride;
        const double mua = 0.5*(dxo[bo-1]*mu[gsm] + dxo[bo]*mu[gsm_own])/dmxo[bo];
        const double mub = 0.5*(dxo[bo-1]*mu[gsp] + dxo[bo]*mu[gsp_own])/dmxo[bo];
        const double mm = 0.5*(dxa[a-1]*muc + dxa[a]*mua)/dmxa[a];
        const double mp = 0.5*(dxa[a+1]*muc + dxa[a]*mub)/dmxa[a+1];
        if (z_staggered) {
            ad = -inh * ir_raw / dmxa[a] * mm / dxa[a-1];
            cd = -inh * ir_raw / dmxa[a] * mp / dxa[a];
        } else {
            ad = -inh * ir_raw / dxa[a] * mm / dmxa[a];
            cd = -inh * ir_raw / dxa[a] * mp / dmxa[a+1];
        }
    }

    // swept-axis minus/plus neighbours (for advecting vel + q gradient)
    const int gm = (d == mpmstd::Direction::X) ? idx3m(i-1,j,k,n2t,n3t)
                 : (d == mpmstd::Direction::Y) ? idx3m(i,j-1,k,n2t,n3t)
                                               : idx3m(i,j,k-1,n2t,n3t);
    const int gp = (d == mpmstd::Direction::X) ? idx3m(i+1,j,k,n2t,n3t)
                 : (d == mpmstd::Direction::Y) ? idx3m(i,j+1,k,n2t,n3t)
                                               : idx3m(i,j,k+1,n2t,n3t);

    double q5, q6;
    double ac, bc, cc;
    if (own) {
        // own-direction: plain face mean (matches u1/v5/w5 in k_mom_rhs_var own branch)
        q5 = 0.5*(q_adv[gm] + q_adv[gi]);
        q6 = 0.5*(q_adv[gi] + q_adv[gp]);
        const double dq5 = (q[gi]-q[gm])/dxa[a-1];
        const double dq6 = (q[gp]-q[gi])/dxa[a];
        ac = cq*(-q5/dxa[a-1] + 0.5*dq5);
        cc = cq*( q6/dxa[a]   + 0.5*dq6);
        bc = cq*( 0.5*dq6 + 0.5*dq5 - q6/dxa[a] + q5/dxa[a-1]);
    } else {
        // cross-direction: advecting velocity interpolated VOLUME-WEIGHTED along the
        // OWN axis (matches k_mom_rhs_var w5/w6/v3/v4 at L444-447 and PaScaL_TCS
        // f90:301-302, reused in the M bands at f90:390/392/394). The previous plain
        // swept-axis mean made the implicit M-operator inconsistent with the faithful
        // −M·qⁿ residual and with PaScaL — an O(dx) convection-linearization divergence.
        q5 = 0.5*(dxo[bo-1]*q_adv[gi] + dxo[bo]*q_adv[gi - ostride])/dmxo[bo];
        q6 = 0.5*(dxo[bo-1]*q_adv[gp] + dxo[bo]*q_adv[gp - ostride])/dmxo[bo];
        if (z_staggered) {
            ac = cq*(-q5/dxa[a-1]);
            cc = cq*( q6/dxa[a]);
            bc = cq*(-q6/dxa[a] + q5/dxa[a-1]);
        } else {
            ac = cq*(-q5/dmxa[a]);
            cc = cq*( q6/dmxa[a+1]);
            bc = cq*(-q6/dmxa[a+1] + q5/dmxa[a]);
        }
    }

    A[p] = ad + ac;
    C[p] = cd + cc;
    B[p] = (1.0 - ad - cd) + bc;
    D[p] = src[gi];
}

// ===========================================================================
//  KERNEL 8: NOB dd* coupling — couple_dd_dW_var
//  rhs_w[c] -= dt*M31(dU) + dt*M32(dV)
//  Both lower-triangular off-diagonals of the wall-normal equation enter the
//  RHS BEFORE the dW ADI solve, so they pass through the implicit z-operator
//  (ref MPM-STD solvedW 1524-1531 M31ddU + 1533-1540 M32ddV). The wall-normal
//  increment must never receive an unfiltered explicit correction — that was
//  the Report/30 reignition pump (Report/RBC arrangement analysis).
// ===========================================================================
__global__ void k_dd_dW_var(
        double* rhs_w,
        const double* dU,
        const double* dV,
        const double* W,
        const double* mu, const double* invrho,
        const double* dx1, const double* dmx1,
        const double* dx2, const double* dmx2,
        const double* dx3, const double* dmx3,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double Cmu, double dt)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int im=i-1, ip=i+1, jm=j-1, jp=j+1, km=k-1;
    const int c   = idx3m(i,  j, k,  n2t, n3t);
    const int cim = idx3m(im, j, k,  n2t, n3t);
    const int cip = idx3m(ip, j, k,  n2t, n3t);
    const int cjm = idx3m(i,  jm, k, n2t, n3t);
    const int cjp = idx3m(i,  jp, k, n2t, n3t);
    const int ckm = idx3m(i,  j, km, n2t, n3t);

    const double irc = mfce(dx3[km], invrho[c], dx3[k], invrho[ckm], dmx3[k]);
    const double inh = 0.5*Cmu*irc;
    const double xc = mfce(dx3[km], mu[c],   dx3[k], mu[ckm],                     dmx3[k]);

    // --- M31: dU increment (ref solvedW 1524-1531) ---
    const double ddu1 = mfce(dx3[km], dU[c],   dx3[k], dU[ckm],                     dmx3[k]);
    const double ddu2 = mfce(dx3[km], dU[cip], dx3[k], dU[idx3m(ip,j,km,n2t,n3t)], dmx3[k]);
    const double dWdx1 = (W[c]   - W[cim])/dmx1[i];
    const double dWdx2 = (W[cip] - W[c]  )/dmx1[ip];
    const double ddudz1 = (dU[c]   - dU[ckm])/dmx3[k];
    const double ddudz2 = (dU[cip] - dU[idx3m(ip,j,km,n2t,n3t)])/dmx3[k];

    const double xa = mfce(dx3[km], mu[cim], dx3[k], mu[idx3m(im,j,km,n2t,n3t)], dmx3[k]);
    const double xb = mfce(dx3[km], mu[cip], dx3[k], mu[idx3m(ip,j,km,n2t,n3t)], dmx3[k]);
    const double mu1 = 0.5*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];
    const double mu2 = 0.5*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];

    rhs_w[c] -= dt * ( 0.25*(ddu2*dWdx2 + ddu1*dWdx1)
                     - inh/dx1[i]*(mu2*ddudz2 - mu1*ddudz1) );

    // --- M32: dV increment (ref solvedW 1533-1540) ---
    const double dvm5 = mfce(dx3[km], dV[c],   dx3[k], dV[ckm],                     dmx3[k]);
    const double dvm6 = mfce(dx3[km], dV[cjp], dx3[k], dV[idx3m(i,jp,km,n2t,n3t)], dmx3[k]);
    const double dWdy5 = (W[c]   - W[cjm])/dmx2[j];
    const double dWdy6 = (W[cjp] - W[c]  )/dmx2[jp];
    const double ddVz5 = (dV[c]   - dV[ckm])/dmx3[k];
    const double ddVz6 = (dV[cjp] - dV[idx3m(i,jp,km,n2t,n3t)])/dmx3[k];

    const double ya = mfce(dx3[km], mu[cjm], dx3[k], mu[idx3m(i,jm,km,n2t,n3t)], dmx3[k]);
    const double yb = mfce(dx3[km], mu[cjp], dx3[k], mu[idx3m(i,jp,km,n2t,n3t)], dmx3[k]);
    const double muy5 = 0.5*(dx2[jm]*xc + dx2[j]*ya)/dmx2[j];
    const double muy6 = 0.5*(dx2[jp]*xc + dx2[j]*yb)/dmx2[jp];

    rhs_w[c] -= dt * ( 0.25*(dvm5*dWdy5 + dvm6*dWdy6)
                     - inh/dx2[j]*(muy6*ddVz6 - muy5*ddVz5) );
}

// ===========================================================================
//  KERNEL 9: NOB dd* coupling — couple_dd_dV_var
//  rhs_v[c] -= dt*M21(dU)   (ref MPM-STD solvedV 1211-1218 M21ddU)
//  The M23(dW) coupling is NOT here: dV is solved BEFORE dW in the reference
//  arrangement; dW feeds dV afterwards via k_block_dV_var (Report/RBC).
// ===========================================================================
__global__ void k_dd_dV_var(
        double* rhs_v,
        const double* dU,
        const double* V,
        const double* mu, const double* invrho,
        const double* dx1, const double* dmx1,
        const double* dx2, const double* dmx2,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double Cmu, double dt)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int im=i-1, ip=i+1, jm=j-1;
    const int c   = idx3m(i,  j,  k,  n2t, n3t);
    const int cim = idx3m(im, j,  k,  n2t, n3t);
    const int cip = idx3m(ip, j,  k,  n2t, n3t);
    const int cjm = idx3m(i,  jm, k,  n2t, n3t);

    // --- μ-faces (identical to dV body, ref 1033-1046) ---
    const double xa = mfce(dx2[jm], mu[cim], dx2[j], mu[idx3m(im,jm,k,n2t,n3t)], dmx2[j]);
    const double xc = mfce(dx2[jm], mu[c],   dx2[j], mu[cjm],                     dmx2[j]);
    const double xb = mfce(dx2[jm], mu[cip], dx2[j], mu[idx3m(ip,jm,k,n2t,n3t)], dmx2[j]);
    const double mu1 = 0.5*(dx1[im]*xc + dx1[i]*xa)/dmx1[i];
    const double mu2 = 0.5*(dx1[ip]*xc + dx1[i]*xb)/dmx1[ip];
    const double irc = mfce(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]);
    const double inh = 0.5*Cmu*irc;

    // V gradients that the increment multiplies
    const double dvdx1 = (V[c]   - V[cim])/dmx1[i];
    const double dvdx2 = (V[cip] - V[c]  )/dmx1[ip];

    // M21: dU increment (ref solvedV 1211-1218)
    const double ddu1   = mfce(dx2[jm], dU[c],   dx2[j], dU[cjm],                     dmx2[j]);
    const double ddu2   = mfce(dx2[jm], dU[cip], dx2[j], dU[idx3m(ip,jm,k,n2t,n3t)], dmx2[j]);
    const double ddudy1 = (dU[c]   - dU[cjm])/dmx2[j];
    const double ddudy2 = (dU[cip] - dU[idx3m(ip,jm,k,n2t,n3t)])/dmx2[j];

    rhs_v[c] -= dt * ( 0.25*(ddu2*dvdx2 + ddu1*dvdx1)
                     - inh/dx1[i]*(mu2*ddudy2 - mu1*ddudy1) );
}

// ===========================================================================
//  KERNEL 10: NOB block coupling — block_couple_dV_var
//  dV[c] -= dt*M23(dW)   (ref MPM-STD blockLdV 1612-1688, M23dWm at 1682)
//  The upper-triangular explicit correction goes to the HORIZONTAL dV only.
//  The wall-normal dW receives no post-hoc correction in the reference
//  arrangement — that unfiltered path was the Report/30 reignition pump
//  (Report/RBC wall-reignition analysis).
// ===========================================================================
__global__ void k_block_dV_var(
        double* dV,
        const double* dW,
        const double* V,
        const double* mu, const double* invrho,
        const double* dx2, const double* dmx2,
        const double* dx3, const double* dmx3,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double Cmu, double dt)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int jm=j-1, km=k-1, kp=k+1;
    const int c   = idx3m(i, j,  k,  n2t, n3t);
    const int cjm = idx3m(i, jm, k,  n2t, n3t);
    const int ckm = idx3m(i, j,  km, n2t, n3t);
    const int ckp = idx3m(i, j,  kp, n2t, n3t);

    // dW interp to the V node at z-faces k/kp (ref dwm5/dwm6 1658-1659)
    const double ddw3   = mfce(dx2[jm], dW[c],   dx2[j], dW[cjm],                     dmx2[j]);
    const double ddw4   = mfce(dx2[jm], dW[ckp], dx2[j], dW[idx3m(i,jm,kp,n2t,n3t)], dmx2[j]);
    const double ddwdy3 = (dW[c]   - dW[cjm])/dmx2[j];
    const double ddwdy4 = (dW[ckp] - dW[idx3m(i,jm,kp,n2t,n3t)])/dmx2[j];
    const double dvdz3  = (V[c]   - V[ckm])/dmx3[k];
    const double dvdz4  = (V[ckp] - V[c]  )/dmx3[kp];

    const double xc = mfce(dx2[jm], mu[c],   dx2[j], mu[cjm],                     dmx2[j]);
    const double za = mfce(dx2[jm], mu[ckm], dx2[j], mu[idx3m(i,jm,km,n2t,n3t)], dmx2[j]);
    const double zb = mfce(dx2[jm], mu[ckp], dx2[j], mu[idx3m(i,jm,kp,n2t,n3t)], dmx2[j]);
    const double muz3 = 0.5*(dx3[km]*xc + dx3[k]*za)/dmx3[k];
    const double muz4 = 0.5*(dx3[kp]*xc + dx3[k]*zb)/dmx3[kp];
    const double irc = mfce(dx2[jm], invrho[c], dx2[j], invrho[cjm], dmx2[j]);

    dV[c] -= dt * ( 0.25*(ddw3*dvdz3 + ddw4*dvdz4)
                  - 0.5*Cmu*irc/dx3[k]*(muz4*ddwdy4 - muz3*ddwdy3) );
}

// ===========================================================================
//  KERNEL 11: NOB block coupling — block_couple_dU_var
//  dU[c] -= dt * (M12(dW) + M13(dV))  full viscous cross-stress
// ===========================================================================
__global__ void k_block_dU_var(
        double* dU,
        const double* dV,
        const double* dW,
        const double* U,
        const double* mu, const double* invrho,
        const double* dx1, const double* dmx1,
        const double* dx2, const double* dmx2,
        const double* dx3, const double* dmx3,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        double Cmu, double dt)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;

    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const int im=i-1, jm=j-1, jp=j+1, km=k-1, kp=k+1;
    const int c   = idx3m(i,  j,  k,  n2t, n3t);
    const int cim = idx3m(im, j,  k,  n2t, n3t);
    const int cjp = idx3m(i,  jp, k,  n2t, n3t);
    const int ckp = idx3m(i,  j,  kp, n2t, n3t);

    const double irc = mfce(dx1[im], invrho[c], dx1[i], invrho[cim], dmx1[i]);

    // --- M12: wall increment (my dW) feeds dU via ∂U/∂z ---
    const double dWm5 = mfce(dx1[i], dW[cim], dx1[im], dW[c],                        dmx1[i]);
    const double dWm6 = mfce(dx1[i], dW[idx3m(im,j,kp,n2t,n3t)], dx1[im], dW[ckp],  dmx1[i]);
    const double dUdz5 = (U[c]   - U[idx3m(i,j,km,n2t,n3t)])/dmx3[k];
    const double dUdz6 = (U[ckp] - U[c]  )/dmx3[kp];
    const double dWmx5 = (dW[c]  - dW[cim])/dmx1[i];
    const double dWmx6 = (dW[ckp]- dW[idx3m(im,j,kp,n2t,n3t)])/dmx1[i];

    const double za = mfce(dx1[im], mu[idx3m(i,j,km,n2t,n3t)],  dx1[i], mu[idx3m(im,j,km,n2t,n3t)], dmx1[i]);
    const double zc = mfce(dx1[im], mu[c],   dx1[i], mu[cim],                        dmx1[i]);
    const double zb = mfce(dx1[im], mu[ckp], dx1[i], mu[idx3m(im,j,kp,n2t,n3t)],    dmx1[i]);
    const double muz5 = 0.5*(dx3[km]*zc + dx3[k]*za)/dmx3[k];
    const double muz6 = 0.5*(dx3[kp]*zc + dx3[k]*zb)/dmx3[kp];

    dU[c] -= dt * ( 0.25*(dWm6*dUdz6 + dWm5*dUdz5)
                  - 0.5*Cmu*irc/dx3[k]*(muz6*dWmx6 - muz5*dWmx5) );

    // --- M13: span increment (my dV) feeds dU via ∂U/∂y ---
    const double dVm3 = mfce(dx1[i], dV[cim], dx1[im], dV[c],                        dmx1[i]);
    const double dVm4 = mfce(dx1[i], dV[idx3m(im,jp,k,n2t,n3t)], dx1[im], dV[cjp],  dmx1[i]);
    const double dUdy3 = (U[c]   - U[idx3m(i,jm,k,n2t,n3t)])/dmx2[j];
    const double dUdy4 = (U[cjp] - U[c]  )/dmx2[jp];
    const double dVmx3 = (dV[c]  - dV[cim])/dmx1[i];
    const double dVmx4 = (dV[cjp]- dV[idx3m(im,jp,k,n2t,n3t)])/dmx1[i];

    const double ya = mfce(dx1[im], mu[idx3m(i,jm,k,n2t,n3t)],  dx1[i], mu[idx3m(im,jm,k,n2t,n3t)], dmx1[i]);
    const double yb = mfce(dx1[im], mu[cjp], dx1[i], mu[idx3m(im,jp,k,n2t,n3t)],    dmx1[i]);
    const double muy3 = 0.5*(dx2[jm]*zc + dx2[j]*ya)/dmx2[j];
    const double muy4 = 0.5*(dx2[jp]*zc + dx2[j]*yb)/dmx2[jp];

    dU[c] -= dt * ( 0.25*(dVm4*dUdy4 + dVm3*dUdy3)
                  - 0.5*Cmu*irc/dx2[j]*(muy4*dVmx4 - muy3*dVmx3) );
}

// ===========================================================================
//  KERNEL 11b: pin the lower z-wall W-face increment to zero.
//  The lower wall W face lives at INTERIOR index k=kH on the MAC grid (the
//  upper wall face sits in the ghost slot), so the z-sweep solves that row
//  like a regular interior row and leaves a spurious increment there. The
//  face is pinned (W=0), so dW must be zero BEFORE it is consumed by
//  k_dd_dV_var / k_block_dU_var; otherwise it leaks into dV/dU with 1/dmx3
//  wall-shear amplification — the bottom-wall-only U spike (Report/27).
// ===========================================================================
__global__ void k_zero_dw_wall_face(double* dW, int n1t, int n2t, int n3t)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n1t || j >= n2t) return;
    dW[(i * n2t + j) * n3t + kH] = 0.0;
}

// ===========================================================================
//  KERNEL 12: copy entire field (device → device, for updating dU/dV/dW fields)
// ===========================================================================
__global__ void k_m_copy_field(double* dst, const double* src, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

// ===========================================================================
//  KERNEL 13: U += dU  (update_velocity, applies to all three components)
// ===========================================================================
__global__ void k_m_add_increment(
        double* q, const double* dq,
        int n1i, int n2i, int n3i,
        int n2t, int n3t)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;
    const int c = idx3m(ii + kH, jj + kH, kk + kH, n2t, n3t);
    q[c] += dq[c];
}

// ---------------------------------------------------------------------------
//  GPU kernel: replace D2H + modify_tdma_row_cpu + H2D for non-periodic dirs.
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

// ---------------------------------------------------------------------------
//  z-staggered wall-face row: on the MAC grid the lower z-wall W face is
//  INTERIOR row 0 of the z-sweep (face zf[0], pinned δW=0). Make it an
//  identity row — a regular row lets its spurious unknown pollute the
//  adjacent rows THROUGH the TDMA during the solve, which the post-solve
//  pin cannot undo (PaScaL excludes this row via j_indexS; Report/30).
// ---------------------------------------------------------------------------
__global__ void k_w_wallface_identity_row(double* A, double* B, double* C, double* D,
                                          int n_sys) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_sys) return;
    A[s] = 0.0; C[s] = 0.0; B[s] = 1.0; D[s] = 0.0;
}

// ---------------------------------------------------------------------------
//  Report/30 §6 instrumentation: interior max|f| split into a top-wall slab
//  (last kMomDbgSlab interior k-planes on the rank owning the upper z-wall)
//  vs the remaining interior. fabs(x) clears the sign bit, so the uint64 bit
//  pattern of a non-negative double orders like the double. NOTE: nvc++ -cuda
//  silently no-ops atomicMax(unsigned long long*) (verified on cc80, NVHPC
//  23.7) — use an atomicCAS loop instead.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dbg_atomic_max_u64(unsigned long long* addr,
                                                   unsigned long long val)
{
    unsigned long long old = *addr;
    while (old < val) {
        const unsigned long long assumed = old;
        old = atomicCAS(addr, assumed, val);
        if (old == assumed) break;
    }
}

__global__ void k_dbg_slabmax(const double* f, unsigned long long* out2,
                              int n1i, int n2i, int n3i, int n2t, int n3t,
                              int k_slab0)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;
    const int k = kk + kH;
    const double v = fabs(f[idx3m(ii + kH, jj + kH, k, n2t, n3t)]);
    dbg_atomic_max_u64(&out2[(k >= k_slab0) ? 0 : 1],
                       (unsigned long long)__double_as_longlong(v));
}

//  Argmax probe (MPMSTD_MOMDBG=2): print interior locations holding ≥99.9% of
//  the field max — locates where a spike lives. Few hits expected.
__global__ void k_dbg_argmax(const double* f, double maxv,
                             int n1i, int n2i, int n3i, int n2t, int n3t,
                             char tag, int rank)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;
    const int i = ii + kH, j = jj + kH, k = kk + kH;
    const double v = f[idx3m(i, j, k, n2t, n3t)];
    if (fabs(v) >= 0.999 * maxv && maxv > 0.0)
        printf("[MOMDBG r%d] argmax %c i=%d j=%d k=%d v=%+.4e\n",
               rank, tag, i, j, k, v);
}

//  E9b bisection switch (MPMSTD_BLKMASK_TOP=1): restore the top-slab planes
//  (k >= k_slab0, full i/j extent incl. halo) from a pre-stage copy, i.e. make
//  the block-coupling stage a no-op there. Diagnostic only.
__global__ void k_dbg_restore_top(double* dst, const double* src,
                                  int n1t, int n2t, int n3t, int k_slab0)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= n1t || j >= n2t || k >= n3t || k < k_slab0) return;
    const int c = (i * n2t + j) * n3t + k;
    dst[c] = src[c];
}

//  E9f global under-relaxation (MPMSTD_BLKRELAX=<theta>): blend the block-
//  coupling correction uniformly — dst = src + theta*(dst - src). No spatial
//  edge (unlike the mask). Diagnostic / interim stabilization.
__global__ void k_dbg_blend_full(double* dst, const double* src, double theta, int n)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    dst[c] = src[c] + theta * (dst[c] - src[c]);
}

//  Slab inner products: out2[0] += df·f, out2[1] += f·f over the top slab.
//  g = out2[0]/out2[1] is the projection of the increment onto the existing
//  pattern — g>0 means the stage pumps the spike, g<0 means it damps it.
__global__ void k_dbg_slabdot(const double* df, const double* f, double* out2,
                              int n1i, int n2i, int n3i, int n2t, int n3t,
                              int k_slab0)
{
    const int ii = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int kk = blockIdx.z * blockDim.z + threadIdx.z;
    if (ii >= n1i || jj >= n2i || kk >= n3i) return;
    const int k = kk + kH;
    if (k < k_slab0) return;
    const int c = idx3m(ii + kH, jj + kH, k, n2t, n3t);
    atomicAdd(&out2[0], df[c] * f[c]);
    atomicAdd(&out2[1], f[c] * f[c]);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  Host side of the Report/30 §6 stage instrumentation. Enabled per-process
//  by MPMSTD_MOMDBG=1; each call prints one line per rank:
//      [MOMDBG r<rank>] <tag> top=<max|f| in top 6 interior k-planes> bulk=...
//  On ranks not owning the upper z-wall the slab is empty (top=0).
// ---------------------------------------------------------------------------
static int mom_dbg_level()
{
    static int lvl = -1;
    if (lvl < 0) {
        const char* e = std::getenv("MPMSTD_MOMDBG");
        lvl = (e && e[0] && e[0] != '0') ? std::atoi(e) : 0;
        if (lvl < 0) lvl = 0;
    }
    return lvl;
}

static bool mom_dbg_on() { return mom_dbg_level() >= 1; }

//  E9f: MPMSTD_BLKRELAX=<theta in (0,1)> → uniformly under-relax the block-
//  coupling correction (1.0 = off/full correction).
static double mom_blkrelax_theta()
{
    static double theta = -1.0;
    if (theta < 0.0) {
        const char* e = std::getenv("MPMSTD_BLKRELAX");
        theta = e && e[0] ? std::atof(e) : 1.0;
        if (theta <= 0.0 || theta >= 1.0) theta = 1.0;
    }
    return theta;
}

//  E9b bisection: MPMSTD_BLKMASK_TOP=<n> → block-coupling stages become no-ops
//  in the top n interior k-planes (n=6 if the value is just "1"; 0 = off).
//  Diagnostic only (see k_dbg_restore_top).
static int mom_blkmask_top_planes()
{
    static int planes = -1;
    if (planes < 0) {
        const char* e = std::getenv("MPMSTD_BLKMASK_TOP");
        if (!e || !e[0] || e[0] == '0') planes = 0;
        else {
            const int v = std::atoi(e);
            planes = (v > 1) ? v : 6;
        }
    }
    return planes;
}

static void mom_dbg_slabmax(const char* tag, const double* d_f,
                            const mpmstd::core::Domain& domain,
                            int n1i, int n2i, int n3i, int n2t, int n3t,
                            char argmax_tag = 0)
{
    if (!mom_dbg_on()) return;
    static unsigned long long* d_out = nullptr;
    if (!d_out) CUDA_CHK(cudaMalloc(&d_out, 3 * sizeof(unsigned long long)));
    CUDA_CHK(cudaMemset(d_out, 0, 3 * sizeof(unsigned long long)));

    constexpr int kMomDbgSlab = 6;
    const bool owns_upper =
        domain.sub.topology().axis(mpmstd::Direction::Z).east_rank == MPI_PROC_NULL;
    // interior k runs [kH, n3i+kH-1]; slab start beyond that range = empty slab
    const int k_slab0 = owns_upper ? (n3i + kH - kMomDbgSlab) : (n3i + 2 * kH);

    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i + 7) / 8, (n2i + 7) / 8, (n3i + 7) / 8);
    k_dbg_slabmax<<<grd, blk>>>(d_f, d_out, n1i, n2i, n3i, n2t, n3t, k_slab0);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    unsigned long long h[2];
    CUDA_CHK(cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost));
    double top, bulk;
    std::memcpy(&top,  &h[0], sizeof top);
    std::memcpy(&bulk, &h[1], sizeof bulk);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::printf("[MOMDBG r%d] %-12s top=%.6e bulk=%.6e\n", rank, tag, top, bulk);
    std::fflush(stdout);

    if (argmax_tag && mom_dbg_level() >= 2) {
        const double maxv = top > bulk ? top : bulk;
        k_dbg_argmax<<<grd, blk>>>(d_f, maxv, n1i, n2i, n3i, n2t, n3t,
                                   argmax_tag, rank);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
    }
}

//  g = <df, f>/<f, f> over the top slab — per-stage pump/damp discriminator.
//  Only the rank owning the upper z-wall prints (the slab is empty elsewhere).
static void mom_dbg_slabcorr(const char* tag, const double* d_df, const double* d_f,
                             const mpmstd::core::Domain& domain,
                             int n1i, int n2i, int n3i, int n2t, int n3t)
{
    if (!mom_dbg_on()) return;
    const bool owns_upper =
        domain.sub.topology().axis(mpmstd::Direction::Z).east_rank == MPI_PROC_NULL;
    if (!owns_upper) return;

    static double* d_out = nullptr;
    if (!d_out) CUDA_CHK(cudaMalloc(&d_out, 2 * sizeof(double)));
    CUDA_CHK(cudaMemset(d_out, 0, 2 * sizeof(double)));

    constexpr int kMomDbgSlab = 6;
    const int k_slab0 = n3i + kH - kMomDbgSlab;

    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i + 7) / 8, (n2i + 7) / 8, (n3i + 7) / 8);
    k_dbg_slabdot<<<grd, blk>>>(d_df, d_f, d_out, n1i, n2i, n3i, n2t, n3t, k_slab0);
    CUDA_CHK(cudaGetLastError());

    double h[2];
    CUDA_CHK(cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost));
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::printf("[MOMDBG r%d] %-12s g=%+.6e (df.f=%+.3e f.f=%.3e)\n",
                rank, tag, h[1] > 0.0 ? h[0] / h[1] : 0.0, h[0], h[1]);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------
//  Helper: map FaceBc to integer kind for k_modify_tdma_row_gpu
// ---------------------------------------------------------------------------
static int modify_ghost_kind_mom(const mpmstd::boundary::FaceBc& face) {
    if (face.kind == mpmstd::boundary::BcKind::Neumann) return 2;
    if (face.ghost_policy == mpmstd::boundary::GhostPolicy::Antisymmetric) return 1;
    return 0;
}

// ---------------------------------------------------------------------------
//  Helper: get PaScaLTDMACudaBackend from registry
// ---------------------------------------------------------------------------
static mpmstd::linear_solver::tdma::PaScaLTDMACudaBackend&
cuda_tdma_mom(mpmstd::linear_solver::tdma::TdmaRegistry& tdma,
              mpmstd::Direction d)
{
    return static_cast<mpmstd::linear_solver::tdma::PaScaLTDMACudaBackend&>(
        tdma.get(d));
}

// ===========================================================================
//  ADI component (OB): 3-sweep ADI, writes result into `out_device`
//  Mirrors the adi_component anonymous function in momentum_cpu.cpp.
//  On entry:  d_rhs_buf holds the explicit RHS (device pointer)
//  On exit:   out_device contains the increment for this velocity component
// ===========================================================================
static void adi_component_ob(
        mpmstd::core::GpuMomentumSystem& mom,
        double* d_rhs_buf,         // device pointer to pre-filled RHS for this component
        const double* d_q,         // own velocity (device)
        const double* d_U, const double* d_V, const double* d_W,  // advecting velocities
        bool z_stag,
        const mpmstd::boundary::FieldBoundary& fbc,
        const mpmstd::core::Domain& domain,
        const mpmstd::boundary::BoundaryCondition& bc,
        double nu, double dt,
        double* out_device,        // destination GpuField.data()
        const mpmstd::core::GpuGridArrays* dg,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t)
{
    using mpmstd::Direction;

    double* in   = d_rhs_buf;
    double* outb = d_m_stage_;

    for (Direction dir : bc.topology.sweep_order()) {
        const double* dxa  = (dir == Direction::X) ? dg->d_dx1()
                           : (dir == Direction::Y) ? dg->d_dx2() : dg->d_dx3();
        const double* dmxa = (dir == Direction::X) ? dg->d_dmx1()
                           : (dir == Direction::Y) ? dg->d_dmx2() : dg->d_dmx3();
        const double* q_adv = (dir == Direction::X) ? d_U
                            : (dir == Direction::Y) ? d_V : d_W;
        const bool zs = (dir == Direction::Z) && z_stag;

        int n_row, n_sys;
        switch (dir) {
            case Direction::X: n_row = n1i; n_sys = n2i * n3i; break;
            case Direction::Y: n_row = n2i; n_sys = n1i * n3i; break;
            default:           n_row = n3i; n_sys = n1i * n2i; break;
        }

        // Build ADI bands (sys in x-thread → coalesced)
        const dim3 blk(32, 16);
        const dim3 grd((n_sys + 31)/32, (n_row + 15)/16);
        k_mom_adi_ob<<<grd, blk>>>(
                dir, d_m_A_, d_m_B_, d_m_C_, d_m_D_,
                in, d_q, q_adv, dxa, dmxa,
                n1i, n2i, n3i, n1t, n2t, n3t,
                nu, dt, zs);

        // BC modification (GPU kernel) + TDMA solve
        if (!bc.topology.is_periodic(dir)) {
            const auto& axis_dir = domain.sub.topology().axis(dir);
            const bool owns_lower = (axis_dir.west_rank == MPI_PROC_NULL);
            const bool owns_upper = (axis_dir.east_rank == MPI_PROC_NULL);
            const int lkind = modify_ghost_kind_mom(fbc.face(dir, mpmstd::Side::Minus));
            const int ukind = modify_ghost_kind_mom(fbc.face(dir, mpmstd::Side::Plus));
            {
                const int blk1d = 256;
                const int grd1d = (n_sys + blk1d - 1) / blk1d;
                k_modify_tdma_row_gpu<<<grd1d, blk1d>>>(d_m_A_, d_m_B_, d_m_C_, n_sys, n_row,
                                                         owns_lower, owns_upper, lkind, ukind);
                if (zs && owns_lower) {   // lower z-wall W-face row → identity (Report/30)
                    k_w_wallface_identity_row<<<grd1d, blk1d>>>(d_m_A_, d_m_B_, d_m_C_, d_m_D_, n_sys);
                }
            }
            cuda_tdma_mom(domain.tdma, dir).solve_many_device(d_m_A_, d_m_B_, d_m_C_, d_m_D_, n_sys, n_row);
        } else {
            cuda_tdma_mom(domain.tdma, dir).solve_many_cyclic_device(d_m_A_, d_m_B_, d_m_C_, d_m_D_, n_sys, n_row);
        }

        // Scatter (same grid: sys in x-thread)
        k_m_scatter<<<grd, blk>>>(dir, outb, d_m_D_,
                                   n1i, n2i, n3i, n1t, n2t, n3t);
        std::swap(in, outb);
    }

    // `in` now holds the final result; copy into the GpuField
    const std::size_t nf = (std::size_t)n1t * n2t * n3t;
    const int blk1 = 256;
    const int grd1 = (int)((nf + blk1 - 1) / blk1);
    k_m_copy_field<<<grd1, blk1>>>(out_device, in, (int)nf);
    CUDA_CHK(cudaDeviceSynchronize());  // sync before returning — caller reads out_device
}

// ===========================================================================
//  ADI component (NOB): 3-sweep var-visc ADI, writes result into out_device
// ===========================================================================
static void adi_component_var(
        mpmstd::core::GpuMomentumSystem& mom,
        double* d_rhs_buf,
        const double* d_q,
        const double* d_U, const double* d_V, const double* d_W,
        const double* d_mu, const double* d_irho,
        mpmstd::Direction own_dir,
        bool z_stag,
        const mpmstd::boundary::FieldBoundary& fbc,
        const mpmstd::core::Domain& domain,
        const mpmstd::boundary::BoundaryCondition& bc,
        double Cmu, double dt,
        double* out_device,
        const mpmstd::core::GpuGridArrays* dg,
        int n1i, int n2i, int n3i,
        int n1t, int n2t, int n3t,
        const char* dbg_tag = nullptr)
{
    using mpmstd::Direction;

    const double* dxo  = (own_dir == Direction::X) ? dg->d_dx1()
                       : (own_dir == Direction::Y) ? dg->d_dx2() : dg->d_dx3();
    const double* dmxo = (own_dir == Direction::X) ? dg->d_dmx1()
                       : (own_dir == Direction::Y) ? dg->d_dmx2() : dg->d_dmx3();

    double* in   = d_rhs_buf;
    double* outb = d_m_stage_;

    for (Direction dir : bc.topology.sweep_order()) {
        const double* dxa  = (dir == Direction::X) ? dg->d_dx1()
                           : (dir == Direction::Y) ? dg->d_dx2() : dg->d_dx3();
        const double* dmxa = (dir == Direction::X) ? dg->d_dmx1()
                           : (dir == Direction::Y) ? dg->d_dmx2() : dg->d_dmx3();
        const double* q_adv = (dir == Direction::X) ? d_U
                            : (dir == Direction::Y) ? d_V : d_W;
        const bool zs = (dir == Direction::Z) && z_stag;

        int n_row, n_sys;
        switch (dir) {
            case Direction::X: n_row = n1i; n_sys = n2i * n3i; break;
            case Direction::Y: n_row = n2i; n_sys = n1i * n3i; break;
            default:           n_row = n3i; n_sys = n1i * n2i; break;
        }

        const dim3 blk(32, 16);
        const dim3 grd((n_sys + 31)/32, (n_row + 15)/16);
        k_mom_adi_var<<<grd, blk>>>(
                dir, own_dir,
                d_m_A_, d_m_B_, d_m_C_, d_m_D_,
                in, d_q, q_adv, d_mu, d_irho,
                dxa, dmxa, dxo, dmxo,
                n1i, n2i, n3i, n1t, n2t, n3t,
                Cmu, dt, zs);

        if (!bc.topology.is_periodic(dir)) {
            const auto& axis_dir = domain.sub.topology().axis(dir);
            const bool owns_lower = (axis_dir.west_rank == MPI_PROC_NULL);
            const bool owns_upper = (axis_dir.east_rank == MPI_PROC_NULL);
            const int lkind = modify_ghost_kind_mom(fbc.face(dir, mpmstd::Side::Minus));
            const int ukind = modify_ghost_kind_mom(fbc.face(dir, mpmstd::Side::Plus));
            {
                const int blk1d = 256;
                const int grd1d = (n_sys + blk1d - 1) / blk1d;
                k_modify_tdma_row_gpu<<<grd1d, blk1d>>>(d_m_A_, d_m_B_, d_m_C_, n_sys, n_row,
                                                         owns_lower, owns_upper, lkind, ukind);
                if (zs && owns_lower) {   // lower z-wall W-face row → identity (Report/30)
                    k_w_wallface_identity_row<<<grd1d, blk1d>>>(d_m_A_, d_m_B_, d_m_C_, d_m_D_, n_sys);
                }
            }
            cuda_tdma_mom(domain.tdma, dir).solve_many_device(d_m_A_, d_m_B_, d_m_C_, d_m_D_, n_sys, n_row);
        } else {
            cuda_tdma_mom(domain.tdma, dir).solve_many_cyclic_device(d_m_A_, d_m_B_, d_m_C_, d_m_D_, n_sys, n_row);
        }

        k_m_scatter<<<grd, blk>>>(dir, outb, d_m_D_,
                                   n1i, n2i, n3i, n1t, n2t, n3t);
        std::swap(in, outb);

        if (dbg_tag && mom_dbg_on()) {
            char tg[32];
            std::snprintf(tg, sizeof tg, "%s.%c", dbg_tag,
                          dir == Direction::X ? 'x' : dir == Direction::Y ? 'y' : 'z');
            mom_dbg_slabmax(tg, in, domain, n1i, n2i, n3i, n2t, n3t);
        }
    }

    const std::size_t nf = (std::size_t)n1t * n2t * n3t;
    const int blk1 = 256;
    const int grd1 = (int)((nf + blk1 - 1) / blk1);
    k_m_copy_field<<<grd1, blk1>>>(out_device, in, (int)nf);
    CUDA_CHK(cudaDeviceSynchronize());  // sync before returning — caller reads out_device
}

#endif // MPMSTD_BACKEND_CUDA

// ============================================================================
//  Public GPU functions
// ============================================================================

namespace mpmstd::equation::momentum {

// ---------------------------------------------------------------------------
//  assemble_momentum_const_visc_gpu  — OB const-property wrapper.
//  *** DEAD FOR DHVC *** (see the OB-kernels banner above). LIVE ONLY for the
//  rbc_gpu_ob binary. DHVC uses assemble_momentum_var_visc_gpu instead.
//  Fills mom.dU/dV/dW with the OB explicit BW-ADI RHS.
//  Buoyancy is added to mom.dW between assemble and solve.
// ---------------------------------------------------------------------------
void assemble_momentum_const_visc_gpu(const core::Domain& domain, core::GpuFields& fields,
                                       core::GpuMomentumSystem& mom, real_t dt)
{
#ifdef MPMSTD_BACKEND_CUDA
    const auto nt = domain.sub.n_total();
    const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
    const auto ni = domain.sub.n_interior();
    const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
    const std::size_t nf = (std::size_t)n1t * n2t * n3t;
    const std::size_t nb = (std::size_t)n1i * n2i * n3i;

    const double nu = static_cast<double>(fields.constant(core::Const::nu));
    ensure_mom_bufs(nf, nb);

    const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
    const double* d_U = fields[core::Var::U].data();
    const double* d_V = fields[core::Var::V].data();
    const double* d_W = fields[core::Var::W].data();

    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i+7)/8, (n2i+7)/8, (n3i+7)/8);

    // rhs staged directly in mom.dU/dV/dW so add_buoyancy_force_gpu sees them
    k_mom_rhs_ob<<<grd, blk>>>(
            mom.dU.data(), d_U, d_U, d_V, d_W,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, nu, static_cast<double>(dt), false);

    k_mom_rhs_ob<<<grd, blk>>>(
            mom.dV.data(), d_V, d_U, d_V, d_W,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, nu, static_cast<double>(dt), false);

    k_mom_rhs_ob<<<grd, blk>>>(
            mom.dW.data(), d_W, d_U, d_V, d_W,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, nu, static_cast<double>(dt), true);
    CUDA_CHK(cudaDeviceSynchronize());  // sync before returning — add_buoyancy_force reads dW
#endif
}

// ---------------------------------------------------------------------------
//  assemble_momentum_var_visc_gpu (NOB)
// ---------------------------------------------------------------------------
void assemble_momentum_var_visc_gpu(const core::Domain& domain, core::GpuFields& fields,
                                     const core::GpuField& mu_field, const core::GpuField& inv_rho_field,
                                     core::GpuMomentumSystem& mom, real_t dt)
{
#ifdef MPMSTD_BACKEND_CUDA
    const auto nt = domain.sub.n_total();
    const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
    const auto ni = domain.sub.n_interior();
    const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
    const std::size_t nf = (std::size_t)n1t * n2t * n3t;
    const std::size_t nb = (std::size_t)n1i * n2i * n3i;

    const double Cmu = static_cast<double>(fields.constant(core::Const::nu));
    ensure_mom_bufs(nf, nb);

    const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
    const double* d_U    = fields[core::Var::U].data();
    const double* d_V    = fields[core::Var::V].data();
    const double* d_W    = fields[core::Var::W].data();
    const double* d_mu   = mu_field.data();
    const double* d_irho = inv_rho_field.data();

    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i+7)/8, (n2i+7)/8, (n3i+7)/8);

    // rhs staged in mom.dU/dV/dW so buoyancy can be added between assemble+solve
    k_mom_rhs_var<<<grd, blk>>>(
            mom.dU.data(), d_U, d_U, d_V, d_W, d_mu, d_irho,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, Cmu, static_cast<double>(dt),
            Direction::X);
    CUDA_CHK(cudaDeviceSynchronize());

    k_mom_rhs_var<<<grd, blk>>>(
            mom.dV.data(), d_V, d_U, d_V, d_W, d_mu, d_irho,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, Cmu, static_cast<double>(dt),
            Direction::Y);
    CUDA_CHK(cudaDeviceSynchronize());

    k_mom_rhs_var<<<grd, blk>>>(
            mom.dW.data(), d_W, d_U, d_V, d_W, d_mu, d_irho,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, Cmu, static_cast<double>(dt),
            Direction::Z);
    CUDA_CHK(cudaDeviceSynchronize());
#endif
}

// ---------------------------------------------------------------------------
//  solve_momentum_gpu (OB)
//
//  Sequence (mirrors solve_momentum_cpu):
//    adi_component(rhs_u, own=X, z_stag=false) → dU
//    adi_component(rhs_v, own=Y, z_stag=false) → dV
//    adi_component(rhs_w, own=Z, z_stag=true)  → dW
//    sync dU (bc.U), sync dV (bc.V), sync dW (bc.W)
//    block_couple_dV(dV ← dW, V, ...)
//    sync dV (bc.V)
//    block_couple_dU(dU ← dV, dW, U, ...)
// ---------------------------------------------------------------------------
void solve_momentum_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                         core::GpuFields& fields, core::GpuMomentumSystem& mom, real_t dt)
{
#ifdef MPMSTD_BACKEND_CUDA
    const auto nt = domain.sub.n_total();
    const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
    const auto ni = domain.sub.n_interior();
    const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
    const std::size_t nf = (std::size_t)n1t * n2t * n3t;
    const std::size_t nb = (std::size_t)n1i * n2i * n3i;

    const double nu = static_cast<double>(fields.constant(core::Const::nu));
    ensure_mom_bufs(nf, nb);

    const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
    const double* d_U = fields[core::Var::U].data();
    const double* d_V = fields[core::Var::V].data();
    const double* d_W = fields[core::Var::W].data();

    // RHS is staged in mom.dU/dV/dW by assemble (+ buoyancy); ADI overwrites them
    adi_component_ob(mom, mom.dU.data(), d_U, d_U, d_V, d_W, false,
                     bc.U, domain, bc, nu, static_cast<double>(dt),
                     mom.dU.data(), dg, n1i, n2i, n3i, n1t, n2t, n3t);

    adi_component_ob(mom, mom.dV.data(), d_V, d_U, d_V, d_W, false,
                     bc.V, domain, bc, nu, static_cast<double>(dt),
                     mom.dV.data(), dg, n1i, n2i, n3i, n1t, n2t, n3t);

    adi_component_ob(mom, mom.dW.data(), d_W, d_U, d_V, d_W, true,
                     bc.W, domain, bc, nu, static_cast<double>(dt),
                     mom.dW.data(), dg, n1i, n2i, n3i, n1t, n2t, n3t);

    // ---- sync increments --------------------------------------------------
    core::sync_field_gpu(mom.dU, bc.U, domain.sub);
    core::sync_field_gpu(mom.dV, bc.V, domain.sub);
    core::sync_field_gpu(mom.dW, bc.W, domain.sub);

    // ---- block_couple_dV  (dV -= dt·0.25·dW·∂V/∂z) -----------------------
    {
        const dim3 blk(8, 8, 8);
        const dim3 grd((n1i+7)/8, (n2i+7)/8, (n3i+7)/8);
        k_block_couple_dV_ob<<<grd, blk>>>(
                mom.dV.data(), mom.dW.data(), d_V,
                dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
                n1i, n2i, n3i, n1t, n2t, n3t, static_cast<double>(dt));
        CUDA_CHK(cudaDeviceSynchronize());
    }
    core::sync_field_gpu(mom.dV, bc.V, domain.sub);

    // ---- block_couple_dU  (dU -= dt·0.25·(dV·∂U/∂y + dW·∂U/∂z)) --------
    {
        const dim3 blk(8, 8, 8);
        const dim3 grd((n1i+7)/8, (n2i+7)/8, (n3i+7)/8);
        k_block_couple_dU_ob<<<grd, blk>>>(
                mom.dU.data(), mom.dV.data(), mom.dW.data(), d_U,
                dg->d_dx1(), dg->d_dmx1(), dg->d_dmx2(), dg->d_dmx3(),
                n1i, n2i, n3i, n1t, n2t, n3t, static_cast<double>(dt));
        CUDA_CHK(cudaDeviceSynchronize());
    }
#endif
}

// ---------------------------------------------------------------------------
//  solve_momentum_var_gpu (NOB)
//
//  Sequence (mirrors solve_momentum_var_cpu; ref MPM-STD submodule.f90:202-227
//  arrangement — wall-normal dW solved LAST, all its couplings pre-injected
//  into the RHS so they pass the implicit z-ADI; explicit post-hoc corrections
//  go to the horizontal dV/dU only. Report/RBC wall-reignition analysis):
//    adi_component_var(rhs_u, own=X) → dU
//    sync dU
//    k_dd_dV_var(rhs_v ← dU, ...)                 [M21]
//    adi_component_var(rhs_v, own=Y) → dV
//    sync dV
//    k_dd_dW_var(rhs_w ← dU, dV, ...)             [M31 + M32]
//    adi_component_var(rhs_w, own=Z, z_stag=true) → dW
//    sync dW, pin lower wall face
//    k_block_dV_var(dV ← dW, ...)                 [M23, post-hoc]
//    sync dV
//    k_block_dU_var(dU ← dV, dW, ...)             [M12 + M13, post-hoc]
// ---------------------------------------------------------------------------
void solve_momentum_var_gpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                              core::GpuFields& fields,
                              const core::GpuField& mu_field, const core::GpuField& inv_rho_field,
                              core::GpuMomentumSystem& mom, real_t dt)
{
#ifdef MPMSTD_BACKEND_CUDA
    const auto nt = domain.sub.n_total();
    const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
    const auto ni = domain.sub.n_interior();
    const int n1i = ni[0], n2i = ni[1], n3i = ni[2];
    const std::size_t nf = (std::size_t)n1t * n2t * n3t;
    const std::size_t nb = (std::size_t)n1i * n2i * n3i;

    const double Cmu = static_cast<double>(fields.constant(core::Const::nu));
    ensure_mom_bufs(nf, nb);

    const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
    const double* d_U    = fields[core::Var::U].data();
    const double* d_V    = fields[core::Var::V].data();
    const double* d_W    = fields[core::Var::W].data();
    const double* d_mu   = mu_field.data();
    const double* d_irho = inv_rho_field.data();

    const dim3 blk3(8, 8, 8);
    const dim3 grd3((n1i+7)/8, (n2i+7)/8, (n3i+7)/8);

    // RHS staged in mom.dU/dV/dW by assemble_var_visc; ADI overwrites them.
    mom_dbg_slabmax("rhsU", mom.dU.data(), domain, n1i, n2i, n3i, n2t, n3t);
    mom_dbg_slabmax("rhsV", mom.dV.data(), domain, n1i, n2i, n3i, n2t, n3t);
    mom_dbg_slabmax("rhsW", mom.dW.data(), domain, n1i, n2i, n3i, n2t, n3t);

    // (1) dU — own=X, no dd* coupling
    adi_component_var(mom, mom.dU.data(), d_U, d_U, d_V, d_W, d_mu, d_irho,
                      Direction::X, false, bc.U, domain, bc,
                      Cmu, static_cast<double>(dt),
                      mom.dU.data(), dg, n1i, n2i, n3i, n1t, n2t, n3t, "adiU");
    core::sync_field_gpu(mom.dU, bc.U, domain.sub);
    mom_dbg_slabcorr("g.adiU", mom.dU.data(), d_U, domain, n1i, n2i, n3i, n2t, n3t);

    // (2) dd* M21: rhs_v (in mom.dV) += coupling(dU); then ADI → dV
    k_dd_dV_var<<<grd3, blk3>>>(
            mom.dV.data(), mom.dU.data(), d_V, d_mu, d_irho,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(),
            n1i, n2i, n3i, n1t, n2t, n3t, Cmu, static_cast<double>(dt));
    CUDA_CHK(cudaDeviceSynchronize());
    mom_dbg_slabmax("rhsV+ddU", mom.dV.data(), domain, n1i, n2i, n3i, n2t, n3t);

    adi_component_var(mom, mom.dV.data(), d_V, d_U, d_V, d_W, d_mu, d_irho,
                      Direction::Y, false, bc.V, domain, bc,
                      Cmu, static_cast<double>(dt),
                      mom.dV.data(), dg, n1i, n2i, n3i, n1t, n2t, n3t, "adiV");
    core::sync_field_gpu(mom.dV, bc.V, domain.sub);
    mom_dbg_slabcorr("g.adiV", mom.dV.data(), d_V, domain, n1i, n2i, n3i, n2t, n3t);

    // (3) dd* M31+M32: rhs_w (in mom.dW) += coupling(dU,dV); then ADI → dW.
    //     The wall-normal solve comes LAST so every coupling into it passes
    //     through its implicit z-ADI (ref solvedW; Report/RBC).
    k_dd_dW_var<<<grd3, blk3>>>(
            mom.dW.data(), mom.dU.data(), mom.dV.data(), d_W, d_mu, d_irho,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, Cmu, static_cast<double>(dt));
    CUDA_CHK(cudaDeviceSynchronize());
    mom_dbg_slabmax("rhsW+dd", mom.dW.data(), domain, n1i, n2i, n3i, n2t, n3t);

    adi_component_var(mom, mom.dW.data(), d_W, d_U, d_V, d_W, d_mu, d_irho,
                      Direction::Z, true, bc.W, domain, bc,
                      Cmu, static_cast<double>(dt),
                      mom.dW.data(), dg, n1i, n2i, n3i, n1t, n2t, n3t, "adiW");
    core::sync_field_gpu(mom.dW, bc.W, domain.sub);
    mom_dbg_slabcorr("g.adiW", mom.dW.data(), d_W, domain, n1i, n2i, n3i, n2t, n3t);

    const bool z_wall_lower = !bc.topology.is_periodic(Direction::Z)
        && (domain.sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
    const dim3 blkw(16, 16);
    const dim3 grdw((n1t + 15)/16, (n2t + 15)/16);
    if (z_wall_lower) {
        k_zero_dw_wall_face<<<grdw, blkw>>>(mom.dW.data(), n1t, n2t, n3t);
        CUDA_CHK(cudaDeviceSynchronize());
    }

    // E9b bisection (MPMSTD_BLKMASK_TOP=1): make the block-coupling stages a
    // no-op in the top-wall slab by restoring pre-stage planes afterwards.
    // d_m_stage_ is free here (adi_component_var is done with it).
    const bool blkmask_top = mom_blkmask_top_planes() > 0
        && !bc.topology.is_periodic(Direction::Z)
        && (domain.sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);
    const double blk_theta = mom_blkrelax_theta();
    const bool blk_copy = blkmask_top || blk_theta < 1.0;
    const int k_slab0_top = n3i + kH - mom_blkmask_top_planes();
    const dim3 blkr(8, 8, 8);
    const dim3 grdr((n1t + 7) / 8, (n2t + 7) / 8, (n3t + 7) / 8);
    const std::size_t nf_full = (std::size_t)n1t * n2t * n3t;
    const int grd1f = (int)((nf_full + 255) / 256);

    // ---- (4) block_couple_dV_var (dV ← dW) --------------------------------
    if (blk_copy) {
        k_m_copy_field<<<grd1f, 256>>>(d_m_stage_, mom.dV.data(), (int)nf_full);
        CUDA_CHK(cudaDeviceSynchronize());
    }
    k_block_dV_var<<<grd3, blk3>>>(
            mom.dV.data(), mom.dW.data(), d_V, d_mu, d_irho,
            dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, Cmu, static_cast<double>(dt));
    CUDA_CHK(cudaDeviceSynchronize());
    core::sync_field_gpu(mom.dV, bc.V, domain.sub);
    if (blk_theta < 1.0) {
        k_dbg_blend_full<<<grd1f, 256>>>(mom.dV.data(), d_m_stage_, blk_theta, (int)nf_full);
        CUDA_CHK(cudaDeviceSynchronize());
    }
    if (blkmask_top) {
        k_dbg_restore_top<<<grdr, blkr>>>(mom.dV.data(), d_m_stage_,
                                          n1t, n2t, n3t, k_slab0_top);
        CUDA_CHK(cudaDeviceSynchronize());
    }
    mom_dbg_slabmax("dV.blk", mom.dV.data(), domain, n1i, n2i, n3i, n2t, n3t);
    mom_dbg_slabcorr("g.blkV", mom.dV.data(), d_V, domain, n1i, n2i, n3i, n2t, n3t);

    // ---- (5) block_couple_dU_var (dU ← dV, dW) ----------------------------
    if (blk_copy) {
        k_m_copy_field<<<grd1f, 256>>>(d_m_stage_, mom.dU.data(), (int)nf_full);
        CUDA_CHK(cudaDeviceSynchronize());
    }
    k_block_dU_var<<<grd3, blk3>>>(
            mom.dU.data(), mom.dV.data(), mom.dW.data(), d_U, d_mu, d_irho,
            dg->d_dx1(), dg->d_dmx1(), dg->d_dx2(), dg->d_dmx2(), dg->d_dx3(), dg->d_dmx3(),
            n1i, n2i, n3i, n1t, n2t, n3t, Cmu, static_cast<double>(dt));
    CUDA_CHK(cudaDeviceSynchronize());
    if (blk_theta < 1.0) {
        k_dbg_blend_full<<<grd1f, 256>>>(mom.dU.data(), d_m_stage_, blk_theta, (int)nf_full);
        CUDA_CHK(cudaDeviceSynchronize());
    }
    if (blkmask_top) {
        k_dbg_restore_top<<<grdr, blkr>>>(mom.dU.data(), d_m_stage_,
                                          n1t, n2t, n3t, k_slab0_top);
        CUDA_CHK(cudaDeviceSynchronize());
    }
    mom_dbg_slabmax("dU.blk", mom.dU.data(), domain, n1i, n2i, n3i, n2t, n3t);
    mom_dbg_slabcorr("g.blkU", mom.dU.data(), d_U, domain, n1i, n2i, n3i, n2t, n3t);
#endif
}

// ---------------------------------------------------------------------------
//  update_velocity_gpu  — U += dU, V += dV, W += dW  (interior cells)
// ---------------------------------------------------------------------------
void update_velocity_gpu(core::GpuFields& fields, const core::GpuMomentumSystem& mom)
{
#ifdef MPMSTD_BACKEND_CUDA
    const auto nt = mom.n_total;
    const int n1t = nt[0], n2t = nt[1], n3t = nt[2];
    const auto ni = mom.n_interior;
    const int n1i = ni[0], n2i = ni[1], n3i = ni[2];

    const dim3 blk(8, 8, 8);
    const dim3 grd((n1i+7)/8, (n2i+7)/8, (n3i+7)/8);

    k_m_add_increment<<<grd, blk>>>(
            fields[core::Var::U].data(), mom.dU.data(),
            n1i, n2i, n3i, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());

    k_m_add_increment<<<grd, blk>>>(
            fields[core::Var::V].data(), mom.dV.data(),
            n1i, n2i, n3i, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());

    k_m_add_increment<<<grd, blk>>>(
            fields[core::Var::W].data(), mom.dW.data(),
            n1i, n2i, n3i, n2t, n3t);
    CUDA_CHK(cudaDeviceSynchronize());
#endif
}

// ---------------------------------------------------------------------------
//  debug_slabmax_gpu — Report/30 §6 instrumentation hook for the time loop
//  (post-update / post-projection probes on U, W). No-op unless MPMSTD_MOMDBG=1.
// ---------------------------------------------------------------------------
void debug_slabmax_gpu(const char* tag, const core::GpuField& f, const core::Domain& domain,
                       char argmax_tag)
{
#ifdef MPMSTD_BACKEND_CUDA
    const auto nt = domain.sub.n_total();
    const auto ni = domain.sub.n_interior();
    mom_dbg_slabmax(tag, f.data(), domain, ni[0], ni[1], ni[2], nt[1], nt[2], argmax_tag);
#else
    (void)tag; (void)f; (void)domain; (void)argmax_tag;
#endif
}

} // namespace mpmstd::equation::momentum
