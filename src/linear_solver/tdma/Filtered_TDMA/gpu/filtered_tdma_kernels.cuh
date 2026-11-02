#ifndef FILTERED_TDMA_KERNELS_CUH
#define FILTERED_TDMA_KERNELS_CUH

#include "../include/filtered_tdma_findmax.cuh"

// Private CUDA kernel definitions shared between filtered_tdma_cuda.cu (production)
// and filtered_tdma_profile_cuda.cu (profiling). Each TU gets its own static copy
// of the device code; no -rdc required.

#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

static constexpr int FTDMA_BLOCK_SYS = 256;

static inline dim3 ftdma_grid1D(int n) {
    return dim3((n + FTDMA_BLOCK_SYS - 1) / FTDMA_BLOCK_SYS);
}

__device__ static inline double ftdma_warp_max(double v) {
    for (int s = 16; s > 0; s >>= 1)
        v = fmax(v, __shfl_down_sync(0xffffffffu, v, s));
    return v;
}

// ---------------------------------------------------------------------------
//  Kernels
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
//  Pre-comm kernel — forward sweep, backward sweep, the row-0 2x2 reduction
//  and both send-buffer packs, fused into ONE kernel.
//
//  This is the structure pascal's tdma_modified_kernel already uses (one kernel
//  carrying rows through registers). The fusion is exact, not an approximation:
//
//    * the backward sweep never touches row n_row-1, so packing C/D of that row
//      from the registers the forward loop ends with is the same value the
//      separate k_pack_lastrow read back from global;
//    * k_pack_row0 read A[0]/D[0] straight after the backward tail wrote them,
//      so packing from those registers is likewise identical;
//    * every step is per-system (thread i touches only column i), so no
//      cross-thread ordering is lost by dropping the kernel boundaries.
//
//  Result: 4 launches -> 1, and the two boundary rows are no longer re-read.
// ---------------------------------------------------------------------------

static __global__ __launch_bounds__(FTDMA_BLOCK_SYS)
void k_pre(double* A, double* B, double* C, double* D,
              double* __restrict__ send_right, double* __restrict__ send_left,
              int n_sys, int n_row, const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;
    const int J = __ldg(d_J);

    // --- forward elimination (full to J, then A-updates skipped) ------------
    double r0 = 1.0 / B[i];
    A[i] *= r0;  C[i] *= r0;  D[i] *= r0;

    std::size_t off1 = (std::size_t)1 * n_sys + i;
    double r1 = 1.0 / B[off1];
    A[off1] *= r1;  C[off1] *= r1;  D[off1] *= r1;

    double a_prev = A[off1], c_prev = C[off1], d_prev = D[off1];

    for (int j = 2; j <= J; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new =  inv * (Dj - Aj * d_prev);
        double c_new =  inv * Cj;
        double a_new = -inv * Aj * a_prev;
        D[off_j] = d_new;  C[off_j] = c_new;  A[off_j] = a_new;
        a_prev = a_new;  c_prev = c_new;  d_prev = d_new;
    }
    // Rows 0 and 1 were normalised above, outside the loop, so the A-carrying
    // pass covers [2, J] and this one must resume at max(J+1, 2).  Writing
    // `J + 1` alone breaks at J = 0: the loop would restart at row 1 and
    // eliminate it a second time against its own C.  The max is loop-invariant
    // and J is a warp-uniform scalar, so it costs one integer op, not a branch.
    for (int j = (J + 1 > 2 ? J + 1 : 2); j < n_row; ++j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Aj = A[off_j], Bj = B[off_j], Cj = C[off_j], Dj = D[off_j];
        double inv  = 1.0 / (Bj - Aj * c_prev);
        double d_new = inv * (Dj - Aj * d_prev);
        double c_new = inv * Cj;
        D[off_j] = d_new;  C[off_j] = c_new;
        c_prev = c_new;  d_prev = d_new;
    }
    send_right[i]         = c_prev;
    send_right[n_sys + i] = d_prev;

    // --- backward substitution (A only below J, C only above lo) ------------
    const int lo = (n_row - 2) - J;
    std::size_t off_nm2 = (std::size_t)(n_row - 2) * n_sys + i;
    double d_next = D[off_nm2];
    double c_next = C[off_nm2];
    double a_next = (J >= 2) ? A[(std::size_t)J * n_sys + i] : 0.0;

    for (int j = n_row - 3; j >= 1; --j) {
        std::size_t off_j = (std::size_t)j * n_sys + i;
        double Cj = C[off_j];
        double d_new = D[off_j] - Cj * d_next;
        D[off_j] = d_new;
        d_next = d_new;
        if (J >= 2 && j < J) {
            double a_new = A[off_j] - Cj * a_next;
            A[off_j] = a_new;
            a_next = a_new;
        }
        if (j > lo) {
            double c_new = -Cj * c_next;
            C[off_j] = c_new;
            c_next = c_new;
        }
    }

    // --- row-0 2x2 reduction + pack for the left neighbour ------------------
    double A1     = A[(std::size_t)1 * n_sys + i];
    double C0     = C[i];
    double D0_val = D[i];
    double D1     = D[(std::size_t)1 * n_sys + i];
    double r      = 1.0 / (1.0 - A1 * C0);
    double d0_new = r * (D0_val - C0 * D1);
    double a0_new = r * A[i];
    D[i] = d0_new;
    A[i] = a0_new;

    send_left[i]         = a0_new;
    send_left[n_sys + i] = d0_new;
}

// ---------------------------------------------------------------------------
//  Post-comm kernel — both interface 2x2 solves and the J-truncated boundary
//  correction, fused. Legal because thread i writes D[0] / D[n_row-1] and then
//  reads back exactly those two values; no other thread reads them. The final
//  correction only writes rows 1..n_row-2, so it never clobbers them either.
//
//  J is known before the exchange, so nothing has to run between the interface
//  solve and the correction — hence the fusion.
// ---------------------------------------------------------------------------

static __global__ __launch_bounds__(FTDMA_BLOCK_SYS, 4)
void k_post(double* D, const double* A, const double* C,
               const double* __restrict__ C_left, const double* __restrict__ D_left,
               const double* __restrict__ A_right, const double* __restrict__ D0_right,
               int has_left, int has_right, int n_sys, int n_row,
               const int* __restrict__ d_J) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_sys) return;

    double d0 = D[i];
    if (has_left) {
        double A0 = A[i];
        d0 = (d0 - A0 * D_left[i]) / (1.0 - C_left[i] * A0);
        D[i] = d0;
    }
    std::size_t offN = (std::size_t)(n_row - 1) * n_sys + i;
    double dn = D[offN];
    if (has_right) {
        double CN = C[offN];
        dn = (dn - CN * D0_right[i]) / (1.0 - A_right[i] * CN);
        D[offN] = dn;
    }

    const int J = __ldg(d_J);
    int jL = J < n_row - 2 ? J : n_row - 2;
    int jR = (n_row - 1) - J;
    if (jR < 1) jR = 1;

    for (int j = 1; j <= jL; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] -= A[off] * d0;
    }
    for (int j = jR; j < n_row - 1; ++j) {
        std::size_t off = (std::size_t)j * n_sys + i;
        D[off] -= C[off] * dn;
    }
}

// 판정 본체.  두 진입점(사후 스캔 / 융합 통계)이 같은 코드를 쓰도록 분리한다.
__device__ static inline void ftdma_select_J_dev(double rho, double d_inf,
                                                 double eps, int n_row,
                                                 int* __restrict__ d_J) {
    const int Jmax = n_row - 2;
    
    int  J        = 0;
    bool fallback = false;

    if (Jmax < 0) {
        fallback = true;
    } else if (!(d_inf > 0.0)) {
        J = 0;                              // zero RHS: exact at any depth
    } else if (!(eps > 0.0) || !(rho < 0.5)) {
        // eps <= 0 permits no truncation; rho >= 1/2 is outside the theory
        // (lambda+ complex, and the 1/(1-2rho) reduced-system bound blows up).
        fallback = true; J = Jmax;
    } else if (rho <= 0.0) {
        J = 0;                              // purely diagonal -> q = 0
    } else {
        const double lam = 0.5 * (1.0 + sqrt(1.0 - 4.0 * rho * rho));
        const double q   = rho / lam;

        if (!(q > 0.0)) {
            J = 0;
        } else if (!(q < 1.0)) {
            fallback = true; J = Jmax;
        } else {
            const double omq  = 1.0 - q;
            const double q2   = q * q;
            const double omq2 = 1.0 - q2;
            const double om2r = 1.0 - 2.0 * rho;
            const double omr  = 1.0 - rho;

            const double C1 = q * (2.0 + q) * omr / (omq * omq2 * om2r * om2r);
            const double C2 = omr * (omq2 * om2r + rho)
                            / (omq2 * omq2 * omq2 * om2r * om2r * om2r);
            const double C3 = omr / (omq2 * om2r * om2r * om2r);

            // Floor (24).  q^(Ns-1) may underflow to 0 for large Ns — that
            // is the floor vanishing, which is correct.
            const double eps_min =
                2.0 * C3 * pow(q, (double)(n_row - 1)) * d_inf / omq;

            if (!(eps > eps_min)) {
                // Feasibility (25) fails: no depth reaches this tolerance.
                fallback = true; J = Jmax;
            } else {
                // Margin the floor leaves (26).
                const double r = (eps - eps_min) / d_inf;
                const double alpha = 2.0 * C2 * q2 / omq;
                // Cancellation-free form of the positive root; see the CPU
                // twin for why the textbook form loses its digits here.
                const double disc   = sqrt(C1 * C1 + 4.0 * alpha * r);
                const double y_plus = 2.0 * r / (C1 + disc);

                if (!(y_plus > 0.0)) {
                    fallback = true; J = Jmax;
                } else if (y_plus >= 1.0) {
                    J = 0;                  // E(0) already within tolerance
                } else {
                    const double Jr = ceil(log(y_plus) / log(q));
                    if (!(Jr > 0.0))            J = 0;
                    else if (Jr > (double)Jmax) { fallback = true; J = Jmax; }
                    else                        J = (int)Jr;
                }
            }
        }
    }

    // Valid range is  0 <= J <= n_row-2 .  Both ends are load-bearing:
    //   upper — the sweeps address rows J+1 and (n_row-2)-J, so a larger J walks
    //           off the ends of A/C/D, and the backward A-recursion would be
    //           seeded from A[n_row-1], skipping row n_row-2;
    //   lower — J = 0 is the legitimate limit "solve the interface system and
    //           apply no interior correction": k_post's two loops go empty on
    //           their own (jL = 0, jR = n_row-1), and the forward pass resumes
    //           at max(J+1, 2) so row 1 is not touched twice.
    // Assumes n_row >= 3, the same constraint the PaScaL port carries.
    if (J > n_row - 2) J = n_row - 2;
    if (J < 0)         J = 0;


    if (J > n_row - 2) J = n_row - 2;
    if (J < 0)         J = 0;
    d_J[0] = J;
    d_J[1] = fallback ? 1 : 0;
}

// Truncation-index criterion — Manuscript §5.4, eqs. (23)-(28).  This mirrors
// FilteredTDMA::cal_J_rhs_bound (cpu/filtered_tdma.cpp) exactly; the two must be
// edited together, or host and device will select different J.
//
//   E(J) = [ C1 q^J + 2/(1-q)( C2 q^(2J+2) + C3 q^(Ns-1) ) ] ||d||_inf   (23)
//
// with C1,C2,C3 of Appendix C.  The J-independent q^(Ns-1) term is a floor:
//   eps_min = 2 C3 q^(Ns-1) ||d||_inf/(1-q)                              (24)
// and eps > eps_min is the feasibility condition (25) — below it no depth can
// meet the tolerance and the solve should revert to PaScaL_TDMA.  Above it,
// with r = (eps-eps_min)/||d||_inf and y = q^J, E(J) <= eps is the quadratic
//   alpha y^2 + C1 y <= r,  alpha = 2 C2 q^2/(1-q)                       (26)
// so y <= y+ (the positive root) and, since ln q < 0,
//   J = max{0, ceil(ln(y+)/ln(q))}                                       (27)
// The outer max is what yields J = 0.  Admissibility is J <= Ns-2 (28); the two
// fallback reasons — infeasible tolerance, or a depth the subdomain cannot
// carry — both fall out of the criterion rather than being special-cased.
//
// d_J[0] = index (always clamped to [0, n_row-2] so an ignoring caller still
//                 gets a runnable value)
// d_J[1] = 1 when the criterion asked for PaScaL_TDMA instead
//
// Both sup-norms are scanned here, in one grid-stride pass over the WHOLE local
// system (n_row x n_sys):
//
//   rho       = max over all systems and all rows k >= 1 of
//               max(|A|,|C|)/|B|                        (row 0 excluded to match
//                                                        the sweep, which starts
//                                                        at k = 1)
//   ||d||_inf = max over the whole RHS
//
// Both are full scans, deliberately.  Earlier versions took shortcuts — |D| over
// the two interface rows only, rho over column 0 only — and both underestimate
// E(J), which selects too large a truncation and voids the tolerance guarantee
// the criterion exists to provide.  The column-0 rho additionally assumed every
// system shares one tridiagonal operator, which fails as soon as the
// coefficients vary across the other two directions.
//
// One block does the whole pass, so no scratch buffer and no second launch: the
// caller that cares about the traffic uses the fused path (k_cal_J_from_stats)
// instead, where the LHS-build kernel accumulates the same two norms for free.
//
// Runs before k_pre, so A/B/C/D still hold the original values.
static __global__ __launch_bounds__(FTDMA_BLOCK_SYS)
void k_cal_J_rhs_bound(int* __restrict__ d_J,
                       const double* __restrict__ A,
                       const double* __restrict__ B,
                       const double* __restrict__ C,
                       const double* __restrict__ D,
                       int n_sys, int n_row, double eps) {
    __shared__ double s_rho[FTDMA_BLOCK_SYS];
    __shared__ double s_d  [FTDMA_BLOCK_SYS];
    int tid = threadIdx.x;

    const std::size_t n = (std::size_t)n_row * n_sys;
    double my_rho = 0.0, my_d = 0.0;
    for (std::size_t i = tid; i < n; i += blockDim.x) {
        my_d = fmax(my_d, fabs(__ldg(&D[i])));
        if (i >= (std::size_t)n_sys) {           // row = i / n_sys >= 1
            const double bk = __ldg(&B[i]);
            if (bk != 0.0) {
                my_rho = fmax(my_rho, fmax(fabs(__ldg(&A[i]) / bk),
                                           fabs(__ldg(&C[i]) / bk)));
            }
        }
    }

    s_rho[tid] = my_rho; s_d[tid] = my_d;
    __syncthreads();
    for (int k = FTDMA_BLOCK_SYS / 2; k >= 32; k >>= 1) {
        if (tid < k) {
            s_rho[tid] = fmax(s_rho[tid], s_rho[tid + k]);
            s_d  [tid] = fmax(s_d  [tid], s_d  [tid + k]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        double a = ftdma_warp_max(s_rho[tid]);
        double b = ftdma_warp_max(s_d  [tid]);
        if (tid == 0) { s_rho[0] = a; s_d[0] = b; }
    }
    __syncthreads();

    if (tid == 0) ftdma_select_J_dev(s_rho[0], s_d[0], eps, n_row, d_J);
}


// ---------------------------------------------------------------------------
//  Fused-statistics entry point.  The caller's LHS-build kernel already
//  accumulated (rho, d_inf) into `stats` via ftdma_fm_block_accum(), so nothing
//  is re-read here: this kernel only folds the nslot slots and applies the
//  criterion.  One block, 2*nslot values.
//
//  Preferred over k_cal_J_rhs_bound, which re-reads A/B/C/D: that post-hoc scan
//  costs 4.6 % of a time step for D alone and 19.5 % for A,B,C,D — and the full
//  criterion needs all four — whereas the fused accumulation measured no slower
//  than the unmodified build.  See include/filtered_tdma_findmax.cuh and
//  apps/re_cal_J.  k_cal_J_rhs_bound stays as the path for callers whose build
//  kernel cannot host the reduction (and as the A/B check, FTDMA_NO_FUSE=1).
//
//  `rho` here and in k_cal_J_rhs_bound are the same quantity — the max over
//  ALL systems and all rows k >= 1 — so the two paths select the same J.
// ---------------------------------------------------------------------------
static __global__ void k_cal_J_from_stats(int* __restrict__ d_J,
                                          const unsigned long long* __restrict__ stats,
                                          int nslot, int n_row, double eps) {
    __shared__ double s_r[FTDMA_BLOCK_SYS];
    __shared__ double s_d[FTDMA_BLOCK_SYS];
    const int tid = threadIdx.x;

    double r = 0.0, d = 0.0;
    for (int i = tid; i < nslot; i += blockDim.x) {
        r = fmax(r, ftdma_fm_unbits(stats[2 * i + 0]));
        d = fmax(d, ftdma_fm_unbits(stats[2 * i + 1]));
    }
    s_r[tid] = r; s_d[tid] = d;
    __syncthreads();
    for (int k = FTDMA_BLOCK_SYS / 2; k >= 32; k >>= 1) {
        if (tid < k) { s_r[tid] = fmax(s_r[tid], s_r[tid + k]);
                       s_d[tid] = fmax(s_d[tid], s_d[tid + k]); }
        __syncthreads();
    }
    if (tid < 32) {
        double a = ftdma_warp_max(s_r[tid]);
        double b = ftdma_warp_max(s_d[tid]);
        if (tid == 0) ftdma_select_J_dev(a, b, eps, n_row, d_J);
    }
}

// ---------------------------------------------------------------------------
//  Multi-block scan — the honest cost of NOT fusing.
//
//  k_cal_J_rhs_bound above does the whole pass from one block, which uses a
//  single SM out of the device's 108 and is therefore not a fair baseline: it
//  measured ~1030 ms on a 514x514x512 subdomain, i.e. 3.2 GB/s, about 1/600 of
//  A100 HBM.  Comparing the fused path against that would be a straw man.
//
//  This version grid-strides over the same four arrays with a full grid and
//  folds per-block maxima through the same slot machinery the fused path uses,
//  so the two differ only in WHERE the sup-norms come from, not in how well the
//  reduction is written.  Pair it with k_cal_J_from_stats.
//
//  `stats` must be zeroed before launch (atomicMax starts from 0).
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(FTDMA_BLOCK_SYS)
void k_scan_abcd_stats(const double* __restrict__ A,
                       const double* __restrict__ B,
                       const double* __restrict__ C,
                       const double* __restrict__ D,
                       std::size_t n, int n_sys, int nslot,
                       unsigned long long* __restrict__ stats) {
    double my_rho = 0.0, my_d = 0.0;
    for (std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (std::size_t)gridDim.x * blockDim.x) {
        const int row = (int)(i / (std::size_t)n_sys);
        my_rho = fmax(my_rho, ftdma_fm_rho_of(__ldg(&A[i]), __ldg(&B[i]),
                                              __ldg(&C[i]), row));
        my_d   = fmax(my_d,   fabs(__ldg(&D[i])));
    }
    ftdma_fm_block_accum(stats, nslot, my_rho, my_d);
}

#endif // FILTERED_TDMA_KERNELS_CUH
