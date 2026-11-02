#ifndef FILTERED_TDMA_FINDMAX_CUH
#define FILTERED_TDMA_FINDMAX_CUH

// ---------------------------------------------------------------------------
//  The two sup-norms the cutoff criterion needs, accumulated INSIDE the caller's
//  LHS-build kernel from coefficients still in registers.
//
//    rho   = max_{k>=1, i} max( |A[k,i]|, |C[k,i]| ) / |B[k,i]|
//    d_inf = max_{k,   i}  |D[k,i]|
//
//  Why fused rather than a separate pass: measured on A100 over the six
//  per-rank shapes of the scalability study (513x513xn_row, n_row = 64..2048,
//  16.8-539 M cells, 3 repetitions each; apps/re_cal_J), obtaining these from a
//  post-hoc scan costs 19.4-19.6 % of a time step when A,B,C,D are all read,
//  and 4.6 % for D alone — which is what this library used to pay.  Fusing them
//  into the build was never slower than the unmodified kernel (-2.6 % to -3.1 %
//  in step time, against a 0.07 % pairing precision from a null comparison of
//  two arithmetically identical kernels).  The saving is the 19.5 %, not the
//  negative sign, which is a grid-alignment-dependent second-order effect.
//
//  This form of `rho` inspects EVERY element of A/B/C, not just column 0.  The
//  column-0 shortcut assumes all n_sys systems share one operator; dropping the
//  assumption is what made the scan expensive, and fusing makes it free.
//
//  Encoding: for v >= 0 the IEEE-754 double bit pattern is monotone in v, so
//  reinterpreting it as u64 turns fmax into atomicMax.  Every quantity here is
//  an absolute value, so non-negativity holds by construction.  NaN maps above
//  every finite value and therefore propagates rather than being dropped.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

// Stats buffer: nslot slots x 2 entries, as u64 bits.
//   stats[2*s + 0] = rho bits,  stats[2*s + 1] = d_inf bits
//
// Slotting is NOT optional.  A 512^3 subdomain at 128 threads/block issues ~1e6
// blocks; funnelling their atomics through one address serialises them at L2
// and costs more than the scan it replaces.  128 slots was past the knee in the
// sweep; folding them afterwards is only 2*nslot values.
#define FTDMA_FM_NSLOT 128

__device__ __forceinline__ unsigned long long ftdma_fm_bits(double v) {
    return (unsigned long long)__double_as_longlong(v);   // precondition: v >= 0
}

__host__ __device__ __forceinline__ double ftdma_fm_unbits(unsigned long long b) {
    long long s = (long long)b;
#ifdef __CUDA_ARCH__
    return __longlong_as_double(s);
#else
    double d; __builtin_memcpy(&d, &s, sizeof(d)); return d;
#endif
}

__device__ __forceinline__ double ftdma_fm_warp_max(double v) {
    for (int s = 16; s > 0; s >>= 1)
        v = fmax(v, __shfl_down_sync(0xffffffffu, v, s));
    return v;
}

// Per-element contribution to rho.  Row 0 is excluded to match the sweep, which
// starts at k = 1.
__device__ __forceinline__ double ftdma_fm_rho_of(double A, double B, double C, int row) {
    if (row == 0 || B == 0.0) return 0.0;
    return fmax(fabs(A), fabs(C)) / fabs(B);
}

// ---------------------------------------------------------------------------
//  Block-wide reduction of one (rho, d_inf) pair per thread, then one atomicMax
//  pair per block.
//
//  EVERY thread of the block must call this, including those whose index fell
//  outside the domain — they pass 0.0.  An early `return` before this point
//  leaves __syncthreads() waiting on threads that no longer exist, so the build
//  kernel must carry an `active` flag instead of returning early.
//
//  Any block shape works: the linear thread id is formed explicitly, in the same
//  order CUDA assigns lanes to warps, so lane/warp agree with the shuffles.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ftdma_fm_block_accum(unsigned long long* __restrict__ stats,
                                                     int nslot, double my_rho, double my_d) {
    __shared__ double s_rho[32];      // one entry per warp; 1024 threads = 32 warps
    __shared__ double s_d  [32];

    const int tid  = threadIdx.x
                   + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int nthr = blockDim.x * blockDim.y * blockDim.z;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (nthr + 31) >> 5;

    double r = ftdma_fm_warp_max(my_rho);
    double d = ftdma_fm_warp_max(my_d);
    if (lane == 0) { s_rho[warp] = r; s_d[warp] = d; }
    __syncthreads();

    if (warp == 0) {
        r = (lane < nwarp) ? s_rho[lane] : 0.0;
        d = (lane < nwarp) ? s_d  [lane] : 0.0;
        r = ftdma_fm_warp_max(r);
        d = ftdma_fm_warp_max(d);
        if (lane == 0) {
            const int blin = blockIdx.x
                           + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
            const int slot = blin % nslot;
            atomicMax(&stats[2 * slot + 0], ftdma_fm_bits(r));
            atomicMax(&stats[2 * slot + 1], ftdma_fm_bits(d));
        }
    }
}

#endif // FILTERED_TDMA_FINDMAX_CUH
