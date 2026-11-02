#include "../include/filtered_tdma_cuda.hpp"
#include "filtered_tdma_kernels.cuh"
#include "ftdma_local_cuda.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// =============================================================================
//  Profiling variants of the Filtered TDMA GPU solves.
//
//  Same numerics as the production solves in filtered_tdma_cuda.cu; only the
//  timing scaffold differs. The scaffold is pascal_tdma's, adopted verbatim
//  (gpu/pascal_tdma_ref_cuda.cu: wall_time / sync_for_timing / add_elapsed):
//
//    * host wall clock (MPI_Wtime) brackets every phase;
//    * each bracket is closed by cudaStreamSynchronize, so the number includes
//      launch overhead and is directly comparable to pascal's;
//    * results accumulate into a named filtered_tdma::SolveTimings;
//    * last_comm_ms()/last_gpu_ms() are derived exactly as
//      PaScaLTDMACUDA::finish_profile() derives them.
//
//  This replaces the previous cudaEvent scheme, which timed only kernel
//  execution across two coarse groups. That was not wrong, but it was not the
//  measurement pascal reports, so the two libraries' numbers were not
//  comparable: pascal was charged one stream sync per phase plus launch
//  overhead that filtered never paid.
//
//  Kernels come from filtered_tdma_kernels.cuh (static __global__); each TU
//  gets its own device-code copy so no -rdc is required.
// =============================================================================

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA] %s:%d %s -> %s\n", __FILE__, __LINE__,\
                         #x, cudaGetErrorString(err));                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

namespace {

inline double wall_time() { return MPI_Wtime(); }

/// Close a phase: drain the stream so the elapsed time covers the work that was
/// launched, then accumulate. Mirrors pascal's sync_for_timing + add_elapsed.
inline void sync_and_add(double& slot, double start) {
    CUDA_CHECK(cudaStreamSynchronize(0));
    slot += wall_time() - start;
}

}  // namespace

void FilteredTDMACUDA::finish_profile_() {
    last_comm_ms_ = timings_.communication() * 1.0e3;
    last_gpu_ms_  = timings_.computation()   * 1.0e3;
}

// =============================================================================
//  J is known before the sweeps, so the post-comm group is a single fused
//  kernel. interface_solve/final_solve stay 0; post_compute carries both.
// =============================================================================

void FilteredTDMACUDA::solve_profile(double* d_A, double* d_B,
                                                 double* d_C, double* d_D) {
    timings_ = filtered_tdma::SolveTimings{};
    const double total_start = wall_time();

    if (nprocs_ == 1) {
        const double t = wall_time();
        ftdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_);
        sync_and_add(timings_.pre, t);
        timings_.total = wall_time() - total_start;
        finish_profile_();
        return;
    }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);
    double t;

    // 1) cutoff estimate (before the sweeps)
    t = wall_time();
    if (d_fm_stats_) {
        // Fused path: the caller already accumulated (rho, d_inf) during its
        // LHS build, so nothing is re-read here.
        k_cal_J_from_stats<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_fm_stats_, fm_nslot_,
                                                   n_row_, eps_);
    } else {
        // Multi-block scan + fold.  One block would leave 107 of 108 SMs idle.
        const std::size_t nD = (std::size_t)n_row_ * n_sys_;
        int nb = (int)((nD + FTDMA_BLOCK_SYS - 1) / FTDMA_BLOCK_SYS);
        if (nb > 4096) nb = 4096;
        if (nb < 1)    nb = 1;
        CUDA_CHECK(cudaMemsetAsync(d_scan_stats_, 0,
                                   sizeof(unsigned long long) * 2 * FTDMA_FM_NSLOT));
        k_scan_abcd_stats<<<nb, FTDMA_BLOCK_SYS>>>(d_A, d_B, d_C, d_D, nD, n_sys_,
                                                   FTDMA_FM_NSLOT, d_scan_stats_);
        k_cal_J_from_stats<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_scan_stats_,
                                                   FTDMA_FM_NSLOT, n_row_, eps_);
    }
    sync_and_add(timings_.cal_J, t);

    // 2) pre-comm group
    t = wall_time();
    k_pre<<<grid, block>>>(d_A, d_B, d_C, d_D,
                              d_send_right_, d_send_left_, n_sys_, n_row_, d_J_);
    sync_and_add(timings_.pre, t);

    // 3) exchange
    t = wall_time();
    mpi_exchange_();
    timings_.comm += wall_time() - t;

    // 4) post-comm group (interface 2x2 + final correction, fused)
    t = wall_time();
    k_post<<<grid, block>>>(d_D, d_A, d_C,
                               d_recv_left_,  d_recv_left_  + n_sys_,
                               d_recv_right_, d_recv_right_ + n_sys_,
                               (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                               (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                               n_sys_, n_row_, d_J_);
    sync_and_add(timings_.post, t);

    timings_.total = wall_time() - total_start;
    finish_profile_();
}

void FilteredTDMACUDA::solve_cyclic_profile(double* d_A, double* d_B,
                                                      double* d_C, double* d_D) {
    timings_ = filtered_tdma::SolveTimings{};
    const double total_start = wall_time();

    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        const double t = wall_time();
        ftdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        sync_and_add(timings_.pre, t);
        timings_.total = wall_time() - total_start;
        finish_profile_();
        return;
    }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);
    double t;

    t = wall_time();
    if (d_fm_stats_) {
        // Fused path: the caller already accumulated (rho, d_inf) during its
        // LHS build, so nothing is re-read here.
        k_cal_J_from_stats<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_fm_stats_, fm_nslot_,
                                                   n_row_, eps_);
    } else {
        // Multi-block scan + fold.  One block would leave 107 of 108 SMs idle.
        const std::size_t nD = (std::size_t)n_row_ * n_sys_;
        int nb = (int)((nD + FTDMA_BLOCK_SYS - 1) / FTDMA_BLOCK_SYS);
        if (nb > 4096) nb = 4096;
        if (nb < 1)    nb = 1;
        CUDA_CHECK(cudaMemsetAsync(d_scan_stats_, 0,
                                   sizeof(unsigned long long) * 2 * FTDMA_FM_NSLOT));
        k_scan_abcd_stats<<<nb, FTDMA_BLOCK_SYS>>>(d_A, d_B, d_C, d_D, nD, n_sys_,
                                                   FTDMA_FM_NSLOT, d_scan_stats_);
        k_cal_J_from_stats<<<1, FTDMA_BLOCK_SYS>>>(d_J_, d_scan_stats_,
                                                   FTDMA_FM_NSLOT, n_row_, eps_);
    }
    sync_and_add(timings_.cal_J, t);

    t = wall_time();
    k_pre<<<grid, block>>>(d_A, d_B, d_C, d_D,
                              d_send_right_, d_send_left_, n_sys_, n_row_, d_J_);
    sync_and_add(timings_.pre, t);

    t = wall_time();
    mpi_exchange_();
    timings_.comm += wall_time() - t;

    t = wall_time();
    k_post<<<grid, block>>>(d_D, d_A, d_C,
                               d_recv_left_,  d_recv_left_  + n_sys_,
                               d_recv_right_, d_recv_right_ + n_sys_,
                               (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                               (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                               n_sys_, n_row_, d_J_);
    sync_and_add(timings_.post, t);

    timings_.total = wall_time() - total_start;
    finish_profile_();
}
