#include "../include/filtered_tdma_cuda.hpp"
#include "filtered_tdma_kernels.cuh"
#include "ftdma_local_cuda.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA] %s:%d %s -> %s\n", __FILE__, __LINE__,\
                         #x, cudaGetErrorString(err));                         \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// =============================================================================
//  Constructor / destructor
// =============================================================================

FilteredTDMACUDA::FilteredTDMACUDA(int n_sys, int n_row,
                                   int myrank, int nprocs, MPI_Comm comm,
                                   int left_rank, int right_rank,
                                   double eps)
    : comm_(comm), n_sys_(n_sys), n_row_(n_row), nprocs_(nprocs),
      left_rank_(left_rank), right_rank_(right_rank)
{
    (void)myrank;

    if (nprocs > 1) {
        CUDA_CHECK(cudaMalloc(&d_send_right_, sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_send_left_,  sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_recv_left_,  sizeof(double) * 2 * n_sys));
        CUDA_CHECK(cudaMalloc(&d_recv_right_, sizeof(double) * 2 * n_sys));
    }

    // [0] = index, [1] = fallback flag.
    CUDA_CHECK(cudaMalloc(&d_J_, 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan_stats_,
                          sizeof(unsigned long long) * 2 * FTDMA_FM_NSLOT));

    // eps is used exactly as given: the caller passes dt*dx^2 (see set_eps).
    eps_ = eps;
}

FilteredTDMACUDA::~FilteredTDMACUDA() {
    if (d_send_right_) cudaFree(d_send_right_);
    if (d_send_left_)  cudaFree(d_send_left_);
    if (d_recv_left_)  cudaFree(d_recv_left_);
    if (d_recv_right_) cudaFree(d_recv_right_);
    if (d_J_)               cudaFree(d_J_);
    if (d_scan_stats_)      cudaFree(d_scan_stats_);
    if (d_E_)               cudaFree(d_E_);
}

int FilteredTDMACUDA::last_J() const {
    int h[2] = {-1, 0};
    CUDA_CHECK(cudaMemcpy(h, d_J_, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    return h[0];
}

bool FilteredTDMACUDA::last_fallback() const {
    int h[2] = {-1, 0};
    CUDA_CHECK(cudaMemcpy(h, d_J_, 2 * sizeof(int), cudaMemcpyDeviceToHost));
    return h[1] != 0;
}

// =============================================================================
//  MPI neighbor exchange (bidirectional, blocking)
// =============================================================================

void FilteredTDMACUDA::mpi_exchange_() {
    MPI_Request req[4];
    int nreq = 0;
    MPI_Isend(d_send_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[nreq++]);
    MPI_Irecv(d_recv_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
    MPI_Isend(d_send_left_,  2 * n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
    MPI_Irecv(d_recv_right_, 2 * n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[nreq++]);
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
}

// =============================================================================
//  solve / solve_cyclic
// =============================================================================

void FilteredTDMACUDA::solve(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) { ftdma_many_cuda(d_A, d_B, d_C, d_D, n_sys_, n_row_); return; }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

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
    k_pre<<<grid, block>>>(d_A, d_B, d_C, d_D,
                              d_send_right_, d_send_left_, n_sys_, n_row_, d_J_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    mpi_exchange_();

    k_post<<<grid, block>>>(d_D, d_A, d_C,
                               d_recv_left_,  d_recv_left_  + n_sys_,
                               d_recv_right_, d_recv_right_ + n_sys_,
                               (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                               (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                               n_sys_, n_row_, d_J_);
}

void FilteredTDMACUDA::solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D) {
    if (nprocs_ == 1) {
        if (!d_E_) CUDA_CHECK(cudaMalloc(&d_E_, sizeof(double) * (std::size_t)n_sys_ * n_row_));
        ftdma_cyclic_many_cuda(d_A, d_B, d_C, d_D, d_E_, n_sys_, n_row_);
        return;
    }

    const dim3 block(FTDMA_BLOCK_SYS);
    const dim3 grid = ftdma_grid1D(n_sys_);

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
    k_pre<<<grid, block>>>(d_A, d_B, d_C, d_D,
                              d_send_right_, d_send_left_, n_sys_, n_row_, d_J_);

    CUDA_CHECK(cudaStreamSynchronize(0));
    mpi_exchange_();

    k_post<<<grid, block>>>(d_D, d_A, d_C,
                               d_recv_left_,  d_recv_left_  + n_sys_,
                               d_recv_right_, d_recv_right_ + n_sys_,
                               (left_rank_  != MPI_PROC_NULL) ? 1 : 0,
                               (right_rank_ != MPI_PROC_NULL) ? 1 : 0,
                               n_sys_, n_row_, d_J_);
}
