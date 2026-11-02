#ifndef FILTERED_TDMA_CUDA_HPP
#define FILTERED_TDMA_CUDA_HPP

#include "filtered_tdma_timings.hpp"

#include <mpi.h>
#include <cuda_runtime.h>

/// CUDA implementation of the Filtered TDMA solver.
///
/// Solves a distributed tridiagonal system split across `nprocs` ranks along
/// the row dimension. Each rank holds `n_row` rows of `n_sys` independent
/// systems. All matrix/RHS arrays are device pointers (row-major, n_row×n_sys).
///
/// Communication uses CUDA-aware MPI (P2P Isend/Irecv with neighbor ranks).
/// Two packed 2×n_sys buffers are exchanged per solve, one in each direction.
///
/// The cutoff depth J is computed before the forward pass; the sweeps and the
/// final correction then skip everything beyond it.
class FilteredTDMACUDA {
public:
    FilteredTDMACUDA(int n_sys, int n_row,
                     int myrank, int nprocs, MPI_Comm comm,
                     int left_rank, int right_rank,
                     double eps);

    ~FilteredTDMACUDA();

    FilteredTDMACUDA(const FilteredTDMACUDA&)            = delete;
    FilteredTDMACUDA& operator=(const FilteredTDMACUDA&) = delete;

    void solve(double* d_A, double* d_B, double* d_C, double* d_D);

    /// Periodic variant: rank 0's left neighbor is the last rank, giving every
    /// rank real interface blocks on both sides. The multi-rank body is identical
    /// to the non-periodic solve; only the nprocs==1 fallback differs.
    void solve_cyclic(double* d_A, double* d_B, double* d_C, double* d_D);

    /// Profiling variants (filtered_tdma_profile_cuda.cu) — same numerics as
    /// the solve_* above, with per-phase timing. The method is pascal_tdma's:
    /// host MPI_Wtime brackets closed by cudaStreamSynchronize, accumulated
    /// into a named filtered_tdma::SolveTimings (see last_timings()).
    /// last_gpu_ms()/last_comm_ms() are derived from it the same way
    /// PaScaLTDMACUDA::finish_profile() derives its own. The non-profile solves
    /// are sync-free for performance and do NOT update those timers.
    void solve_profile(double* d_A, double* d_B, double* d_C, double* d_D);
    void solve_cyclic_profile(double* d_A, double* d_B, double* d_C, double* d_D);

    int n_sys()  const { return n_sys_; }
    int n_row()  const { return n_row_; }
    int nprocs() const { return nprocs_; }

    /// Absolute truncation tolerance used by the cutoff estimate.  The CALLER
    /// supplies the finished value — the paper's criterion is
    ///     eps = dt * dx^2
    /// with dx the grid spacing along THIS solve direction (L/N for a stretched
    /// axis).  The library deliberately does no scaling of its own: it used to
    /// divide by (n_row*nprocs)^2, which silently assumed a unit-length domain
    /// and was wrong by L^2 for every real case (L=2 in heat, up to 12.6 in the
    /// channel).
    void set_eps(double eps) { eps_ = eps; }

    /// Truncation depth J computed by the most recent solve (blocking
    /// device->host copy of d_J_[0]).
    int last_J() const;

    /// True when the criterion asked for PaScaL_TDMA instead of the filtered
    /// branch — the tolerance is infeasible (eps <= eps_min), or the depth it
    /// needs exceeds n_row-2.  Blocking device->host copy of d_J_[1].  The
    /// solve still ran the filtered branch at the clamped index, so this is a
    /// signal to the caller, not an error.
    bool last_fallback() const;

    /// Take (rho, ||d||_inf) from statistics the CALLER accumulated inside its
    /// own LHS-build kernel, instead of re-reading the arrays here.
    ///
    /// `d_stats` is a device buffer of 2*nslot u64 slots in the layout of
    /// include/filtered_tdma_findmax.cuh; the caller must zero it before the
    /// build and fill it with ftdma_fm_block_accum().  Passing nullptr restores
    /// the self-computed path.
    ///
    /// Why: the self-computed path (k_cal_J_rhs_bound) streams A/B/C/D once per
    /// solve — the general rho needs all of them — which measured 19.5 % of a
    /// heat_gpu time step on A100 (4.6 % for D alone).  Accumulating the same
    /// two sup-norms from coefficients already in registers during the build
    /// was never slower than the unmodified build.  See apps/re_cal_J.
    void set_norms_device(const unsigned long long* d_stats, int nslot) {
        d_fm_stats_ = d_stats;
        fm_nslot_   = nslot;
    }

private:
    void mpi_exchange_();   // bidirectional neighbour MPI exchange (no timing)

    // --- MPI plan data ---
    MPI_Comm     comm_;
    int          n_sys_, n_row_, nprocs_;
    int          left_rank_, right_rank_;

    // Packed 2-message exchange buffers (each 2*n_sys doubles).
    // send_right: [C_lastrow | D_lastrow] → right neighbor (tags 1)
    // send_left : [A_row0   | D_row0  ] → left  neighbor (tag 2)
    // recv_left : [C | D] from left  (C_left = recv_left, D_left = recv_left+n_sys)
    // recv_right: [A | D] from right (A_right= recv_right, D0_right= recv_right+n_sys)
    double* d_send_right_ = nullptr;
    double* d_send_left_  = nullptr;
    double* d_recv_left_  = nullptr;
    double* d_recv_right_ = nullptr;

    // Device scratch for the cal_J kernel result: [0] = index, [1] = fallback.
    int*    d_J_     = nullptr;
    // Caller-supplied fused sup-norm statistics (see set_norms_device).  When
    // non-null k_cal_J_rhs_bound is not launched at all — the caller's LHS
    // build already produced both sup-norms.
    const unsigned long long* d_fm_stats_ = nullptr;
    int                       fm_nslot_   = 0;

    // Slots for the non-fused path's own multi-block scan (k_scan_abcd_stats).
    // Allocated once; zeroed before each scan because atomicMax needs a floor.
    unsigned long long*       d_scan_stats_ = nullptr;
    // Lazily-allocated [n_row × n_sys] workspace for the nprocs==1 cyclic
    // (Sherman-Morrison) local solve. Unused by the non-cyclic path.
    double* d_E_     = nullptr;

    double eps_;

    // --- per-solve timing ---
    filtered_tdma::SolveTimings timings_;   // filled by solve_*_profile() only
    void finish_profile_();                 // timings_ -> last_comm_ms_/last_gpu_ms_

    double last_comm_ms_ = 0.0;
    double last_gpu_ms_  = 0.0;

public:
    /// Pure MPI wall-clock time (ms) of the most recent profiled solve.
    /// == timings_.communication(); same definition as PaScaLTDMACUDA's.
    double last_comm_ms() const { return last_comm_ms_; }

    /// Compute time (ms) of the most recent profiled solve.
    /// == timings_.computation() = cutoff_J + local_compute + post_compute;
    /// same definition as PaScaLTDMACUDA's. Being host wall clock around
    /// synchronised kernel groups, it includes launch overhead — that is what
    /// makes it comparable to pascal's number (the old cudaEvent value did not
    /// include it, which flattered filtered).
    double last_gpu_ms()  const { return last_gpu_ms_; }

    /// Phase breakdown of the most recent solve_*_profile() call.
    const filtered_tdma::SolveTimings& last_timings() const { return timings_; }
};

#endif // FILTERED_TDMA_CUDA_HPP
