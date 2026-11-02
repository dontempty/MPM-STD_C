#ifndef FILTERED_TDMA_HPP
#define FILTERED_TDMA_HPP

#include "filtered_tdma_timings.hpp"

#include <mpi.h>
#include <vector>

/// Parallel tridiagonal solver for many right-hand sides — Filtered TDMA.
///
/// The cutoff depth J is estimated from the RHS bound before the sweeps, and
/// the forward sweep, backward sweep and final correction all stop at it: the
/// influence of one subdomain's boundary decays like q^j, so beyond J it is
/// below the tolerance and the work can be skipped outright.
///
/// Encapsulates the MPI plan (one bidirectional neighbour exchange per solve).
/// Memory layout: row-major [n_row x n_sys], row j starts at offset j*n_sys.
class FilteredTDMA {
public:
    /// Construct solver for `n_sys` independent systems, each of size `n_row`.
    FilteredTDMA(int n_sys, int n_row,
                 int myrank, int nprocs, MPI_Comm comm,
                 int left_rank, int right_rank,
                 double eps);

    ~FilteredTDMA();

    // Non-copyable
    FilteredTDMA(const FilteredTDMA&)            = delete;
    FilteredTDMA& operator=(const FilteredTDMA&) = delete;

    /// Solve n_sys tridiagonal systems of n_row rows each. A, B, C: lower /
    /// diagonal / upper coefficients; D: RHS, overwritten with the solution.
    /// A, C and D double as workspace.
    void solve(double* __restrict A, double* __restrict B,
               double* __restrict C, double* __restrict D);

    /// Periodic variant: row 0 couples to row n_row-1. For nprocs > 1 the
    /// communicator itself is periodic, so both interfaces are ordinary ones
    /// and the body matches solve(); only the single-rank fallback differs.
    void solve_cyclic(double* __restrict A, double* __restrict B,
                      double* __restrict C, double* __restrict D);

    // --- Profile: per-phase timing (7 entries, see last_timings()) ---
    //
    // Timing follows pascal_tdma's method exactly (MPI_Barrier + MPI_Wtime per
    // phase, accumulating). `time_list` keeps the 7 slots the heat driver
    // reports; the full breakdown is available via last_timings().
    //   [0] forward  [1] backward  [2] pack  [3] exchange
    //   [4] interface 2x2  [5] unused (filtered has no back-communication)
    //   [6] final local solve
    // The J estimate is folded into the slot where it actually runs:
    // [0] for v2 (before the forward sweep), [6] for v1 (after the exchange).
    void solve_profile(double* __restrict A, double* __restrict B,
                       double* __restrict C, double* __restrict D,
                       std::vector<double>& time_list);

    /// Phase breakdown of the most recent solve_*_profile() call.
    const filtered_tdma::SolveTimings& last_timings() const { return timings_; }

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

    /// Truncation index chosen by the most recent solve.  -1 before the first.
    int  last_J() const { return last_J_; }

    /// True when the criterion asked for PaScaL_TDMA instead of the filtered
    /// branch — either the tolerance is infeasible (eps <= eps_min, no depth
    /// reaches it) or the depth it needs exceeds n_row-2.  See
    /// filtered_tdma_jcriterion.hpp.  The solve still runs the filtered branch
    /// at the clamped index, so this is a signal to the caller, not an error.
    bool last_fallback() const { return last_fallback_; }

    /// How many solves have raised last_fallback() since construction.
    long fallback_count() const { return fallback_count_; }

private:
    /// Select the truncation index by the quadratic criterion of Manuscript
    /// §5.4 (eqs. 23-28).  Two reductions feed it:
    ///   rho       — max over the WHOLE local operator, all rows k >= 1
    ///   ||d||_inf — over the WHOLE local RHS, n_row x n_sys.
    /// Both are full scans, in one fused pass over A/B/C/D.  The shortcuts this
    /// used to take (rho from column 0, ||d||_inf over the two interface rows)
    /// underestimate E(J), which selects too large a truncation and voids the
    /// tolerance guarantee the criterion exists to give; the column-0 rho also
    /// assumed every system shares one operator.
    /// Called before the sweeps, so A/B/C/D still hold the original values.
    /// k_cal_J_rhs_bound (gpu/filtered_tdma_kernels.cuh) mirrors this; the two
    /// must be edited together.
    int cal_J_rhs_bound(const double* A, const double* B, const double* C,
                        const double* D);

    /// Single bidirectional neighbour exchange of both interfaces' reduced rows,
    /// one comm round. Both directions are packed into one contiguous buffer
    /// each, so the exchange is 4 requests rather than 8 — the same layout the
    /// CUDA path uses (d_send_right_/d_send_left_).
    ///   send_right = [ C[n_row-1] | D[n_row-1] ] -> right neighbour
    ///   send_left  = [ A[0]       | D[0]       ] -> left  neighbour
    ///   recv_left  = [ C | D ] from left, recv_right = [ A | D ] from right
    void exchange_interfaces_(const double* A, const double* C, const double* D);

    // --- MPI plan data ---
    MPI_Comm     comm_;
    int          n_sys_, n_row_, nprocs_;
    int          left_rank_, right_rank_;

    // Packed 2-message exchange buffers (each 2*n_sys doubles).
    std::vector<double> send_right_;
    std::vector<double> send_left_;
    std::vector<double> recv_left_;
    std::vector<double> recv_right_;

    // Workspace for the nprocs==1 cyclic (Sherman-Morrison) local solve.
    // Allocated on first use; unused by the non-cyclic path.
    std::vector<double> E_;
    double* cyclic_workspace_();

    double eps_;

    // Most recent criterion outcome (see last_J()/last_fallback()).
    int  last_J_         = -1;
    bool last_fallback_  = false;
    long fallback_count_ = 0;

    // Filled by the solve_*_profile() entry points only.
    filtered_tdma::SolveTimings timings_;
};

#endif // FILTERED_TDMA_HPP
