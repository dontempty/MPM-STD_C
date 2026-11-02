#include "../include/filtered_tdma.hpp"
#include "filtered_tdma_kernels.hpp"
#include "tdma_local.hpp"

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>

// =============================================================================
//  Per-phase timed variants of the Filtered TDMA solves.
//
//  Same numerics as the production solves in filtered_tdma.cpp; only the timing
//  scaffold differs. The scaffold is pascal_tdma's, adopted verbatim so the two
//  libraries' phase numbers are directly comparable:
//
//    * PhaseTimer below has the same shape as pascal_tdma::PhaseTimer
//      (cpu/pascal_tdma_ref.cpp) — MPI_Barrier, then MPI_Wtime, accumulate into
//      the slot with +=;
//    * results land in a named filtered_tdma::SolveTimings rather than bare
//      vector indices, and `total` spans the whole call;
//    * the 7-slot `time_list` the heat driver expects is derived from the
//      struct at the end, exactly as PaScaLTDMA::solve_profile does.
//
//  The algorithm steps are reused from filtered_tdma_kernels.hpp (the same
//  building blocks the production solves call).
// =============================================================================

using namespace ftdma_kernel;

namespace {

/// Phase bracket. Mirrors pascal_tdma::PhaseTimer: the barrier lines the ranks
/// up so a phase's time is the slowest rank's rather than a rank-local number.
struct PhaseTimer {
    MPI_Comm comm;
    double   t0 = 0.0;

    explicit PhaseTimer(MPI_Comm c) : comm(c) {}

    void begin() {
        MPI_Barrier(comm);
        t0 = MPI_Wtime();
    }

    void end(double& slot) { slot += MPI_Wtime() - t0; }
};

/// Derive the 7 slots the heat driver reports. The J estimate is charged to
/// slot 0: it runs before the forward sweep, so that is the phase it belongs to.
void to_time_list(const filtered_tdma::SolveTimings& t,
                  std::vector<double>& time_list) {
    time_list.assign(7, 0.0);
    time_list[0] = t.local_forward;
    time_list[1] = t.local_backward;
    time_list[2] = t.local_pack;
    time_list[3] = t.comm;
    time_list[4] = t.interface_solve;
    time_list[5] = 0.0;              // filtered has no back-communication
    time_list[6] = t.final_solve;
    time_list[0] += t.cal_J;
}

}  // namespace

// ============================================================================
//  solve_profile
// ============================================================================

void FilteredTDMA::solve_profile(double* __restrict A, double* __restrict B,
                                 double* __restrict C, double* __restrict D,
                                 std::vector<double>& time_list) {
    timings_ = filtered_tdma::SolveTimings{};
    const double total_start = MPI_Wtime();
    PhaseTimer phase(comm_);

    if (nprocs_ == 1) {
        phase.begin();
        ftdma_many(A, B, C, D, n_sys_, n_row_);
        phase.end(timings_.pre);
        timings_.local_forward = timings_.pre;
        timings_.total = MPI_Wtime() - total_start;
        to_time_list(timings_, time_list);
        return;
    }

    // 1) Cutoff estimate — J is needed before the sweeps. Timed, not free setup:
    //    the production path computes it at exactly this point.
    phase.begin();
    const int J  = cal_J_rhs_bound(A, B, C, D);
    const int lo = (n_row_ - 2) - J;
    phase.end(timings_.cal_J);

    // 2) Forward elimination (split at J)
    phase.begin();
    fwd_elim(A, B, C, D, n_sys_, n_row_, J);
    phase.end(timings_.local_forward);

    // 3) Backward substitution (3-phase)
    phase.begin();
    bwd_subst(A, C, D, n_sys_, n_row_, J, lo);
    phase.end(timings_.local_backward);

    // 4) Pack reduced system
    phase.begin();
    pack_reduced(A, C, D, n_sys_);
    phase.end(timings_.local_pack);

    // 5) Single bidirectional exchange
    phase.begin();
    exchange_interfaces_(A, C, D);
    phase.end(timings_.comm);

    // 6) Solve both interface 2x2 blocks locally
    phase.begin();
    solve_interfaces(A, C, D, recv_left_.data(),  recv_left_.data()  + n_sys_,
                     recv_right_.data(), recv_right_.data() + n_sys_,
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);
    phase.end(timings_.interface_solve);

    // 7) Final local solve
    phase.begin();
    final_solve(A, C, D, n_sys_, n_row_, /*left_end=*/J,
                                         /*right_beg=*/std::max(1, lo + 1));
    phase.end(timings_.final_solve);

    timings_.pre = timings_.local_forward + timings_.local_backward
                           + timings_.local_pack;
    timings_.post  = timings_.interface_solve + timings_.final_solve;
    timings_.total         = MPI_Wtime() - total_start;
    to_time_list(timings_, time_list);
}
