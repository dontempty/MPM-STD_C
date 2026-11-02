#include "../include/filtered_tdma.hpp"
#include "filtered_tdma_kernels.hpp"
#include "tdma_local.hpp"

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace ftdma_kernel;

// ============================================================================
//  Constructor / Destructor
// ============================================================================

FilteredTDMA::FilteredTDMA(int n_sys, int n_row,
                           int myrank, int nprocs, MPI_Comm comm,
                           int left_rank, int right_rank,
                           double eps)
    : comm_(comm), n_sys_(n_sys), n_row_(n_row), nprocs_(nprocs),
      left_rank_(left_rank), right_rank_(right_rank)
{
    (void)myrank;

    // Packed exchange buffers. Row j of the [n_row x n_sys] row-major arrays is
    // already contiguous, so no MPI derived datatype is needed: each direction
    // is one contiguous 2*n_sys message.
    if (nprocs > 1) {
        send_right_.resize(2 * n_sys);
        send_left_.resize(2 * n_sys);
        recv_left_.resize(2 * n_sys);
        recv_right_.resize(2 * n_sys);
    }

    // eps is used exactly as given: the caller passes dt*dx^2 (see set_eps).
    eps_ = eps;
}

FilteredTDMA::~FilteredTDMA() = default;

double* FilteredTDMA::cyclic_workspace_() {
    const std::size_t need = (std::size_t)n_sys_ * n_row_;
    if (E_.size() < need) E_.resize(need);
    return E_.data();
}

// ============================================================================
//  Cutoff-J estimators
// ============================================================================

// Truncation-index criterion — Manuscript §5.4, eqs. (23)-(28).
//
// The total local error at index J is bounded by (23)
//
//     E(J) = [ C1 q^J + 2/(1-q) ( C2 q^(2J+2) + C3 q^(Ns-1) ) ] ||d||_inf
//
// and the three powers of q are kept at their own orders rather than folded
// under a single q^J:
//   q^J        what the interior update discards
//   q^(2J+2)   the closure-factor perturbation.  It reaches the solution only
//              through the already-truncated aB_1, so it is second order in
//              the truncation.
//   q^(Ns-1)   the coupling pair the reduced system discards outright.  J has
//              no effect on it, so it is a floor.
//
// Constants (Appendix C):
//   C1 = q(2+q)(1-rho) / [ (1-q)(1-q^2)(1-2rho)^2 ]
//   C2 = (1-rho)[(1-q^2)(1-2rho) + rho] / [ (1-q^2)^3 (1-2rho)^3 ]
//   C3 = (1-rho) / [ (1-q^2)(1-2rho)^3 ]
//
// The floor gives the feasibility condition (24)-(25):
//   eps_min = 2 C3 q^(Ns-1) ||d||_inf / (1-q);  a tolerance at or below it
// cannot be met at any depth, and the solve reverts to PaScaL_TDMA without
// forming an index.  Otherwise, with margin r = (eps - eps_min)/||d||_inf and
// y = q^J, the requirement E(J) <= eps is the quadratic inequality (26)
//   alpha y^2 + C1 y <= r,   alpha = 2 C2 q^2/(1-q),
// which holds exactly for y <= y+, the positive root.  Since ln q < 0, (27)
//   J = max{ 0, ceil( ln(y+)/ln(q) ) }.
// The outer max is what returns J = 0: when y+ >= 1, E(0) already meets the
// tolerance and no interior row needs an interface correction.
//
// Admissibility (28) is J <= Ns-2, so a solve falls back for one of exactly two
// reasons — the tolerance is infeasible, or the index it needs is deeper than
// the subdomain carries.  Both drop out of the criterion; neither is a special
// case bolted on.  The reported J is clamped either way, so a caller that
// ignores last_fallback() still gets a runnable (maximally retentive) index.
//
// k_cal_J_rhs_bound (gpu/filtered_tdma_kernels.cuh) mirrors this exactly.
// Called before the sweeps, so A/B/C/D still hold the original values.
int FilteredTDMA::cal_J_rhs_bound(const double* A, const double* B,
                                  const double* C, const double* D) {
    // Both norms are full scans over the WHOLE local system.  The earlier
    // shortcuts — rho from column 0 only, ||d||_inf over the two interface rows
    // only — both underestimate E(J) and so pick too large a truncation, which
    // is exactly the guarantee this criterion exists to provide.  The column-0
    // rho additionally assumed every system shares one tridiagonal operator,
    // which fails as soon as the coefficients vary across the other two
    // directions.  Row 0 stays excluded from rho: the sweep starts at k = 1.
    const std::size_t nD = (std::size_t)n_row_ * n_sys_;
    double rho = 0.0, d_inf = 0.0;
    #pragma omp simd reduction(max: rho, d_inf)
    for (std::size_t i = 0; i < nD; ++i) {
        const double v = std::abs(D[i]);
        if (v > d_inf) d_inf = v;
        if (i >= (std::size_t)n_sys_) {          // row = i / n_sys >= 1
            const double bk = B[i];
            if (bk != 0.0) {
                const double r = std::max(std::abs(A[i] / bk),
                                          std::abs(C[i] / bk));
                if (r > rho) rho = r;
            }
        }
    }

    const int Jmax = n_row_ - 2;
    int  J        = 0;
    bool fallback = false;

    if (Jmax < 0) {
        fallback = true;
    } else if (!(d_inf > 0.0)) {
        J = 0;                                  // zero RHS: exact at any depth
    } else if (!(eps_ > 0.0) || !(rho < 0.5)) {
        // eps <= 0 permits no truncation at all; rho >= 1/2 puts the problem
        // outside the theory (lambda+ would be complex, and the reduced-system
        // bound 1/(1-2rho) blows up).
        fallback = true; J = Jmax;
    } else if (rho <= 0.0) {
        J = 0;                                  // purely diagonal -> q = 0
    } else {
        const double lam = 0.5 * (1.0 + std::sqrt(1.0 - 4.0 * rho * rho));
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

            // Floor (24): eps_min = 2 C3 q^(Ns-1) ||d||_inf / (1-q).
            // q^(Ns-1) may underflow to 0 for large Ns — that is the floor
            // vanishing, which is correct.
            const double eps_min =
                2.0 * C3 * std::pow(q, (double)(n_row_ - 1)) * d_inf / omq;

            if (!(eps_ > eps_min)) {
                // Feasibility (25) fails: no depth reaches this tolerance.
                fallback = true; J = Jmax;
            } else {
                // Margin the floor leaves (26).
                const double r = (eps_ - eps_min) / d_inf;
                const double alpha = 2.0 * C2 * q2 / omq;
                // y+ in the cancellation-free form.  When 4*alpha*r << C1^2 —
                // the common case where a shallow cut suffices — the textbook
                // (-C1 + sqrt(...))/(2 alpha) subtracts two near-equal numbers
                // and loses most of its digits.  This form is algebraically
                // identical and stays finite as alpha -> 0.
                const double disc   = std::sqrt(C1 * C1 + 4.0 * alpha * r);
                const double y_plus = 2.0 * r / (C1 + disc);

                if (!(y_plus > 0.0)) {
                    fallback = true; J = Jmax;
                } else if (y_plus >= 1.0) {
                    J = 0;                      // E(0) already within tolerance
                } else {
                    const double Jr = std::ceil(std::log(y_plus) / std::log(q));
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
    if (J > n_row_ - 2) J = n_row_ - 2;
    if (J < 0)          J = 0;

    last_J_        = J;
    last_fallback_ = fallback;
    if (fallback) ++fallback_count_;
    return J;
}

// ============================================================================
//  MPI neighbour exchange (one bidirectional comm round)
//    my row N-1 (C,D) -> right ; my row 0 (A,D) -> left
//    recv left's row N-1 (LEFT 2x2) and right's row 0 (RIGHT 2x2).
//    The reduced system decouples into independent interface 2x2 blocks, so
//    each rank solves BOTH its interfaces locally (no back-communication).
//    For a PERIODIC communicator every rank has both neighbours, so the
//    wrap-around interface (rank N-1 <-> rank 0) is just another interface.
// ============================================================================

void FilteredTDMA::exchange_interfaces_(const double* A, const double* C,
                                        const double* D) {
    const std::size_t last = (std::size_t)(n_row_ - 1) * n_sys_;

    // Pack: one contiguous message per direction.
    std::copy_n(C + last, n_sys_, send_right_.begin());
    std::copy_n(D + last, n_sys_, send_right_.begin() + n_sys_);
    std::copy_n(A,        n_sys_, send_left_.begin());
    std::copy_n(D,        n_sys_, send_left_.begin()  + n_sys_);

    MPI_Request req[4];
    int nreq = 0;
    MPI_Isend(send_right_.data(), 2 * n_sys_, MPI_DOUBLE, right_rank_, 1, comm_, &req[nreq++]);
    MPI_Irecv(recv_left_.data(),  2 * n_sys_, MPI_DOUBLE, left_rank_,  1, comm_, &req[nreq++]);
    MPI_Isend(send_left_.data(),  2 * n_sys_, MPI_DOUBLE, left_rank_,  2, comm_, &req[nreq++]);
    MPI_Irecv(recv_right_.data(), 2 * n_sys_, MPI_DOUBLE, right_rank_, 2, comm_, &req[nreq++]);
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
}

// ============================================================================
//  solve / solve_cyclic — no timing.
//  J is computed up front; the forward and backward sweeps then skip the work
//  that falls beyond the cutoff.
// ============================================================================

void FilteredTDMA::solve(double* __restrict A, double* __restrict B,
                                     double* __restrict C, double* __restrict D) {
    if (nprocs_ == 1) { ftdma_many(A, B, C, D, n_sys_, n_row_); return; }

    // cal_J_rhs_bound returns J in [0, n_row-2]; see the clamp there for why
    // both ends matter.  last_fallback() says whether the criterion would have
    // preferred PaScaL_TDMA.
    const int J  = cal_J_rhs_bound(A, B, C, D);
    const int lo = (n_row_ - 2) - J;

    fwd_elim(A, B, C, D, n_sys_, n_row_, J);
    bwd_subst(A, C, D, n_sys_, n_row_, J, lo);
    pack_reduced(A, C, D, n_sys_);
    exchange_interfaces_(A, C, D);
    solve_interfaces(A, C, D, recv_left_.data(),  recv_left_.data()  + n_sys_,
                     recv_right_.data(), recv_right_.data() + n_sys_,
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);

    final_solve(A, C, D, n_sys_, n_row_, /*left_end=*/J,
                                         /*right_beg=*/std::max(1, lo + 1));
}

void FilteredTDMA::solve_cyclic(double* __restrict A, double* __restrict B,
                                          double* __restrict C, double* __restrict D) {
    if (nprocs_ == 1) { ftdma_cyclic_many(A, B, C, D, cyclic_workspace_(), n_sys_, n_row_); return; }

    const int J  = cal_J_rhs_bound(A, B, C, D);
    const int lo = (n_row_ - 2) - J;

    fwd_elim(A, B, C, D, n_sys_, n_row_, J);
    bwd_subst(A, C, D, n_sys_, n_row_, J, lo);
    pack_reduced(A, C, D, n_sys_);
    exchange_interfaces_(A, C, D);
    solve_interfaces(A, C, D, recv_left_.data(),  recv_left_.data()  + n_sys_,
                     recv_right_.data(), recv_right_.data() + n_sys_,
                     left_rank_  != MPI_PROC_NULL,
                     right_rank_ != MPI_PROC_NULL, n_sys_, n_row_);

    final_solve(A, C, D, n_sys_, n_row_, /*left_end=*/J,
                                         /*right_beg=*/std::max(1, lo + 1));
}
