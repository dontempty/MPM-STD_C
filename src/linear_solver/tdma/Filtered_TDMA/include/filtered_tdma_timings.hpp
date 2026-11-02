#ifndef FILTERED_TDMA_TIMINGS_HPP
#define FILTERED_TDMA_TIMINGS_HPP

namespace filtered_tdma {

/// Per-phase timing of one Filtered TDMA solve.
///
/// This deliberately mirrors `pascal_tdma::SolveTimings` so the two libraries
/// are measured the same way and their numbers can be compared directly.
/// The measurement method is upstream PaScaL_TDMA's, adopted verbatim:
///
///   * host wall clock (`MPI_Wtime`) brackets every phase — no cudaEvents;
///   * CPU: `MPI_Barrier(comm)` opens each bracket, so a phase's time is the
///     slowest rank's, not a rank-local number nobody can add up;
///   * GPU: `cudaStreamSynchronize` closes each bracket before the clock is
///     read, so the number includes launch overhead and is directly
///     comparable to pascal's;
///   * fields accumulate with `+=`; the profiled entry points zero the struct
///     on entry, so one call == one solve unless the caller wants otherwise;
///   * `total` spans the whole profiled call, measured without a barrier.
///
/// One field per kernel/communication step, in execution order:
///
///   #  label   what it covers
///   0  cal_J   k_cal_J_rhs_bound — one full scan of A/B/C/D for rho and
///              max|D|, reduced to the cutoff depth J (or
///              k_cal_J_from_stats when the caller supplied fused norms)
///   1  pre     k_pre  — forward sweep, backward sweep, row-0 2x2 reduction
///                       and both send-buffer packs, fused
///   2  comm    mpi_exchange_ — the single bidirectional neighbour exchange
///                       (4 requests: one packed message each way)
///   3  post    k_post — both interface 2x2 solves and the J-truncated
///                       boundary correction, fused
///
/// The phase set differs from pascal's because the algorithms differ: filtered
/// has one neighbour exchange and no reduced-system transpose, so there is no
/// pack/unpack pair and no reduced-system solve. Watch the boundaries when
/// comparing columns one-to-one:
///
///   * filtered's `pre` INCLUDES the send-buffer pack; pascal reports packing
///     separately (pack1/unpack1/pack2/unpack2). Compare pre vs
///     tdma_modified + packing().
///   * filtered's `post` INCLUDES the interface solve; in pascal that work is
///     the `reduced` phase and `update` is only the solution rebuild. Compare
///     post vs reduced + update.
///   * `cal_J` has no pascal counterpart — it is what the cutoff costs.
///
/// The only wholly comparable numbers are `total` and `communication()`.
struct SolveTimings {
    double total = 0.0;

    double cal_J = 0.0;
    double pre   = 0.0;
    double comm  = 0.0;
    double post  = 0.0;

    // --- EXTENSION: split of pre/post (host path only; 0 on the CUDA path) --
    // The CUDA path fuses each group into a single kernel (k_pre, k_post), so
    // it can only report the group totals. Same convention pascal uses for its
    // local_forward/local_backward/local_reduce split.
    double local_forward   = 0.0;
    double local_backward  = 0.0;
    double local_pack      = 0.0;
    double interface_solve = 0.0;
    double final_solve     = 0.0;

    // --- labelled event view, in execution order -------------------------
    static constexpr int kNumEvents = 4;

    static const char* event_name(int i) {
        static const char* const n[kNumEvents] = {"cal_J", "pre", "comm", "post"};
        return (i >= 0 && i < kNumEvents) ? n[i] : "";
    }

    double event_time(int i) const {
        switch (i) {
            case 0: return cal_J;
            case 1: return pre;
            case 2: return comm;
            case 3: return post;
            default: return 0.0;
        }
    }

    double communication() const { return comm; }

    /// Host-path only: the CUDA path fuses the pack into `pre`, so this is 0
    /// there and the pack time is already inside `pre` either way.
    double packing() const { return local_pack; }

    /// Everything that is not MPI. `pre` already contains the send-buffer pack,
    /// so this matches pascal's packing-inclusive computation().
    double computation() const { return cal_J + pre + post; }
};

}  // namespace filtered_tdma

#endif  // FILTERED_TDMA_TIMINGS_HPP
