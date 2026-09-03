#pragma once

// Scaling-run timing layer — a port of the KISTI Scalability harness recorder
// (~/workspace/KISTI_6_test/Scalability/timing.{hpp,cu}), generalized so the
// event list is supplied by the caller instead of a hard-coded enum.  It
// implements ~/workspace/KISTI_6_test/mpi_gpu_timing_spec.md:
//
//   * each rank stores its own measurements in local memory (spec 2.1, 3);
//   * the measurement loop contains no timing collective, no timing barrier and
//     no file I/O (spec 2.2, 13);
//   * the whole iteration x event array is reduced once, after the loop
//     (spec 2.3, 8.1);
//   * only rank 0 writes, one CSV per run (spec 2.4, 11.1);
//   * the official scaling number is the rank-max of the measurement window
//     (spec 2.5, 9);
//   * no cudaDeviceSynchronize is inserted per event just to measure
//     (spec 2.8, 13).
//
// HOW THE CLOCK WORKS.  GPU work is asynchronous, so a host timestamp after a
// kernel launch measures the launch, not the work; synchronising at every
// boundary is what the spec forbids.  So an iteration is timed as a TIMELINE of
// CUDA events recorded on the stream: one mark at every region boundary,
// recorded asynchronously (nothing waits).  After the window closes, the
// elapsed time between consecutive marks is read back and accumulated into
// whichever event that interval was attributed to.  A rank stalled inside MPI
// is charged to the event containing that MPI, because the next mark only
// executes once the stream gets there.  The marks tile the iteration end to
// end; total_iteration is their (telescoping) sum per rank per iteration —
// never a sum of per-event maxima (spec 4).
//
// The same event may be marked several times per iteration (a kernel launched
// once per sweep direction, say); the intervals accumulate.  The mark PATTERN
// must be identical every iteration — position k must name the same event —
// so a column means one thing on every row; violations abort.
//
// CUDA-only: include this header from .cu translation units (and the GPU app
// mains).  Public solver headers that merely pass a pointer through should
// forward-declare `mpmstd::post::TimingRecorder` instead of including this.

#include <mpi.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace mpmstd::post {

class TimingRecorder {
public:
    /// `event_names` are the partition events; a derived "total_iteration" is
    /// appended automatically.  `marks_per_iteration` is how many mark() calls
    /// the caller will issue each iteration (fixes the CUDA-event pool size).
    /// `detailed` additionally reduces MIN and sum-of-squares (spec 7.2).
    TimingRecorder(int n_measured_iteration, int marks_per_iteration,
                   std::vector<std::string> event_names,
                   MPI_Comm comm, bool detailed = false);
    ~TimingRecorder();

    TimingRecorder(const TimingRecorder&) = delete;
    TimingRecorder& operator=(const TimingRecorder&) = delete;

    /// One device sync + one barrier, then the window clock starts.  This is
    /// the only barrier the timing layer performs (spec 5).
    void start_window();
    /// Device sync, then the window clock stops (spec 6.3).
    void stop_window();

    void begin_iteration(int iteration);
    /// Attribute everything since the previous mark to `event`, open the next
    /// interval.  Asynchronous (one cudaEventRecord).
    void mark(int event);
    void end_iteration();

    /// Harvest the CUDA-event timeline, then reduce across ranks: one MAX and
    /// one SUM (plus MIN/SUMSQ when detailed) over the whole array, one MAX on
    /// the window.  Call once, after stop_window (spec 8).
    void reduce();

    /// Rank 0 writes one CSV: `# key=value` metadata rows, then one row per
    /// iteration with {event}_avg,{event}_max columns (detailed adds
    /// min/std/imbalance).  Other ranks return immediately (spec 11).
    void write_csv(const std::string& path,
                   const std::vector<std::pair<std::string, std::string>>& metadata) const;

    /// Per-event median/mean of the rank-max plus imbalance, and optional named
    /// groups printed as sums of member events (rank 0 only).
    void print_summary(const std::vector<std::pair<std::string, std::vector<int>>>& groups = {}) const;

    int    total_event() const { return n_partition_event_; }   // derived index
    double window_max() const { return window_max_; }
    double window_per_iteration() const {
        return n_iteration_ > 0 ? window_max_ / n_iteration_ : 0.0;
    }
    double reduction_overhead() const { return reduction_overhead_; }

    // Rank-0 iteration-direction statistics over an event's rank-max (spec 10).
    double median_max(int event) const;
    double mean_max(int event) const;
    double median_imbalance(int event) const;

private:
    std::size_t index(int iteration, int event) const {
        return static_cast<std::size_t>(iteration) * n_event_ + event;
    }
    void harvest();   // CUDA-event timeline -> time_local_

    MPI_Comm comm_;
    int rank_ = 0, nprocs_ = 1;
    int n_iteration_ = 0;
    int n_partition_event_ = 0;   // events the caller marks
    int n_event_ = 0;             // + derived total_iteration
    bool detailed_ = false;

    std::vector<std::string> event_names_;   // [n_event_], last = total_iteration

    std::vector<cudaEvent_t> marks_;   // (marks_per_iter_+1) per iteration
    std::vector<int> mark_event_;      // locked event id of each interval
    int marks_per_iteration_ = 0;
    int current_iteration_ = -1;
    int current_mark_ = 0;

    std::vector<double> time_local_;   // [iteration][event], seconds
    std::vector<double> time_max_, time_sum_, time_min_, time_sumsq_;

    double window_start_ = 0.0;
    double window_local_ = 0.0;
    double window_max_ = 0.0;
    double reduction_overhead_ = 0.0;
    bool harvested_ = false;
};

}  // namespace mpmstd::post
