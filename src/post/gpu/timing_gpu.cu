#include "post/timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>

#ifdef MPMSTD_BACKEND_CUDA

namespace mpmstd::post {

namespace {

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[timing] %s -> %s\n", what, cudaGetErrorString(status));
        std::abort();
    }
}

void mpi_check(int status, const char* what) {
    if (status != MPI_SUCCESS) {
        std::fprintf(stderr, "[timing] %s -> MPI error %d\n", what, status);
        MPI_Abort(MPI_COMM_WORLD, status);
    }
}

double median_of(std::vector<double> values) {
    values.erase(std::remove_if(values.begin(), values.end(),
                                [](double x) { return !std::isfinite(x); }),
                 values.end());
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return (n % 2) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

}  // namespace

TimingRecorder::TimingRecorder(const int n_measured_iteration, const int marks_per_iteration,
                               std::vector<std::string> event_names,
                               MPI_Comm comm, const bool detailed)
    : comm_(comm), n_iteration_(std::max(n_measured_iteration, 0)), detailed_(detailed) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &nprocs_);

    n_partition_event_ = static_cast<int>(event_names.size());
    event_names_ = std::move(event_names);
    event_names_.push_back("total_iteration");   // derived, always last
    n_event_ = n_partition_event_ + 1;

    marks_per_iteration_ = std::max(marks_per_iteration, 1);

    // spec 14: iteration*event must fit an MPI count.
    const long long n_data = static_cast<long long>(n_iteration_) * n_event_;
    if (n_data > static_cast<long long>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "[timing] n_iteration * n_event exceeds MPI count range\n");
        MPI_Abort(comm_, 1);
    }

    time_local_.assign(static_cast<std::size_t>(n_iteration_) * n_event_, 0.0);

    const std::size_t n_marks =
        static_cast<std::size_t>(n_iteration_) * (marks_per_iteration_ + 1);
    marks_.resize(n_marks);
    for (auto& event : marks_) cuda_check(cudaEventCreate(&event), "cudaEventCreate");
    mark_event_.assign(static_cast<std::size_t>(marks_per_iteration_), -1);
}

TimingRecorder::~TimingRecorder() {
    for (auto& event : marks_)
        if (event) cudaEventDestroy(event);
}

void TimingRecorder::start_window() {
    // The one alignment the spec allows (spec 5): drain warm-up work, line the
    // ranks up once, start the clock.
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (window open)");
    MPI_Barrier(comm_);
    window_start_ = MPI_Wtime();
}

void TimingRecorder::stop_window() {
    // spec 6.3: every measured GPU operation must have finished, or the window
    // is shorter than the work.
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize (window close)");
    window_local_ = MPI_Wtime() - window_start_;
}

void TimingRecorder::begin_iteration(const int iteration) {
    current_iteration_ = iteration;
    current_mark_ = 0;
    if (iteration < 0 || iteration >= n_iteration_) { current_iteration_ = -1; return; }
    const std::size_t m =
        static_cast<std::size_t>(iteration) * (marks_per_iteration_ + 1);
    cuda_check(cudaEventRecord(marks_[m], 0), "cudaEventRecord (open)");
}

void TimingRecorder::mark(const int event) {
    if (current_iteration_ < 0) return;
    if (event < 0 || event >= n_partition_event_) {
        std::fprintf(stderr, "[timing] mark with event id %d out of range\n", event);
        MPI_Abort(comm_, 1);
    }
    if (current_mark_ >= marks_per_iteration_) {
        std::fprintf(stderr, "[timing] more marks than the %d declared\n",
                     marks_per_iteration_);
        MPI_Abort(comm_, 1);
    }
    // The pattern must repeat identically, or a column would mean one thing on
    // some rows and another on others.
    if (mark_event_[current_mark_] < 0) {
        mark_event_[current_mark_] = event;
    } else if (mark_event_[current_mark_] != event) {
        std::fprintf(stderr, "[timing] mark %d is '%s' this iteration but was '%s'\n",
                     current_mark_, event_names_[event].c_str(),
                     event_names_[mark_event_[current_mark_]].c_str());
        MPI_Abort(comm_, 1);
    }
    const std::size_t m = static_cast<std::size_t>(current_iteration_) *
                              (marks_per_iteration_ + 1) + current_mark_ + 1;
    cuda_check(cudaEventRecord(marks_[m], 0), "cudaEventRecord");
    ++current_mark_;
}

void TimingRecorder::end_iteration() {
    if (current_iteration_ >= 0 && current_mark_ != marks_per_iteration_) {
        std::fprintf(stderr, "[timing] iteration %d closed after %d of %d marks\n",
                     current_iteration_, current_mark_, marks_per_iteration_);
        MPI_Abort(comm_, 1);
    }
    current_iteration_ = -1;
}

void TimingRecorder::harvest() {
    if (harvested_ || n_iteration_ == 0) return;
    // Everything was recorded asynchronously; one wait on the last mark drains
    // the lot.  Runs after the window closed, so it costs the run nothing.
    cuda_check(cudaEventSynchronize(marks_.back()), "cudaEventSynchronize");

    for (int i = 0; i < n_iteration_; ++i) {
        const std::size_t base =
            static_cast<std::size_t>(i) * (marks_per_iteration_ + 1);
        for (int k = 0; k < marks_per_iteration_; ++k) {
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds,
                                            marks_[base + k], marks_[base + k + 1]),
                       "cudaEventElapsedTime");
            time_local_[index(i, mark_event_[k])] +=
                static_cast<double>(milliseconds) * 1e-3;
        }
        // total_iteration = telescoping sum of the partition, per rank per
        // iteration — never a later sum of per-event maxima (spec 4).
        double total = 0.0;
        for (int e = 0; e < n_partition_event_; ++e) total += time_local_[index(i, e)];
        time_local_[index(i, n_partition_event_)] = total;
    }
    harvested_ = true;
}

void TimingRecorder::reduce() {
    harvest();

    const double reduce_start = MPI_Wtime();
    const int n_data = n_iteration_ * n_event_;

    time_max_.assign(n_data, 0.0);
    time_sum_.assign(n_data, 0.0);

    // spec 8.1: one collective over the whole array, after the loop.
    mpi_check(MPI_Reduce(time_local_.data(), time_max_.data(), n_data, MPI_DOUBLE,
                         MPI_MAX, 0, comm_), "MPI_Reduce(MAX)");
    mpi_check(MPI_Reduce(time_local_.data(), time_sum_.data(), n_data, MPI_DOUBLE,
                         MPI_SUM, 0, comm_), "MPI_Reduce(SUM)");

    if (detailed_) {
        time_min_.assign(n_data, 0.0);
        time_sumsq_.assign(n_data, 0.0);
        std::vector<double> local_squares(n_data);
        for (int k = 0; k < n_data; ++k)
            local_squares[k] = time_local_[k] * time_local_[k];
        mpi_check(MPI_Reduce(time_local_.data(), time_min_.data(), n_data, MPI_DOUBLE,
                             MPI_MIN, 0, comm_), "MPI_Reduce(MIN)");
        mpi_check(MPI_Reduce(local_squares.data(), time_sumsq_.data(), n_data, MPI_DOUBLE,
                             MPI_SUM, 0, comm_), "MPI_Reduce(SUMSQ)");
    }

    mpi_check(MPI_Reduce(&window_local_, &window_max_, 1, MPI_DOUBLE, MPI_MAX, 0,
                         comm_), "MPI_Reduce(window)");

    // spec 8.3: reported as metadata, never folded into the solver time.
    reduction_overhead_ = MPI_Wtime() - reduce_start;
}

double TimingRecorder::median_max(const int event) const {
    if (rank_ != 0 || time_max_.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> values(n_iteration_);
    for (int i = 0; i < n_iteration_; ++i) values[i] = time_max_[index(i, event)];
    return median_of(std::move(values));
}

double TimingRecorder::mean_max(const int event) const {
    if (rank_ != 0 || time_max_.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    for (int i = 0; i < n_iteration_; ++i) sum += time_max_[index(i, event)];
    return n_iteration_ ? sum / n_iteration_ : std::numeric_limits<double>::quiet_NaN();
}

double TimingRecorder::median_imbalance(const int event) const {
    if (rank_ != 0 || time_max_.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> values(n_iteration_, std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < n_iteration_; ++i) {
        const double rank_max = time_max_[index(i, event)];
        const double rank_avg = time_sum_[index(i, event)] / nprocs_;
        if (rank_avg > 0.0) values[i] = (rank_max - rank_avg) / rank_avg;   // fraction (spec 7.1)
    }
    return median_of(std::move(values));
}

void TimingRecorder::write_csv(const std::string& path,
                               const std::vector<std::pair<std::string, std::string>>& metadata) const {
    if (rank_ != 0) return;

    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "[timing] cannot open %s for writing\n", path.c_str());
        MPI_Abort(comm_, 1);
    }
    out.setf(std::ios::scientific);
    out.precision(9);   // spec 11.1: >= 9 significant digits

    for (const auto& [key, value] : metadata) out << "# " << key << '=' << value << '\n';
    out << "# time_unit=second\n# imbalance_unit=fraction\n";
    out << "# window_max=" << window_max_ << '\n';
    out << "# window_time_per_iteration=" << window_per_iteration() << '\n';
    out << "# timing_reduction_overhead=" << reduction_overhead_ << '\n';

    // Header: total first, then every partition event (spec 11.2).
    out << "iteration";
    auto columns = [&](int e) {
        const std::string& name = event_names_[e];
        if (detailed_) out << ',' << name << "_min";
        out << ',' << name << "_avg" << ',' << name << "_max";
        if (detailed_) out << ',' << name << "_std" << ',' << name << "_imbalance";
    };
    columns(n_partition_event_);
    for (int e = 0; e < n_partition_event_; ++e) columns(e);
    out << '\n';

    for (int i = 0; i < n_iteration_; ++i) {
        out << i;
        auto values = [&](int e) {
            const double rank_max = time_max_[index(i, e)];
            const double rank_avg = time_sum_[index(i, e)] / nprocs_;
            if (detailed_) out << ',' << time_min_[index(i, e)];
            out << ',' << rank_avg << ',' << rank_max;
            if (detailed_) {
                const double mean_square = time_sumsq_[index(i, e)] / nprocs_;
                const double variance = std::max(0.0, mean_square - rank_avg * rank_avg);
                out << ',' << std::sqrt(variance);
                out << ',' << (rank_avg > 0.0 ? (rank_max - rank_avg) / rank_avg
                                              : std::numeric_limits<double>::quiet_NaN());
            }
        };
        values(n_partition_event_);
        for (int e = 0; e < n_partition_event_; ++e) values(e);
        out << '\n';
    }
    out.close();
    if (!out) {
        std::fprintf(stderr, "[timing] write failed for %s\n", path.c_str());
        MPI_Abort(comm_, 1);
    }
    std::printf("[timing] wrote %s  (%d iterations x %d events)\n",
                path.c_str(), n_iteration_, n_event_);
}

void TimingRecorder::print_summary(
        const std::vector<std::pair<std::string, std::vector<int>>>& groups) const {
    if (rank_ != 0 || time_max_.empty()) return;
    std::printf("\n[timing] window_max                = %.9e s   <- the official scaling time\n",
                window_max_);
    std::printf("[timing] window_time_per_iteration = %.9e s\n", window_per_iteration());
    std::printf("[timing] timing_reduction_overhead = %.9e s   (not part of the run)\n",
                reduction_overhead_);

    std::printf("  %-28s %14s %14s %12s\n", "event", "median_max(s)", "mean_max(s)", "imbalance");
    auto row = [&](const char* name, double median, double mean, double imbalance) {
        std::printf("  %-28s %14.6e %14.6e %12.4f\n", name, median, mean, imbalance);
    };
    row(event_names_[n_partition_event_].c_str(), median_max(n_partition_event_),
        mean_max(n_partition_event_), median_imbalance(n_partition_event_));

    // Named groups: per-iteration sum of the member events' rank-max — an upper
    // bound on the group's true rank-max (different events may peak on
    // different ranks), printed for orientation only.
    for (const auto& [group_name, members] : groups) {
        std::vector<double> sums(n_iteration_, 0.0);
        for (int i = 0; i < n_iteration_; ++i)
            for (int e : members) sums[i] += time_max_[index(i, e)];
        double mean = 0.0;
        for (double v : sums) mean += v;
        mean = n_iteration_ ? mean / n_iteration_ : 0.0;
        row(("[" + group_name + "]").c_str(), median_of(sums), mean,
            std::numeric_limits<double>::quiet_NaN());
    }

    for (int e = 0; e < n_partition_event_; ++e)
        row(event_names_[e].c_str(), median_max(e), mean_max(e), median_imbalance(e));
}

}  // namespace mpmstd::post

#endif  // MPMSTD_BACKEND_CUDA
