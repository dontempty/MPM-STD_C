#pragma once

#include <mpi.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace pascal_tdma {

enum class MpiBufferMode {
    DeviceDirect,
    HostStaging
};

struct CudaError : public std::runtime_error {
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

void cuda_check(cudaError_t status, const char* expr, const char* file, int line);

#define PASCAL_TDMA_CUDA_CHECK(expr) \
    ::pascal_tdma::cuda_check((expr), #expr, __FILE__, __LINE__)

inline std::size_t index2(const int sys, const int row, const int nsys) {
    return static_cast<std::size_t>(sys) + static_cast<std::size_t>(row) * nsys;
}

void partition_1d(int start, int end, int nprocs, int rank, int& first, int& last);

MpiBufferMode mpi_mode_from_env();

struct SolveTimings {
    double total = 0.0;

    double local_compute = 0.0;
    double pack_forward = 0.0;
    double mpi_forward = 0.0;
    double unpack_forward = 0.0;

    double reduced_compute = 0.0;

    double pack_backward = 0.0;
    double mpi_backward = 0.0;
    double unpack_backward = 0.0;

    double update_compute = 0.0;

    double communication() const {
        return mpi_forward + mpi_backward;
    }

    double packing() const {
        return pack_forward + unpack_forward + pack_backward + unpack_backward;
    }

    double computation() const {
        return local_compute + reduced_compute + update_compute;
    }
};

struct PascalTdmaPlan {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int nprocs = 1;
    int nsys = 0;
    int local_nsys = 0;
    int local_first_sys = 0;
    int local_last_sys = -1;

    int nrd_global[2] = {0, 0};
    int nrd_local[2] = {0, 0};
    int ntr_global[2] = {0, 0};
    int ntr_local[2] = {0, 0};

    std::vector<int> gather_nrd_local;
    std::vector<int> gather_ntr_local;
    std::vector<int> gather_nrd_start;
    std::vector<int> gather_ntr_start;

    std::vector<int> counts_a;
    std::vector<int> displs_a;
    std::vector<int> counts_b;
    std::vector<int> displs_b;
    std::vector<int> big_counts_a;
    std::vector<int> big_displs_a;
    std::vector<int> big_counts_b;
    std::vector<int> big_displs_b;

    double* ard = nullptr;
    double* brd = nullptr;
    double* crd = nullptr;
    double* drd = nullptr;
    double* atr = nullptr;
    double* btr = nullptr;
    double* ctr = nullptr;
    double* dtr = nullptr;
    double* bigbuf_a = nullptr;
    double* bigbuf_b = nullptr;
    // [최적화 2026-10-23] 순환(solve_cycle) Sherman–Morrison 보조 RHS 는 plan 마다 두지 않고
    //   모든 plan 이 공용 스크래치 하나를 쓴다 (pascal_tdma_cuda.cu cyclic_scratch) → 주기 축 plan 수만큼 −장

    std::size_t bigbuf_a_size = 0;
    std::size_t bigbuf_b_size = 0;
    std::size_t buf_a_size = 0;
    std::size_t buf_b_size = 0;

    dim3 threads_tdma = dim3(128, 1, 1);
    dim3 blocks_tdma = dim3(1, 1, 1);
    dim3 threads_reduced = dim3(128, 1, 1);
    dim3 blocks_reduced = dim3(1, 1, 1);
    dim3 threads_pack = dim3(128, 1, 1);
    dim3 threads_init = dim3(128, 1, 1);

    MpiBufferMode mpi_mode = MpiBufferMode::DeviceDirect;
    bool created = false;

    PascalTdmaPlan() = default;
    PascalTdmaPlan(const PascalTdmaPlan&) = delete;
    PascalTdmaPlan& operator=(const PascalTdmaPlan&) = delete;
    PascalTdmaPlan(PascalTdmaPlan&& other) noexcept;
    PascalTdmaPlan& operator=(PascalTdmaPlan&& other) noexcept;
    ~PascalTdmaPlan();

    void create(int nsys_in,
                MPI_Comm comm_in,
                int tdma_threads = 128,
                int reduced_threads = 128,
                MpiBufferMode mode = mpi_mode_from_env());

    void destroy() noexcept;
};

/// `comm_seconds`, when given, accumulates the wall time this call spends
/// inside MPI.  It costs nothing to measure: the algorithm already
/// synchronises the stream before handing device buffers to MPI, so the
/// bracket adds no synchronisation of its own and the measured loop stays free
/// of instrumentation (see ../../Scalability/timing.hpp).
void solve(PascalTdmaPlan& plan,
           double* a_dev,
           double* b_dev,
           double* c_dev,
           double* d_dev,
           int nsys,
           int nrow,
           cudaStream_t stream = nullptr,
           double* comm_seconds = nullptr);

// 주기(순환) 삼중대각 — x/y 주기 축 스윕용. 로컬 소거는 solve 와 동일하고
// 축약계만 순환 Thomas 커널로 푼다 (MPM-STD 2026-10-04 신설; 참조 구현 =
// MPM-STD 의 pascal_tdma_cpu.cpp + apps/test_tdma).
void solve_cycle(PascalTdmaPlan& plan,
                 double* a_dev,
                 double* b_dev,
                 double* c_dev,
                 double* d_dev,
                 int nsys,
                 int nrow,
                 cudaStream_t stream = nullptr,
                 double* comm_seconds = nullptr);

void solve_profiled(PascalTdmaPlan& plan,
                    double* a_dev,
                    double* b_dev,
                    double* c_dev,
                    double* d_dev,
                    int nsys,
                    int nrow,
                    SolveTimings* timings,
                    cudaStream_t stream = nullptr);

}  // namespace pascal_tdma
