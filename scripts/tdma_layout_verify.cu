// Verify PaScaL_TDMA layout change: [2×n_sys] → [n_sys×2].
// Tests that distributed GPU solve (MPI_Ialltoallv path) gives the correct
// analytical solution for a standard 1D Poisson tridiagonal.
//
// System: a=-1, b=2, c=-1, d=1, n_row=8, Dirichlet (a[0]=c[n-1]=0).
// Analytical: D[j] = (j+1)*(n_row-j)/2 = {4,7,9,10,10,9,7,4}.
//
// Run with np=2 to exercise the distributed alltoallv path.
//   sbatch scripts/tdma_layout_verify.sh
#include "pascal_tdma_many_cuda.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHK(e) do { \
    cudaError_t _err=(e); \
    if(_err!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n", \
        cudaGetErrorString(_err),__FILE__,__LINE__); MPI_Abort(MPI_COMM_WORLD,1);} \
} while(0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    CUDA_CHK(cudaSetDevice(rank % 2));

    const int n_sys  = (argc > 1) ? atoi(argv[1]) : 64;   // independent systems
    const int n_row  = (argc > 2) ? atoi(argv[2]) : 8;    // rows per system (must be even for np=2)

    if (rank == 0)
        printf("[tdma_layout_verify] np=%d  n_sys=%d  n_row=%d\n", nprocs, n_sys, n_row);

    // Build reference analytical solution: x[j] = (j+1)*(n_row-j)/2
    std::vector<double> ref(n_row);
    for (int j = 0; j < n_row; ++j)
        ref[j] = (double)(j + 1) * (n_row - j) / 2.0;

    if (rank == 0) {
        printf("  Expected solution (first 8 rows):");
        int show = std::min(n_row, 8);
        for (int j = 0; j < show; ++j) printf(" %.1f", ref[j]);
        printf("\n");
    }

    // Allocate host arrays [n_row × n_sys]
    std::vector<double> hA(n_row * n_sys, -1.0);
    std::vector<double> hB(n_row * n_sys,  2.0);
    std::vector<double> hC(n_row * n_sys, -1.0);
    std::vector<double> hD(n_row * n_sys,  1.0);
    // Dirichlet boundaries
    for (int i = 0; i < n_sys; ++i) {
        hA[0 * n_sys + i]            = 0.0;
        hC[(n_row - 1) * n_sys + i]  = 0.0;
    }

    // Upload to device
    double *dA, *dB, *dC, *dD;
    const size_t sz = (size_t)n_row * n_sys * sizeof(double);
    CUDA_CHK(cudaMalloc(&dA, sz)); CUDA_CHK(cudaMalloc(&dB, sz));
    CUDA_CHK(cudaMalloc(&dC, sz)); CUDA_CHK(cudaMalloc(&dD, sz));
    CUDA_CHK(cudaMemcpy(dA, hA.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, hB.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dC, hC.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dD, hD.data(), sz, cudaMemcpyHostToDevice));

    double global_max = 0.0;

    // Scope: solver must be destroyed before MPI_Finalize (CUDA dtor only, no MPI calls)
    {
        PaScaLTDMAManyCUDA solver(n_sys, rank, nprocs, MPI_COMM_WORLD);
        solver.solve(dA, dB, dC, dD, n_sys, n_row);
        CUDA_CHK(cudaDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Download result
    CUDA_CHK(cudaMemcpy(hD.data(), dD, sz, cudaMemcpyDeviceToHost));

    // Validate: every system should match the reference
    double max_err = 0.0;
    for (int j = 0; j < n_row; ++j)
        for (int i = 0; i < n_sys; ++i) {
            double err = std::fabs(hD[j * n_sys + i] - ref[j]);
            if (err > max_err) max_err = err;
        }

    MPI_Reduce(&max_err, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("  max error vs analytical = %.3e  (%s)\n",
               global_max, global_max < 1e-10 ? "PASS" : "FAIL");
        if (global_max >= 1e-10) {
            printf("  First system result (rows 0..min(n_row,8)-1):");
            int show = std::min(n_row, 8);
            for (int j = 0; j < show; ++j)
                printf(" %.6f", hD[j * n_sys + 0]);
            printf("\n");
        }
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    MPI_Finalize();
    return (rank == 0 && global_max >= 1e-10) ? 1 : 0;
}
