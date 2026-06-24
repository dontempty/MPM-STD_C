// Is the runtime MPI actually CUDA-aware, or does it host-stage device-pointer
// collectives? Profile MPI_Alltoall on DEVICE buffers. If nsys shows H2D/D2H per
// call → MPI host-stages (NOT cuda-aware) = the dhvc memcpy-storm source.
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (rank == 0) {
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        printf("[query] MPIX_Query_cuda_support() = %d  (compile MPIX_CUDA_AWARE_SUPPORT=1)\n",
               MPIX_Query_cuda_support());
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
        printf("[query] MPIX_CUDA_AWARE_SUPPORT defined=0\n");
#else
        printf("[query] MPIX_CUDA_AWARE_SUPPORT NOT defined at compile\n");
#endif
        printf("[query] ranks=%d\n", size);
    }
    const int n = (argc>1)?atoi(argv[1]):37056;   // per-peer chunk (≈ pressure slab)
    const int nstep = (argc>2)?atoi(argv[2]):50;
    double *sbuf=nullptr,*rbuf=nullptr;
    cudaMalloc(&sbuf,(size_t)n*size*sizeof(double));
    cudaMalloc(&rbuf,(size_t)n*size*sizeof(double));
    cudaMemset(sbuf,0,(size_t)n*size*sizeof(double));
    cudaDeviceSynchronize();
    // warmup
    MPI_Alltoall(sbuf,n,MPI_DOUBLE,rbuf,n,MPI_DOUBLE,MPI_COMM_WORLD);
    cudaDeviceSynchronize(); MPI_Barrier(MPI_COMM_WORLD);
    // measured: 50 device-pointer alltoalls
    for (int s=0;s<nstep;++s) {
        MPI_Alltoall(sbuf,n,MPI_DOUBLE,rbuf,n,MPI_DOUBLE,MPI_COMM_WORLD);
        cudaDeviceSynchronize();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) printf("[done] %d device-ptr MPI_Alltoall (n=%d/peer)\n", nstep, n);
    cudaFree(sbuf); cudaFree(rbuf);
    MPI_Finalize();
    return 0;
}
