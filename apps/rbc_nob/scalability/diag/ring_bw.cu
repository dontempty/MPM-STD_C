#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv); int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ndev; cudaGetDeviceCount(&ndev); cudaSetDevice(rank % ndev);
  const size_t n = 256 * 256; double *s, *r; cudaMalloc(&s, n * 8); cudaMalloc(&r, n * 8); cudaMemset(s, 0, n * 8);
  const int plus = rank + 1 < size ? rank + 1 : MPI_PROC_NULL, minus = rank > 0 ? rank - 1 : MPI_PROC_NULL;   // z 벽: 랩 없음
  for (int it = 0; it < 10; ++it) { MPI_Sendrecv(s, n, MPI_DOUBLE, minus, 0, r, n, MPI_DOUBLE, plus, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    MPI_Sendrecv(s, n, MPI_DOUBLE, plus, 1, r, n, MPI_DOUBLE, minus, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
  MPI_Barrier(MPI_COMM_WORLD); const double t0 = MPI_Wtime(); const int iters = 100;
  for (int it = 0; it < iters; ++it) { MPI_Sendrecv(s, n, MPI_DOUBLE, minus, 0, r, n, MPI_DOUBLE, plus, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                       MPI_Sendrecv(s, n, MPI_DOUBLE, plus, 1, r, n, MPI_DOUBLE, minus, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
  double dt = (MPI_Wtime() - t0) / iters, mx; MPI_Reduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) printf("ring halo exchange (2 Sendrecv of 0.5 MB), %d ranks: %.3f ms per exchange (max over ranks)\n", size, mx * 1e3);
  MPI_Finalize(); return 0; }
