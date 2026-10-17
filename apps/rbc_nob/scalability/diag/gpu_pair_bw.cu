// GPU 간 MPI_Sendrecv 대역폭 진단 — rbc_nob 의 halo 교환과 같은 패턴 (디바이스 버퍼, 0.5 MB 면, blocking Sendrecv 쌍방향).
// 랭크 2개. 랭크 r 은 GPU r (CUDA_VISIBLE_DEVICES 로 쌍을 고른다: "0,1" = NVLink, "3,4" = SYS).
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
  int ndev = 0; cudaGetDeviceCount(&ndev); cudaSetDevice(rank % ndev);
  cudaDeviceProp prop; cudaGetDeviceProperties(&prop, rank % ndev);
  char bus[32]; cudaDeviceGetPCIBusId(bus, 32, rank % ndev);
  const int other = (rank + 1) % size;
  int peer = 0; cudaDeviceCanAccessPeer(&peer, rank % ndev, other % ndev);
  const size_t n = (argc > 1 ? atol(argv[1]) : 256 * 256);   // doubles per message (기본 0.5 MB)
  double *send, *recv; cudaMalloc(&send, n * 8); cudaMalloc(&recv, n * 8); cudaMemset(send, 0, n * 8);
  const int iters = 200;
  for (int it = 0; it < 10; ++it)
    MPI_Sendrecv(send, n, MPI_DOUBLE, other, 0, recv, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();
  for (int it = 0; it < iters; ++it)
    MPI_Sendrecv(send, n, MPI_DOUBLE, other, 0, recv, n, MPI_DOUBLE, other, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  const double dt = (MPI_Wtime() - t0) / iters;
  // 호스트 버퍼 비교 (같은 경로를 호스트로)
  double* hs = (double*)malloc(n * 8); double* hr = (double*)malloc(n * 8);
  MPI_Barrier(MPI_COMM_WORLD);
  const double t1 = MPI_Wtime();
  for (int it = 0; it < iters; ++it)
    MPI_Sendrecv(hs, n, MPI_DOUBLE, other, 1, hr, n, MPI_DOUBLE, other, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  const double dth = (MPI_Wtime() - t1) / iters;
  printf("rank %d  GPU %d (%s, bus %s)  peer-access to rank %d's GPU: %s\n", rank, rank % ndev, prop.name, bus, other, peer ? "YES" : "NO");
  if (rank == 0)
    printf("Sendrecv %.2f MB device->device: %.3f ms  (%.2f GB/s per direction)   host->host: %.3f ms (%.2f GB/s)\n",
           n * 8 / 1e6, dt * 1e3, n * 8 / dt / 1e9, dth * 1e3, n * 8 / dth / 1e9);
  MPI_Finalize();
  return 0;
}
