// PaScaL 축소계 Alltoallv 와 같은 크기의 순수 MPI 벤치 (디바이스 버퍼 vs 호스트 경유).
//   앞: 각 랭크가 상대마다 3배열 × 2행 × (nsys/np) double, 뒤: 1배열 × 2행 × (nsys/np).
//   인자: nsys (기본 65536 = 256², 512³ 는 262144)
#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
static double bench(const double* s, double* r, const std::vector<int>& cnt, const std::vector<int>& dsp, int iters) {
  for (int i = 0; i < 5; ++i) MPI_Alltoallv(s, cnt.data(), dsp.data(), MPI_DOUBLE, r, cnt.data(), dsp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); const double t0 = MPI_Wtime();
  for (int i = 0; i < iters; ++i) MPI_Alltoallv(s, cnt.data(), dsp.data(), MPI_DOUBLE, r, cnt.data(), dsp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
  double dt = (MPI_Wtime() - t0) / iters, mx; MPI_Reduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); return mx;
}
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv); int rank, np; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &np);
  int ndev; cudaGetDeviceCount(&ndev); cudaSetDevice(rank % ndev);
  const long nsys = argc > 1 ? atol(argv[1]) : 65536;
  for (int arrays : {3, 1}) {
    const int per = static_cast<int>(arrays * 2 * (nsys / np)); const size_t n = static_cast<size_t>(per) * np;
    std::vector<int> cnt(np, per), dsp(np); for (int r = 0; r < np; ++r) dsp[r] = r * per;
    double *ds, *dr, *hs, *hr; cudaMalloc(&ds, n * 8); cudaMalloc(&dr, n * 8); cudaMemset(ds, 0, n * 8);
    cudaMallocHost(&hs, n * 8); cudaMallocHost(&hr, n * 8);
    const double mb = n * 8 / 1e6;
    const double t_dev = bench(ds, dr, cnt, dsp, 50);
    // 호스트 경유: D2H → Alltoallv(host) → H2D (PaScaL host 모드와 같은 순서)
    for (int i = 0; i < 5; ++i) { cudaMemcpy(hs, ds, n * 8, cudaMemcpyDeviceToHost); MPI_Alltoallv(hs, cnt.data(), dsp.data(), MPI_DOUBLE, hr, cnt.data(), dsp.data(), MPI_DOUBLE, MPI_COMM_WORLD); cudaMemcpy(dr, hr, n * 8, cudaMemcpyHostToDevice); }
    MPI_Barrier(MPI_COMM_WORLD); const double t0 = MPI_Wtime(); const int iters = 50;
    for (int i = 0; i < iters; ++i) { cudaMemcpy(hs, ds, n * 8, cudaMemcpyDeviceToHost); MPI_Alltoallv(hs, cnt.data(), dsp.data(), MPI_DOUBLE, hr, cnt.data(), dsp.data(), MPI_DOUBLE, MPI_COMM_WORLD); cudaMemcpy(dr, hr, n * 8, cudaMemcpyHostToDevice); }
    double dt = (MPI_Wtime() - t0) / iters, t_host; MPI_Reduce(&dt, &t_host, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const double t_hostonly = bench(hs, hr, cnt, dsp, 50);
    if (rank == 0) printf("nsys %ld np %d arrays %d: %.2f MB/rank | device-buffer %.3f ms (%.1f GB/s/rank) | host-staged %.3f ms (%.1f GB/s) | host Alltoallv only %.3f ms\n",
                          nsys, np, arrays, mb, t_dev * 1e3, mb / 1e3 / t_dev, t_host * 1e3, mb / 1e3 / t_host, t_hostonly * 1e3);
    cudaFree(ds); cudaFree(dr); cudaFreeHost(hs); cudaFreeHost(hr);
  }
  MPI_Finalize(); return 0; }
