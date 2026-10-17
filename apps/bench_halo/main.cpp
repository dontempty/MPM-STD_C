// halo 교환 진단 벤치 — 솔버 없이 Subdomain::exchange_halo 만 반복해 비용을 잰다 (docs/Scalability/scalar_scalability.md §4 의 np 8 문제).
//   mpirun -np N ./build_gpu/bin/bench_halo nx ny nz_global [iters]
// z-슬랩(npz = N), x·y 주기(npx = npy = 1 → 이웃 = 자기 자신 → x·y 면은 자기 자신에게 MPI_Sendrecv), z 벽.
// 출력: 교환 1회 ms (랭크 max·min), 그리고 랭크 사이 어긋남(skew)을 보기 위해 Barrier 뒤 교환만 잰 값도 같이.
#include <mpi.h>
#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "field/field.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"

int main(int argc, char** argv) {
  mpmstd::parallel::MpiContext mpi(&argc, &argv);
#ifdef MPMSTD_GPU
  int n_devices = 0;
  cudaGetDeviceCount(&n_devices);
  if (n_devices > 0) cudaSetDevice(mpi.node_rank() % n_devices);
#endif
  if (argc < 4) {
    if (mpi.is_root()) std::fprintf(stderr, "usage: bench_halo nx ny nz_global [iters]\n");
    return 1;
  }
  const int nx = std::atoi(argv[1]), ny = std::atoi(argv[2]), nz = std::atoi(argv[3]);
  const int iters = argc > 4 ? std::atoi(argv[4]) : 100;
  const mpmstd::parallel::Topology topo(mpi, 1, 1, mpi.world_size(), true, true, false);
  const mpmstd::parallel::Subdomain sub(topo, nx, ny, nz);
  mpmstd::field::Field f(sub.x().n_total, sub.y().n_total, sub.z().n_total, "f");
  f.fill(1.0);
  for (int i = 0; i < 10; ++i) sub.exchange_halo(f.data());
  MPI_Barrier(MPI_COMM_WORLD);

  // ① 연속 교환 (솔버 패턴과 가장 가까움: 랭크들이 서로 기다리는 시간 포함)
  double t0 = MPI_Wtime();
  for (int i = 0; i < iters; ++i) sub.exchange_halo(f.data());
  double dt_free = (MPI_Wtime() - t0) / iters;
  // ② 매번 Barrier 로 맞춘 뒤 교환만 (교환 자체의 비용)
  double sum = 0;
  for (int i = 0; i < iters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double a = MPI_Wtime();
    sub.exchange_halo(f.data());
    sum += MPI_Wtime() - a;
  }
  double dt_sync = sum / iters;
  double mx_free, mn_free, mx_sync, mn_sync;
  MPI_Reduce(&dt_free, &mx_free, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&dt_free, &mn_free, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&dt_sync, &mx_sync, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&dt_sync, &mn_sync, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  if (mpi.is_root())
    std::printf("bench_halo %dx%dx%d np %d (nz_local %d): x-plane %.2f MB, y-plane %.2f MB, z-face %.2f MB | "
                "exchange: free %.3f ms (min %.3f), barrier-synced %.3f ms (min %.3f)\n",
                nx, ny, nz, mpi.world_size(), sub.z().n_local,
                sub.y().n_total * sub.z().n_total * 8e-6, sub.x().n_total * sub.z().n_total * 8e-6,
                sub.x().n_total * sub.y().n_total * 8e-6, mx_free * 1e3, mn_free * 1e3, mx_sync * 1e3, mn_sync * 1e3);
  return 0;
}
