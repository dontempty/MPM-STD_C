#!/bin/bash
# gpu04 GPU 쌍 통신 진단: 우리 halo 패턴(디바이스 버퍼 0.5 MB blocking Sendrecv)으로 NVLink 쌍(0,1)·SYS 쌍(3,4)·(0,4) 대역폭 + UCX 가 고른 transport.
#SBATCH -p cas_v100nv_8 -w gpu04 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=2 --mem-per-cpu=8000M --time=00:15:00
#SBATCH -J diag_gpu04 -o diag_gpu04_%j.out --comment="field=mae;appl=in_house"
source "${MODULESHOME}/init/bash"; module purge; module load nvhpc/25.11_cuda12
cd "$SLURM_SUBMIT_DIR"
echo "# host=$(hostname)"; nvidia-smi topo -m | head -10
mpic++ -cuda -gpu=cc70 -O2 gpu_pair_bw.cu -o gpu_pair_bw || exit 1
for pair in "0,1" "2,3" "3,4" "0,4" "0,7"; do
  echo "=== pair $pair (default UCX) ==="
  CUDA_VISIBLE_DEVICES=$pair mpirun -np 2 -x CUDA_VISIBLE_DEVICES ./gpu_pair_bw
done
echo "=== pair 0,1 with UCX_TLS=cuda_ipc,cuda_copy,self,sm ==="
CUDA_VISIBLE_DEVICES=0,1 UCX_TLS=cuda_ipc,cuda_copy,self,sm mpirun -np 2 -x CUDA_VISIBLE_DEVICES -x UCX_TLS ./gpu_pair_bw
echo "=== pair 0,1 protocol info (UCX_PROTO_INFO=y) ==="
CUDA_VISIBLE_DEVICES=0,1 UCX_PROTO_INFO=y mpirun -np 2 -x CUDA_VISIBLE_DEVICES -x UCX_PROTO_INFO ./gpu_pair_bw 2>&1 | grep -iE "cuda|rendezvous|eager|GB/s|peer" | head -40
echo "=== 8-rank ring (our halo pattern: Sendrecv to r+1 and r-1), default order vs NVLink order ==="
cat > ring_bw.cu <<'EOT'
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
EOT
mpic++ -cuda -gpu=cc70 -O2 ring_bw.cu -o ring_bw || exit 1
for np in 2 4 8; do echo "-- default order np $np"; mpirun -np $np ./ring_bw; done
echo "-- NVLink order 0,1,2,3,7,6,5,4 np 8"; CUDA_VISIBLE_DEVICES=0,1,2,3,7,6,5,4 mpirun -np 8 -x CUDA_VISIBLE_DEVICES ./ring_bw
echo "-- np 8 with UCX_TLS=cuda_ipc,cuda_copy,self,sm"; UCX_TLS=cuda_ipc,cuda_copy,self,sm mpirun -np 8 -x UCX_TLS ./ring_bw
echo "# done"
