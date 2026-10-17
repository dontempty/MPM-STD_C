#!/bin/bash
# 4번 측정: PaScaL 축소계 Alltoallv 경로 — 순수 MPI 벤치(디바이스 vs 호스트 경유, UCX 변형) + 솔버 host 모드 비교는 별도 잡(TAG=v100hst).
#SBATCH -p cas_v100nv_8 -w gpu04 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=2 --mem-per-cpu=8000M --time=00:20:00
#SBATCH -J a2a_gpu04 -o alltoall_gpu04_%j.out --comment="field=mae;appl=in_house"
source "${MODULESHOME}/init/bash"; module purge; module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
cd "$SLURM_SUBMIT_DIR"
echo "# host=$(hostname)"; nvidia-smi topo -m | head -10
mpic++ -cuda -gpu=cc70 -O2 alltoall_bw.cu -o alltoall_bw || exit 1
for nsys in 65536 262144; do
  for np in 2 4 8; do echo "=== np $np nsys $nsys (default UCX) ==="; mpirun -np $np ./alltoall_bw $nsys; done
  echo "=== np 8 nsys $nsys UCX_TLS=cuda_ipc,cuda_copy,self,sm ==="; UCX_TLS=cuda_ipc,cuda_copy,self,sm mpirun -np 8 -x UCX_TLS ./alltoall_bw $nsys
  echo "=== np 8 nsys $nsys UCX_RNDV_SCHEME=get_zcopy ==="; UCX_RNDV_SCHEME=get_zcopy mpirun -np 8 -x UCX_RNDV_SCHEME ./alltoall_bw $nsys
  echo "=== np 8 nsys $nsys 4+4 GPU 한 소켓만(0,1,2,3 ×2 ranks each 불가) → 소켓 내 4 ranks ==="; CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 -x CUDA_VISIBLE_DEVICES ./alltoall_bw $nsys
  echo "=== 소켓 사이 4 ranks (0,1,4,5) ==="; CUDA_VISIBLE_DEVICES=0,1,4,5 mpirun -np 4 -x CUDA_VISIBLE_DEVICES ./alltoall_bw $nsys
done
echo "# done"
