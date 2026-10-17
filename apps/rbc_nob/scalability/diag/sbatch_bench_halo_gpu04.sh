#!/bin/bash
#SBATCH -p cas_v100nv_8 -w gpu04 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=2 --mem-per-cpu=8000M --time=00:15:00
#SBATCH -J bench_halo -o bench_halo_gpu04_%j.out --comment="field=mae;appl=in_house"
source "${MODULESHOME}/init/bash"; module purge; module load nvhpc/25.11_cuda12
B=/scratch/r908a06/KDH/MPM-STD_C/build_gpu/bin/bench_halo
echo "# host=$(hostname)"
for np in 1 2 4 8; do echo "-- strong 256^3 np $np"; mpirun -np $np $B 256 256 256; done
for np in 1 2 4 8; do echo "-- weak 256^3/rank np $np"; mpirun -np $np $B 256 256 $((256*np)); done
echo "-- weak np 8, NVLink order"; CUDA_VISIBLE_DEVICES=0,1,2,3,7,6,5,4 mpirun -np 8 -x CUDA_VISIBLE_DEVICES $B 256 256 2048
echo "-- weak np 8, bind-to none / map-by numa"; mpirun -np 8 --bind-to none $B 256 256 2048; mpirun -np 8 --map-by numa --bind-to core $B 256 256 2048
echo "-- weak np 8, UCX_TLS=cuda_ipc,cuda_copy,self,sm"; UCX_TLS=cuda_ipc,cuda_copy,self,sm mpirun -np 8 -x UCX_TLS $B 256 256 2048
echo "# done"
