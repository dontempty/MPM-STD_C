#!/bin/bash
#SBATCH -p cas_v100nv_8 -w gpu04 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=2 --mem-per-cpu=8000M --time=00:50:00
#SBATCH -J halo_fix -o halo_fix_gpu04_%j.out --comment="field=mae;appl=in_house"
source "${MODULESHOME}/init/bash"; module purge; module load nvhpc/25.11_cuda12
ROOT=/scratch/r908a06/KDH/MPM-STD_C; B=$ROOT/build_gpu/bin/bench_halo
echo "# host=$(hostname)  (self-neighbor fast path)"
for np in 1 2 4 8; do echo "-- strong 256^3 np $np"; mpirun -np $np $B 256 256 256; done
for np in 1 2 4 8; do echo "-- weak 256^3/rank np $np"; mpirun -np $np $B 256 256 $((256*np)); done
echo "# bench done; now rbc_nob gate + np 4/8 measurement (tag v100h)"
cd $ROOT/apps/rbc_nob
export TAG=v100h NP_LIST="1 2 4 8" CASES="strong_256 weak_256" REPEAT=2 BIN=$ROOT/build_gpu/bin/rbc_nob
bash scalability/sbatch_scal_a100_1node.sh
