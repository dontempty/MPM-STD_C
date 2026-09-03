#!/bin/bash
#SBATCH -J fft_verify_base
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/fft_verify_baseline.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
BASE=apps/rbc/_fft_verify
echo "### RUN 1 (current HEAD binary) ###"
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_fft_verify.toml 2>&1 | grep -E "step=|nan|error" | tail -3
cp $BASE/tmp/{U,V,W,T,P}.bin $BASE/ref_run1/
echo "### RUN 2 (current HEAD binary, identical config) ###"
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_fft_verify.toml 2>&1 | grep -E "step=|nan|error" | tail -3
cp $BASE/tmp/{U,V,W,T,P}.bin $BASE/ref_run2/
echo "### md5 run1 vs run2 ###"
cd $BASE && md5sum ref_run1/*.bin ref_run2/*.bin
echo "### DONE ###"
