#!/bin/bash
#SBATCH -J g20_compile
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/g20_compile.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

touch apps/rbc/main_gpu_nob.cu

make BACKEND=cuda lib 2>&1
rm -f build/cuda/bin/apps/rbc_gpu_nob
make BACKEND=cuda apps/rbc_gpu_nob 2>&1 || make BACKEND=cuda apps 2>&1
BUILD_RC=$?
echo "BUILD_EXIT:$BUILD_RC"
if [ $BUILD_RC -eq 0 ]; then
  echo "=== Submitting g23/g24 long runs with new GPU-stats binary ==="
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/g23_fig9_gpu_20_long.sh
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/g24_fig9_gpu_60_long.sh
fi
