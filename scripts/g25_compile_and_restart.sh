#!/bin/bash
#SBATCH -J g25_comp
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH -t 00:12:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/g25_compile.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

touch apps/rbc/main_gpu_nob.cu apps/rbc/rbc_common.hpp

make BACKEND=cuda lib 2>&1
rm -f build/cuda/bin/apps/rbc_gpu_nob
make BACKEND=cuda build/cuda/bin/apps/rbc_gpu_nob 2>&1
BUILD_RC=$?
echo "BUILD_EXIT:$BUILD_RC"

if [ $BUILD_RC -eq 0 ]; then
  echo "=== Submitting g25a (20K extended) and g25b (60K extended) ==="
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/g25a_fig9_gpu_20_x.sh
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/g25b_fig9_gpu_60_x.sh
fi
