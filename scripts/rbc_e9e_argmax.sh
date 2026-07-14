#!/bin/bash
#SBATCH -J rbc_e9e_argmax
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/rbc_e9e_argmax.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== E9e: 16-plane block mask + argmax locator — where does the regrown spike live? ==="
make BACKEND=cuda apps 2>&1 | tail -3
mkdir -p apps/rbc/e9e_argmax
mpirun -np 2 -x MPMSTD_MOMDBG=2 -x MPMSTD_BLKMASK_TOP=16 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_e9e_argmax.toml 2>&1 \
  | grep --line-buffered -E "step=|argmax|restart|nan|NaN|error|Error"
echo "=== E9e done ==="
