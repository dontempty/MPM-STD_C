#!/bin/bash
#SBATCH -J g8_ob_fix
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/g8_ob_fix_test.log
#SBATCH -e /shared/home/wel1come1234/workspace/MPM-STD_C/g8_ob_fix_test.log

set -e
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7 2>/dev/null || true

echo "=== G8: Fix kappa=1.0 for OB GPU scalar, rebuild + test ==="

touch src/equation/scalar/gpu/scalar_gpu.cu

echo "--- Build lib ---"
make BACKEND=cuda lib 2>&1; BRET=$?
echo "LIB_EXIT:${BRET}"
if [ $BRET -ne 0 ]; then echo "LIB BUILD FAILED"; exit 1; fi

echo "--- Build apps ---"
make BACKEND=cuda apps 2>&1; BRET=$?
echo "APPS_EXIT:${BRET}"
if [ $BRET -ne 0 ]; then echo "APPS BUILD FAILED"; exit 1; fi

echo ""
echo "=== GPU OB 32^3 (500 steps) — expect T in [-0.52, 0.52] throughout ==="
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_ob apps/rbc/input_gpu_32_ob_test.toml

echo ""
echo "=== GPU NOB 32^3 regression (100 steps, DeltaT=20) — must still match CPU ==="
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_gpu_32_nob_test.toml
