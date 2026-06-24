#!/bin/bash
#SBATCH -J tdma_verify
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/tdma_layout_verify.log

set -e
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

PASCAL_INC=/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include
PASCAL_LIB=/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib

echo "=== build ==="
nvcc -O3 -std=c++17 -arch=sm_80 -ccbin mpic++ \
  -I"${PASCAL_INC}" \
  scripts/tdma_layout_verify.cu \
  -L"${PASCAL_LIB}" -lpascal_tdma \
  -o build/tdma_layout_verify \
  2>&1

echo "=== run (np=2, n_sys=256, n_row=8) ==="
mpirun -np 2 build/tdma_layout_verify 256 8

echo "=== run (np=2, n_sys=1024, n_row=16) ==="
mpirun -np 2 build/tdma_layout_verify 1024 16

echo "=== DONE ==="
