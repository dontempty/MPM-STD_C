#!/bin/bash
#SBATCH -J cufft_v
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/cufft_verify.log

# Build + nsys-profile the cuFFT microbench. Counts H2D/D2H + cuMemcpyAsync for
# 50 steps × 96 cufftExecZ2Z. Per-step memcpy = total/50. If ≈ the dhvc 628/step,
# cuFFT is confirmed as the memcpy-storm source.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/prof
echo "=== build microbench ==="
nvcc -O3 -arch=sm_80 scripts/cufft_microbench.cu -lcufft -o build/cufft_microbench 2>&1 | tail -10
[ -x build/cufft_microbench ] || { echo "BUILD FAIL"; exit 1; }

echo; echo "=== nsys profile (50 steps × 96 FFT) ==="
nsys profile --trace=cuda --force-overwrite true -o apps/dhvc/prof/cufft_mb \
  build/cufft_microbench 192 193 48 50 2>&1 | tail -6

echo; echo "############ GPU MemOps (H2D/D2H/D2D) — cuFFT alone ############"
nsys stats --report cuda_gpu_mem_time_sum --format table apps/dhvc/prof/cufft_mb.nsys-rep 2>&1 | grep -A12 "MemOps" | head -14
echo; echo "############ CUDA API (cuMemcpyAsync count) ############"
nsys stats --report cuda_api_sum --format table apps/dhvc/prof/cufft_mb.nsys-rep 2>&1 | grep -iE "Name|Memcpy|cuMemcpy|Time \(%\)|----" | head -14
echo "=== per-step: divide counts by 50. dhvc ref: ~628 H2D/D2H per step, 96 FFT/step ==="
echo "=== CUFFT_VERIFY DONE ==="
