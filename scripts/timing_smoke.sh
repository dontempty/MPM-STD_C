#!/bin/bash
#SBATCH -J timing_smoke
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:25:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/rbc/logs/timing_smoke.log
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "### COMPILE ###"
make BACKEND=cuda apps 2>&1 | grep -E "error|Error|APP  " | tail -15
[ -x build/cuda/bin/apps/rbc_gpu_nob ] || { echo "BUILD FAILED"; exit 1; }
echo "### [A] timing OFF — md5 must equal baseline ###"
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_fft_verify.toml 2>&1 | grep -E "^step=20|restart" | tail -2
md5sum apps/rbc/_fft_verify/tmp/{U,V,W,T}.bin | sed 's#apps/rbc/_fft_verify/tmp/##'
echo "--- baseline ---"
grep -E "U.bin|V.bin|W.bin|T.bin" apps/rbc/_fft_verify/BASELINE_md5.txt | sed 's#ref_run1/##'
echo "### [B] timing ON (warmup=5, measure=10) — summary + CSV + md5 ###"
mpirun -np 2 build/cuda/bin/apps/rbc_gpu_nob apps/rbc/input_timing_smoke.toml 2>&1 | grep -vE "^\[FWD|^\[NOB_PRES|^  \[bc|^step=[0-9]+ t=0\.[0-3]" | tail -75
md5sum apps/rbc/_fft_verify/tmp/{U,V,W,T}.bin | sed 's#apps/rbc/_fft_verify/tmp/##'
ls -la apps/rbc/_fft_verify/timing/
echo "### DONE ###"
