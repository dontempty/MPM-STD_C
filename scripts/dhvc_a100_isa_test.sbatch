#!/bin/bash
#SBATCH -J isa_test
#SBATCH --comment inhouse
#SBATCH -p amd_a100nv_8
#SBATCH -w gpu43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/dhvc/logs/isa_test.%j.log
set -uo pipefail
lscpu | grep -E "Model name|Flags" | cut -c1-120
NVR=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11
export PATH=$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/bin:$PATH
export LD_LIBRARY_PATH=$NVR/cuda/12.9/lib64:$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/lib:$NVR/math_libs/12.9/lib64:$NVR/compilers/lib:/apps/libraries/fftw3/3.3.10/gcc/11.5.0/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
cd /scratch/x3319a05/MPM-STD_C
echo "=== nvc++ CPU dhvc tiny on gpu43 ==="
mkdir -p apps/dhvc/out_isatest; timeout 120 mpirun --bind-to none -np 1 build/cuda/bin/apps/dhvc apps/dhvc/input_tiny_isatest.toml 2>&1 | tail -6
echo "rc=$?"
