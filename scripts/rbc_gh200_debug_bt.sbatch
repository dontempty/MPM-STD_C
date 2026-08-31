#!/bin/bash
#SBATCH -J rbc_gh200_dbg
#SBATCH --comment inhouse
#SBATCH -p gh200_1
#SBATCH -w gpu51
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_gh200_debug_bt.%j.log
# Get a backtrace for the startup segfault of rbc_gpu_nob on GH200.
set -uo pipefail

ROOT=/scratch/x3319a05/MPM-STD_C
NVR=/apps/ARM_node/ARM_nvidia_hpc_sdk/24.5/Linux_aarch64/24.5
FFTW_ARM=/apps/ARM_node/ARM_gcc/11.4.1/applib1/fftw3/3.3.7
export PATH=$NVR/cuda/bin:$NVR/comm_libs/mpi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVR/cuda/lib64:$NVR/comm_libs/mpi/lib:$NVR/math_libs/lib64:$NVR/compilers/lib:$FFTW_ARM/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0

BIN=$ROOT/build/cuda_gh200/bin/apps/rbc_gpu_nob
INPUT=$ROOT/apps/rbc/input_gh200_deltat40_256_cfl1_theta1.toml
cd $ROOT

echo "=== unbuffered run (last lines before crash) ==="
stdbuf -o0 -e0 $BIN $INPUT 2>&1 | tail -30 || true

echo "=== cuda-gdb run ==="
$NVR/cuda/bin/cuda-gdb --batch \
    -ex "set pagination off" \
    -ex run \
    -ex "bt" \
    --args $BIN $INPUT 2>&1 | tail -60
echo "=== done ==="
