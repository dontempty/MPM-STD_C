#!/bin/bash
#SBATCH -J rbc_gh200_nordc
#SBATCH --comment inhouse
#SBATCH -p gh200_1
#SBATCH -w gpu51
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:40:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_gh200_nordc.%j.log
# V1: full rbc_gpu_nob rebuilt with -gpu=cc90,nordc (SDK 24.5) + 200-step smoke.
# V2 (only if V1 crashes): same with SDK 24.1 (default rdc).
set -uo pipefail
ROOT=/scratch/x3319a05/MPM-STD_C
FFTW_ARM=/apps/ARM_node/ARM_gcc/11.4.1/applib1/fftw3/3.3.7
PASCAL_BUILD=build_arm_sm90
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
cd $ROOT
mkdir -p apps/rbc/gh200_smoke apps/rbc/restart_gh200_smoke

build_and_smoke () { # tag nvr gpucc
  local tag=$1 nvr=$2 gpucc=$3
  export PATH=$nvr/cuda/bin:$nvr/comm_libs/mpi/bin:$nvr/compilers/bin:$PATH
  export LD_LIBRARY_PATH=$nvr/cuda/lib64:$nvr/comm_libs/mpi/lib:$nvr/math_libs/lib64:$nvr/compilers/lib:$FFTW_ARM/lib:${LD_LIBRARY_PATH:-}
  echo "=== [$tag] build (GPU_CC=$gpucc) ==="
  make BACKEND=cuda GPU_CC="$gpucc" BUILD_TAG=$tag \
       PASCAL_TDMA_BUILD=$PASCAL_BUILD \
       NVHPC_MATH_LIB=$nvr/math_libs/lib64 \
       NVHPC_MATH_INC=$nvr/math_libs/include \
       FFTW3_DIR=$FFTW_ARM \
       lib apps -j16 2>&1 | grep -vE "warning|Remark|\^" | tail -8
  local bin=build/$tag/bin/apps/rbc_gpu_nob
  [ -x $bin ] || { echo "[$tag] BUILD FAILED"; return 1; }
  echo "=== [$tag] smoke run 200 steps ==="
  if timeout 900 mpirun -np 1 $bin apps/rbc/input_gh200_smoke.toml 2>&1 | tail -25; then
    echo "[$tag] SMOKE OK"
  else
    echo "[$tag] SMOKE FAILED rc=$?"
    return 1
  fi
}

NV245=/apps/ARM_node/ARM_nvidia_hpc_sdk/24.5/Linux_aarch64/24.5
NV241=/apps/ARM_node/ARM_nvidia_hpc_sdk/24.1/Linux_aarch64/24.1

if build_and_smoke cuda_gh200_nordc $NV245 "90,nordc"; then
  echo "### RESULT: nordc/24.5 works"
else
  echo "### nordc/24.5 failed -> trying SDK 24.1 rdc"
  if build_and_smoke cuda_gh200_241 $NV241 "90"; then
    echo "### RESULT: SDK 24.1 works"
  else
    echo "### RESULT: both variants failed"
  fi
fi
