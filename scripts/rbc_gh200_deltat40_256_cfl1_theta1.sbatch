#!/bin/bash
#SBATCH -J rbc_dt40_gh200
#SBATCH --comment inhouse
#SBATCH -p gh200_1
#SBATCH -w gpu51
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_gh200_deltat40_256_cfl1_theta1.%j.log
# ---------------------------------------------------------------------------
# RBC NOB DeltaT=40K, 256^3, cold start, MaxCFL=1.0, theta=1 (no MPMSTD_BLKRELAX),
# n_steps=100000. Restart: step 50000 -> restart_.../ckpt_b/, step 100000 (end)
# -> restart_.../ (final write).
#
# gpu51 is a GH200 node (aarch64 + H100 cc90): the x86 build/cuda tree cannot
# run here, so this job compiles PaScaL_TDMA (build_arm_sm90) and MPM-STD_C
# (build/cuda_gh200) on the node with ARM NVHPC 24.5 before running.
# ---------------------------------------------------------------------------
set -eo pipefail

ROOT=/scratch/x3319a05/MPM-STD_C
PASCAL=/scratch/x3319a05/TDMA/PaScaL_TDMA
NVR=/apps/ARM_node/ARM_nvidia_hpc_sdk/24.5/Linux_aarch64/24.5
FFTW_ARM=/apps/ARM_node/ARM_gcc/11.4.1/applib1/fftw3/3.3.7

export PATH=$NVR/cuda/bin:$NVR/comm_libs/mpi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$NVR/cuda/lib64:$NVR/comm_libs/mpi/lib:$NVR/math_libs/lib64:$NVR/compilers/lib:$FFTW_ARM/lib:$LD_LIBRARY_PATH

echo "=== node: $(hostname) / $(uname -m) ==="
nvidia-smi -L || true
which mpic++ nvcc

# --- 1) PaScaL_TDMA: aarch64 + sm_90 (skip if already built on a prior run) ---
if [ ! -f $PASCAL/build_arm_sm90/lib/libpascal_tdma.a ]; then
  echo "=== building PaScaL_TDMA build_arm_sm90 ==="
  make -C $PASCAL lib BUILDDIR=build_arm_sm90 USE_CUDA=1 CUDA_ARCH=90 USE_NVTX=0 \
       CXXFLAGS="-O3 -std=c++17 -fPIC"
fi

# --- 2) MPM-STD_C: BACKEND=cuda, cc90, separate build tree ------------------
echo "=== building MPM-STD_C build/cuda_gh200 ==="
cd $ROOT
make BACKEND=cuda GPU_CC="90,nordc" BUILD_TAG=cuda_gh200_nordc HOST_TP= \
     PASCAL_TDMA_BUILD=build_arm_sm90 \
     NVHPC_MATH_LIB=$NVR/math_libs/lib64 \
     NVHPC_MATH_INC=$NVR/math_libs/include \
     FFTW3_DIR=$FFTW_ARM \
     lib apps -j16

BIN=$ROOT/build/cuda_gh200_nordc/bin/apps/rbc_gpu_nob
ls -l $BIN

# --- 3) run ------------------------------------------------------------------
mkdir -p $ROOT/apps/rbc/logs \
         $ROOT/apps/rbc/gh200_deltat40_256_cfl1_theta1 \
         $ROOT/apps/rbc/restart_gh200_deltat40_256_cfl1_theta1

# hpcx/nvhpc mpirun vs Slurm-injected PMIx conflict — same fix as the proven
# Filtered_TDMA H200 sbatch scripts.
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export UCX_TLS=cuda_copy,cuda_ipc,sm,self
export UCX_MEMTYPE_CACHE=n

echo "=== RUN start $(date) ==="
INPUT=apps/rbc/input_gh200_deltat40_256_cfl1_theta1.toml
if ! mpirun -np 1 $BIN $INPUT 2>&1; then
  echo "=== mpirun failed (rc=$?) — retrying as direct singleton exec ==="
  $BIN $INPUT 2>&1
fi
echo "=== RUN done $(date) ==="
ls -lR $ROOT/apps/rbc/restart_gh200_deltat40_256_cfl1_theta1 | head -30
