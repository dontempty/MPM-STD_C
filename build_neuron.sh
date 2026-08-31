#!/bin/bash
# Build MPM-STD_C on KISTI Neuron.
#   ./build_neuron.sh cpu    # gcc/11.5.0 + openmpi-4.1.8 (login-node runnable)
#   ./build_neuron.sh cuda   # nvhpc/25.11_cuda12 hpcx, -gpu=cc80 (A100)
#
# Dependencies (paths wired into Makefile.inc):
#   PaScaL_TDMA_C : /scratch/x3319a05/TDMA/PaScaL_TDMA
#                     build_gcc      <- make lib BUILDDIR=build_gcc USE_CUDA=0 USE_NVTX=0   (gcc env)
#                     build_nv_sm80  <- make lib BUILDDIR=build_nv_sm80 USE_CUDA=1 CUDA_ARCH=80 USE_NVTX=0 (nvhpc env)
#   FFTW3         : /apps/libraries/fftw3/3.3.10/gcc/11.5.0 (module fftw3/3.3.10)
#
# Runtime (CPU build): export LD_LIBRARY_PATH=/apps/libraries/fftw3/3.3.10/gcc/11.5.0/lib:$LD_LIBRARY_PATH
# For V100/H200 GPU nodes rebuild PaScaL_TDMA with CUDA_ARCH=70/90 and change
# -gpu=cc80 in Makefile.inc (plus PASCAL_TDMA_BUILD=build_nv_smXX).
set -e
cd "$(dirname "$0")"
source /etc/profile.d/modules.sh 2>/dev/null || true

case "${1:-cpu}" in
  cpu)
    module purge 2>/dev/null
    module load gcc/11.5.0 mpi/openmpi-4.1.8
    make BACKEND=cpu all -j8
    ;;
  cuda)
    module purge 2>/dev/null
    module load nvhpc/25.11_cuda12 2>/dev/null
    NVR=/apps/compiler/nvidia_hpc_sdk/25.11/Linux_x86_64/25.11
    export PATH=$NVR/cuda/12.9/bin:$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/bin:$NVR/compilers/bin:$PATH
    export LD_LIBRARY_PATH=$NVR/cuda/12.9/lib64:$NVR/comm_libs/12.9/hpcx/hpcx-2.25.1/ompi/lib:$NVR/math_libs/12.9/lib64:$NVR/compilers/lib:$LD_LIBRARY_PATH
    make BACKEND=cuda all -j8
    ;;
  *)
    echo "usage: $0 [cpu|cuda]" >&2; exit 1
    ;;
esac
