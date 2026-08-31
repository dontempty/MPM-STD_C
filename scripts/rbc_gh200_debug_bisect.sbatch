#!/bin/bash
#SBATCH -J rbc_gh200_bisect
#SBATCH --comment inhouse
#SBATCH -p gh200_1
#SBATCH -w gpu51
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /scratch/x3319a05/MPM-STD_C/apps/rbc/logs/rbc_gh200_bisect.%j.log
# Bisect the pre-main __pgi_cuda_register_fat_binaryA segfault:
#  T1: minimal nvc++ -cuda -gpu=cc90 kernel            -> toolchain sanity
#  T2: T1 + link libpascal_tdma (nvcc sm_90 objects)   -> mixed-runtime link
#  T3: T1 with -gpu=cc90,nordc                          -> RDC registration path
#  T4: T1 with ARM SDK 24.1                             -> SDK version
set -uo pipefail
S=/tmp/bisect_$$
mkdir -p $S; cd $S

cat > hello.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k_add(double* a, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]+=1.0; }
int main(){
  int n=1024; double *d; cudaMalloc(&d,n*sizeof(double)); cudaMemset(d,0,n*sizeof(double));
  k_add<<<(n+255)/256,256>>>(d,n);
  double h[4]; cudaMemcpy(h,d,4*sizeof(double),cudaMemcpyDeviceToHost);
  printf("ok %g %g\n", h[0], h[3]);
  return 0;
}
EOF

run_t () { # name, sdk_root, extra flags..., link...
  local name=$1 nvr=$2; shift 2
  export PATH=$nvr/cuda/bin:$nvr/comm_libs/mpi/bin:$nvr/compilers/bin:$PATH
  export LD_LIBRARY_PATH=$nvr/cuda/lib64:$nvr/comm_libs/mpi/lib:$nvr/math_libs/lib64:$nvr/compilers/lib:${LD_LIBRARY_PATH:-}
  echo "--- $name compile ---"
  if $nvr/comm_libs/mpi/bin/mpic++ -std=c++17 -O2 -cuda "$@" hello.cu -o $name.x 2>&1 | tail -5; then
    echo "--- $name run ---"
    ./$name.x && echo "$name: RUN OK" || echo "$name: RUN FAILED rc=$?"
  else
    echo "$name: COMPILE FAILED"
  fi
}

NV245=/apps/ARM_node/ARM_nvidia_hpc_sdk/24.5/Linux_aarch64/24.5
NV241=/apps/ARM_node/ARM_nvidia_hpc_sdk/24.1/Linux_aarch64/24.1
PASCAL=/scratch/x3319a05/TDMA/PaScaL_TDMA

unset PMIX_INSTALL_PREFIX

run_t t1_cc90        $NV245 -gpu=cc90
run_t t2_pascal      $NV245 -gpu=cc90 -L$PASCAL/build_arm_sm90/lib -lpascal_tdma
run_t t3_nordc       $NV245 -gpu=cc90,nordc
[ -d $NV241 ] && run_t t4_sdk241 $NV241 -gpu=cc90 || echo "t4: 24.1 not present"

echo "=== bisect done ==="
