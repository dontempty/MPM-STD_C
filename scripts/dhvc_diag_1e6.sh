#!/bin/bash
#SBATCH -J dhvc_diag1e6
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_gpu_1e6.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

# Recompile dhvc_gpu on the GPU node (login node cannot compile .cu).
# main_gpu.cu now has: instantaneous Re/KE/Nu/theta_rms in monitor.csv +
# periodic cumulative-stats fileout (every nstat_out steps from nstat_start).
touch apps/dhvc/main_gpu.cu
make BACKEND=cuda lib 2>&1 | tail -3

CXX=mpic++
DEV_FLAGS="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"
mkdir -p "$OBJ" build/cuda/bin/apps

rm -f build/cuda/bin/apps/dhvc_gpu
$CXX $DEV_FLAGS $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o"
CRC=$?
echo "COMPILE_OBJ:$CRC"
BUILD_RC=$CRC
if [ $CRC -eq 0 ]; then
  $CXX $DEV_FLAGS "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu
  BUILD_RC=$?
fi
echo "BUILD_EXIT:$BUILD_RC"

if [ $BUILD_RC -eq 0 ]; then
  echo "=== DHVC GPU Ra=1e6 (150K steps, nstat_start=50K, nstat_out=10K, inst monitor + periodic stats) ==="
  mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e6.toml 2>&1 \
    | grep --line-buffered -E "step=|dhvc_gpu\]|stats out|rror|Killed|CANCELLED|BLOW|nan"
fi
