#!/bin/bash
#SBATCH -J build_inc
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:40:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/build_incremental.log

# GPU-node build of the incremental-projection edits (.cu needs the driver → gres).
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== make lib (incremental pressure + conv/buoyancy fixes) ==="
make BACKEND=cuda lib 2>&1 | tail -8

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
  echo "=== SANITY (incremental, np 1x1x2, 32x16x32, 400 steps) — div must stay ~1e-13, no nan ==="
  mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_test_inc.toml 2>&1 \
    | grep --line-buffered -E "step=|dhvc_gpu\]|nan|rror|Killed|CANCELLED"
fi
