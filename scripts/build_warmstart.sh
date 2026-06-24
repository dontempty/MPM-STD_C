#!/bin/bash
#SBATCH -J build_ws
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:40:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/build_warmstart.log

# Build-only: verify the warm-start (cross-mesh interpolation) GPU build compiles.
# interp_cpu.cpp is new in the lib; main_gpu.cu gained the warm-start IC + projection.
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== make lib (BACKEND=cuda; compiles new interp_cpu.cpp + gpu .cu) ==="
make BACKEND=cuda lib 2>&1 | tail -8

CXX=mpic++
DEV_FLAGS="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"; mkdir -p "$OBJ" build/cuda/bin/apps

rm -f build/cuda/bin/apps/dhvc_gpu
$CXX $DEV_FLAGS $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o"; CRC=$?; echo "COMPILE_OBJ:$CRC"
BUILD_RC=$CRC
if [ $CRC -eq 0 ]; then
  $CXX $DEV_FLAGS "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu; BUILD_RC=$?
fi
echo "BUILD_EXIT:$BUILD_RC"
ls -la build/cuda/bin/apps/dhvc_gpu 2>/dev/null
