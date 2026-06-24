#!/bin/bash
#SBATCH -J dhvc_f1e5
#SBATCH -w gpu02
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 06:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_gpu_1e5_faithful.log

cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

# Rebuild the LIB — momentum_gpu.cu (k_mom_adi_var cross-band advecting velocity now
# volume-weighted along the own axis, matching k_mom_rhs_var / PaScaL_TCS f90:301-302)
# and momentum_var_kernels_cpu.cpp changed. Then relink the dhvc_gpu app.
echo "=== make lib (recompiling faithful momentum kernels) ==="
make BACKEND=cuda lib 2>&1 | tail -6

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
  echo "=== DHVC GPU Ra=1e5 FAITHFUL run (full t_end=330, stats from t=60) ==="
  mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_fig7_gpu_1e5.toml 2>&1 \
    | grep --line-buffered -E "step=|dhvc_gpu\]|rror|Killed|CANCELLED|BLOW|nan"
fi
