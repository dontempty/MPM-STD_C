#!/bin/bash
#SBATCH -J dhvc_a4
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_a4.log

# A4: __launch_bounds__(512,3) on heavy var kernels (k_mom_adi_var, k_mom_rhs_var,
# k_build_adi_var, k_energy_residual_var) to cap registers → raise occupancy
# (ncu: k_mom_adi_var was 64 reg → 50% occ). Bit-identical math.
# Build + run prof config (20 steps) + report TIMING_BREAKDOWN vs baseline.
#   BASELINE [TIMING_BREAKDOWN]: cfl=0.1 scalar=10.3 mom=28.6 corr=0.2 sync=1.9 pres=8.5
#   BASELINE clean steady ms/step (a1_ab OFF, 300 steps): 35.90
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== BUILD (launch_bounds in momentum/scalar var kernels) ==="
touch src/equation/momentum/gpu/momentum_gpu.cu src/equation/scalar/gpu/scalar_gpu.cu apps/dhvc/main_gpu.cu
make BACKEND=cuda lib 2>&1 | tail -4
CXX=mpic++
DEV_FLAGS="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"; mkdir -p "$OBJ" build/cuda/bin/apps
rm -f build/cuda/bin/apps/dhvc_gpu
$CXX $DEV_FLAGS $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o" 2>&1 | tail -20
CRC=${PIPESTATUS[0]}; RC=$CRC
[ $CRC -eq 0 ] && { $CXX $DEV_FLAGS "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu 2>&1 | tail -20; RC=${PIPESTATUS[0]}; }
echo "BUILD_EXIT:$RC"
[ $RC -eq 0 ] && [ -x build/cuda/bin/apps/dhvc_gpu ] || { echo "=== A4: BUILD FAILED ==="; exit 1; }

echo; echo "=== RUN prof config (20 steps) — A4 ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_prof.toml 2>&1 | grep -E "TIMING|step=" | tail -12
echo; echo "=== COMPARE ==="
echo "BASELINE [TIMING_BREAKDOWN] cfl=0.1 scalar=10.3 mom=28.6 corr=0.2 sync=1.9 pres=8.5 | steady 35.90 ms/step"
echo "=== A4 DONE ==="
