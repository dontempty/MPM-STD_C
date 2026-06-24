#!/bin/bash
#SBATCH -J dhvc_base
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_baseline.log

# Clean BASELINE after reverting A1/A4. Also verifies the surgical reverts compile.
# cold-start 300 steps (clock-ramp amortized) → record ms/step + phase breakdown.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
echo "=== BUILD (reverted baseline) ==="
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
[ $RC -eq 0 ] && [ -x build/cuda/bin/apps/dhvc_gpu ] || { echo "=== BASELINE: BUILD FAILED (revert broke compile) ==="; exit 1; }
# keep a copy as the reference baseline binary for later fair A/B
cp build/cuda/bin/apps/dhvc_gpu build/cuda/bin/apps/dhvc_gpu_baseline

echo; echo "=== RUN baseline (cold-start 300 steps) ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu apps/dhvc/input_prof300.toml 2>&1 | grep -E "TIMING|step=" | tail -12
echo "=== BASELINE DONE ==="
