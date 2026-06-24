#!/bin/bash
#SBATCH -J dhvc_lean
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:40:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_leansync.log

# LEANSYNC: drop defensive cudaDeviceSynchronize in momentum+scalar (default stream,
# no D2H → all syncs redundant). Build BASE (no flag) + LEAN (-DMPMSTD_LEANSYNC) from
# one source via object-override; run both cold-300 same session (fair clock) →
# correctness (div/KE/Re must track) + ms/step. Baseline ref = 34.7 ms/step.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
CXX=mpic++
DEV="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"; mkdir -p "$OBJ" build/cuda/bin/apps

echo "=== BUILD base lib + main ==="
touch src/equation/momentum/gpu/momentum_gpu.cu src/equation/scalar/gpu/scalar_gpu.cu apps/dhvc/main_gpu.cu
make BACKEND=cuda lib 2>&1 | tail -3
$CXX $DEV $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o" 2>&1 | tail -15
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "main build FAIL"; exit 1; }
$CXX $DEV "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu_base 2>&1 | tail -8
[ -x build/cuda/bin/apps/dhvc_gpu_base ] || { echo "base link FAIL"; exit 1; }

echo "=== BUILD lean objects (-DMPMSTD_LEANSYNC) + link (override) ==="
$CXX $DEV -DMPMSTD_LEANSYNC $INCS -c src/equation/momentum/gpu/momentum_gpu.cu -o "$OBJ/mom_lean.o" 2>&1 | tail -15
$CXX $DEV -DMPMSTD_LEANSYNC $INCS -c src/equation/scalar/gpu/scalar_gpu.cu  -o "$OBJ/sca_lean.o" 2>&1 | tail -15
$CXX $DEV "$OBJ/dhvc_gpu.o" "$OBJ/mom_lean.o" "$OBJ/sca_lean.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu_lean 2>&1 | tail -8
[ -x build/cuda/bin/apps/dhvc_gpu_lean ] || { echo "lean link FAIL"; exit 1; }

echo; echo "=== RUN base (cold-300) ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu_base apps/dhvc/input_prof300.toml 2>&1 | tee /tmp/lean_base.out | grep -E "TIMING_SS|TIMING_TOTAL|TIMING_BREAKDOWN"
echo "=== RUN lean (cold-300) ==="
mpirun -np 2 build/cuda/bin/apps/dhvc_gpu_lean apps/dhvc/input_prof300.toml 2>&1 | tee /tmp/lean_lean.out | grep -E "TIMING_SS|TIMING_TOTAL|TIMING_BREAKDOWN"

echo; echo "=== SPEED ==="
echo "base: $(grep TIMING_SS /tmp/lean_base.out | tail -1)"
echo "lean: $(grep TIMING_SS /tmp/lean_lean.out | tail -1)"
echo; echo "=== CORRECTNESS (final step KE/div/Re/Trms must track; lean must have no nan) ==="
echo "base step300: $(grep 'step=300 ' /tmp/lean_base.out | tail -1)"
echo "lean step300: $(grep 'step=300 ' /tmp/lean_lean.out | tail -1)"
if grep -qiE "nan|inf|error|Killed" /tmp/lean_lean.out; then echo "*** LEAN UNSTABLE (nan/crash) — a needed sync was removed ***"; else echo "lean: no nan/crash"; fi
echo "=== LEANSYNC DONE ==="
