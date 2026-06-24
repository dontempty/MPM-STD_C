#!/bin/bash
#SBATCH -J a1_ab
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:40:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/a1_ab.log

# A1 (k-major block remap) verification:
#  (1) build dhvc_gpu (lib + app)  — also first compile of the timing changes
#  (2) run 300-step warm-start case with MPMSTD_KMAJOR OFF  → restart_a1_off
#  (3) run same with MPMSTD_KMAJOR=1 (ON)                   → restart_a1_on
#  (4) cmp restart fields byte-for-byte  → MUST be identical (A1 is bit-identical)
#  (5) report ms/step OFF vs ON (the speedup)
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7

echo "=== [1/5] BUILD (lib + dhvc_gpu, incl. A1 + timing changes) ==="
touch apps/dhvc/main_gpu.cu src/equation/momentum/gpu/momentum_gpu.cu src/equation/scalar/gpu/scalar_gpu.cu
make BACKEND=cuda lib 2>&1 | tail -4
CXX=mpic++
DEV_FLAGS="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"
mkdir -p "$OBJ" build/cuda/bin/apps
rm -f build/cuda/bin/apps/dhvc_gpu
$CXX $DEV_FLAGS $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o" 2>&1 | tail -25
CRC=${PIPESTATUS[0]}; RC=$CRC
if [ $CRC -eq 0 ]; then
  $CXX $DEV_FLAGS "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu 2>&1 | tail -25
  RC=${PIPESTATUS[0]}
fi
echo "BUILD_EXIT:$RC"
if [ $RC -ne 0 ] || [ ! -x build/cuda/bin/apps/dhvc_gpu ]; then
  echo "=== A1 RESULT: BUILD FAILED ==="; exit 1
fi
BIN=build/cuda/bin/apps/dhvc_gpu

echo; echo "=== [2/5] RUN OFF (default mapping) ==="
rm -rf apps/dhvc/restart_a1_test apps/dhvc/restart_a1_off apps/dhvc/restart_a1_on
mpirun -np 2 $BIN apps/dhvc/input_a1_ab.toml 2>&1 | tee /tmp/a1_off.log | grep -E "step=|TIMING|warm-start|restart\]"
mv apps/dhvc/restart_a1_test apps/dhvc/restart_a1_off

echo; echo "=== [3/5] RUN ON (MPMSTD_KMAJOR=1, k-major) ==="
MPMSTD_KMAJOR=1 mpirun -x MPMSTD_KMAJOR -np 2 $BIN apps/dhvc/input_a1_ab.toml 2>&1 | tee /tmp/a1_on.log | grep -E "step=|TIMING|warm-start|restart\]"
mv apps/dhvc/restart_a1_test apps/dhvc/restart_a1_on

echo; echo "=== [4/5] BYTE-FOR-BYTE FIELD COMPARE (must be identical) ==="
IDENT=1
for f in U V W T P; do
  if cmp -s apps/dhvc/restart_a1_off/$f.bin apps/dhvc/restart_a1_on/$f.bin; then
    echo "  $f.bin: IDENTICAL"
  else
    echo "  $f.bin: ***DIFFERS***"; IDENT=0
  fi
done

echo; echo "=== [5/5] ms/step OFF vs ON ==="
echo "OFF: $(grep -oE 'avg [0-9.]+ ms/step' /tmp/a1_off.log | tail -1)  | last monitor: $(grep -oE 'ms/step=[0-9.]+' /tmp/a1_off.log | tail -1)"
echo "ON : $(grep -oE 'avg [0-9.]+ ms/step' /tmp/a1_on.log  | tail -1)  | last monitor: $(grep -oE 'ms/step=[0-9.]+' /tmp/a1_on.log  | tail -1)"
grep -E "TIMING_BREAKDOWN" /tmp/a1_off.log | tail -1 | sed 's/^/OFF /'
grep -E "TIMING_BREAKDOWN" /tmp/a1_on.log  | tail -1 | sed 's/^/ON  /'

if [ $IDENT -eq 1 ]; then
  echo "=== A1 RESULT: PASS — bit-identical (correct) + see ms/step above ==="
else
  echo "=== A1 RESULT: FAIL — fields differ, A1 mapping has a BUG ==="; exit 2
fi
