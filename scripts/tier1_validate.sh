#!/bin/bash
#SBATCH -J tier1_val
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=4
#SBATCH -t 02:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/tier1_validate.log

# Tier 1 distributed-slab warm-start I/O — build + A/B validation.
#   (1) build dhvc_gpu with the slab/replicate paths
#   (2) dry-run REPLICATE  → interpolated-field checksums
#   (3) dry-run SLAB       → interpolated-field checksums
#   (4) diff: must be identical (pure I/O refactor, same trilinear math)
#   (5) if identical: short 2000-step slab run → early KE/div/Re trajectory
# 4xA100 / 2 nodes (np 1x1x4), grid 832x416x192, source = restart_1e6_t240.
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/out_dryrun apps/dhvc/out_shortrun apps/dhvc/logs

echo "=== [1/5] BUILD (lib + dhvc_gpu) ==="
touch apps/dhvc/main_gpu.cu src/post/cpu/interp_cpu.cpp
make BACKEND=cuda lib 2>&1 | tail -3
CXX=mpic++
DEV_FLAGS="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"
mkdir -p "$OBJ" build/cuda/bin/apps
rm -f build/cuda/bin/apps/dhvc_gpu
$CXX $DEV_FLAGS $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o" 2>&1 | tail -20
CRC1=${PIPESTATUS[0]}
BUILD_RC=$CRC1
if [ $CRC1 -eq 0 ]; then
  $CXX $DEV_FLAGS "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu 2>&1 | tail -20
  BUILD_RC=${PIPESTATUS[0]}
fi
echo "BUILD_EXIT:$BUILD_RC"
if [ $BUILD_RC -ne 0 ] || [ ! -x build/cuda/bin/apps/dhvc_gpu ]; then
  echo "=== TIER1 RESULT: BUILD FAILED — aborting ==="
  exit 1
fi

BIN=build/cuda/bin/apps/dhvc_gpu
echo; echo "=== [2/5] DRY-RUN REPLICATE ==="
mpirun -np 4 $BIN apps/dhvc/input_dryrun_replicate.toml 2>&1 | tee /tmp/tier1_rep.out | grep -E "warm-start"
echo; echo "=== [3/5] DRY-RUN SLAB ==="
mpirun -np 4 $BIN apps/dhvc/input_dryrun_slab.toml 2>&1 | tee /tmp/tier1_slab.out | grep -E "warm-start"

echo; echo "=== [4/5] CHECKSUM DIFF (replicate vs slab) ==="
grep -E "dryrun\] [UVWTP]:" /tmp/tier1_rep.out  | sort > /tmp/tier1_rep.sums
grep -E "dryrun\] [UVWTP]:" /tmp/tier1_slab.out | sort > /tmp/tier1_slab.sums
echo "--- replicate ---"; cat /tmp/tier1_rep.sums
echo "--- slab ---";      cat /tmp/tier1_slab.sums
if diff -q /tmp/tier1_rep.sums /tmp/tier1_slab.sums >/dev/null && [ -s /tmp/tier1_rep.sums ]; then
  echo "CHECKSUM_MATCH: YES — slab interpolation == replicate (bit-for-bit)"
else
  echo "CHECKSUM_MATCH: NO  — DIFFERENCE DETECTED:"
  diff /tmp/tier1_rep.sums /tmp/tier1_slab.sums
  echo "=== TIER1 RESULT: CHECKSUM MISMATCH — slab read is WRONG ==="
  exit 2
fi

echo; echo "=== [5/5] SHORT SLAB RUN (2000 steps) — trajectory ==="
mpirun -np 4 $BIN apps/dhvc/input_shortrun_slab.toml 2>&1 \
  | tee /tmp/tier1_short.out | grep --line-buffered -E "step=|warm-start|rror|nan|Killed"
# guard: the short run must reach step>=2000 cleanly (no nan / crash) to count as PASS
LASTSTEP=$(grep -oE "step=[0-9]+" /tmp/tier1_short.out | tail -1 | grep -oE "[0-9]+")
if grep -qiE "nan|killed|out of memory" /tmp/tier1_short.out || [ "${LASTSTEP:-0}" -lt 2000 ]; then
  echo "=== TIER1 RESULT: SHORT RUN FAILED (laststep=${LASTSTEP:-0}, nan/crash?) — NOT launching 1e8 ==="
  exit 3
fi

echo; echo "=== TIER1 RESULT: PASS — checksums identical + short run reached step $LASTSTEP ==="
echo "Compare the step= lines above to the early lines of apps/dhvc/logs/dhvc_gpu_1e7.log"

# --- PASS path only: clean up validation outputs, then launch 1e8 from restart_1e7_t240 ---
echo "=== cleanup: removing validation result data ==="
rm -rf apps/dhvc/out_dryrun apps/dhvc/out_shortrun
rm -f /tmp/tier1_rep.out /tmp/tier1_slab.out /tmp/tier1_rep.sums /tmp/tier1_slab.sums /tmp/tier1_short.out
echo "  (kept: tier1_validate.log = PASS record; input_dryrun_*.toml / input_shortrun_slab.toml / tier1_validate.sh = test defs)"

echo "=== launching 1e8 (warm-start slab from restart_1e7_t240) ==="
if [ -f apps/dhvc/restart_1e7_t240/meta.txt ]; then
  sbatch --dependency=afterok:$SLURM_JOB_ID scripts/dhvc_gpu_1e8.sh
  echo "  1e8 submitted (afterok:$SLURM_JOB_ID)."
else
  echo "  ERROR: restart_1e7_t240/meta.txt missing — 1e7 did not write its restart. 1e8 NOT submitted."
  exit 4
fi
