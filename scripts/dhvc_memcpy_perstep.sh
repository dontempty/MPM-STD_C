#!/bin/bash
#SBATCH -J dhvc_mc
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_memcpy_perstep.log

# Settle the H2D/D2H question: profile dhvc at 5 vs 40 steps (NO monitor, NO stats).
# per-step memcpy = (count@40 - count@5)/35  → cancels startup. Decides whether the
# ~628 H2D/D2H is per-step (real) or startup-dominated (my earlier mis-division).
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/prof apps/dhvc/out_prof
echo "=== rebuild clean dhvc_gpu (reverted source) ==="
touch src/equation/momentum/gpu/momentum_gpu.cu src/equation/scalar/gpu/scalar_gpu.cu apps/dhvc/main_gpu.cu
make BACKEND=cuda lib 2>&1 | tail -2
CXX=mpic++
DEV="-std=c++17 -O3 -fast -cuda -gpu=cc80 -DOMPI_SKIP_MPICXX -DMPMSTD_BACKEND_CUDA"
INCS="-I/shared/home/wel1come1234/workspace/MPM-STD_C/src -I/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/include -I/shared/home/wel1come1234/local/fftw3/include"
LIB="build/cuda/lib/libmpmstd.a"
EXT="-L/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/build/lib -lpascal_tdma -cudalib=cufft -lm -L/shared/home/wel1come1234/local/fftw3/lib -lfftw3"
OBJ="build/cuda/obj/apps"; mkdir -p "$OBJ"
$CXX $DEV $INCS -c apps/dhvc/main_gpu.cu -o "$OBJ/dhvc_gpu.o" 2>&1 | tail -8
$CXX $DEV "$OBJ/dhvc_gpu.o" "$LIB" $EXT -o build/cuda/bin/apps/dhvc_gpu 2>&1 | tail -8
BIN=build/cuda/bin/apps/dhvc_gpu
[ -x "$BIN" ] || { echo "BUILD FAIL"; exit 1; }

mc () {  # $1=cfg $2=tag : profile rank0, print H2D/D2H/D2D counts
  mpirun -np 2 nsys profile --trace=cuda --force-overwrite true -o "apps/dhvc/prof/mc_$2_%q{OMPI_COMM_WORLD_RANK}" "$BIN" "$1" >/dev/null 2>&1
  echo "--- $2 (rank0) ---"
  nsys stats --report cuda_gpu_mem_time_sum --format table "apps/dhvc/prof/mc_$2_0.nsys-rep" 2>/dev/null \
    | grep -E "HtoD|DtoH|DtoD" | awk '{print $0}'
}
echo; echo "=== profile 5 steps ==="; mc apps/dhvc/input_prof5.toml s5
echo; echo "=== profile 40 steps ==="; mc apps/dhvc/input_prof40.toml s40
echo; echo "=== per-step = (count@40 - count@5)/35 ; compute from the Count columns above ==="
echo "=== MEMCPY_PERSTEP DONE ==="
