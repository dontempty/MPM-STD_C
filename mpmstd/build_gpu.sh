#!/bin/bash
#SBATCH -J mpmstd_gpu
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=00:15:00

# ============================================================
#  P0 — full refactored libmpmstd GPU dual-build + smoke.
#  Proves: all gpu/ .cu stubs compile under NVHPC, the lib carries BOTH _cpu and
#  _gpu families, apps + tests link, and a binary runs on the A100 node.
#
#  Submit:  cd .../MPM-STD_C && mkdir -p log && sbatch mpmstd/build_gpu.sh
# ============================================================

ROOT=/shared/home/wel1come1234/workspace/MPM-STD_C
[ -n "${SLURM_SUBMIT_DIR}" ] && ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}" || exit 1

# Batch shells are non-login: define `module` first (see P-0.5 spike_gpu.sh).
source /etc/profile.d/modules.sh
source /etc/profile.d/z95_nvhpc_modules.sh
module purge
module load nvhpc/23.7

echo "=== toolchain ==="
echo "nvc++ : $(command -v nvc++ || echo none)"
nvidia-smi -L 2>&1 | head

echo "=== build: make BACKEND=cuda -C mpmstd (lib + apps + tests) ==="
make BACKEND=cuda -C mpmstd clean >/dev/null 2>&1
make BACKEND=cuda -C mpmstd all
BUILD=$?
echo "BUILD_EXIT=${BUILD}"
if [ ${BUILD} -ne 0 ]; then echo "[FAIL] gpu dual-build FAILED"; exit ${BUILD}; fi
echo "[OK] gpu dual-build succeeded"

echo "=== lib carries BOTH _cpu and _gpu families ==="
LIB=build/cuda/mpmstd/lib/libmpmstd.a
echo "exchange_halo_cpu : $(nm $LIB 2>/dev/null | grep -c exchange_halo_cpu)"
echo "exchange_halo_gpu : $(nm $LIB 2>/dev/null | grep -c exchange_halo_gpu)"
echo "solve_momentum_gpu: $(nm $LIB 2>/dev/null | grep -c solve_momentum_gpu)"

echo "=== run smoke + app on the A100 node ==="
mpirun -np 1 build/cuda/mpmstd/bin/tests/unit/test_smoke_cpu
mpirun -np 2 build/cuda/mpmstd/bin/apps/channel
echo "RUN_EXIT=$?"
