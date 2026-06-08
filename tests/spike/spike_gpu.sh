#!/bin/bash
#SBATCH -J spike_gpu
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=00:10:00

# ============================================================
#  P-0.5 spike — GPU dual-build LINK + device halo RUN.
#  Proves: _gpu symbols compile under NVHPC, link with CUDA-aware MPI,
#  and run with 1 MPI rank == 1 A100 (device-to-device halo exchange).
#
#  Submit:
#    cd /shared/home/wel1come1234/workspace/MPM-STD_C && mkdir -p log
#    sbatch tests/spike/spike_gpu.sh
# ============================================================

ROOT=/shared/home/wel1come1234/workspace/MPM-STD_C
[ -n "${SLURM_SUBMIT_DIR}" ] && ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}" || exit 1

# Batch shells are non-login: `module` is undefined until we source the init
# scripts (and z95 sets MODULEPATH to the nvhpc modulefiles). See memory:
# feedback_make_rm_before_sbatch / prior "module: command not found" fix.
source /etc/profile.d/modules.sh
source /etc/profile.d/z95_nvhpc_modules.sh
module purge
module load nvhpc/23.7

echo "=== toolchain ==="
echo "nvc++ : $(command -v nvc++ || echo none)"
echo "mpic++: $(command -v mpic++ || echo none)"
nvidia-smi -L 2>&1 | head

echo "=== build: make BACKEND=cuda gpu ==="
make BACKEND=cuda -C tests/spike clean >/dev/null 2>&1
make BACKEND=cuda -C tests/spike gpu
BUILD=$?
echo "BUILD_EXIT=${BUILD}"
if [ ${BUILD} -ne 0 ]; then
  echo "[FAIL] gpu dual-build link FAILED"
  exit ${BUILD}
fi
echo "[OK] gpu dual-build LINK succeeded -> ./build/cuda/spike/link_gpu"

echo "=== run: mpirun -np 2 link_gpu (1 rank = 1 A100) ==="
mpirun -np 2 ./build/cuda/spike/link_gpu
RUN=$?
echo "RUN_EXIT=${RUN}"
if [ ${RUN} -eq 0 ]; then
  echo "[OK] gpu device halo RUN succeeded (CUDA-aware MPI, 1 rank/GPU)"
else
  echo "[WARN] link OK but run returned ${RUN} (CUDA-aware MPI runtime issue; link is the DoD)"
fi
exit ${RUN}
