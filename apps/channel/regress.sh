#!/bin/bash
#SBATCH -J mpm_regress
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=00:25:00

# P1e — Re_tau=180 turbulent regression for the refactored free-function channel.
# Compute nodes have MPI only via nvhpc (HPC-X), not the login node's Ubuntu
# OpenMPI, so we BUILD with nvhpc here (BACKEND=cuda = nvc++ host code + FFTW
# Poisson; the channel calls only _cpu functions → pure CPU run, cuda libs
# linked-but-unused) and run with HPC-X mpirun. Loads the frozen turbulent
# restart (apps/channel/restart_in/) and checks it SUSTAINS (div small, Ub~1).

ROOT=/shared/home/wel1come1234/workspace/MPM-STD_C
[ -n "${SLURM_SUBMIT_DIR}" ] && ROOT="${SLURM_SUBMIT_DIR}"
cd "${ROOT}" || exit 1

source /etc/profile.d/modules.sh
source /etc/profile.d/z95_nvhpc_modules.sh
module purge
module load nvhpc/23.7
export LD_LIBRARY_PATH=/shared/home/wel1come1234/local/fftw3/lib:${LD_LIBRARY_PATH}

echo "=== host=$(hostname)  nvc++=$(command -v nvc++)  mpirun=$(command -v mpirun) ==="

echo "=== build BACKEND=cuda (nvc++ host channel + FFTW Poisson) ==="
make BACKEND=cuda -C mpmstd clean >/dev/null 2>&1
make BACKEND=cuda -C mpmstd lib apps 2>&1 | tail -6
BIN=build/cuda/mpmstd/bin/apps/channel
if [ ! -x "$BIN" ]; then echo "[FAIL] no channel binary built"; exit 1; fi

echo "=== run: mpirun -np 32 channel input_regress (frozen Re_tau=180, 30 steps) ==="
mpirun -np 32 "$BIN" mpmstd/apps/channel/input_regress.toml
echo "RUN_EXIT=$?"
echo "=== stats out ===" ; ls -la mpmstd/apps/channel/regress_out/ 2>&1 | head
