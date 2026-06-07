#!/bin/bash
#SBATCH -J mpm_stats
#SBATCH -p batch
#SBATCH -w cpu01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=24:00:00

# ============================================================
#  MPM-STD_C — turbulent channel flow Re_tau≈180 validation
#  Fresh start: Poiseuille + 5% perturbation, no restart.
#
#  Submit:
#    cd /shared/home/wel1come1234/workspace/MPM-STD_C
#    mkdir -p log
#    sbatch apps/channel/channel_Re180.sh
# ============================================================

ROOT=/shared/home/wel1come1234/workspace/MPM-STD_C

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    ROOT="${SLURM_SUBMIT_DIR}"
fi

cd "${ROOT}" || exit 1

# `module` is a shell function defined by the modules init; non-interactive
# sbatch shells don't source it. Also need z95_nvhpc_modules.sh to put the
# nvhpc modulefiles on MODULEPATH (otherwise `module load nvhpc/23.7` fails).
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null
source /etc/profile.d/z95_nvhpc_modules.sh 2>/dev/null
module purge 2>/dev/null
module load nvhpc/23.7

# FFTW3 (built against the user-local install) must be on the runtime lib path;
# compute-node non-login shells don't have it otherwise → libfftw3.so.3 missing.
export LD_LIBRARY_PATH=/shared/home/wel1come1234/local/fftw3/lib:${LD_LIBRARY_PATH}

NP=64   # np1=4 * np2=4 * np3=4
INPUT="apps/channel/input_stats.toml"

mkdir -p log apps/channel/restart_out apps/channel/statistics apps/channel/instant

echo "=============================================="
echo " Job   : ${SLURM_JOB_NAME:-interactive}  (ID=${SLURM_JOB_ID:-none})"
echo " Root  : ${ROOT}"
echo " Input : ${INPUT}"
echo " MPI   : np=${NP}  (np1=4, np2=4, np3=4)"
echo " Binary: ./build/cpu/bin/channel"
echo "=============================================="

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting"
mpirun -np ${NP} ./build/cpu/bin/channel "${INPUT}"
EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished (exit=${EXIT_CODE})"
exit ${EXIT_CODE}
