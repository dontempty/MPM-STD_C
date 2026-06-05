#!/bin/bash
#SBATCH -J mpm_rbc
#SBATCH -p batch
#SBATCH -w cpu01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o log/%x_%j.out
#SBATCH -e log/%x_%j.err
#SBATCH --time=04:00:00

# ============================================================
#  MPM-STD_C — DHVC / RBC (Ra=100, Pr=1, 32x32x16)
#
#  Submit:
#    cd /shared/home/wel1come1234/workspace/MPM-STD_C
#    mkdir -p log
#    sbatch apps/rbc/rbc.sh [input.toml]
# ============================================================

ROOT=/shared/home/wel1come1234/workspace/MPM-STD_C

if [ -n "${SLURM_SUBMIT_DIR}" ]; then
    ROOT="${SLURM_SUBMIT_DIR}"
fi

cd "${ROOT}" || exit 1

module purge
module load nvhpc/23.7

INPUT="${1:-apps/rbc/input.toml}"

# Parse NP from input file (np1 * np2 * np3)
_get() { grep -E "^\s*${1}\s*=" "${INPUT}" | sed 's/[^=]*=//;s/#.*//' | tr -d '[:space:]' | head -1; }
NP1=$(_get np1); NP2=$(_get np2); NP3=$(_get np3)
NP=$(( ${NP1:-1} * ${NP2:-1} * ${NP3:-1} ))

echo "=============================================="
echo " Job   : ${SLURM_JOB_NAME:-interactive}  (ID=${SLURM_JOB_ID:-none})"
echo " Root  : ${ROOT}"
echo " Input : ${INPUT}"
echo " MPI   : np=${NP}  (np1=${NP1:-1}, np2=${NP2:-1}, np3=${NP3:-1})"
echo " Binary: ./build/cpu/bin/rbc"
echo "=============================================="

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting"
mpirun -np ${NP} ./build/cpu/bin/rbc "${INPUT}"
EXIT_CODE=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished (exit=${EXIT_CODE})"
exit ${EXIT_CODE}
