#!/bin/bash
#SBATCH -J dhvc_ncu
#SBATCH -w gpu01
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH -t 00:50:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD_C/apps/dhvc/logs/dhvc_ncu.log

# Nsight Compute kernel-level analysis (per nsys.txt §2 / reference pattern).
# Roofline of the two hottest kernels → memory-bound vs compute-bound vs occupancy.
#   k_mom_adi_var          : our biggest controllable kernel (ADI band assembly)
#   tdma_cyclic_many_kernel: dominant library kernel (PaScaL_TDMA periodic solve)
set -o pipefail
cd /shared/home/wel1come1234/workspace/MPM-STD_C
module load nvhpc/23.7
mkdir -p apps/dhvc/prof
BIN=build/cuda/bin/apps/dhvc_gpu
[ -x "$BIN" ] || { echo "no binary"; exit 1; }
CFG=apps/dhvc/input_prof.toml
SUM="Duration|Compute \(SM\)|Memory Throughput|DRAM Throughput|Achieved Occupancy|Achieved Active Warps|Registers Per Thread|Block Limit|Theoretical Occupancy|L1/TEX|L2 |Waves Per"

profile_kernel () {
  local KERN="$1"
  echo; echo "########################################################"
  echo "## ncu: $KERN"
  echo "########################################################"
  srun --ntasks=2 ncu --kernel-name "regex:$KERN" --set full \
       --launch-skip 1 --launch-count 1 --target-processes all -f \
       -o "apps/dhvc/prof/ncu_${KERN}_%q{SLURM_PROCID}" \
       "$BIN" "$CFG" 2>&1 | tail -12
  local REP="apps/dhvc/prof/ncu_${KERN}_0.ncu-rep"
  if [ -f "$REP" ]; then
    echo "--- key metrics (rank 0) ---"
    ncu --import "$REP" --page details 2>&1 | grep -iE "$SUM" | head -40
  else
    echo "  (no report $REP — check ncu permission/match above)"
  fi
}

profile_kernel "k_mom_adi_var"
profile_kernel "tdma_cyclic_many_kernel"
echo "=== NCU DONE ==="
