#!/bin/bash
# cv12 캠페인 ③ gpu02: ΔT 60 냉시작 (dt ≈ 1.5e-3, ~4일)
#SBATCH -J cv12_gpu02
#SBATCH -p batch
#SBATCH -w gpu02
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 7-00:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc_nob/log/cv12_gpu02_%j.log
set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
J=$SLURM_JOB_ID
echo "# host=$(hostname) gpus=$(nvidia-smi -L | wc -l) start=$(date '+%F %T')"
run() {
  local name=$1 app=$2 deck=$3
  echo "=== $name ($app $deck): $(date '+%F %T') ==="
  ( cd "$ROOT/apps/$app" && "$HPCX/ompi/bin/mpirun" -np 2 "$ROOT/build_gpu/bin/$app" "$deck" > "log/${name}_$J.log" 2>&1 )
  local rc=$?
  grep -E "FINAL|error \(rank|MPI_ABORT|blew up" "$ROOT/apps/$app/log/${name}_$J.log" | tail -3
  echo "=== $name rc=$rc $(date '+%F %T') ==="
  return $rc
}
run cv12_dt60 rbc_nob input_rbc_nob_cv12_dt60.ini || echo "DT60 FAILED"
echo "# done $(date '+%F %T')"
