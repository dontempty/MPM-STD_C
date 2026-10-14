#!/bin/bash
# gpu02 2 GPU (2026-10-14): ① rbc_ob Kerr 검증(Pr 0.7, 192×192×96, ~15분) ② 가변물성 경로 ΔT=0 정확 OB 극한
#   냉시작 NOB 프로토콜(→ t 27000, ~3 h). 바이너리는 gpu01 잡 36041/36038 이 빌드한 build_gpu/bin/{rbc_ob,rbc_nob}.
#SBATCH -J rbc_ob_gpu02
#SBATCH -p batch
#SBATCH -w gpu02
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc_ob/log/gpu02_%j.log
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
run kerr_pr07        rbc_ob  input_rbc_ob_ra1e6_pr07_kerr.ini      || echo "KERR FAILED"
echo "# legacy rbc reference (Oct 4, same deck):"; grep FINAL "$ROOT/apps/rbc/log/rbc_gpu_fine_35326.log"
run oblim_cold_full  rbc_nob input_rbc_nob_ob_limit_cold_full.ini  || echo "OBLIM_COLD FAILED"
echo "# done $(date '+%F %T')"
