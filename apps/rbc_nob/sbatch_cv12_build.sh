#!/bin/bash
# cv12 캠페인 ① 빌드 + 스모크 게이트 (gpu01). 통계 모듈 확장(시간평균 T·W 장) 뒤 해가 바뀌지 않았음을 FINAL 일치로 강제.
#SBATCH -J cv12_build
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc_nob/log/cv12_build_%j.log
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
cd "$ROOT"
echo "=== build GPU rbc_nob rbc_ob: $(date '+%F %T') ==="
make GPU=1 rbc_nob rbc_ob -j8 > "apps/rbc_nob/log/cv12_build_make_$J.log" 2>&1 || { echo "BUILD FAILED"; grep -n -i -E " error" "apps/rbc_nob/log/cv12_build_make_$J.log" | head -20; exit 1; }
ls -la build_gpu/bin/rbc_nob build_gpu/bin/rbc_ob
run smoke_nob_ext rbc_nob input_rbc_nob_smoke_ext.ini || { echo "SMOKE nob FAILED"; exit 1; }
run smoke_ob      rbc_ob  input_rbc_ob_smoke.ini      || { echo "SMOKE ob FAILED"; exit 1; }
fn=$(grep FINAL "$ROOT/apps/rbc_nob/log/smoke_nob_ext_$J.log"); fo=$(grep FINAL "$ROOT/apps/rbc_ob/log/smoke_ob_$J.log")
echo "# nob smoke: $fn"; echo "# ob  smoke: $fo"
[[ "$fn" == *"theta_c = 0.01512"* && "$fn" == *"Nu = 46.3863"* ]] || { echo "GATE FAILED: nob smoke FINAL changed (ref theta_c 0.01512, Nu 46.3863)"; exit 1; }
[[ "$fo" == *"theta_c = -0.00001"* && "$fo" == *"Nu = 77.0954"* ]] || { echo "GATE FAILED: ob smoke FINAL changed (ref theta_c -0.00001, Nu 77.0954)"; exit 1; }
grep -H "U_pl=" "$ROOT"/apps/rbc_nob/stats_nob_smoke_ext/rbc_nob_stats_*.dat | tail -1 | cut -c1-160 || { echo "GATE FAILED: no U_pl in smoke stats"; exit 1; }
grep -H "pdf_timemean" "$ROOT"/apps/rbc_nob/stats_nob_smoke_ext/rbc_nob_tpdf_*.dat | tail -1 | cut -c1-120
echo "# gate passed $(date '+%F %T')"
