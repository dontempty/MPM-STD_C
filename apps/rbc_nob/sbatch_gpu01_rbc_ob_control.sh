#!/bin/bash
# NOB 윗벽 rms 스파이크의 OB 대조 — gpu01 2 GPU (2026-10-13)
#   ① GPU 빌드(rbc·rbc_nob 앱 패치: OB 앱 T/u/v rms + 벽 max|U| 모니터, NOB 앱 ΔT=0 정확 OB 극한)
#   ② 스모크: rbc(OB) · rbc_nob smoke_ext(ΔT 40, 참조 FINAL 대조) · rbc_nob smoke_ob_limit(ΔT 0)
#   ③ 가변물성 경로 정확 OB 극한을 NOB ΔT40 스파이크 상태(t 25000)에서 warm restart — dt_max 0.05(D_× 35) / 0.10(D_× 70)
#   ④ 냉시작 t→5000 (통계 3000~5000): rbc 앱 OB(dt_max 0.05 / 0.10) · rbc_nob 앱 ΔT 0
#SBATCH -J rbc_ob_ctrl
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc_nob/log/ob_control_%j.log
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

cd "$ROOT"
echo "=== build GPU rbc rbc_nob: $(date '+%F %T') ==="
make GPU=1 rbc rbc_nob -j8 > "apps/rbc_nob/log/ob_control_build_$J.log" 2>&1 || { echo "BUILD FAILED"; tail -30 "apps/rbc_nob/log/ob_control_build_$J.log"; exit 1; }
grep -c "mpic++" "apps/rbc_nob/log/ob_control_build_$J.log" | sed 's/^/# compile+link commands: /'
ls -la build_gpu/bin/rbc build_gpu/bin/rbc_nob

# run <이름> <앱> <덱>  — 앱 폴더에서 실행(덱의 상대 경로 기준), 로그는 그 앱의 log/
run() {
  local name=$1 app=$2 deck=$3
  echo "=== $name ($app $deck): $(date '+%F %T') ==="
  ( cd "$ROOT/apps/$app" && "$HPCX/ompi/bin/mpirun" -np 2 "$ROOT/build_gpu/bin/$app" "$deck" > "log/ob_control_${name}_$J.log" 2>&1 )
  local rc=$?
  grep -E "FINAL|error \(rank|MPI_ABORT|blew up" "$ROOT/apps/$app/log/ob_control_${name}_$J.log" | tail -3
  echo "=== $name rc=$rc $(date '+%F %T') ==="
  return $rc
}
run smoke_rbc        rbc     input_rbc_smoke.ini              || { echo "SMOKE rbc FAILED"; exit 1; }
run smoke_nob_ext    rbc_nob input_rbc_nob_smoke_ext.ini      || { echo "SMOKE nob_ext FAILED"; exit 1; }
echo "# smoke_ext reference (잡 35989, 패치 전 GPU):"; grep FINAL "$ROOT/apps/rbc_nob/log/dt40_smoke_35989.log"
run smoke_nob_oblim  rbc_nob input_rbc_nob_smoke_ob_limit.ini || { echo "SMOKE nob_oblim FAILED"; exit 1; }

run oblim_restart      rbc_nob input_rbc_nob_ob_limit_restart.ini      || echo "oblim_restart FAILED"
run oblim_restart_dt10 rbc_nob input_rbc_nob_ob_limit_restart_dt10.ini || echo "oblim_restart_dt10 FAILED"
run ob_cold            rbc     input_rbc_ob_pr2547_z256.ini            || echo "ob_cold FAILED"
run ob_cold_dt10       rbc     input_rbc_ob_pr2547_z256_dt10.ini       || echo "ob_cold_dt10 FAILED"
run oblim_cold         rbc_nob input_rbc_nob_ob_limit_cold.ini         || echo "oblim_cold FAILED"
echo "# done $(date '+%F %T')"
