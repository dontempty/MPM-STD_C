#!/bin/bash
# ΔT=40 K 윗벽 rms 스파이크 dt 스터디 — gpu01 2 GPU (2026-10-09)
#   ① GPU 스모크(확장 통계·D_× 배선, CPU 스모크 log/smoke_ext_35983.log 과 대조)
#   ② base  : CFL dt 그대로 warm restart (t 25000→27000, 통계 25300~) — 잡 35985 완료
#   ②b 임계 스캔 visc30/24/18 (D_× 30/24/18, t 25000→25100, 통계 25050~; 스파이크 유무만)
#   ③ visc12: viscous_coupling_number 12 (dt≈3.6e-3, t→26500)
#   ④ visc6 : viscous_coupling_number 6  (dt≈1.8e-3, t→26500)
#SBATCH -J nob_dt40_dt
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 1-12:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc_nob/log/dt40_dtstudy_%j.log
set -u
ROOT=/shared/home/wel1come1234/workspace/MPM-STD
NVR=/opt/nvidia/hpc-sdk/Linux_x86_64/23.7
HPCX=$NVR/comm_libs/12.2/hpcx/hpcx-2.15
export OPAL_PREFIX=$HPCX/ompi
export PATH=$NVR/cuda/12.2/bin:$HPCX/ompi/bin:$NVR/compilers/bin:$PATH
export LD_LIBRARY_PATH=$HPCX/ucx/lib:$HPCX/ompi/lib:$NVR/cuda/12.2/lib64:$NVR/math_libs/12.2/lib64:$NVR/compilers/lib:${LD_LIBRARY_PATH:-}
unset PMIX_INSTALL_PREFIX
export OMPI_MCA_opal_warn_on_missing_libcuda=0
cd "$ROOT/apps/rbc_nob"
echo "# host=$(hostname) gpus=$(nvidia-smi -L | wc -l) start=$(date '+%F %T')"
run() { echo "=== $1: $(date '+%F %T') ==="; "$HPCX/ompi/bin/mpirun" -np 2 "$ROOT/build_gpu/bin/rbc_nob" "$2" > "log/dt40_$1_${SLURM_JOB_ID}.log" 2>&1; rc=$?; grep -E "FINAL|rror" "log/dt40_$1_${SLURM_JOB_ID}.log" | tail -3; return $rc; }
run smoke input_rbc_nob_smoke_ext.ini || { echo "SMOKE FAILED"; exit 1; }
echo "# CPU smoke reference:"; grep FINAL log/smoke_ext_35983.log
# base 는 잡 35985 에서 완료(stats_nob_dt40_base/, t 25000→27000). 임계 스캔(각 100 time, ~5 분) 먼저.
for CV in 30 24 18; do run visc$CV input_rbc_nob_dt40_visc$CV.ini || echo "VISC$CV FAILED"; done
run visc12 input_rbc_nob_dt40_visc12.ini || echo "VISC12 FAILED"
run visc6  input_rbc_nob_dt40_visc6.ini  || echo "VISC6 FAILED"
echo "# done $(date '+%F %T')"
