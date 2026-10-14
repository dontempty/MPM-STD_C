#!/bin/bash
# rbc_ob(충실 상수물성) Pr 2547 본런 — NOB 윗벽 rms 스파이크의 OB 대조 (2026-10-13), gpu01 2 GPU.
#   ① GPU 빌드(rbc_ob) ② 스모크(가변경로 ΔT=0 스모크와 대조) ③ 본런 냉시작 → t 27000
#SBATCH -J rbc_ob_cold
#SBATCH -p batch
#SBATCH -w gpu01
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /shared/home/wel1come1234/workspace/MPM-STD/apps/rbc_ob/log/ob_cold_%j.log
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
echo "=== build GPU rbc_ob: $(date '+%F %T') ==="
make GPU=1 rbc_ob -j8 > "apps/rbc_ob/log/build_$J.log" 2>&1 || { echo "BUILD FAILED"; grep -n -i -E "error" "apps/rbc_ob/log/build_$J.log" | head -20; exit 1; }
ls -la build_gpu/bin/rbc_ob
cd "$ROOT/apps/rbc_ob"
echo "=== smoke: $(date '+%F %T') ==="
"$HPCX/ompi/bin/mpirun" -np 2 "$ROOT/build_gpu/bin/rbc_ob" input_rbc_ob_smoke.ini > "log/smoke_$J.log" 2>&1 || { echo "SMOKE FAILED"; tail -5 "log/smoke_$J.log"; exit 1; }
grep FINAL "log/smoke_$J.log"; echo "# reference (rbc_nob variable path, DeltaT=0):"; grep FINAL "$ROOT/apps/rbc_nob/log/ob_control_smoke_nob_oblim_36038.log"
echo "=== main run: $(date '+%F %T') ==="
"$HPCX/ompi/bin/mpirun" -np 2 "$ROOT/build_gpu/bin/rbc_ob" input_rbc_ob_pr2547_z256.ini
rc=$?
echo "# rc=$rc end=$(date '+%F %T')"
exit $rc
