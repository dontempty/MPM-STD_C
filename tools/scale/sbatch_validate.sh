#!/bin/bash
# 최적화 전/후 통계 검증 런 — cas_v100_4 (V100 ×4, 2일 한도). APP, DECK, BIN(절대경로), NAME 을 --export 로.
#   sbatch -p cas_v100_4 --gres=gpu:4 --ntasks-per-node=4 --export=ALL,APP=rbc_nob,DECK=input_rbc_nob_cv12_dt40_val_new.ini,BIN=.../rbc_nob,NAME=dt40_new tools/scale/sbatch_validate.sh
#SBATCH --time=2-00:00:00 --nodes=1 --cpus-per-task=4 --mem-per-cpu=9000M --hint=nomultithread
#SBATCH -J validate -o tools/scale/log/validate_%j.out --comment="field=mae;appl=in_house"
set -u
if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then source "${MODULESHOME}/init/bash"; fi
module purge; module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export MPMSTD_PHASE_START=0   # 계측 main 이면 타이머 끔
ROOT=/scratch/r908a06/KDH/MPM-STD_C
cd "$ROOT/apps/$APP" || exit 1; mkdir -p log
echo "# job=$SLURM_JOB_ID host=$(hostname) app=$APP deck=$DECK bin=$BIN ($(stat -c %y "$BIN")) start=$(date '+%F %T')"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
RUN_DECK=$DECK
if [[ -n "${WARMSTART_DIR:-}" ]]; then   # dhvc 1e7: 1e6 검증 런의 최신 체크포인트로 warm-start (원본 sbatch_dhvc_1e7.sh 와 같은 방식)
  LATEST=$(ls -t "$WARMSTART_DIR"/dhvc_ck_*.bin 2>/dev/null | head -1)
  [[ -z "$LATEST" ]] && { echo "ERROR: no checkpoint in $WARMSTART_DIR"; exit 1; }
  RUN_DECK="${DECK%.ini}_run.ini"; sed "s|WARMSTART_BIN|$LATEST|" "$DECK" > "$RUN_DECK"; echo "# warm-start: $LATEST"
fi
mpirun -x MPMSTD_PHASE_START -np 4 "$BIN" "$RUN_DECK" > "log/${NAME}_$SLURM_JOB_ID.log" 2>&1; rc=$?
grep -E "FINAL|error \(rank|MPI_ABORT|blew up" "log/${NAME}_$SLURM_JOB_ID.log" | tail -3
echo "# rc=$rc end=$(date '+%F %T')"
