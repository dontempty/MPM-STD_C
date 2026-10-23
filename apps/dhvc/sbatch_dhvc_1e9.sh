#!/bin/bash
# DHVC Ra 1e9 (및 속도 시험) — 1노드 N GPU, z-슬랩 npz = N. 1e7 체크포인트 격자보간 warm-start.
#   sbatch -p amd_h200nv_8 -w gpu56 --gres=gpu:6 --ntasks-per-node=6 --time=00:30:00 \
#     --export=ALL,DECK=input_dhvc_1e9_speed.ini,NAME=speed_h200,WARMSTART_DIR=restart_val_1e7_new apps/dhvc/sbatch_dhvc_1e9.sh
#SBATCH --nodes=1 --cpus-per-task=4 --mem-per-cpu=8000M --hint=nomultithread
#SBATCH -J dhvc_1e9 -o apps/dhvc/log/dhvc_1e9_%j.out --comment="field=mae;appl=in_house"
set -u
if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then source "${MODULESHOME}/init/bash"; fi
module purge; module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
ROOT=/scratch/r908a06/KDH/MPM-STD_C
BIN=${BIN:-$ROOT/build_gpu/bin/dhvc}
NP=${SLURM_NTASKS_PER_NODE:-$SLURM_NTASKS}
cd "$ROOT/apps/dhvc" || exit 1; mkdir -p log
echo "# job=$SLURM_JOB_ID host=$(hostname) np=$NP deck=$DECK bin=$BIN ($(stat -c %y "$BIN")) start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
LATEST=$(ls -t "$WARMSTART_DIR"/dhvc_ck_*.bin 2>/dev/null | head -1)
[[ -z "$LATEST" ]] && { echo "ERROR: no checkpoint in $WARMSTART_DIR"; exit 1; }
# DECOMPS: 분할 목록 (예: "1x1x6 3x1x2") — 각각 차례로 돌린다. 기본 = z-슬랩 1x1xNP.
for DEC in ${DECOMPS:-1x1x$NP}; do
  IFS=x read PX PY PZ <<< "$DEC"
  [[ $((PX*PY*PZ)) -eq $NP ]] || { echo "ERROR: $DEC != np $NP"; continue; }
  TAGD="${NAME}_${DEC}"
  RUN_DECK="${DECK%.ini}_run_${SLURM_JOB_ID}_${DEC}.ini"   # 잡·분할마다 다른 이름 (동시 제출 시 npz 가 섞인 사고, 940449)
  sed -e "s|WARMSTART_BIN|$LATEST|" -e "s|NPX_VAL|$PX|" -e "s|NPY_VAL|$PY|" -e "s|NPZ_VAL|$PZ|" "$DECK" > "$RUN_DECK"
  echo "# === $DEC  (WARMSTART_BIN 치환값: $LATEST — 덱이 restart_from 이면 쓰이지 않음)  $(date '+%T')"
  LOG="log/${TAGD}_$SLURM_JOB_ID.log"
  ( while sleep 20; do echo "$(date '+%T') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"; done ) > "log/${TAGD}_${SLURM_JOB_ID}_mem.txt" &
  MON=$!
  mpirun -np "$NP" "$BIN" "$RUN_DECK" > "$LOG" 2>&1; rc=$?
  kill $MON 2>/dev/null
  grep -E "warm-start|error \(rank|MPI_ABORT|blew up|out of memory|FINAL" "$LOG" | tail -4
  echo "# last monitor lines (마지막 열 = s/step):"; grep -v "^#" "$LOG" | tail -3
  echo "# peak GPU memory (MiB per GPU):"; awk '{for(i=2;i<=NF;i++) if($i+0>m[i]) m[i]=$i+0} END{for(i=2;i<=NF;i++) printf "%s ", m[i]; print ""}' "log/${TAGD}_${SLURM_JOB_ID}_mem.txt"
  echo "# $DEC rc=$rc end=$(date '+%F %T')"
done
