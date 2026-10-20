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
RUN_DECK="${DECK%.ini}_run_${SLURM_JOB_ID}.ini"   # 잡마다 다른 이름 (동시 제출 시 npz 가 섞인 사고, 940449)
sed -e "s|WARMSTART_BIN|$LATEST|" -e "s|NPZ_VAL|$NP|" "$DECK" > "$RUN_DECK"; echo "# warm-start: $LATEST  npz=$NP"
LOG="log/${NAME}_$SLURM_JOB_ID.log"
( while sleep 20; do echo "$(date '+%T') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"; done ) > "log/${NAME}_${SLURM_JOB_ID}_mem.txt" &
MON=$!
mpirun -np "$NP" "$BIN" "$RUN_DECK" > "$LOG" 2>&1; rc=$?
kill $MON 2>/dev/null
grep -E "warm-start|error \(rank|MPI_ABORT|blew up|out of memory|FINAL" "$LOG" | tail -4
echo "# last monitor lines (마지막 열 = s/step):"; grep -v "^#" "$LOG" | tail -3
echo "# peak GPU memory (MiB per GPU):"; awk '{for(i=2;i<=NF;i++) if($i+0>m[i]) m[i]=$i+0} END{for(i=2;i<=NF;i++) printf "%s ", m[i]; print ""}' "log/${NAME}_${SLURM_JOB_ID}_mem.txt"
echo "# rc=$rc end=$(date '+%F %T')"
