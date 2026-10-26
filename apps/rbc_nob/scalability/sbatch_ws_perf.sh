#!/bin/bash
# Workspace(솔버 버퍼 공유) 전/후 성능 동일성 — 같은 GPU 에서 before_ws 와 현재 바이너리를 번갈아 3회씩 (docs/Memory/plan.md §5).
#   rbc_nob strong_256 np1: 내장 PhaseTimer → phase_timing.csv (구간별 ms/step)
#   dhvc    strong_256 np1: 모니터 sec/step 열 (전체 s/step)
#   sbatch -p cas_v100nv_8 -w gpu04 --gres=gpu:1 apps/rbc_nob/scalability/sbatch_ws_perf.sh
#SBATCH --time=00:50:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=2 --mem-per-cpu=8000M --hint=nomultithread
#SBATCH -J ws_perf -o apps/rbc_nob/scalability/log/ws_perf_%j.out --comment="field=mae;appl=in_house"
set -u
if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then source "${MODULESHOME}/init/bash"; fi
module purge; module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
ROOT=/scratch/r908a06/KDH/MPM-STD_C; J=$SLURM_JOB_ID
OUT=$ROOT/apps/rbc_nob/scalability/out/ws_perf_$J; mkdir -p $OUT
echo "# job=$J host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) start=$(date '+%F %T')"
echo "# before=$ROOT/build_gpu/bin/before_ws ($(stat -c %y $ROOT/build_gpu/bin/before_ws/rbc_nob | cut -c1-19))  new=$ROOT/build_gpu/bin ($(stat -c %y $ROOT/build_gpu/bin/rbc_nob | cut -c1-19))"
for rep in 1 2 3; do
  for v in before new; do
    B=$ROOT/build_gpu/bin; [[ $v == before ]] && B=$ROOT/build_gpu/bin/before_ws
    # rbc_nob 256³ np1 — PhaseTimer CSV
    d=$OUT/rbc_nob_${v}_r$rep; mkdir -p $d
    sed -e "s|^stats_dir = .*|stats_dir = \"$d/\"|" $ROOT/apps/rbc_nob/scalability/input_scal_strong_256_np1.ini > $d/deck.ini
    ( cd $ROOT/apps/rbc_nob/scalability && mpirun -np 1 $B/rbc_nob $d/deck.ini > $d/run.log 2>&1 ); echo "rbc_nob $v r$rep rc=$? $(grep '# step_total' $d/run.log | awk '{print "step_total", $3, "ms"}')"
    # dhvc 256³ np1 — 모니터 sec/step
    d=$OUT/dhvc_${v}_r$rep; mkdir -p $d
    sed -e "s|^stats_dir = .*|stats_dir = \"$d/\"|" -e "s|^checkpoint_dir = .*|checkpoint_dir = \"$d/ck\"|" $ROOT/apps/dhvc/scale/decks/input_strong_256_np1.ini > $d/deck.ini
    ( cd $ROOT/apps/dhvc && mpirun -np 1 $B/dhvc $d/deck.ini > $d/run.log 2>&1 ); echo "dhvc $v r$rep rc=$? $(grep -E '^ +(100|150|200) ' $d/run.log | awk '{s+=$NF; n++} END{printf "sec/step(mean of 3 monitors) %.4f", s/n}')"
  done
done
python3 - "$OUT" <<'PY'
import csv, glob, os, sys, statistics as st
out = sys.argv[1]
def phases(path):
    d = {}
    for row in csv.reader(l for l in open(path) if not l.startswith("#")):
        if row and row[0] != "phase": d[row[0]] = float(row[3])   # max_ms (np1: min=avg=max)
    return d
runs = {v: [phases(p) for p in sorted(glob.glob(f"{out}/rbc_nob_{v}_r*/phase_timing.csv"))] for v in ("before", "new")}
if all(runs.values()):
    keys = [k for k, _ in sorted(runs["before"][0].items(), key=lambda kv: -kv[1]) if k != "step_total"]
    print(f"\n{'rbc_nob 256^3 np1 (ms/step, median of 3)':44} {'before':>9} {'new':>9} {'diff':>8}")
    for k in ["step_total"] + keys[:14]:
        b = st.median(r[k] for r in runs["before"]); n = st.median(r[k] for r in runs["new"])
        print(f"{k:44} {b:9.3f} {n:9.3f} {100*(n/b-1):+7.1f}%")
    print("rep spread step_total: before", [round(r["step_total"],2) for r in runs["before"]], "new", [round(r["step_total"],2) for r in runs["new"]])
for app in ("dhvc",):
    vals = {}
    for v in ("before", "new"):
        xs = []
        for p in sorted(glob.glob(f"{out}/{app}_{v}_r*/run.log")):
            rows = [l.split() for l in open(p) if l.strip() and l.split()[0] in ("100", "150", "200")]
            if rows: xs.append(sum(float(r[-1]) for r in rows) / len(rows))
        vals[v] = xs
    if vals["before"] and vals["new"]:
        b, n = st.median(vals["before"]), st.median(vals["new"])
        print(f"\n{app} 256^3 np1 wall s/step (median of 3): before {b:.4f}  new {n:.4f}  diff {100*(n/b-1):+.1f}%   reps before {[round(x,4) for x in vals['before']]} new {[round(x,4) for x in vals['new']]}")
PY
echo "# end=$(date '+%F %T')"
