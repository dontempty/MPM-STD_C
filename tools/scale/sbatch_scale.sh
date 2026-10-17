#!/bin/bash
# 앱별 전/후 확장성 측정 — build_gpu/bin/orig/<app>_scale(최적화 전, 원본 커밋) vs build_gpu/bin/<app>_scale(현재).
#   sbatch -p cas_v100nv_8 --gres=gpu:N --ntasks-per-node=N --export=ALL,APPS="dhvc channel",NP_LIST="1 2 4",CASES="strong_256 weak_256" tools/scale/sbatch_scale.sh
#   CSV: apps/<app>/scale/out/<case>_np<N>_<variant>_<job>_r<rep>.csv, 로그: apps/<app>/scale/log/
#SBATCH --time=02:30:00 --hint=nomultithread --nodes=1 --cpus-per-task=2 --mem-per-cpu=11000M
#SBATCH -J app_scale -o tools/scale/log/app_scale_%j.out --comment="field=mae;appl=in_house"
set -u
if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then source "${MODULESHOME}/init/bash"; fi
module purge; module load nvhpc/25.11_cuda12
export OMPI_MCA_opal_warn_on_missing_libcuda=0
ROOT=/scratch/r908a06/KDH/MPM-STD_C
APPS=${APPS:-"dhvc channel boundary_layer thermal_boundary_layer"}
VARIANTS=${VARIANTS:-"orig new"}
NP_LIST=${NP_LIST:-"1 2 4"}
CASES=${CASES:-"strong_256 weak_256"}
REPEAT=${REPEAT:-2}
export MPMSTD_PHASE_START=${MPMSTD_PHASE_START:-50}
J=$SLURM_JOB_ID
for APP in $APPS; do
  cd "$ROOT/apps/$APP" || exit 1
  mkdir -p scale/out scale/log scale/out_tmp
  { echo "# job=$J host=$(hostname) date=$(date -Is) app=$APP"; nvidia-smi --query-gpu=name,memory.total --format=csv; nvidia-smi topo -m 2>/dev/null | head -10
    for v in $VARIANTS; do b=$ROOT/build_gpu/bin/$([ $v = orig ] && echo orig/)${APP}_scale; echo "# bin[$v]=$b ($(stat -c %y "$b"))"; done; } > "scale/out/environment_${J}.txt"
  for v in $VARIANTS; do
    BIN=$ROOT/build_gpu/bin/$([ $v = orig ] && echo orig/)${APP}_scale
    for case in $CASES; do for np in $NP_LIST; do
      deck="scale/decks/input_${case}_np${np}.ini"; [[ -f $deck ]] || { echo "no deck $deck"; continue; }
      for rep in $(seq 1 $REPEAT); do
        csv="scale/out/${case}_np${np}_${v}_${J}_r${rep}.csv"; log="scale/log/${case}_np${np}_${v}_${J}_r${rep}.log"
        echo "=== $APP $v $case np $np rep $rep $(date '+%T') ==="
        MPMSTD_PHASE_CSV=$csv mpirun -x MPMSTD_PHASE_CSV -x MPMSTD_PHASE_START -np $np "$BIN" "$deck" > "$log" 2>&1; echo "=== rc=$? ==="
        grep -E "^(step_total|momentum_solve|pressure_solve)," "$csv" 2>/dev/null | head -3
      done
    done; done
  done
  rm -rf scale/out_tmp
done
echo "# done"
