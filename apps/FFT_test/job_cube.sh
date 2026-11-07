#!/bin/bash
#SBATCH --job-name=fft_cube
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --output=log/cube_%j.out
# =============================================================================
#  cube 분할에서 MPM-STD 압력 경로의 통신 방식 비교 (plan.md §12). 평면묶음·펜슬을 한 런에서 잰다.
#  제출은 apps/FFT_test/ 의 submit_cube.sh 로. 노드 수가 np 를 정한다: np = 노드 × 4.
#
#  방식 (기존 소스 수정 없이 PMPI 로 바꾼다, transpose_transport.hpp):
#    alltoallv    전치·z TDMA 모두 MPI_Alltoallv (지금의 MPM-STD)
#    pairwise     x·y 전치만 pairwise
#    pairwise_z   x·y 전치와 z TDMA 축소계 all-to-all 모두 pairwise
#
#  환경변수: MODES "alltoallv pairwise pairwise_z", REPS 3, STEP_TIMEOUT 300 (np 128·256 은 600 으로, --time 도 함께 늘려 제출)
#  결과: results/cube/weak512/np<N>/j<잡>/  → plot_cube.py
#  재제출 자동화 없음. 2 스텝 실패하면 중단한다.
# =============================================================================
set -u
APP=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "${APP}/env_hangang.sh" ] || { echo "ERROR: apps/FFT_test/ 에서 제출할 것 (지금 ${APP})"; exit 1; }
REPO=$(cd "${APP}/../.." && pwd)
BIN=${REPO}/build_gpu/bin/FFT_test
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} 없음 — build.sh 먼저"; exit 1; }

MODES=${MODES:-"alltoallv pairwise pairwise_z"}
REPS=${REPS:-3}
STEP_TIMEOUT=${STEP_TIMEOUT:-300}
NNODES=${SLURM_JOB_NUM_NODES:-1}
NP=$(( NNODES * 4 ))
JOB=${SLURM_JOB_ID:-local}
SRUN_ARGS="-N ${NNODES} -n ${NP} --ntasks-per-node=4 --cpus-per-task=72 --kill-on-bad-exit=1"
deck=${APP}/decks/cube_weak512/cube_weak512_np${NP}.ini
[ -f "${deck}" ] || { echo "ERROR: ${deck} 없음"; exit 1; }
base=${APP}/results/cube/weak512/np${NP}/j${JOB}
mkdir -p "${base}"

echo "job ${JOB}  nodes ${NNODES} (${SLURM_JOB_NODELIST:-?})  np ${NP}  $(date '+%F %T')  bin $(stat -c '%y' "${BIN}" | cut -d. -f1)"
echo "modes ${MODES}  reps ${REPS}"

nbad=0
fail_step() {   # fail_step <이름> <rc> <로그>
    nbad=$((nbad + 1))
    echo "  FAIL ${1} rc=${2}  last stage: $(grep -E '^# stage' "${3}" 2>/dev/null | tail -1 | cut -c1-60)"
    grep -iE "error|abort|fault|FFT_test rank|FFT_test pairwise" "${3}" 2>/dev/null | head -5 | cut -c1-150 | sed 's/^/    | /'
    if [ ${nbad} -ge 2 ]; then echo "!!! 2 스텝 실패 — 중단 (원인 확인 후 다시 제출)"; exit 1; fi
}

run_mpm() {   # run_mpm <방식> <접두사>
    ( source "${APP}/env_hangang.sh" >/dev/null 2>&1
      case "$1" in
          alltoallv)  export FFT_TEST_TRANSPOSE=alltoallv FFT_TEST_TDMA_A2A=alltoallv ;;
          pairwise)   export FFT_TEST_TRANSPOSE=pairwise  FFT_TEST_TDMA_A2A=alltoallv ;;
          pairwise_z) export FFT_TEST_TRANSPOSE=pairwise  FFT_TEST_TDMA_A2A=pairwise ;;
          *) echo "unknown mode $1"; exit 8 ;;
      esac
      timeout -k 30 "${STEP_TIMEOUT}" srun ${SRUN_ARGS} "${BIN}" "${deck}" "$2" ) > "$2.log" 2>&1
}

echo "==== cube weak512 np ${NP}  $(date '+%T') ===="
for k in $(seq 1 "${REPS}"); do    # 같은 반복 안에서 번갈아 — 시간대 변동을 고르게
    for mode in ${MODES}; do
        prefix=${base}/${mode}_r${k}
        run_mpm "${mode}" "${prefix}"; rc=$?
        if [ ${rc} -ne 0 ] || ! grep -q "^PASS FFT_test" "${prefix}.log"; then
            fail_step "${mode} r${k}" "${rc}" "${prefix}.log"
            continue
        fi
        grep -E '^\[MPM-CONTRACT\]' "${prefix}.log" | cut -c16-85 | sed "s/^/  r${k} $(printf '%-10s' "${mode}") /"
    done
done
echo "==== DONE $(date '+%F %T')  실패 ${nbad} 스텝 ===="
exit ${nbad}
