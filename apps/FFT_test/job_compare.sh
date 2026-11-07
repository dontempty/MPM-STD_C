#!/bin/bash
#SBATCH --job-name=fft_compare
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=log/compare_%j.out
# =============================================================================
#  같은 문제를 MPM-STD(평면묶음·펜슬)·FFTU·CaNS·PaScaL 로 한 잡 안에서 잰다 (plan.md §10).
#  노드 배치에 따라 같은 코드도 시간이 크게 달라서, 비교는 반드시 같은 할당 안에서 한다.
#  제출은 apps/FFT_test/ 의 submit_compare.sh 로 한다 (멀티 노드 --exclusive).
#  노드 수가 np 를 정한다: 1 노드 = np 1 2 4 (먼저 시험 덱), N 노드 = np N×4.
#
#  CaNS·PaScaL 은 CMP_ROOT 에 이미 빌드된 실행파일과 입력 생성기를 그대로 쓴다 (읽기 전용, 결과는 여기에 쓴다).
#  실행 조건은 그 패키지의 slurm/run_paired_case.slurm 과 같다 (환경변수, 백엔드 mpi_p2p_pl, 예열 5 + 측정 20).
#  단, 공정성을 위해 세 코드 모두 사이트 권장 --cpus-per-task=72 로 돌린다 (그 패키지의 기록은 8).
#  CaNS·PaScaL 은 총시간과 RMS 만 잰다 (프로파일 실행 없음, 사용자 결정 2026-11-06). event 분해는 MPM-STD·FFTU 만.
#
#  환경변수: PROBLEMS "weak256 strong1024", NP_LIST (자동), REPS 3, STEP_TIMEOUT 420
#  결과: results/compare/<문제>/np<N>/j<잡>/{mpm,cans,pascal}/...  → plot_compare.py
#  재제출 자동화 없음. 2 스텝 실패하면 중단한다.
# =============================================================================
set -u
APP=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "${APP}/env_hangang.sh" ] || { echo "ERROR: apps/FFT_test/ 에서 제출할 것 (지금 ${APP})"; exit 1; }
REPO=$(cd "${APP}/../.." && pwd)
BIN=${REPO}/build_gpu/bin/FFT_test
CMP_ROOT=${CMP_ROOT:-/scratch/t525a06/KDH/CaNS_vs_PaScaL_Hangang_GH200_NP256_3STEP}
CMP_BIN=${CMP_ROOT}/_install/bin
BACKEND=mpi_p2p_pl
for f in "${BIN}" "${CMP_BIN}/cans_perf" "${CMP_BIN}/pascal_perf" \
         "${CMP_ROOT}/tools/generate_case.py" "${CMP_ROOT}/load_hangang_env.sh"; do
    [ -e "${f}" ] || { echo "ERROR: ${f} 없음"; exit 1; }
done

PROBLEMS=${PROBLEMS:-"weak256 strong1024"}
REPS=${REPS:-3}
STEP_TIMEOUT=${STEP_TIMEOUT:-420}
NNODES=${SLURM_JOB_NUM_NODES:-1}
JOB=${SLURM_JOB_ID:-local}
if [ -z "${NP_LIST:-}" ]; then
    if [ "${NNODES}" -eq 1 ]; then NP_LIST="1 2 4"; else NP_LIST=$(( NNODES * 4 )); fi
fi

echo "job ${JOB}  nodes ${NNODES} (${SLURM_JOB_NODELIST:-?})  np ${NP_LIST}  $(date '+%F %T')"
echo "problems ${PROBLEMS}  reps ${REPS}  FFT_test $(stat -c '%y' "${BIN}" | cut -d. -f1)  git $(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null)"

nbad=0
fail_step() {   # fail_step <이름> <rc> <로그>
    nbad=$((nbad + 1))
    echo "  FAIL ${1} rc=${2}  (${3})"
    grep -iE "error|abort|fault|FFT_test rank|FAIL" "${3}" 2>/dev/null | head -5 | sed 's/^/    | /'
    if [ ${nbad} -ge 2 ]; then echo "!!! 2 스텝 실패 — 중단 (원인 확인 후 다시 제출)"; exit 1; fi
}

srun_args() { local np=$1; echo "-N $(( (np + 3) / 4 )) -n ${np} --ntasks-per-node=$(( np < 4 ? np : 4 )) --cpus-per-task=72 --kill-on-bad-exit=1"; }

# MPM-STD·FFTU 한 스텝 — 환경은 서브셸 안에서만 바꾼다 (두 패키지의 모듈이 다르다)
run_mpm() {   # run_mpm <np> <덱> <접두사>
    ( source "${APP}/env_hangang.sh" >/dev/null 2>&1
      timeout -k 30 "${STEP_TIMEOUT}" srun $(srun_args "$1") "${BIN}" "$2" "$3" ) > "$3.log" 2>&1
}

# CaNS·PaScaL 한 스텝 — slurm/run_paired_case.slurm 의 환경과 인자
run_cmp() {   # run_cmp <np> <실행파일> <디렉터리> <입력 디렉터리> <cans|pascal> [VAR=값 ...]
    local np=$1 exe=$2 dir=$3 inputs=$4 solver=$5
    shift 5
    mkdir -p "${dir}"
    ( export SUITE_ROOT="${CMP_ROOT}" HANGANG_ENV_QUIET=1
      source "${CMP_ROOT}/tools/common.sh" >/dev/null 2>&1
      source "${CMP_ROOT}/load_hangang_env.sh" >/dev/null 2>&1
      export LD_LIBRARY_PATH="${CMP_ROOT}/_install/cudecomp/lib:${LD_LIBRARY_PATH:-}"
      export MPICH_GPU_SUPPORT_ENABLED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=WARN
      export CANS_WARMUP=5 CANS_ITERS=20 POISSON_WARMUP=5 POISSON_ITERS=20 POISSON_DEBUG_WARMUP=5 POISSON_DEBUG_COLD_PROFILE=0
      export POISSON_CUDECOMP_BACKEND="${BACKEND}" POISSON_TDMA_2RHS=1 POISSON_TDMA_STATIC_OPERATOR=1 POISSON_WRITE_SOLUTION=0
      export POISSON_COARSE_PROFILE=0 HANGANG_GPU_BIND_MODE_RESOLVED=none
      for assignment in "$@"; do export "${assignment}"; done
      cd "${dir}" || exit 9
      if [ "${solver}" = cans ]; then
          cp "${inputs}/cans_input.nml" .
          timeout -k 30 "${STEP_TIMEOUT}" srun $(srun_args "${np}") "${exe}" "${dir}/cans_input.nml"
      else
          cp "${inputs}/PARA_INPUT_${np}.dat" .
          timeout -k 30 "${STEP_TIMEOUT}" srun $(srun_args "${np}") "${exe}"
      fi ) > "${dir}/solver.log" 2>&1
}

if [ "${NNODES}" -eq 1 ] && [ "${RUN_TESTS:-1}" = 1 ]; then
    out=${APP}/results/test/j${JOB}
    mkdir -p "${out}"
    for deck in "${APP}"/decks/test/*.ini; do
        name=$(basename "${deck}" .ini)
        run_mpm 4 "${deck}" "${out}/${name}"
        rc=$?
        printf "test %-26s rc=%d  %s\n" "${name}" "${rc}" "$(grep -E '^(PASS|FAIL) FFT_test' "${out}/${name}.log" | tail -1)"
        if [ ${rc} -ne 0 ]; then tail -5 "${out}/${name}.log" | sed 's/^/    | /'; echo "!!! 시험 덱 실패 — 측정하지 않고 끝낸다"; exit 1; fi
    done
fi

for problem in ${PROBLEMS}; do
    for np in ${NP_LIST}; do
        deck=${APP}/decks/compare_${problem}/compare_${problem}_np${np}.ini
        [ -f "${deck}" ] || { echo "skip ${problem} np ${np}: no deck"; continue; }
        base=${APP}/results/compare/${problem}/np${np}/j${JOB}
        inputs=${base}/inputs
        mkdir -p "${base}/mpm" "${inputs}"
        # 생성기는 python ≥ 3.7 이 필요하다 (시스템 python3 는 3.6). 그 패키지 환경의 python 으로 돌린다.
        ( export SUITE_ROOT="${CMP_ROOT}" HANGANG_ENV_QUIET=1
          source "${CMP_ROOT}/tools/common.sh" >/dev/null 2>&1
          source "${CMP_ROOT}/load_hangang_env.sh" >/dev/null 2>&1
          python3 "${CMP_ROOT}/tools/generate_case.py" generate --problem "${problem}" --np "${np}" --backend "${BACKEND}" \
              --gpus-per-node 4 --out "${inputs}" ) > "${base}/generator.log" 2>&1 \
            || { fail_step "generate ${problem} np${np}" $? "${base}/generator.log"; continue; }
        scontrol show hostnames "${SLURM_JOB_NODELIST:-}" > "${base}/nodes.txt" 2>/dev/null
        echo "==== ${problem} np ${np}  $(date '+%T') ===="
        for k in $(seq 1 "${REPS}"); do    # 같은 반복 안에서 세 코드를 번갈아 — 시간대 변동을 고르게
            prefix=${base}/mpm/r${k}
            run_mpm "${np}" "${deck}" "${prefix}"; rc=$?
            [ ${rc} -eq 0 ] && grep -q "^PASS FFT_test" "${prefix}.log" || fail_step "mpm r${k}" "${rc}" "${prefix}.log"
            run_cmp "${np}" "${CMP_BIN}/cans_perf" "${base}/cans/perf/r${k}" "${inputs}" cans; rc=$?
            [ ${rc} -eq 0 ] || fail_step "cans_perf r${k}" "${rc}" "${base}/cans/perf/r${k}/solver.log"
            run_cmp "${np}" "${CMP_BIN}/pascal_perf" "${base}/pascal/perf/r${k}" "${inputs}" pascal; rc=$?
            [ ${rc} -eq 0 ] || fail_step "pascal_perf r${k}" "${rc}" "${base}/pascal/perf/r${k}/solver.log"
            grep -E "^\[MPM-CONTRACT\]" "${prefix}.log" | cut -c15-70 | sed "s/^/  r${k} /"
            grep -hE "Steady-state average" "${base}/cans/perf/r${k}/solver.log" "${base}/pascal/perf/r${k}/solver.log" | cut -c1-80 | sed "s/^/  r${k} /"
        done
    done
done
echo "==== DONE $(date '+%F %T')  실패 ${nbad} 스텝 — plot_compare.py ===="
exit ${nbad}
