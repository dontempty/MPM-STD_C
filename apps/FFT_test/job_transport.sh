#!/bin/bash
#SBATCH --job-name=fft_transport
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=log/transport_%j.out
# =============================================================================
#  MPM-STD 펜슬 경로의 x·y 전치 통신 방식 비교 (plan.md §11): MPI_Alltoallv, pairwise, NCCL, 그리고 같은 잡의 PaScaL.
#  기존 소스를 고치지 않고 PMPI 로 바꾼다 (transpose_transport.hpp). z TDMA·halo 는 모든 방식에서 MPI 다.
#  제출은 apps/FFT_test/ 의 submit_transport.sh 로. 노드 수가 np 를 정한다: np = 노드 × 4.
#
#  NCCL 런은 NCCL_PROTO=^LL128 (사이트 규정)을 쓰고, 멀티노드면 aws-ofi-nccl 모듈도 쓴다. 이 모듈의 FI_CXI_* 설정은 같은 런의 MPI 통신에도 걸린다.
#  멀티노드 NCCL 런의 로그에 "Selected provider is cxi" 가 없으면 TCP 대체로 보고 실패로 처리한다.
#
#  환경변수: PROBLEMS "weak256 strong1024", MODES "alltoallv pairwise nccl", REPS 3, STEP_TIMEOUT 300
#  결과: results/transport/<문제>/np<N>/j<잡>/  → plot_transport.py
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
for f in "${BIN}" "${CMP_BIN}/pascal_perf" "${CMP_ROOT}/tools/generate_case.py" "${CMP_ROOT}/load_hangang_env.sh"; do
    [ -e "${f}" ] || { echo "ERROR: ${f} 없음"; exit 1; }
done

PROBLEMS=${PROBLEMS:-"weak256 strong1024"}
MODES=${MODES:-"alltoallv pairwise nccl"}
REPS=${REPS:-3}
STEP_TIMEOUT=${STEP_TIMEOUT:-300}
NNODES=${SLURM_JOB_NUM_NODES:-1}
NP=$(( NNODES * 4 ))
JOB=${SLURM_JOB_ID:-local}
SRUN_ARGS="-N ${NNODES} -n ${NP} --ntasks-per-node=4 --cpus-per-task=72 --kill-on-bad-exit=1"

echo "job ${JOB}  nodes ${NNODES} (${SLURM_JOB_NODELIST:-?})  np ${NP}  $(date '+%F %T')  bin $(stat -c '%y' "${BIN}" | cut -d. -f1)"
echo "modes ${MODES}  problems ${PROBLEMS}  reps ${REPS}"

nbad=0
fail_step() {   # fail_step <이름> <rc> <로그>
    nbad=$((nbad + 1))
    echo "  FAIL ${1} rc=${2}  last stage: $(grep -E '^# stage' "${3}" 2>/dev/null | tail -1 | cut -c1-60)"
    grep -iE "error|abort|fault|FFT_test rank|NCCL WARN" "${3}" 2>/dev/null | head -5 | cut -c1-150 | sed 's/^/    | /'
    if [ ${nbad} -ge 2 ]; then echo "!!! 2 스텝 실패 — 중단 (원인 확인 후 다시 제출)"; exit 1; fi
}

run_mpm() {   # run_mpm <방식> <덱> <접두사>
    ( source "${APP}/env_hangang.sh" >/dev/null 2>&1
      export FFT_TEST_TRANSPOSE="$1"
      if [ "$1" = nccl ]; then
          # 멀티노드만 네트워크 플러그인을 쓴다. 1 노드 스텝에서는 플러그인의 CXI 도메인 생성이 실패한다 (RC -38, 잡 1229571·1229581;
          # Slurm 설정에 single_node_vni 가 없어 1 노드 스텝에 VNI 가 없는 것으로 보인다). 같은 노드 안은 NVLink 라 플러그인이 필요 없다.
          if [ "${NNODES}" -gt 1 ]; then
              module load aws-ofi-nccl/1.20.0 >/dev/null 2>&1 || { echo "module aws-ofi-nccl failed"; exit 8; }
          fi
          export NCCL_PROTO='^LL128' NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
      fi
      timeout -k 30 "${STEP_TIMEOUT}" srun ${SRUN_ARGS} "${BIN}" "$2" "$3" ) > "$3.log" 2>&1
}

run_pascal() {   # run_pascal <디렉터리> <입력 디렉터리>  — slurm/run_paired_case.slurm 과 같은 환경·인자
    mkdir -p "$1"
    ( export SUITE_ROOT="${CMP_ROOT}" HANGANG_ENV_QUIET=1
      source "${CMP_ROOT}/tools/common.sh" >/dev/null 2>&1
      source "${CMP_ROOT}/load_hangang_env.sh" >/dev/null 2>&1
      export LD_LIBRARY_PATH="${CMP_ROOT}/_install/cudecomp/lib:${LD_LIBRARY_PATH:-}"
      export MPICH_GPU_SUPPORT_ENABLED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_DEBUG=WARN
      export POISSON_WARMUP=5 POISSON_ITERS=20 POISSON_DEBUG_WARMUP=5 POISSON_DEBUG_COLD_PROFILE=0
      export POISSON_CUDECOMP_BACKEND="${BACKEND}" POISSON_TDMA_2RHS=1 POISSON_TDMA_STATIC_OPERATOR=1 POISSON_WRITE_SOLUTION=0
      export POISSON_COARSE_PROFILE=0 HANGANG_GPU_BIND_MODE_RESOLVED=none
      cd "$1" || exit 9
      cp "$2/PARA_INPUT_${NP}.dat" .
      timeout -k 30 "${STEP_TIMEOUT}" srun ${SRUN_ARGS} "${CMP_BIN}/pascal_perf" ) > "$1/solver.log" 2>&1
}

for problem in ${PROBLEMS}; do
    deck=${APP}/decks/transport_${problem}/transport_${problem}_np${NP}.ini
    [ -f "${deck}" ] || { echo "skip ${problem}: no deck for np ${NP}"; continue; }
    base=${APP}/results/transport/${problem}/np${NP}/j${JOB}
    inputs=${base}/inputs
    mkdir -p "${base}" "${inputs}"
    ( export SUITE_ROOT="${CMP_ROOT}" HANGANG_ENV_QUIET=1
      source "${CMP_ROOT}/tools/common.sh" >/dev/null 2>&1
      source "${CMP_ROOT}/load_hangang_env.sh" >/dev/null 2>&1
      python3 "${CMP_ROOT}/tools/generate_case.py" generate --problem "${problem}" --np "${NP}" --backend "${BACKEND}" \
          --gpus-per-node 4 --out "${inputs}" ) > "${base}/generator.log" 2>&1 \
        || { fail_step "generate ${problem}" $? "${base}/generator.log"; continue; }
    echo "==== ${problem} np ${NP}  $(date '+%T') ===="
    for k in $(seq 1 "${REPS}"); do    # 같은 반복 안에서 번갈아 — 시간대 변동을 고르게
        for mode in ${MODES}; do
            prefix=${base}/${mode}_r${k}
            run_mpm "${mode}" "${deck}" "${prefix}"; rc=$?
            if [ ${rc} -ne 0 ] || ! grep -q "^PASS FFT_test" "${prefix}.log"; then
                fail_step "${mode} r${k}" "${rc}" "${prefix}.log"
                continue
            fi
            note=""
            if [ "${mode}" = nccl ]; then
                cxi=$(grep -c "Selected provider is cxi" "${prefix}.log")
                note="cxi lines ${cxi}"
                if [ "${NNODES}" -gt 1 ] && [ "${cxi}" -eq 0 ]; then
                    mv "${prefix}.log" "${prefix}.no_cxi.log"
                    fail_step "${mode} r${k} no cxi provider (TCP fallback)" 0 "${prefix}.no_cxi.log"
                    continue
                fi
            fi
            printf "  r%d %-10s %s  %s\n" "${k}" "${mode}" "$(grep -E '^\[MPM-CONTRACT\]' "${prefix}.log" | cut -c16-100)" "${note}"
        done
        run_pascal "${base}/pascal/r${k}" "${inputs}"; rc=$?
        if [ ${rc} -ne 0 ]; then fail_step "pascal r${k}" "${rc}" "${base}/pascal/r${k}/solver.log"; continue; fi
        printf "  r%d %-10s %s\n" "${k}" "pascal" "$(grep -E 'Steady-state average' "${base}/pascal/r${k}/solver.log" | cut -c1-80)"
    done
done
echo "==== DONE $(date '+%F %T')  실패 ${nbad} 스텝 ===="
exit ${nbad}
