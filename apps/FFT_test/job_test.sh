#!/bin/bash
#SBATCH --job-name=fft_test
#SBATCH --comment="field=mae;appl=in_house"
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=log/test_%j.out
# =============================================================================
#  FFT_test 시험 잡 — 작은 격자 6 종, 4 랭크 (decks/test/*.ini, gen_decks.py). 캠페인 전에 이것이 COMPLETED + 전부 PASS 여야 한다.
#  제출: apps/FFT_test/ 에서  mkdir -p log && sbatch job_test.sh
#  1 노드 GPU 잡: --ntasks·--cpus-per-task·--mem 은 쓰지 않는다 (사이트 규정). --cpus-per-task 는 srun 줄에만.
#  GPU 바인딩은 코드가 local rank 로 한다 — --gpus-per-task·CUDA_VISIBLE_DEVICES 금지 (Cray GTL IPC 가 깨진다).
# =============================================================================
set -u
APP=${SLURM_SUBMIT_DIR:-$PWD}
[ -f "${APP}/env_hangang.sh" ] || { echo "ERROR: apps/FFT_test/ 에서 제출할 것 (지금 ${APP})"; exit 1; }
REPO=$(cd "${APP}/../.." && pwd)
source "${APP}/env_hangang.sh"
BIN=${REPO}/build_gpu/bin/FFT_test
[ -x "${BIN}" ] || { echo "ERROR: ${BIN} 없음 — apps/FFT_test/build.sh 먼저"; exit 1; }

OUT=${APP}/results/test/j${SLURM_JOB_ID:-local}
mkdir -p "${OUT}"
echo "job ${SLURM_JOB_ID:-local}  node ${SLURM_JOB_NODELIST:-?}  $(date '+%F %T')  bin $(stat -c '%y' "${BIN}" | cut -d. -f1)"
nvidia-smi -L

fail=0
for deck in "${APP}"/decks/test/*.ini; do
    name=$(basename "${deck}" .ini)
    echo ""
    echo "==== ${name} ===="
    timeout -k 30 240 srun -n 4 --ntasks-per-node=4 --cpus-per-task=72 "${BIN}" "${deck}" "${OUT}/${name}" \
        > "${OUT}/${name}.log" 2>&1
    rc=$?
    grep -E "^(fft check|manufactured|random rhs|new vs old|# cyclic|# existing|PASS|FAIL|  FAIL)|old.solve_total|new.solve_total|FFT_test rank" "${OUT}/${name}.log"
    echo "rc=${rc}"
    [ ${rc} -eq 0 ] || fail=$((fail + 1))
    if [ ${fail} -ge 2 ]; then
        echo "!!! 두 덱 이상 실패 — 중단. 로그: ${OUT}"
        tail -20 "${OUT}/${name}.log"
        exit 1
    fi
done
echo ""
echo "==== DONE $(date '+%F %T')  실패 덱 ${fail} / 결과 ${OUT} ===="
exit ${fail}
