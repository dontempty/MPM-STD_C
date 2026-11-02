# rbc_nob 확장성 측정 — KISTI-6 (HANGANG, GH200)

`rbc_nob` 의 시간 루프를 **main 호출 단위 구간**으로 재서 strong / weak 확장성을 측정한다.
측정 코드는 본 앱에서 분리해 전용 앱 `apps/rbc_nob_scalability/` 에 있다 (2026-11-02).
아래 1 → 2 → 3 을 그대로 실행하면 된다. 측정 계획·구간 정의·순위 규칙은 `docs/Scalability/plan.md`,
빌드 함정의 자세한 이유는 `BUILD_HANGANG.md` §5 에 있다.

| 항목 | 값 |
|---|---|
| 노드 | GH200 120GB × 4 / 노드, Grace 72 코어 × 4 (aarch64) |
| 파티션 | `natStr` (그룹 `bs0356`), 최대 1일, **사용자당 실행 5 잡** |
| 바인딩 | 1 랭크 = 1 GPU = Grace 72 코어 (`cudaSetDevice(node_rank % 4)`, `--cpus-per-task=72`) |
| 실행 | `srun` (cray-mpich 9.0.1, GPU-aware) |

---

## 1. 환경 — module load, export

빌드(로그인 노드)와 실행(잡 안)에 같은 환경을 쓴다.

```bash
module purge
module load craype-arm-grace PrgEnv-nvidia
module swap cray-mpich cray-mpich/9.0.1          # 기본 9.1.0 은 CUDA 13 GTL — 반드시 바꾼다
module load cuda/12.9_26.3 craype-accel-nvidia90
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH   # 실행 때 9.0.1 라이브러리가 먼저 잡히게
export MPICH_GPU_SUPPORT_ENABLED=1                              # 디바이스 포인터를 MPI 로 직접 보낸다
```

같은 내용이 `apps/rbc_nob_scalability/env_kisti6.sh` 에 있다 (`source` 로 대신할 수 있고, 잡 스크립트는 이 파일을 쓴다).

## 2. 빌드 — make

로그인 노드에서 리포 루트에서 한다 (Grace 라 계산 노드와 아키텍처가 같다. 전체 빌드 약 20초).

```bash
cd /scratch/t639a05/MPM-STD_C

make GPU=1 -j32 \
  CXX="CC --gcc-toolchain=/usr/bin/gcc-14" \
  NVHPC_DIR=$NVIDIA_PATH CUDA_VER=12.9 GPU_ARCH=cc90 \
  EXT_INC=-I$NVIDIA_PATH/math_libs/12.9/include \
  rbc_nob rbc_nob_scalability
```

- 결과: `build_gpu/bin/rbc_nob_scalability` (측정), `build_gpu/bin/rbc_nob` (게이트 대조용 — 둘 다 필요).
- TDMA 백엔드는 **PaScaL 만** 쓴다 (기본 빌드). `TDMA=filtered` 는 붙이지 않는다 — 잡은 `build_gpu/bin/` 만 본다.
- 덮어쓰는 변수 5개의 이유는 `BUILD_HANGANG.md` §3 (Makefile.inc 는 구 클러스터 기본값이라 고치지 않는다).

## 3. 실행 — sbatch

**`apps/rbc_nob_scalability/` 에서 제출한다** (잡 로그 `log/` 가 제출 폴더 기준이다).

```bash
cd /scratch/t639a05/MPM-STD_C/apps/rbc_nob_scalability
mkdir -p log

bash submit_scal.sh          # 노드 1·2·4·8 잡 4개 → np 1 2 4 | 8 | 16 | 32, strong·weak 256³ slab
```

노드 수 하나씩 내도 된다. 잡은 **자기 노드 수를 꽉 채우는 np 만** 돈다.

```bash
sbatch -N 1  job_scal.sh     # np 1 2 4
sbatch -N 2  job_scal.sh     # np 8
sbatch -N 4  job_scal.sh     # np 16
sbatch -N 8  job_scal.sh     # np 32
sbatch -N 16 job_scal.sh     # np 64
```

무엇을 잴지는 환경변수로 고른다 (`VAR=… bash submit_scal.sh` 또는 `VAR=… sbatch …` — sbatch 는 기본으로 환경을 넘긴다).

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `STUDIES` | `strong_256_slab weak_256_slab` | `decks/<study>/` 이름. `<strong\|weak>_<256\|512>_<slab\|pencil>` |
| `REPS` | `2` | 같은 구성 반복 횟수 (요약은 step_total 중앙값 런) |
| `NP_LIST` | 노드 수로 자동 | 예: `NP_LIST="1 2"`. 항목을 `np/rpn` 으로 쓰면 그 런만 노드당 rpn 랭크로 펼친다 (아래 예) |
| `RPN` | `4` | 노드당 랭크 수 기본값 (1–4). `RPN=1` 이면 `-N 2` → np 2, `-N 4` → np 4 (전부 노드 간 통신) |
| `GATE` | `1` | `0` 이면 rbc_nob 대조 게이트를 건너뛴다 |
| `RUN_TIMEOUT` | `900` | 런당 상한(초) |
| `RESULTS` | `results/` | 결과 폴더. 시험 런은 `RESULTS=$PWD/results/_test` |
| `NODES` | `1 2 4 8` | (`submit_scal.sh` 만) 낼 노드 수 목록 |

예:

```bash
STUDIES="strong_512_slab weak_512_slab" bash submit_scal.sh               # 512³
STUDIES="strong_256_pencil weak_256_pencil" NODES="1 2 4 8 16" bash submit_scal.sh   # 3차원 분할
REPS=3 NP_LIST="1 2" sbatch -N 1 job_scal.sh                               # np 1·2 만 3회씩

# 노드 안 랭크끼리의 통신을 피해 np 2·4 를 노드당 1 랭크로 (4 노드 잡 하나에서 np 8 은 2노드 x 4)
STUDIES=weak_512_slab NP_LIST="1/1 2/1 4/1 8/4" RESULTS=$PWD/results/weak512_internode \
  sbatch -N 4 --time=00:40:00 job_scal.sh
```

배치(노드 수 × 노드당 랭크)는 결과 파일 이름의 `n<노드>x<rpn>` 에 들어가고, `summarize.py` 는 같은 np 라도 배치가 다르면 다른 행으로 보인다.

## 4. 결과 보기

```bash
cd /scratch/t639a05/MPM-STD_C/apps/rbc_nob_scalability
python3 summarize.py                 # 전체
python3 summarize.py --study strong_256_slab --phases    # 한 study, 구간 표까지
```

`summarize.py` 는 study 마다 다음을 찍는다.

- **효율표**: np, 분할, ms/step(랭크 max), speedup, 효율 (strong: `T(np0)/T(np)/(np/np0)`, weak: `T(np0)/T(np)`),
  comm% (분산 TDMA MPI 시간 / step_total), untimed% (step_total − 구간 합, 2% 를 넘으면 구간 누락), 반복 편차.
- **손실 순위** @최대 np: strong `T_p(np) − T_p(np0)·np0/np`, weak `T_p(np) − T_p(np0)` 내림차순 (plan.md §5).
- **결과 불변** (strong): 마지막 모니터 줄의 np0 대비 최대 상대차 — 분할이 계산을 바꾸지 않았는지.
- 대표 런 전부의 구간 표를 `results/summary_phases.csv` 로 쓴다 (노트북용).

파일 위치:

```
results/<study>/<study>_np<N>_n<노드>x<rpn>_r<k>_j<job>.csv   구간 표 (ms/step: min, avg, max over ranks, share, imbalance)
results/<study>/<study>_np<N>_n<노드>x<rpn>_r<k>_j<job>.log   런 출력 (설정 에코, 모니터, 구간 표)
results/env/environment_j<job>.txt                            git 커밋·모듈·GPU·노드 목록
log/scal_<job>.out                                            잡 진행 (바인딩, 게이트, 런별 ms/step)
log/j<job>/gate_*.log                                         게이트 런 로그
```

`results/` 와 `log/` 는 커밋하지 않는다 (.gitignore).

그림은 `plot/strong.ipynb`, `plot/weak.ipynb` (커널 `channel-plot`, 공용 코드 `plot/scaling_plot.py`).
첫 코드 셀의 `RESULTS` (결과 폴더)와 `STUDIES` 를 바꿔 다른 측정을 그린다. 시간·speedup·효율, 구간 분해(누적 막대),
통신 시간(TDMA MPI, halo) 순서이고, 숫자는 `summarize.py` 와 같은 계산이다. png 는 `plot/figures/` (커밋 안 함).

---

## 5. 무엇을 어떻게 재는가

### 5.1 구간 — rbc_nob 시간 루프의 호출 단위 (plan.md §1 표가 정본)

| 구간 | 부르는 것 |
|---|---|
| `dt_cfl`, `dt_viscous` | CFL dt, 가변 점성 dt 한계 (리덕션 + Allreduce) |
| `properties_1` | 물성 갱신 (Tⁿ) |
| `scalar_solve`, `sync_T` | 에너지 가변-κ̂ ADI (z 분산 TDMA), T halo + 벽 |
| `momentum_rhs`, `momentum_solve`, `sync_UVW_1` | 운동량 우변, 가변 운동량 ADI ×3 + 커플링, 속도 halo |
| `properties_2` | 투영 직전 물성 재갱신 (Tⁿ⁺¹) |
| `pressure_rhs`, `pressure_solve`, `pressure_correct`, `sync_UVW_2` | Poisson 우변, FFT + z TDMA, 보정 + P 누적, 속도 halo |
| `predictor`, `clip` | p̂ 외삽 + halo, T 클립 |
| `monitor` | 모니터 줄 (`monitor_interval` 마다) |
| `step_total` | 한 스텝 전체 |
| `tdma_comm_<scalar\|momentum\|pressure>_<x\|y\|z>` | 분산 TDMA(PaScaL) 의 MPI(Alltoallv) 시간 — 위 풀이 구간에 포함된 몫 |

- 각 구간 경계에서 `cudaDeviceSynchronize()` 후 `MPI_Wtime()` (`src/common/phase_timer.hpp`). 비동기 커널을 자기 구간에 귀속시키기 위해서다.
- 비용 지표는 **랭크 max** (가장 늦은 랭크가 스텝을 정한다). min·avg·불균형도 CSV 에 있다.
- 덱: 냉시작 200 스텝, 앞 50 스텝 제외(`[timing] warmup_steps`) → 150 스텝 누적. 통계·체크포인트는 없다.
- `[timing] solver_phases = true` 로 덱을 바꾸면 풀이 안의 호출별 구간(`momentum.dU.build_x` 등)까지 잰다 —
  구간이 1위로 올라왔을 때 그 안을 가르는 다음 단계용 (plan.md §1.1).

### 5.2 study — 덱은 `gen_decks.py` 가 만든다

물리는 cv12 ΔT 40 본런 덱과 같다 (Ra 1e6, Pr 2547, ΔT 40 K, γ_z 2.6, CFL 0.3, viscous_coupling_number 12).

| | strong (전역 n³ 고정) | weak (랭크당 n³ 고정) |
|---|---|---|
| **slab** (1×1×np) | 6×6×1 박스, nz_local = n/np | 전역 n×n×(n·np), lz = np — np 따라 변하는 건 z 분산 TDMA 축소계뿐 |
| **pencil** (2×1×1, 2×2×1, 2×2×2, 4×2×2 …) | 6×6×1 박스를 3축으로 | 나눈 축마다 격자·도메인을 늘린다 (셀 크기 유지) |

n = 256, 512, np = 1 … 64. 축당 랭크 셀이 8 보다 적은 조합은 만들지 않는다 (지금은 strong_256_slab np 64 하나).
덱을 바꾸려면 `python3 gen_decks.py --help` (스텝 수, 크기, np 목록).

### 5.3 게이트 — 측정 앱 = 본 앱

측정 앱의 루프는 `apps/rbc_nob/main.cpp` 를 옮겨 온 것이라 둘이 어긋날 수 있다. 그래서 잡마다 측정 전에
같은 작은 계산(64×64×16·np, 60 스텝, 그 잡의 최대 np)을 `rbc_nob` 과 `rbc_nob_scalability` 로 돌려
모니터 줄을 `check_gate.py` 로 비교한다 (상대차 ≤ 1e-12). 다르면 측정하지 않고 잡을 끝낸다.
**rbc_nob 루프를 고치면 `apps/rbc_nob_scalability/main.cpp` 도 같이 고친다.**

---

## 6. 코드 구성 (`apps/rbc_nob_scalability/`)

| 파일 | 역할 |
|---|---|
| `main.cpp` | 측정 드라이버. rbc_nob 시간 루프 + PhaseTimer 구간 + TDMA 통신 훅. 사용: `rbc_nob_scalability <deck.ini> [timing.csv]` |
| `settings.{hpp,cpp}` | 이 앱 소유 설정 — rbc 공용 설정에서 측정에 쓰는 키만 + `[timing]` (`warmup_steps`, `solver_phases`) |
| `rbc_nob_statistics.cpp` | rbc_nob 의 Statistics 를 빌려오는 1줄 TU (모니터의 θ_c·벽 Nu 를 같은 식으로) |
| `gen_decks.py` | `decks/<study>/`, `decks/gate/` 생성 |
| `env_kisti6.sh` | 1절 환경 |
| `job_scal.sh` | 잡: 프리플라이트 → 바인딩 출력 → 게이트 → 측정 루프 → 실패 집계 |
| `submit_scal.sh` | 노드 수별 일괄 제출 (natStr 5 잡 한도 경고) |
| `summarize.py`, `check_gate.py` | 4절 요약, 5.3절 게이트 |

분리하면서 바뀐 것 (2026-11-02):

- `apps/rbc_nob/main.cpp` 에서 구간 타이머 코드를 뺐다 (수치 불변 — 타이머는 켤 때만 동기화를 넣었다).
- rbc 공용 설정의 `[output] phase_timing_start_step` 을 뺐다 → 이 앱의 `[timing] warmup_steps`.
  이 키가 들어 있는 옛 덱(`apps/rbc_nob/scalability/input_scal_*.ini`, 구 클러스터용)은 이제 rbc_nob 에서 unknown-key 에러다.
- 풀이 안의 단계 타이머(`Solver::timer`, `src/equation/*/solve.*`)는 아직 src 에 있다 — 다음 분리 대상.

## 7. 주의

- natStr 는 **사용자당 실행 5 잡**. `submit_scal.sh` 기본(4 잡)도 다른 잡이 돌고 있으면 대기한다.
- 잡은 `--exclusive` 로 노드를 통째로 잡는다 (np 1·2 런도 같은 노드 안에서 — 이웃 잡 간섭 없음).
- 구간 경계 동기화(스텝당 약 34회)의 오버헤드는 이 클러스터에서 아직 따로 재지 않았다 (구 클러스터 ≤ 3%, plan.md §2.1).
- 빌드·실행 오류(`libcudart.so.13`, `filesystem`, CUDA driver insufficient …)는 `BUILD_HANGANG.md` §5 함정 ①–③.
