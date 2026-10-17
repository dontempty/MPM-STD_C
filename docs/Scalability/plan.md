# rbc_nob 확장성(scalability) 측정 계획 — 256³·512³ strong/weak, main 호출 단위 (2026-10-17)

## 0. 한 줄 요약

rbc_nob 는 검증이 끝났지만 느리다. **커널 내부를 들여다보기 전에**, `apps/rbc_nob/main.cpp` 의 시간 루프가
부르는 호출 단위로만 구간 시간을 재서 (1) 한 스텝이 어디에 쓰이는지, (2) 랭크 수를 늘릴 때 어느 구간이
줄지 않는지를 256³·512³ strong / weak 로 측정하고, **병목 순위표**를 만든 뒤 1위부터 하나씩 고친다.
최적화는 이 문서의 범위 밖이다 — 이 문서는 "무엇을, 어떻게 재서, 어떤 기준으로 순위를 매기는가"까지다.

원칙 (사용자 지시 2026-10-17):
- 처음부터 커널 내부(nsys/ncu) 프로파일링은 하지 않는다. main 에서 호출되는 부분만 먼저 잰다.
- 병목 지역에 순위를 매기고, 순서대로 최적화한다. 한 번에 하나.

---

## 1. 측정 대상 — main.cpp 시간 루프의 호출 목록

`apps/rbc_nob/main.cpp` 의 `while (t < end_time …)` 한 바퀴를 호출 순서대로 쪼갠다. 타이머 구간 이름은
아래 표의 **구간** 열이 정본이다 (코드·CSV·그림이 전부 이 이름을 쓴다).

| # | 구간 | main 이 부르는 것 | 통신 | 성격 |
|---|---|---|---|---|
| 1 | `dt_cfl` | `compute_cfl_dt` | 리덕션 D2H 1 + Allreduce 1 | 동기화 지점 |
| 2 | `dt_viscous` | `compute_viscous_coupling_dt` | 리덕션 D2H 1 + Allreduce 1 | 동기화 지점 |
| 3 | `properties_1` | `update_properties`(Tⁿ) | 없음 | 로컬 커널 |
| 4 | `scalar_solve` | `scalar::solve_variable_property` | z 분산 TDMA 1회 (Alltoallv 전진+후진) | ADI x·y 는 랭크 내 순환 TDMA, z 만 분산 |
| 5 | `sync_T` | `sync_temperature` | halo 1 | halo + 벽 ghost |
| 6 | `momentum_rhs` | `rhs_*.fill(0)` + `add_advection_diffusion_{u,v,w}_variable_property` + `add_buoyancy_z_variable_property` + `pressure::add_gradient_variable_property` | 없음 | 로컬 커널 (가변물성 스텐실 3 + 부력 + ∇P) |
| 7 | `momentum_solve` | `momentum::solve_variable_property` | z 분산 TDMA 3회 + halo 4 | ADI ×3 (dU→dV→dW 전진 커플링) + 사후 커플링 |
| 8 | `sync_UVW_1` | `sync_velocity` | halo 3 | halo + 벽 ghost |
| 9 | `properties_2` | `update_properties`(Tⁿ⁺¹, 투영 전 재갱신) | 없음 | 로컬 커널 |
| 10 | `pressure_rhs` | `rhs_dP.fill(0)` + `add_divergence` + `add_extra_term_variable_property` | 없음 | 로컬 커널 |
| 11 | `pressure_solve` | `pressure::solve` | z 분산 TDMA 1회 + Allreduce 1(평균 제거) + halo 1 | cuFFT x·y 전진/후진 4 + 재배열 + z TDMA |
| 12 | `pressure_correct` | `correct_velocity_variable_property` + `accumulate` | 없음 | 로컬 커널 |
| 13 | `sync_UVW_2` | `sync_velocity` | halo 3 | halo + 벽 ghost |
| 14 | `predictor` | `advance_pressure_predictor` + halo + `fill_z_wall_zero_gradient` | halo 1 | 로컬 + halo |
| 15 | `clip` | `clip_interior` | 없음 | 로컬 커널 |
| 16 | `monitor` | 모니터 블록 (`max_divergence`, `plane_average`, `plane_maximum`, viscous 진단, printf) | D2H 수 회 + Allreduce 수 회 | `monitor_interval` 마다 — 분리해서 잰다 |
| — | (제외) | `statistics.accumulate/write`, 체크포인트 | — | 벤치마크 덱에서 끈다 (§4) |

한 스텝의 통신 총량 (z-슬랩 분할, 위 표의 합):
- **z 분산 TDMA 5회** (scalar 1 + momentum 3 + pressure 1). 각각 PaScaL_TDMA 의 `MPI_Alltoallv` 2회(전진·후진).
  축소계(reduced system)는 랭크당 시스템 수 × 2행 — **np 가 늘수록 축소계가 커진다** (strong 에서 줄지 않는 항).
- **halo 교환 13회** (T 1, momentum 내부 4, UVW ×2 = 6, pressure 1, predictor 1). 전부 blocking `MPI_Sendrecv`,
  z-슬랩이라 면 크기 = nx·ny·8B — **strong 에서 np 와 무관하게 고정** (nz_local 이 줄어도 안 준다).
- **Allreduce 3회** (dt 2 + 압력 평균 1), 각각 앞에 D2H 동기화.

이 세 항이 "np 를 늘려도 줄지 않는 후보"다. 단, **후보일 뿐이고 순위는 측정이 정한다** (§5).

### 1.1 처음에는 더 쪼개지 않는다

`momentum_solve`·`pressure_solve`·`scalar_solve` 는 호출 하나가 크다. 이들이 1위로 올라오면 **그때** 그 함수
안을 같은 방식(구간 타이머)으로 한 단계만 더 쪼갠다 (예: momentum → ADI-x / ADI-y / ADI-z / halo / 커플링).
커널 프로파일러(nsys)는 구간 타이머로 더 이상 못 가를 때만 쓴다 — 지금 계획에는 없다.

---

## 2. 측정 방법 — 구간 타이머

### 2.1 설계 (`src/common/phase_timer.hpp`, 헤더 하나, CPU/GPU 공용)

```
PhaseTimer timer(names);           // 구간 이름 고정 목록
timer.begin("momentum_solve");     // GPU: cudaDeviceSynchronize() 후 MPI_Wtime()
...
timer.end("momentum_solve");       // 동일, 누적 += 경과
timer.report(comm, n_steps, csv);  // 랭크별 합 → MPI_Reduce(min/avg/max) → 루트가 표 + CSV
```

- 각 경계에서 `cudaDeviceSynchronize()` 를 넣는다. 비동기 커널을 구간에 귀속시키려면 필요하고,
  main 단위 입도에서는 오버랩 손실이 작다. **타이머 오버헤드는 수치로 남긴다**: 같은 덱을 타이머 on/off 로
  돌려 모니터의 sec/step 차이를 기록한다 (허용 ≤ 3%).
- 워밍업 제외: `[output] phase_timing_start_step = 50` (0 = 끔). 그 스텝부터 누적을 시작한다.
  설정 키로 두는 이유: 이 프로젝트는 모르는 키를 에러로 하고 런 로그 머리에 설정을 에코하므로, 런이 실제로
  쓴 값이 로그에 남는다 (readme §4).
- 보고: 구간별 **sec/step (랭크 max)** 를 기본 지표로, 랭크 min·avg 와 불균형 `(max−min)/avg` 를 같이 낸다.
  통신 구간은 max 가 의미 있고(가장 늦은 랭크가 스텝을 정한다), 계산 구간은 불균형이 부하 분배를 말한다.
- 출력: 로그 끝에 `# phase <이름> <min> <avg> <max> <share%>` 줄들 + CSV 한 파일
  (`scalability/timing_<tag>.csv`: `mode,nx,ny,nz,np,nodes,phase,min,avg,max,share`). 노트북이 CSV 만 읽는다.
- 합계 검증: 구간 합 vs 루프 전체 경과(별도 타이머 `step_total`). 차이 = 재지 않은 곳 (허용 ≤ 2%, 넘으면 구간 누락).

### 2.2 PaScaL_TDMA 내부 시간은 공짜로 얻는다

`pascal_tdma_cuda.cu` 의 `alltoallv_double(comm_seconds, …)` 와 `SolveTimings`(local_compute / pack / mpi_forward /
reduced_compute / mpi_backward …) 훅이 이미 있다. **구간 타이머와 별개로** 플랜마다 `SolveTimings` 를 켜서 스텝당
합을 CSV 에 같이 쓴다 (`tdma_<plan>_<항목>`). 이는 "TDMA 가 병목이면 그 안의 MPI 가 문제인지 축소계 계산이
문제인지"를 커널 프로파일링 없이 가른다. z 플랜 3개(scalar·momentum·pressure)를 구분한다.

---

## 3. 격자·랭크 조합

제약: 압력 솔버가 **z-슬랩 전용(npx = npy = 1)** (`pressure/solve.hpp` 머리말) → np = npz ≤ nz, nz 가 np 로 나뉘어야 한다.

### 3.1 노드·파티션 (이 클러스터, 2026-10-17 `sinfo`)

| 파티션 | 노드 | GPU | 메모리 | 용도 |
|---|---|---|---|---|
| `amd_a100nv_8` | gpu36–43 (8) | A100-80GB ×8/노드 (NVLink) | 985 GB | **주 측정**: 노드 내 1–8, 노드 간 16·32·64 |
| `amd_h200nv_8` | gpu54–56 (3) | H200-141GB ×8/노드 | 2.2 TB | 보조: 같은 조합을 세대 차로 비교 (TDMA 연구와 동일 축) |
| `cas_v100nv_8` | gpu01–05 (5) | V100-32GB ×8/노드 | 371 GB | 보조/대기열 여유용. 512³ 는 메모리상 np ≥ 4 |

GPU 할당은 `cudaSetDevice(node_rank % n_devices)` 라 `--ntasks-per-node = GPU 수` 로 1 랭크 = 1 GPU.
노드 간 조합(16 이상)은 NVLink → InfiniBand 로 바뀌는 지점이라 **별도 열로 표시**한다.

### 3.2 메모리 추정 (랭크 메모리 = 전역 / np)

필드 1장 = nx·ny·nz·8B: 256³ → 134 MB, 512³ → 1.07 GB.
rbc_nob 가 상주시키는 전체 필드 수 ≈ 상태 12(U V W T P, rhs 3, rhs_dP, dP, p̂, p̂₀) + 물성 6 + momentum δ 3
+ 통계 2 + ADI 밴드 4–8 + 압력(라인·스펙트럼·밴드) ≈ 7 → **약 35–40장**.
→ 256³ ≈ 5 GB, 512³ ≈ 40 GB (전역). A100-80GB 1장에 512³ 가 **간신히** 들어간다; V100-32GB 는 np ≥ 4.
Phase 0 에서 `cudaMemGetInfo` 로 실측해 이 표를 교체한다.

### 3.3 Strong scaling (전역 격자 고정)

| 전역 | np (= npz) | nz_local | 노드 |
|---|---|---|---|
| 256³ | 1, 2, 4, 8 | 256, 128, 64, 32 | A100 1노드 |
| ~~256³~~ | ~~16, 32~~ | | ~~A100 2, 4노드~~ — **취소: 다노드 잡은 이 큐에서 배정되지 않는다 (사용자 결정 10-17). 상한 = 1노드 8 GPU** |
| 512³ | 1(메모리 확인 후), 2, 4, 8 | 512 … 64 | A100 1노드 |
| ~~512³~~ | ~~16, 32, 64~~ | | ~~A100 2, 4, 8노드~~ — 취소 (같은 이유) |

nz_local 8 까지 내려가면 halo 2면/8면 = 25% 가 ghost 라 strong 의 끝은 여기서 끊는다.
x·y 는 6×6 박스(lx = ly = 6, lz = 1), γ_z 2.6 — 검증 덱과 같은 비율 (cv12 덱 기반, 격자만 교체).

### 3.4 Weak scaling (랭크당 격자 고정)

z-슬랩이라 **z 를 늘리는 방식**으로 랭크당 로컬 블록을 똑같이 유지한다 (TDMA 연구 `run_scaling_compare.sh` 의
weak 정의와 동일: 전역 nz = nz_local × np). 각 랭크의 계산량·halo 면·FFT 크기가 전부 같고, **np 에 따라 변하는
것은 z 분산 TDMA 의 축소계뿐**이라 weak 손실 = 분산 TDMA 통신이라는 해석이 깨끗하다.

| 랭크당 | np | 전역 | 노드 |
|---|---|---|---|
| 256³ | 1, 2, 4, 8 | 256×256×(256·np) | A100 1노드 (16·32 는 취소 — 다노드 없음) |
| 512³ | 1, 2, 4, 8 | 512×512×(512·np) | A100 1노드 (랭크당 ≈ 40 GB) |

lz 는 np 배로 늘려(셀 종횡비 유지) 격자 생성이 실패하지 않게만 한다 — 물리 의미는 없고 성능 런이다.

### 3.5 예상 비용 (제출 전 어림, 실측으로 교체)

cv12 본런 128×128×256(4.2M 셀)이 A100 2장에서 6.9M 스텝/32 h ≈ **17 ms/step**. 셀 수 비례로
256³(16.8M) ≈ 70 ms/step @2GPU, 512³(134M) ≈ 0.6 s/step @2GPU. 200 스텝 × 3 반복이면 가장 큰 조합도 10분 안.
전체 조합 ≈ 20 런 × 3 반복 → 잡 시간 합 2–3시간, 노드 대기열이 지배적.

---

## 4. 벤치마크 프로토콜

### 4.1 덱 (`apps/rbc_nob/scalability/input_scal_<mode>_<격자>_np<N>.ini`, 생성 스크립트 `gen_decks.sh`)

cv12 ΔT 40 덱(`input_rbc_nob_cv12_dt40.ini`)을 기반으로 **격자·npz·lz·출력만** 바꾼다. 물리(Ra, Pr, ΔT 40,
viscous_coupling_number 12, CFL 0.3)는 그대로 — 측정하는 경로가 본런 경로와 같아야 한다.
- `max_step = 200`, `end_time` 은 충분히 크게. 냉시작(IC 는 호스트 루프 + upload 라 루프 밖, 측정 제외).
- `phase_timing_start_step = 50` (워밍업 50 스텝 제외, 150 스텝 누적).
- `monitor_interval = 50` (monitor 구간이 보이게 하되 드물게), `checkpoint_interval = 0`,
  `stats_start_time = 1e9` (통계 누적 안 함).
- dt 값은 비용과 무관(스텝당 일이 같다)하므로 손대지 않는다. sec/step 로만 비교한다.

### 4.2 반복과 집계

- 같은 조합 **3회** (가능하면 다른 노드 세트). 대표값 = `step_total` 의 중앙값 런, 구간값은 그 런의 것.
- 런마다 환경 파일: git 커밋, `module list`, `nvidia-smi`, mpirun 플래그·UCX 환경, SLURM 잡 id·노드 목록
  (TDMA 연구 `environment_<tag>.txt` 형식 재사용).
- 비교는 **같은 바이너리·같은 모듈** 안에서만. 빌드가 바뀌면 전체 재측정.

### 4.3 결과 불변 게이트 (벤치마크 런이 맞는 계산인지)

step 200 의 모니터 줄(θ_c, Nu, Tmin/Tmax, max|div|)이 np 에 따라 **상대 1e-8 안**에서 같아야 한다
(분산 TDMA 축소계의 반올림 차 이상이면 분할 버그). 기준 = np 2 (검증 본런과 같은 분할).
Phase 0 의 스모크(`input_rbc_nob_smoke.ini`, FINAL θ_c 0.01512 / Nu 46.3863 — cv12 설계노트 §3) 도 통과해야 한다.

### 4.4 잡 스크립트 (`apps/rbc_nob/scalability/sbatch_scal_<파티션>_<노드수>.sh`)

TDMA 연구의 `job_scaling_py_h200_gpu8.sh` 양식을 따른다: `module purge; module load nvhpc/25.11_cuda12`,
`--ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=2 --hint=nomultithread`, `--comment="field=mae;appl=in_house"`.
한 잡 안에서 np 루프를 돌려(1→2→4→8) 노드 1개로 1노드 조합을 전부 끝내고, 노드 간 조합은 별도 잡.
`--exclusive`/`--mem=0` 금지(readme 운용 메모)는 그대로.

---

## 5. 병목 순위 매기기

측정이 끝나면 조합마다 구간별 표가 나온다. 순위는 다음 두 수로 매긴다.

1. **점유율** `share_p = T_p / T_total` (np 별). 어디에 시간이 가는가.
2. **확장 손실** — strong: `loss_p(np) = T_p(np) − T_p(1)/np`, weak: `loss_p(np) = T_p(np) − T_p(1)`.
   np 를 늘려도 안 줄어드는(또는 느는) 양. 이상적이면 0.

순위 = 목표 np(1노드 = 8, 다노드 = 32)에서 **loss_p 내림차순**, 동률이면 share_p. 여기에 랭크 불균형
`(max−min)/avg` 를 주석으로 단다. 통신 구간은 §2.2 의 TDMA 내부 항목(mpi vs reduced_compute)으로 한 단계 더 가른다.

순위표 양식 (`docs/Scalability/ranking_<date>.md`):

| 순위 | 구간 | share @np1 | share @np8 | loss @np8 (ms/step) | 불균형 | 분류(통신/동기화/계산) | 가설 |
|---|---|---|---|---|---|---|---|

"가설" 열은 §1 의 후보(분산 TDMA 축소계, 고정 크기 halo 13회, Allreduce 전 D2H 동기화 3회, 모니터 D2H)
중 어느 것인지 적되, **측정이 뒷받침하는 것만** 적는다.

single-GPU(np 1)의 share 도 따로 본다: np 1 에서 이미 큰 구간은 통신이 아니라 커널 자체의 문제라
확장성 작업과 별개의 (나중의) 커널 최적화 목록으로 보낸다.

---

## 6. 순서대로 최적화 — 루프

1. 순위 1위 구간을 고른다. 그 구간이 큰 호출 하나면(§1.1) 먼저 한 단계 쪼개서 다시 잰다 — 원인이 함수 안
   어디인지 구간 타이머로 확정한 뒤에 손댄다.
2. 고친다. 한 번에 **하나의 변경**.
3. 게이트: (a) 스모크 FINAL 불변 + §4.3 결과 불변, (b) `make test`, (c) 1노드 조합(np 1·2·8) 재측정.
4. 기록: 대상별 문서(`momentum_scalability.md` 처럼)에 날짜·변경·전후 표(구간 sec/step, 효율)·커밋. 효과 없으면 되돌리고 그것도 기록.
5. 전체 순위표를 갱신하고 1 로 돌아간다.

종료 기준(제안, 사용자 확정 필요): strong 256³ np 8 병렬 효율 ≥ 70%, weak 256³/랭크 np 32 효율 ≥ 80%.
이 수치에 닿거나, 남은 1위가 "np 1 에서도 큰 계산 구간"이 되면 확장성 단계를 끝내고 커널 단계로 넘어간다.

---

## 7. 단계

| 단계 | 할 일 | 산출물 | 게이트 |
|---|---|---|---|
| **0. 빌드 포팅** | `Makefile.inc` 의 nvhpc 23.7 `/opt/…` 경로를 이 클러스터(`nvhpc/25.11_cuda12`, `/apps/compiler/nvidia_hpc_sdk/25.11`, hpc-x 2.25.1)로; `-gpu=cc80`(A100) + cc90(H200) 선택; CUDA-aware UCX 확인(PaScaL `MpiBufferMode::DeviceDirect` 가 기본이라 필수); `cudaMemGetInfo` 로 §3.2 실측 | 빌드, 스모크 로그 | 스모크 FINAL 일치, `make test` |
| **1. 타이머** | `src/common/phase_timer.hpp` + main.cpp 구간 16개 + `[output] phase_timing_start_step` + TDMA `SolveTimings` 연결 + CSV | 커밋 | 타이머 on/off sec/step 차 ≤ 3%, 구간 합 vs step_total ≤ 2% |
| **2. 기준점** | np 1·2, 256³ (A100) — 구간 표 첫 버전, §3.5 어림 교체 | `timing_*.csv` | §4.3 |
| **3. strong/weak** | §3.3·3.4 전 조합 ×3 반복 (1노드 먼저, 다노드 다음) | CSV + 노트북 `apps/rbc_nob/scalability/plot_scalability.ipynb` (strong 속도향상·효율, weak 효율, 구간 누적막대, 통신 비율 vs np) | 반복 간 중앙값 편차 ≤ 5% |
| **4. 순위표** | §5 | `ranking_<date>.md` | — |
| **5. 최적화 루프** | §6 | 대상별 문서(`momentum_scalability.md`), 커밋들 | §6.3 |

보조(시간이 남으면): H200 1노드 같은 조합(세대 비교), CPU 노드(`cpu01–07`, 48코어) 256³ np 8–64 — GPU 와
통신/계산 비율이 어떻게 다른지.

---

## 8. 미결·리스크

- **512³ np 1** 이 A100-80GB 에 안 들어갈 수 있다 (추정 40 GB + cuFFT 작업공간 + PaScaL 버퍼). 안 들어가면 strong 의
  기준점을 np 2 로 두고 효율을 np 2 대비로 정의한다.
- CUDA-aware MPI 가 이 클러스터의 hpc-x 에서 켜져 있지 않으면 PaScaL `DeviceDirect` 가 실패하거나 느리다.
  Phase 0 에서 `DeviceDirect` vs host-staged 를 둘 다 재서 기록한다 — 이것 자체가 TDMA 구간의 변수다.
- z-슬랩 전용이라 np 상한 = nz. 512³ 이상 또는 np > 64 가 필요해지면 펜슬(2D) 분할이 필요한데, 그건 순위표가
  "halo/TDMA 가 아니라 분할 자체" 를 가리킬 때의 후속 설계 항목이지 지금 범위가 아니다.
- `cudaDeviceSynchronize` 기반 타이머는 커널 간 오버랩을 없앤다. main 단위에서는 영향이 작다고 보지만
  on/off 차(§2.1)로 확인하고, 크면 구간 수를 줄인다.
- 노드 간 조합은 대기열에 묶일 수 있다. 1노드(np ≤ 8) 결과만으로도 순위 1차 판정은 가능하게 단계를 나눴다.

---

## 9. 진행 기록
- **2026-10-17**: Phase 0–3 (1노드 np ≤ 4, H200·V100 4장 + A100 1장) 완료. 전체 스텝 결과·순위 → `main_scalability.md`.
  momentum_solve 를 호출 단위로 쪼갠 측정과 최적화 1회차(z 스윕 조립 타일 전치 커널 `build_kernel_z`, V100 스텝 −27 %, H200 512³ −11 %) → `momentum_scalability.md`.
  측정 상한은 1노드 8 GPU (다노드 잡은 이 큐에서 배정되지 않는다 — 사용자 결정). A100 8 GPU 잡은 배정이 안 돼 포기 (10-17) — 측정은 H200·V100 4장과 A100 1장 결과로 닫는다.
  덱·잡·집계 스크립트 `apps/rbc_nob/scalability/`, 노트북 `apps/rbc_nob/plot/{main,momentum,momentum_opt}_scalability.ipynb`.
  런 산출물(`scalability/out`, `scalability/log`)과 그림은 커밋하지 않는다.
- **2026-10-17 (밤)**: 최적화 2회차 = 순환 Thomas 커널 레지스터 두 행 버퍼링(논문 2.1 방식). 게이트 통과. 누적 H200 256³ −10 %, 512³ −14 %, V100 −29 %.
  `momentum_scalability.md` §5.
- **2026-10-17 (밤 2)**: 최적화 3회차 = 라인이 적을 때 조각 분할 순환 Thomas(블록 안 Thomas–축소계, p 자동). H200 256³ np 4 누적 −30 %, 효율 61 → 75 %.
  V100 의 np ≥ 2 step 비교는 노드 배정이 매번 달라 MPI 항이 흔들려 무효 (계산 항은 개선). `momentum_scalability.md` §6.
- **2026-10-17 (밤 3)**: 3회차(조각 분할)는 본런 격자에서 2~3 % 라 **본 코드에서 제외**, `apps/rbc_nob/scalability/backup/` 에 백업·기록만. 본 코드는 2회차까지
  (build_kernel_z + 순환 Thomas 레지스터화 = 논문 커널과 통일). 다음 = scalar z 조립·scatter_z 전치, 커플링 커널.
- **2026-10-17 (밤 4)**: scalar_solve 를 호출 단위로 분해(`scalar::Solver::timer`) → build_z 가 같은 전치 문제(V100 에서 scalar 의 55 %). 같은 타일 전치 커널 적용, 게이트 통과.
  V100 256³ 스텝 −15 %, H200 512³ −7 %. 누적(원본 대비) H200 512³ −20 %, V100 256³ −40 %. gpu04 8장이 비어 np 8 측정: 256³ 효율 41 %(통신 — 소켓 간 링크 3↔4),
  512³ 76 %. 문서 `scalar_scalability.md` (문서 4개째 — scalar 노트북과 짝).
- **2026-10-17 (밤 5)**: 노트북은 `apps/rbc_nob/plot/{main,momentum,scalar}_scalability.ipynb` 세 개만 (momentum_opt 삭제, PNG 저장 끔). 세 노트북 모두 같은 틀
  (이벤트별 막대 → 이벤트별 scaling → 효율), 제목·범례는 환경 파일에서 읽은 실제 기종(모델·장 수·노드·꼬리표) + 문제 크기. 기존 결과만 사용.
- **2026-10-17 (밤 6)**: V100 gpu04 np 8 붕괴 원인 확정 — 하드웨어·UCX 아님. 주기 축 npx=npy=1 에서 이웃 = 자기 자신이라 교환마다 x·y 면 4장을 자기에게 MPI_Sendrecv
  (UCX 호스트 경유, np² 로 악화). `subdomain_halo.cu` 에서 자기 이웃은 디바이스 복사로 → weak np 8 효율 57 → 86 %, strong np 8 36 → 47 %. 진단 도구 `apps/bench_halo`,
  `apps/rbc_nob/scalability/diag/`. H200 재측정(h200h) 진행. `scalar_scalability.md` §4.
- **2026-10-17 (밤 7)**: 세 노트북을 "최적화 전 / 후 / 전·후 비교(이벤트별 감소량)" 양식으로 통일. 최종 바이너리로 H200(gpu55 4장) 재측정(`h200h`), V100 gpu04 8장
  256³ np 1–8·512³ np 2–8(`v100h`), H200 weak 512³/랭크 np 1–4 와 8장 np 8 잡 진행/대기. 전 → 후: H200 −15~21 %, V100 −32~41 % (`main_scalability.md` §2c).
- **2026-10-17 (밤 8)**: momentum_rhs 최적화 — 격자 역수(나눗셈 58회/셀 제거), 전용 RHS 빌드 커널(8 → 3 패스), j coarsening(실험 후 끔). V100 momentum_rhs ×2.5, 스텝 −6~9 %.
  기여 분리: 역수 50 %, 융합 20 %, coarsening 손해. H200 잡은 전부 취소(사용자: 배정 불가) — 이후 측정은 V100 gpu04 8장만. `main_scalability.md` §2d.
- **2026-10-17 (밤 9)**: 격자 역수를 나머지 스텐실로 확장 — momentum 조립(`sweep_band`·`own_face_blend`)·커플링 4개·scalar `axis_band`. 게이트 허용오차 통과. V100 gpu04(`v100s`):
  스텝 −5~−8 %(256³ np 1 108.3 → 100.1), 커플링 −30~−40 %, scalar build_x −29 %, 조립 −10~−16 %. 누적 원본 대비 256³ np 1 **198 → 100 ms**. `main_scalability.md` §2e.
  다음 후보: pressure_solve 분해(np 1 의 13 %), np 8 의 PaScaL Alltoallv.
- **2026-10-18**: 패스 제거 회차 — scatter_z 타일 전치, 밴드 핑퐁(scatter_x/y 제거, `band_layout.hpp`), scalar T += 융합·momentum W += 융합. 비트 동일. V100 256³ np 1 100.1 → 93.2 ms(−7 %),
  momentum −10 %, scalar −13 %. 꼬리표 `v100p`. 누적 원본 대비 **198 → 93 ms (×2.1)**. 설계 제약: 나중에 3차원 분할이 들어오므로 z-슬랩 전제 최적화(조립+풀이 융합)는 제외.
  pressure_solve 단계별 타이머(`pressure::Solver::timer`) 추가, np 8 Alltoallv 경로 측정(순수 MPI 벤치 `diag/alltoall_bw.cu` + `PASCAL_TDMA_MPI_MODE=host` 런 `v100hst`) 진행.
  4번 결과: np 8 Alltoallv 1.9 GB/s 는 소켓 사이 링크를 16쌍이 나눠 쓰는 토폴로지 한계(순수 MPI 벤치 = 솔버 값, UCX 설정 무효, host 모드는 솔버에서 2배 느림) → 코드로는 안 줄고 3차원 분할이 해결. `main_scalability.md` §2g.
  pressure 분해(`v100q`): np 1 의 13.4 ms 중 reorder_x_to_y 4.3 + reorder_y_to_x 1.3 + pack/scatter_lines 1.8 = 전치 복사가 55 %, FFT 는 10 %. 다음 = 네 커널 타일 전치. `main_scalability.md` §2h.

## 10. 최적화 코드 위치 (주석 표식 `[최적화 …]` 와 짝)

| 회차 | 무엇 | 파일 (주석 위치) |
|---|---|---|
| 1 (10-17) | momentum z 조립 32×32 타일 전치 `build_kernel_z` | `src/equation/momentum/system/system.cu` |
| 2 | 순환 Thomas 레지스터 두 행 버퍼링 `tdma_cycl_many_kernel` | `src/linear_solver/tdma/PaScaL_TDMA/pascal_tdma_cuda.cu` |
| 4 | scalar z 조립 타일 전치 `build_z_kernel` | `src/equation/scalar/system/system.cu` |
| 6 | halo 자기 이웃 디바이스 복사 | `src/parallel/subdomain/subdomain_halo.cu` |
| 8 | 격자 역수 배열 `inverse_*` | `src/grid/grid.{hpp,cpp}` |
| 8 | momentum RHS 역수(`rhs_*_cell`, `face_blend_inverse`) + 전용 융합 빌드 커널 `build_rhs_kernel` (kCoarsenJ=1) | `src/equation/momentum/momentum_stencil.hpp`, `rhs/rhs.cu`, `apps/rbc_nob/main.cpp` |
| 9 | 조립 `sweep_band`·`own_face_blend`, 커플링 `couple_*_cell`, scalar `axis_band` 역수 | `momentum_stencil.hpp`, `scalar_stencil.hpp`, 두 `system.{cu,cpp}`, `momentum/solve_kernels.{cu,cpp}` |
| 10 (10-18) | scatter_z 타일 전치 + 선택적 `q += δ` | `src/equation/sweep_kernels.{hpp,cu,cpp}` |
| 10 | 밴드 핑퐁(`SourceLayout`, scatter_x/y 제거) | `src/equation/band_layout.hpp`, momentum·scalar `system.{cu,cpp}`·`solve.cpp` |
| 11 | pressure 전치 복사 4개 타일화 | `src/equation/pressure/solve_kernels.cu` |
| 계측 | PhaseTimer·단계 타이머(수치 불변) | `src/common/phase_timer.hpp`, momentum·scalar·pressure `solve.{hpp,cpp}` |
- **2026-10-18 (2)**: pressure 전치 복사 4개 타일화(`pressure/solve_kernels.cu`). 비트 동일. V100 pressure_solve 256³ np 1 13.4 → 7.5 (−44 %), 512³ np 2 85.5 → 38.8 (−55 %);
  스텝 −7 % / −9 %. 꼬리표 `v100pt`. 누적 원본 대비 256³ np 1 **198 → 87.5 ms (×2.26)**. `main_scalability.md` §2i. 코드의 최적화 지점은 전부 `[최적화 날짜]` 주석 + plan §10 표.
