# RBC GPU 스텝 타이밍 계획 — main_gpu_nob.cu 시간 루프

대상: `apps/rbc/main_gpu_nob.cu`의 시간 루프(단계 [1]~[10]).
기준 스펙: `~/workspace/KISTI_6_test/mpi_gpu_timing_spec.md` (이하 "스펙").
목표: 많은 GPU(수백~2048랭크급)에서 strong/weak scaling 시간 측정 — 스텝당
event별·iteration별·rank별 시간을 **측정 루프에 아무것도 삽입하지 않는 방식**으로
수집하고, 종료 후 일괄 reduce → rank 0 단일 CSV.

참고 구현: `~/workspace/KISTI_6_test/Scalability/timing.{hpp,cu}` —
이미 스펙을 구현한 범용(MPI+CUDA만 의존) 레코더. **그대로 이식해서 재사용한다.**

---

## 1. 스텝 해부 — 무엇이 어디서 일어나는가

프로덕션 분해는 `np = 1 x 1 x N`(z-슬랩)이다. 이 전제에서 스텝당 일어나는 일:

| 단계 | 내용 | 실행 위치 | MPI | 자연 동기화 지점 |
|---|---|---|---|---|
| [1] CFL dt | 디바이스 리덕션 → D2H 8B → **`MPI_Allreduce(MAX)`** (`cfl_gpu.cu`) | GPU+host | **매 스텝 전역 collective 1회** | ✅ dt가 host로 와야 하므로 sync 내장 |
| [2] 물성(Tⁿ) | 전장 다항식 커널 | GPU | — | — |
| [3] 에너지 | 3-sweep ADI(x·y 로컬 TDMA, **z 분산 TDMA**) + `sync_field(T)` 할로 | GPU+host | z-TDMA comm 1 + 할로 1 | ✅ MPI 전 스트림 동기화 |
| [4] 물성(Tⁿ⁺¹) | [2]와 동일 커널 재실행 | GPU | — | — |
| [5] 운동량 | assemble+부력+∇P → 3-sweep ADI×3속도(블록커플링) → update → `sync_field`×3 | GPU+host | z-TDMA comm 3 + 할로 3 | ✅ 각 MPI 전 |
| [6] 벽 W=0 | 작은 커널 + `sync_field(W)` | GPU+host | 할로 1 | ✅ |
| [7] 압력 | div RHS → **FFT x·y(로컬, fast-path 디바이스 상주)** → **z 분산 TDMA** → 역FFT → 투영 → P 누적 | GPU+host | z-TDMA comm 1 | ✅ TDMA 경계 |
| [8] 클리핑 | 작은 커널 (DeltaT>0) | GPU | — | — |
| [9a,9b] 통계 | 디바이스 리덕션 + D2H ≤48B (+nmonitor마다 출력/CSV) | GPU+host | 통계 Allreduce | ✅ |
| [9c,9d] 파일 IO | stats 파일·체크포인트 | host | gather | (측정창에서는 발생 금지) |

**스텝당 MPI 인벤토리** (np=1×1×N, 측정창 기준): 전역 `Allreduce` 1([1]) +
z-이웃 할로 5(sync_field: T,U,V,W,W) + 분산 z-TDMA 5(스칼라1·운동량3·압력1) +
통계 리덕션. **이 목록이 곧 스케일링 병목 후보이며, 측정이 분리해서 보여줘야 할 대상이다.**

---

## 2. 현재 계측의 문제 (스펙 위반 목록) — 교체 대상

현 코드의 `t_scalar/t_mom/…` + `[TIMING]`/`[TIMING_SS]` 방식:

1. **phase마다 `cudaDeviceSynchronize()` 7회 삽입** — 스펙 §2.8/§13 금지.
   host가 스트림을 따라잡을 때까지 매번 세워서, 계측이 곧 실행 흐름을 바꾼다.
   (7회 전부 계측 목적이며 알고리즘상 필요 없음 → 제거 대상.)
2. **누적합만 저장** — iteration별 데이터가 없어 변동성·스텝별 분석 불가(스펙 §3).
3. **rank 0 로컬 시간만 출력** — rank 방향 max/avg/imbalance 없음(§7, §13 금지
   "rank 0의 로컬 시간만 전체 실행시간으로 사용").
4. **합 ≠ 전체**: [1] CFL, [2]/[4] 물성, [9c/9d] IO가 어느 타이머에도 안 들어가
   `total`이 스텝 시간이 아니다. measurement window(§5,9)도 없다.
5. reduce/CSV 없음 — printf뿐(§8, §11).

---

## 3. 제안 — 3계층 측정 설계 (Scalability 하네스와 동일 구조)

### 계층 0 — measurement window (공식 스케일링 시간, §5·6.3·9)

```
warm-up n_warmup 스텝 (계측 완전 off, 동일 코드경로)
cudaDeviceSynchronize + MPI_Barrier 1회 → window_start = MPI_Wtime()
  … n_measured 스텝 (아래 mark 타임라인만) …
cudaDeviceSynchronize → window_local = MPI_Wtime() - window_start
→ 종료 후 MPI_MAX = T_run(P)   ← strong/weak scaling의 공식 값
```

### 계층 1 — 분할 CUDA-event 타임라인 (기본, 스케일 런 표준)

mark = `cudaEventRecord`(비동기, ~µs). phase 경계마다 1개, **스텝당 15개 고정**.
sync 없음 — MPI에서 기다린 시간은 그 MPI를 포함한 event에 자동 귀속.
`total_iteration` = 분할의 합(망원합 = 처음→마지막 mark 직접 측정, rank-로컬로
계산 후 reduce — §4의 "max들의 합" 금지 준수).
단일 소형 커널(부력·∇P·update·P누적)도 mark 비용이 ~0이고 CUDA-event 해상도
(~0.5µs)로 충분히 분해되므로 따로 재는 데 손해 없음.

| # | mark 지점 (루프 내 위치) | event | 비고 |
|---|---|---|---|
| 1 | [1] CFL 직후 | `cfl_dt` | 리덕션+D2H+**전역 Allreduce** — 스케일 핵심이라 단독 event |
| 2 | [2] 물성(Tⁿ) 직후 | `properties` | |
| 3 | [3] 스칼라 solve+sync_field(T) 직후 | `scalar` | 3-sweep ADI, z-TDMA comm 1 + 할로 1 |
| 4 | [4] 물성(Tⁿ⁺¹) 직후 | `properties` (누적 2회) | |
| 5 | [5] assemble 직후 | `mom_rhs` | 대류+변수-μ 확산 명시 RHS |
| 6 | [5] 부력 직후 | `mom_buoyancy` | 단일 커널 |
| 7 | [5] ∇P 직후 | `mom_pgrad` | 단일 커널 |
| 8 | [5] solve 직후 | `mom_solve` | 3-sweep×3속도+블록커플링, z-TDMA comm 3 |
| 9 | [5] update_velocity 직후 | `mom_update` | |
| 10 | [5]~[6] sync_field(U,V,W) + zero_w + sync_field(W) 직후 | `vel_boundary` | **할로 4회** 집중 구간 |
| 11 | [7] 압력 solve 직후 | `poisson_solve` | div→FFT(x,y)→z-TDMA→역FFT→투영 |
| 12 | [7] `accumulate_pressure_gpu` 직후 | `pressure_update` | P_acc += δp |
| 13 | [8] 클리핑 직후 | `clip` | DeltaT>0에서만 (OB면 ~0) |
| 14 | [9a] 통계 누적 직후 | `statistics` | 디바이스 리덕션 + D2H ≤48B |
| 15 | [9b] monitor_if_due 직후 | `statistics` (누적 2회) | 측정창에선 no-op이어야 함(§5 정책) |

→ 파티션 event 13종: `cfl_dt, properties, scalar, mom_rhs, mom_buoyancy,
mom_pgrad, mom_solve, mom_update, vel_boundary, poisson_solve, pressure_update,
clip, statistics` (+파생 `total_iteration`). 시간 루프의 물리 단계 12개
(dt→물성→에너지→물성→운동량 rhs/부력/∇P/solve/update→속도경계→poisson solve→
P갱신)를 그대로 event로 쓰고, 루프에 실재하는 클리핑·통계를 더해 스텝을 빈틈없이
분할한 것. 스펙 §4 "이름은 코드 구조에 맞게 변경 가능".

### 계층 2 — nested host event (타임라인이 못 보는 host/MPI 시간, §6.1·§4 중첩규칙)

알고리즘이 **이미 동기화한 지점**에서 `MPI_Wtime` 차이만 out-param으로 받는다
(Scalability의 `halo_comm`/`SolveCost` 패턴 — nullptr이면 완전 무비용):

| nested event | 출처 | 필요한 API 확장 |
|---|---|---|
| `communication` | 할로 5회의 MPI_Sendrecv 구간(`halo_gpu.cu`) + z-TDMA 내부 MPI 5회(`PaScaLTDMACudaBackend`) | `sync_field_gpu`/TDMA backend에 `double* comm_seconds` out-param |
| `reduction` | [1] CFL `MPI_Allreduce` + [9a/9b] 통계 collective | 같은 패턴 |

이 두 개가 파티션 event와 **중첩**된다는 사실은 CSV 메타(`event_semantics`)에 명시.

### 계층 3(세부) — 어디까지 들어갈 것인가

| 깊이 | 대상 | 방법 | API 변경 | 판단 |
|---|---|---|---|---|
| L1 | 위 15-mark(13 event) — 운동량 5분할·poisson 2분할 포함 | main만 수정 | 없음 | ✅ **스케일 런 기본** |
| L2b | nested comm/reduction | out-param | 소규모(할로·TDMA·CFL) | ✅ 기본 포함 — 스케일링 논문의 핵심 열 |
| L2c | 압력 내부 분해: div_rhs/FFT_fwd/**tdma_z**/FFT_bwd/project | `PressureSolverGpu`에 cost 구조체 | 중간 | ⚪ **진단 모드(옵션)** — 기존 `[FWD #]` hrc+sync 계측은 스펙 위반 방식이라 이걸로 대체 |
| L2d | 스칼라/운동량 sweep별(x/y/z) | scalar/momentum solve에 cost | 중간 | ⚪ 옵션 — z-sweep만 comm을 갖so L2b의 TDMA comm으로 대부분 커버됨 |
| L3 | 커널 단위 | CUDA event 상세/replay | 침습 큼 | ❌ 스케일 런에서 안 함 — 일회성 분석은 nsys/ncu (기존 Report/21 방식) |

**권장 기본 = L1(15 mark, 13 event) + L2b(comm/reduction out-param)**.
L2c/L2d는 `timing_detailed`류 플래그로 켜는 진단 모드로만.

---

## 4. 측정 런 정책 (측정창 오염 방지)

- **warm-start 권장**: 발달된 필드(기존 restart)에서 시작해야 dt·통신량이 정상상태
  대표값. cold 초기 구간은 비대표적.
- warm-up 20 / measured 100 스텝 권장(§5).
- 측정창 안에서 다음이 **발생하지 않도록 덱 설정**: `nmonitor` > 측정스텝수,
  `t_stat_start = 1e18`, `t_stat_out = 1e18`, `nrestart_write = 0`,
  `MPMSTD_CFLDBG`/`MPMSTD_MOMDBG` 해제. (§13: 루프 내 파일 IO·표준출력 금지)
- 적응 dt는 문제 없음 — 스텝당 작업량(커널·통신)은 dt와 무관하게 동일.
- 비침습 검증(§15): window 직후 계측 off 20스텝 레퍼런스 →
  `instrumentation_overhead` 자동 산출·1% 목표 판정 (Scalability `timing_check` 그대로).

## 5. 출력 (IO) — 스펙 §8·§11 그대로

- 측정 루프 안: **아무것도 없음** (mark 기록뿐).
- 종료 후: `MPI_Reduce` MAX+SUM(+detailed 시 MIN/SUMSQ) 일괄 1회 + window MAX,
  reduce 소요는 `timing_reduction_overhead` 메타로만.
- rank 0 단일 CSV: `timing_{scaling}_{gx}x{gy}x{gz}_{ngpu}gpu_rbc_run{id}.csv`
  — 행=iteration, 열=event별 `avg,max`(+detailed: `min,std,imbalance`),
  머리에 `# key=value` 메타(그리드/분해/Ra/Pr/DeltaT/커밋/CUDA/MPI/window_max/…).
- 2048랭크 안전성: reduce 데이터 = 100 iter × 16 event(파티션13+nested2+total)
  ≈ 12.8 KB/rank, CUDA event 풀 1,600개, CSV 1개 — 전부 무시할 수준.
  per-rank 원본 gather(§8.2)는 기본 off.

## 6. 커널 단위 세분화 (v2 — 4개 다중커널 함수 내부)

`solve_scalar / solve_momentum / solve_pressure`(요청) + **`assemble_momentum`**
(재확인 결과 추가: `k_mom_rhs_var`×3, momentum이 스텝의 최대 비중)의 내부를
같은 mark-타임라인으로 세분화한다. 함수 시그니처에 `perf::Recorder* rec = nullptr`
를 추가(기본 nullptr → 기존 호출·수치 완전 불변)하고, 내부에서 `if (rec) rec->mark(…)`.
단일 커널 함수(물성·부력·∇P·update·clip·zero_w·CFL·통계)는 top-level 이벤트 유지.

집계 원칙: **방향(x/y/z)은 구분**(z만 분산 TDMA/MPI를 가짐), **성분(U/V/W)·반복은
같은 event에 누적** — event 수를 다스리면서 통신 있는 축은 분리 유지.

| 함수 | sub-event | 커널/구간 |
|---|---|---|
| scalar | `sc_build_{x,y,z}` | `k_build_A_rhs_nob` (sweep별) |
| | `sc_rowmod_z` | `k_modify_tdma_row_gpu` (z 벽 클로저; x/y 주기라 z만) |
| | `sc_tdma_{x,y,z}` | TDMA 백엔드 solve (**z = 분산·MPI 포함**) |
| | `sc_scatter_{x,y,z}` | `k_scatter_from_tdma` |
| | `sc_update` | `k_add_increment` (T += δT) |
| assemble | `ma_rhs_{u,v,w}` | `k_mom_rhs_var` ×3 |
| mom solve | `ms_ddcouple` | `k_dd_dV_var`+`k_dd_dW_var`+`k_zero_dw_wall_face` (누적) |
| | `ms_build_{x,y,z}` | `k_mom_adi_var` (성분×3 누적) |
| | `ms_rowmod` | `k_modify_tdma_row`+`k_w_wallface_identity_row` (누적) |
| | `ms_tdma_{x,y,z}` | TDMA solve (성분×3 누적; **z=MPI**) |
| | `ms_scatter_{x,y,z}` | `k_m_scatter` (성분×3 누적) |
| | `ms_copyout` | `k_m_copy_field` (성분×3 누적) |
| | `ms_halo` | **solve 내부 `sync_field(dU/dV/dW)` 할로 3회** (누적) |
| | `ms_blockcouple` | `k_block_dV_var`+`k_block_dU_var`(+stage copy) |
| pressure | `pr_div`, `pr_extra` | `k_divergence_rhs_pres`, `k_extra_term_nob_pres` |
| | `pr_fftx_f`, `pr_ffty_f` | cuFFT D2Z / Z2Z (forward) |
| | `pr_tdma_split`, `pr_tdma_z`, `pr_tdma_merge` | `k_tdma_split_hat_Y` / **분산 z-TDMA(MPI)** / `k_tdma_merge_hat_Y` |
| | `pr_ffty_b`, `pr_fftx_b`, `pr_scale` | 역FFT + `k_gpu_scale` |
| | `pr_unpack`, `pr_project` | `k_unpack_dp_C_to_P`, `k_project_nob_pres` |
| | `pr_dphat_ext`, `pr_dphat_halo` | `k_dphat_extrapolate`, **dPhat 할로(MPI)** |

- 세분화 후 top-level의 `scalar/mom_rhs/mom_solve/poisson_solve` 이벤트는 sub-event
  합으로 **파생**된다(직접 mark하지 않음 — 이중계상 금지). 나머지 top-level
  이벤트(§3의 cfl_dt·properties·부력·∇P·update·vel_boundary·pressure_update·
  clip·statistics)는 그대로 mark.
- 규모: event ≈ 50, mark ≈ 75~80/스텝 — CUDA event 풀 100 iter × ~80 ≈ 8,000개,
  reduce 데이터 ≈ 42 KB/rank. 여전히 2048랭크에서 무시할 수준. mark는 비동기
  record라 비침습성 원칙 불변(§2.8), 오버헤드는 timing_check가 매 런 검증.
- 기존 엔진 내부 계측(`[NOB_PRES #]`, `[FWD #]` — hrc+sync 방식)은 이것으로 대체.
- nested `communication`/`reduction` out-param(§3 L2b)은 다음 단계로 분리
  (타임라인이 이미 MPI 포함 구간을 `*_tdma_z`/`*_halo`로 격리하므로 우선순위 낮음).

## 7. 구현 순서

1. `KISTI_6_test/Scalability/timing.{hpp,cu}` → `src/post/timing/`으로 이식
   (MPI+CUDA에만 의존, heat 코드 비종속 — 수정 없이 컴파일 가능).
2. `main_gpu_nob.cu`: 기존 `t_*` 누적/`[TIMING]` printf/**계측용
   `cudaDeviceSynchronize()` 7회 제거**, 15-mark 배선(warm-up은 recorder=nullptr).
3. out-param 추가: `sync_field_gpu`(할로 comm), `PaScaLTDMACudaBackend::solve`
   (TDMA comm), `compute_cfl_dt_gpu`(Allreduce) — 모두 nullptr 기본값으로
   기존 호출부 불변.
4. 덱 키: `[timing] dir / warmup / measure / detailed / check`.
5. 검증: ① 계측 off 경로 byte-동일(validate/스모크 재현) ② timing_check 오버헤드
   ≤1% ③ 작은 랭크에서 인공 시간으로 max/avg/imbalance 수작업 대조(§15).
