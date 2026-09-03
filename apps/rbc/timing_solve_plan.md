# solve 함수 커널 단위 시간 측정 — 분해 명세

`timing_plan.md`의 계층 3(L2c/L2d)을 실제 구현으로 확정한 문서.
어떤 함수를 어떻게 분해해서 어떤 구간의 시간을 재는지 정의한다.

## 측정 방식 (공통)

- 각 함수 시그니처에 `post::TimingRecorder* recorder = nullptr, int timing_event_base = 0`
  추가 — **nullptr(기본)이면 mark가 전혀 없고 실행 경로·수치가 완전 동일**.
- recorder가 있으면 스테이지 경계마다 `recorder->mark(base + 이벤트)` —
  비동기 `cudaEventRecord` 1회(~µs), sync 삽입 없음. 구간 시간은 window 종료 후
  CUDA event 타임라인에서 일괄 회수(`src/post/timing.hpp` 참조, KISTI 스펙 준수).
- 이벤트 블록(로컬 enum + 이름 + mark 수)은 각 함수의 공개 헤더에 정의되어 있고,
  main이 블록들을 이어 붙여 recorder의 이벤트 목록을 구성한 뒤 각 함수에
  `base` 오프셋을 넘긴다.
- 방향(x/y/z)은 **구분**한다 — `np = 1×1×N` 분해에서 **z만 분산 TDMA(MPI)** 를
  가지므로. 성분(U/V/W)·sweep 반복은 같은 이벤트에 **누적**한다.
- **Phase 1 정책**: 함수 내부에 이미 있던 `cudaDeviceSynchronize()`는 그대로 두고
  (실행 거동 불변 → 검증 md5 게이트 자명), 기존 hrc+printf 계측(`[SCA #]` 등)만
  mark로 교체. sync 제거는 Phase 2에서 md5 게이트로 검증하며 별도 진행.

## 1. `solve_scalar_variable_property_gpu` — 11 이벤트, 11 mark/호출

헤더: `src/equation/scalar/solve.hpp` (`GpuTimingEvent`)

| 이벤트 | 구간(커널) | 비고 |
|---|---|---|
| `scalar_build_{x,y,z}` | `k_build_A_rhs_nob` | sweep별 융합 밴드+RHS 조립 (stage-1은 명시잔차 R 인라인) |
| `scalar_wall_rows` | `k_modify_tdma_row_gpu` | 벽 방향(z)만 — BC를 행렬에 fold |
| `scalar_tdma_{x,y,z}` | PaScaL cuda TDMA `solve_many_{cyclic_,}device` | x,y=순환(로컬) / **z=분산·CUDA-aware MPI** |
| `scalar_scatter_{x,y,z}` | `k_scatter_from_tdma` | 밴드→3D 필드 산란 (핑퐁) |
| `scalar_update` | `k_add_increment` | Tⁿ⁺¹ = Tⁿ + δθ |

기존 `[SCA #]` hrc 계측은 제거(대체)됨.

## 2. `assemble_momentum_variable_property_gpu` — 3 이벤트, 3 mark/호출

헤더: `src/equation/momentum/assemble.hpp` (`GpuAssembleTimingEvent`)

| 이벤트 | 구간(커널) |
|---|---|
| `momentum_rhs_{u,v,w}` | `k_mom_rhs_var` — 성분별 명시 RHS(대류+변수-μ 확산+교차응력) |

## 3. `solve_momentum_variable_property_gpu` — 14 이벤트, 41 mark/호출

헤더: `src/equation/momentum/solve.hpp` (`GpuSolveTimingEvent`)
구조: 성분 U→V→W 순서로 각각 [교차커플링 → 3-sweep ADI → δ 할로], 마지막에 블록커플링.

| 이벤트 | 구간(커널) | 비고 |
|---|---|---|
| `momentum_cross_couple` | `k_dd_dV_var`, `k_dd_dW_var`(+`k_zero_dw_wall_face`) | M21/M31+M32 — 앞 성분 δ의 명시 커플링 (V,W에서 1회씩 누적) |
| `momentum_build_{x,y,z}` | `k_mom_adi_var` | 성분×3 누적 |
| `momentum_wall_rows` | `k_modify_tdma_row_gpu`(+W의 `k_w_wallface_identity_row`) | 벽 방향(z)만 |
| `momentum_tdma_{x,y,z}` | PaScaL cuda TDMA | 성분×3 누적; **z=MPI** — 스텝 내 z-TDMA 3회 |
| `momentum_scatter_{x,y,z}` | `k_m_scatter` | 성분×3 누적 |
| `momentum_copy_out` | `k_m_copy_field` | δ를 출력 버퍼로 (성분×3 누적) |
| `momentum_halo` | `core::sync_field_gpu(dU/dV/dW/dV)` | **solve 내부 할로 4회** — 각 성분 ADI 직후 3회 + 블록커플링 dV 갱신 직후 1회 |
| `momentum_block_couple` | `k_block_dV_var`, `k_block_dU_var` | 증분 교차대류의 블록 하삼각 되풀이 |

하단랭크 전용 `k_zero_dw_wall_face`와 환경변수 게이트 debug 커널은 **mark 없이**
인접 구간에 흡수된다 — rank/환경 조건부 mark는 고정 패턴을 깨기 때문.

## 4. `solve_pressure_variable_property_gpu` — 13 이벤트, 13 mark/호출

헤더: `src/equation/pressure/solve.hpp` (`GpuPressureTimingEvent`)
`PressureSolverGpu::solve_nob`의 실행 순서 그대로 (fast path: np1=np2=1, GPU 상주 z-TDMA).

| 이벤트 | 구간(커널) | 비고 |
|---|---|---|
| `pressure_divergence` | `k_divergence_rhs_pres` | ∇·U* → Poisson RHS |
| `pressure_extra_term` | `k_extra_term_nob_pres` | NOB ∇·[(1−1/ρ̂)∇P̂] |
| `pressure_fft_x_forward` | cuFFT `D2Z` | 주기 x, r2c |
| `pressure_fft_y_forward` | cuFFT `Z2Z` forward | 주기 y, c2c |
| `pressure_tdma_split` | `k_tdma_split_hat_Y` | 복소→실수 2계(Re/Im) 분리 |
| `pressure_tdma_z` | 분산 z-TDMA | **MPI** — 벽수직 삼중대각 |
| `pressure_tdma_merge` | `k_tdma_merge_hat_Y` | 실수 2계→복소 재조합 |
| `pressure_fft_y_backward` | cuFFT `Z2Z` inverse | |
| `pressure_fft_x_backward` | cuFFT `Z2D` + `k_gpu_scale` | 역변환+정규화(1/N₁N₂) |
| `pressure_unpack` | `k_unpack_dp_C_to_P` + `sync_field(dP)` | δp 배치 + **dP 할로(MPI)** |
| `pressure_project` | `k_project_nob_pres` | 속도 발산-제로 투영 |
| `pressure_dphat_extrapolate` | `k_dphat_extrapolate` | 다음 스텝 P̂ 외삽 |
| `pressure_halo` | dPhat + U/V/W 할로 교환 ×4 | **MPI** — dPhat(다음 스텝 ExtraTerm용) + 투영된 속도 3성분 |

기존 `[NOB_PRES #]`/`[FWD #]` hrc print는 Phase 1에서는 그대로 두었다(처음
5~10 호출 = warmup 중에만 출력되므로 측정창 무해; Phase 2 sync 제거 때 함께
정리). fast path가 아닌 경로(np1·np2>1)에서는 추가 전치/H↔D 복사가 인접
구간에 흡수됨(메타에 명시).

## 5. 메인 루프 top-level 이벤트 — 10종, 12 mark/스텝

solve 내부로 들어가지 않는 나머지 단계 (main_gpu_nob.cu에서 직접 mark):

| 이벤트 | 구간 |
|---|---|
| `cfl_dt` | CFL 리덕션+D2H+**전역 Allreduce** |
| `properties` (×2) | 물성 다항식 갱신 (Tⁿ, Tⁿ⁺¹) |
| `scalar_halo` | 에너지 solve 직후 `sync_field(T)` |
| `momentum_buoyancy` | 부력 커널 |
| `momentum_pressure_gradient` | −dt·(1/ρ̂)∇P_acc 커널 |
| `momentum_velocity_update` | `update_velocity_gpu` |
| `velocity_boundary` | `sync_field(U,V,W)` + `k_zero_w_wall` + `sync_field(W)` — 할로 4회 |
| `pressure_update` | `accumulate_pressure_gpu` (P_acc += δp) |
| `temperature_clip` | `k_clip_T_gpu` (DeltaT>0) |
| `statistics` (×2) | 통계 누적 + monitor(측정창에선 no-op) |

## 합계

| | 이벤트 | mark/스텝 |
|---|---|---|
| top-level | 10 | 12 |
| scalar | 11 | 11 |
| momentum assemble | 3 | 3 |
| momentum solve | 14 | 41 |
| pressure | 13 | 13 |
| **파티션 합** | **51** (+파생 `total_iteration`) | **80** |

CUDA event 풀 = 측정스텝 × 80 (100스텝 기준 8,000개), reduce 데이터 ≈ 42 KB/rank,
CSV 열 ≈ 106(기본 avg/max) — 2048랭크에서도 전부 무시할 수준.

## 스케일링 해석 가이드 (무엇을 보면 되나)

- **z-분산 TDMA 총비용** = `scalar_tdma_z + momentum_tdma_z + pressure_tdma_z`
  (스텝당 분산 TDMA 5회 중 z만 MPI) — 랭크 수 증가의 1차 민감 항목
- **할로 총비용** = `scalar_halo + momentum_halo + velocity_boundary + pressure_unpack(일부) + pressure_halo`
  (스텝당 할로 14회: T 1 + momentum 내부 δ 4 + 속도경계 4 + dP 1 + dPhat·U/V/W 4)
- **전역 collective** = `cfl_dt`(매 스텝 Allreduce) + `statistics`
- 순수 로컬 커널(`*_build_*`, `*_scatter_*`, FFT 등)은 랭크 수와 무관해야 정상 —
  이게 흔들리면 GPU 모델 혼합/노이즈를 의심

## 검증 결과 (2026-09-03, job 33388 — 32×32×64, np 1 1 2)

- ✅ 계측 **off**: U/V/W/T restart 덤프 md5가 리팩터 전 베이스라인
  (`_fft_verify/BASELINE_md5.txt`)과 **byte-동일**
- ✅ 계측 **on**(warmup=5, measure=10): md5 **여전히 byte-동일** — mark는 순수
  기록, 수치·실행 경로 불변
- ✅ 80 mark/step 고정 패턴 통과, 52 events × 10 iter CSV 생성,
  `timing_reduction_overhead` 65 µs
- 첫 측정 (6.7 ms/step, 2 GPU): `tdma_z_all` 2.80 ms(≈42%),
  `halo_all` 1.54 ms — momentum_tdma_z 1.30 / pressure_tdma_z 0.83 /
  scalar_tdma_z 0.44 ms

사용법: 덱에 `[timing]` 섹션 — `warmup`(기본 20) / `measure`(기본 0=off) /
`detailed`(0/1) / `dir`(기본 apps/rbc/timing/). CSV는
`timing_rbc_{grid}_{n}gpu_run{SLURM_JOB_ID}.csv`.
