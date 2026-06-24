# 22. dhvc (GPU) 코드 워크스루 — main 실행 순서 & time loop

> 목적: `apps/dhvc/main_gpu.cu`를 **위에서 아래로 따라가며** 각 단계가 무슨 일을 하는지 정리.
> 나중에 코드와 대조하면서 확인하기 위한 체크리스트.
> 대상 파일: `apps/dhvc/main_gpu.cu` (GPU 백엔드). CPU 경로는 `apps/dhvc/main.cpp`로 구조 동일.
> 작성: 2026-06-24

---

## 0. 개요 — 이 코드가 푸는 것

- **문제**: OB 차등가열 수직대류 (PaScaL_TCS Fig 7, DHVC = Differentially Heated Vertical Channel)
- **축 매핑** (PaScaL_TCS와 다름, 주의):
  | 축 | 역할 | BC |
  |---|---|---|
  | x | streamwise / 연직 / 중력방향 → 부력이 U에 작용 (`gravity_axis=0`) | periodic |
  | y | spanwise | periodic |
  | z | wall-normal (hot z=0, cold z=Lz) | wall |
- **시간전진 1스텝 순서**: `CFL → 스칼라(T) → 운동량+부력 → 질량유량보정 → (W 벽 0) → 압력투영`
- **핵심 수치기법**:
  - 운동량·에너지: Beam-Warming delta-form ADI (NOB "var" 커널을 **단위물성으로 OB 환원**해서 사용)
  - 압력: **증분형(van Kan) 투영** — 예측자에서 `−dt·∇P_acc`를 빼고, 풀이 후 `δp`를 누적
  - 통계: 매 스텝 GPU 커널로 누적 (시간루프 중 전체장 D2H 없음)

---

## 1. main 셋업 단계 (시간루프 진입 전)

순서대로. 각 항목 = `main_gpu.cu` 라인 / 하는 일.

| # | 라인 | 단계 | 설명 |
|---|---|---|---|
| 1 | 351-354 | MPI + 설정 로드 | `MpiContext`, `Logger`, `Config::load(input.toml)` |
| 2 | 356-363 | 토폴로지 + 서브도메인 | `np1/np2/np3` 분할, x·y=periodic / z=wall, `n1m/n2m/n3m` 전역격자 → `Subdomain` |
| 3 | 365-372 | 격자 생성 | 축별 `AxisConfig` (균일/tanh 스트레치), `Grid` |
| 4 | 373-376 | BC + TDMA + Domain | `load_problem_from_config` → `bc.validate()`, `TdmaRegistry::make_default`, `Domain` 묶음 |
| 5 | 378-381 | **물성 파라미터** | `nu = √(Pr/Ra)`, `alpha_T = 1/√(Ra·Pr)` |
| 6 | 383-398 | 시간/출력 파라미터 | `t_end, dt_cap(dt_start), max_cfl, n_steps, nmonitor`, 통계는 **물리시간 기준** (`t_stat_start/out`) |
| 7 | 402-410 | CPU 필드 생성 | U,V,W,P,T + 상수(nu, alpha_T). **IC 셋업 + 통계 D2H 용도로만** 존재 |
| 8a | 411-430 | **IC: 콜드스타트** | 해석적 기저상태(U: 3차 프로파일, T: 선형) + 사인 교란(V,W). `restart_in` 비었을 때 |
| 8b | 431-496 | **IC: 웜스타트** | 이전 (다른 격자 가능) restart를 읽어 trilinear 보간. `slab`(기본, MPI-IO 슬랩) vs `replicate`(전역+Bcast). `warmstart_dryrun`이면 체크섬만 찍고 종료 |
| 9 | 497-505 | W 벽면 0 | z 하/상벽 인접 face의 W를 0으로 |
| 10 | 506-510 | halo 동기화 (CPU) | U,V,W,P,T 각각 `sync_field_cpu` |
| 11 | 513-528 | **GPU 셋업** | `bind_gpu_to_local_rank`, `GpuGridArrays`(격자메트릭 D2H), `GpuFields` 생성 + CPU IC를 `to_device` |
| 12 | 530-532 | 솔버 시스템 | `GpuMomentumSystem`, `GpuScalarSystem`, `PressureSystem` |
| 13 | 534-540 | **누적압력 P_acc** | 증분투영용 Pⁿ. 콜드=0, 웜=보간된 P로 시드 |
| 14 | 542-549 | **물성장 (OB 모드)** | `GpuPropertyFields` + `update_properties_gpu(PropertyModel{})` → DeltaT=0이라 **μ=κ̂=1/ρ=1/ρcp≡1, dκ=d(1/ρcp)=0**. 이게 var 커널을 OB로 환원시키는 핵심 |
| 15 | 553-558 | 웜스타트 투영 | 보간장은 이산 비발산이 아님 → 압력투영 1회로 청소 |
| 16 | 561-571 | 통계/모니터 준비 | `GpuDhvcAccum`(디바이스 통계버퍼), `Stats`, `monitor.csv` 헤더 |
| 17 | 573-580 | 부력/보정 파라미터 | `buoy.gravity_axis=0, coeff=1, T_ref=0` (OB), `total_vol`, 벽 플래그 `has_lo/has_hi` |

> **핵심 포인트**: 14번 때문에 dhvc는 "NOB var 커널을 OB 단위물성으로 돌리는" 케이스. 진짜 NOB(가변물성)는 `apps/rbc/main_gpu_nob.cu`.

---

## 2. Time loop — 한 스텝에서 일어나는 일

루프 헤더: `while (time < t_end && step < n_steps)` (`main_gpu.cu:602`). 처음 10스텝은 타이밍 측정(`do_timing`).

순서대로:

### CFL — `main_gpu.cu:607`
```
dt = compute_cfl_dt_gpu(domain, gpu_fields, max_cfl, dt_cap)
```
- 함수: `src/driver/gpu/cfl_gpu.cu:74`
- 셀별 속도 `|u|/dx+|v|/dy+|w|/dz`의 전역 max → `dt = min(max_cfl/max, dt_cap)`
- **대류가 명시적이라 dt는 이류 CFL을 따라야 함** (dt_cap에 고정하면 속도 커질 때 폭발). 큰 고정 dt를 원하면 대류를 음해화해야 함 (Report 18).

### (1) 스칼라 T: Tⁿ → Tⁿ⁺¹ — `main_gpu.cu:611-617`
```
assemble_scalar_var_diff_gpu(...)   // 명시 잔차 RHS 계산
solve_scalar_var_gpu(...)           // 3-sweep ADI + 증분 ΔT 가산
sync_field_gpu(T)
```
- 함수: `src/equation/scalar/gpu/scalar_gpu.cu` (assemble=386, solve=422), 셀 스텐실 = `src/equation/scalar/scalar_var_stencil.hpp`
- 대류: 중심차분, 반음해 delta-form, `−M·Tⁿ` 잔차. 단위물성이라 확산은 순수 라플라시안(κ=alpha_T).
- 내부: X→Y→Z sweep(`sweep_order`), 비주기축은 `k_modify_tdma_row_gpu`로 벽 마감 후 TDMA.

### (2) 운동량 예측자 + 부력 — `main_gpu.cu:621-627`
```
assemble_momentum_var_visc_gpu(...)        // 3성분 RHS (점성 명시 + −M·qⁿ + 명시 교차대류)
add_buoyancy_force_gpu(... buoy ...)        // RHS_U += dt·coeff·(T_face − T_ref)
add_pressure_gradient_gpu(... P_acc ...)    // RHS -= dt·∇Pⁿ  (증분 예측자)
solve_momentum_var_gpu(...)                 // ADI + 블록 하삼각 결합 → dU,dV,dW
update_velocity_gpu(...)                     // U += dU, V += dV, W += dW
```
- 운동량: `src/equation/momentum/gpu/momentum_gpu.cu` (assemble=1285, solve=1424). RHS커널 `k_mom_rhs_var`(419), ADI `k_mom_adi_var`(633), 블록결합 `k_dd_dW/dV_var`·`k_block_dW/dU_var`
- 부력: `src/physics/buoyancy/gpu/buoyancy_gpu.cu:46` — 중력축(x) face에 보간된 T 사용 (PaScaL f90:348)
- 압력경사: `src/equation/pressure/gpu/pressure_gpu.cu:42` — `k_pred_pgrad`, `dU -= dt·(Pc−P_west)/dmx1` 등 (PaScaL mom.f90:344-345)
- 풀이 순서: dU(own=X) → dW(own=Z) → dV(own=Y), 그 사이 `k_dd_*`로 이미 푼 증분을 결합 (delta-form 하삼각). 벽=z이므로 PaScaL의 y↔z 스왑 적용.

### (3) 질량유량 보정 — `main_gpu.cu:631`
```
apply_mass_flow_correction_gpu(... Ub_target=0 ...)
```
- 함수: `src/physics/forcing/gpu/forcing_gpu.cu:81`
- 전역 bulk U를 0으로 (streamwise 순유량 0). U face 전체에 상수 shift.

### sync U,V,W — `main_gpu.cu:634-636`
- 예측속도 halo 교환.

### (4) W 벽면 0 — `main_gpu.cu:639-646`
```
k_zero_w_wall<<<>>>(W, ... has_lo, has_hi)   // 벽 face의 W=0
sync_field_gpu(W)
```
- `apply_ghost`가 못 잡는 z벽 W face를 직접 0. (메모리: MAC W face 벽 처리 이슈)

### (5) 압력투영 — `main_gpu.cu:650-651`
```
solve_pressure_gpu(... pressure ...)           // ∇²δp = ∇·u*/dt, u -= dt·∇δp
accumulate_pressure_gpu(P_acc, P)              // Pⁿ⁺¹ = Pⁿ + δp
```
- 함수: `src/equation/pressure/gpu/pressure_gpu.cu:65` (OB 오버로드, inv_rho 없음 → **NOB 가변밀도 경로 미사용**)
- 엔진: `src/equation/pressure/gpu/pressure_engine_gpu.cuh` (FFT(x,y) + TDMA(z))
- 누적: `k_pacc_add`, `src/equation/pressure/gpu/pressure_gpu.cu:56`

### 스텝 카운터 — `main_gpu.cu:654`
```
time += dt; ++step;
```

### (6) 통계 누적 — `main_gpu.cu:657-658`
- `time >= t_stat_start`이면 `gpu_accum_dhvc` (매 스텝, dt-가중 → 참 시간평균). **D2H 없음**.

### (7) 모니터 — `main_gpu.cu:661-669`
- `step % nmonitor == 0`이면 `gpu_monitor_dhvc`: div_max, Re_δ*(누적/순간), KE, Nu, θ_rms를 ~1.6KB D2H로 계산해 stdout + `monitor.csv`.

### (8) 통계 파일출력 — `main_gpu.cu:673-678`
- 물리시간 `t_stat_out`마다 누적평균 프로파일 `stats_t####.dat` 기록.

### 타이밍 리포트 — `main_gpu.cu:681-693`
- 10스텝째에 ms/step + 단계별 분해(cfl/scalar/mom/corr/sync/pres) 출력.

---

## 3. 루프 종료 후 — `main_gpu.cu:696-744`

| 라인 | 단계 | 설명 |
|---|---|---|
| 696-708 | 총 compute 시간 | barrier 후 전체 벽시계, avg ms/step |
| 716-738 | **체크포인트 쓰기** | `restart_out` 있으면 U,V,W,T + P(=P_acc) D2H → 전역 C-order MPI-IO + `meta.txt`(step/time/mesh/Ra) |
| 740-743 | 최종 Re_δ* | `compute_re_delta_star_cpu` vs 타겟 `0.23·Ra^0.28` |

---

## 4. 부록 — 따라갈 때 헷갈리는 핵심 개념

1. **var 커널의 OB 환원**: dhvc는 `*_var_*`(NOB) 커널만 호출하지만 물성장이 단위(μ=κ̂=1/ρ=1/ρcp=1, 도함수=0)라 OB와 수치적으로 동일. OB 상수 커널(`k_mom_rhs_ob`, `k_energy_residual`, `solve_scalar_gpu`, `solve_momentum_gpu`)은 `DEAD FOR DHVC` 주석으로 표시됨.

2. **증분형(van Kan) 압력투영**: 예측자에서 `−dt·∇P_acc`를 빼고(step 2), 투영은 압력 *증분* δp만 풂(step 5), 그 뒤 `P_acc += δp`. 그래서 `P_acc`가 진짜 압력 Pⁿ, restart의 `P.bin`도 P_acc.

3. **sweep 순서**: `bc.topology.sweep_order()` = X→Y→Z. 각 sweep은 (band 빌드 → 벽 마감 `k_modify_tdma_row_gpu` → TDMA → scatter).

4. **벽 = z**: PaScaL_TCS는 벽=y(V)인데 dhvc는 벽=z(W). 운동량 풀이 순서/블록결합/부력축에 y↔z, V↔W 스왑이 일관 적용됨. PaScaL의 벽 마스크(jwp/jwm)는 GPU에선 (a)벽 W=0 (b)ghost 정책+`modify_tdma_row`로 대체 (Report 18·20 참고).

5. **통계는 물리시간 기준**: 가변 dt와 무관하도록 dt-가중 누적(`t_accum=Σdt`로 나눔). step 기준 아님.

6. **CPU 필드의 역할**: IC 셋업과 (체크포인트 시) D2H 목적지로만. 시간루프 중 실제 계산은 전부 `GpuFields`.

---

## 관련 파일 빠른참조

| 기능 | 파일:라인 |
|---|---|
| 메인 (GPU) | `apps/dhvc/main_gpu.cu` |
| CFL | `src/driver/gpu/cfl_gpu.cu:74` |
| 스칼라 assemble/solve | `src/equation/scalar/gpu/scalar_gpu.cu:386,422` |
| 스칼라 셀 스텐실 | `src/equation/scalar/scalar_var_stencil.hpp` |
| 운동량 assemble/solve | `src/equation/momentum/gpu/momentum_gpu.cu:1285,1424` |
| 운동량 RHS/ADI 커널 | 같은 파일 `k_mom_rhs_var:419`, `k_mom_adi_var:633` |
| 부력 | `src/physics/buoyancy/gpu/buoyancy_gpu.cu:46` |
| 압력경사/투영/누적 | `src/equation/pressure/gpu/pressure_gpu.cu:42,65,56` |
| 압력 엔진 | `src/equation/pressure/gpu/pressure_engine_gpu.cuh` |
| 질량유량 보정 | `src/physics/forcing/gpu/forcing_gpu.cu:81` |
| 물성 (OB 단위) | `src/physics/properties/gpu/properties_gpu.cu:48` |
| halo 동기화 | `src/core/gpu/boundary_ops_gpu.cu:131` |
