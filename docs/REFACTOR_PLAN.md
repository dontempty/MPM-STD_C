# MPM-STD_C 리팩토링 계획서 (rev.2)

## 0. 배경 / 목표

교수님 피드백:
- 추상화 과다 → 가독성 저하 (단, **객체지향 방향 유지**)
- **Ax=b 관점 3섹션**: `[전처리/assemble] → [풀이/solve] → [후처리/post]`
- 이름으로 기능이 드러나게 → main이 레시피처럼
- **Multi-GPU 빌드**: CPU(정리)·GPU(신규), `_cpu`/`_gpu` 구분, 한 번에 빌드
- 나중에 RBC·IBM·LES 확장

고정 전제:
- 항상 **MPM-STD implicit**: (T 선택) → momentum → Poisson. explicit 안 씀
- **solve(x=A⁻¹b)는 거의 불변·공용** (단, Poisson은 임의 BC 대응으로 변환 일반화 필요 — §9c)
- **최종 목표 = PaScaL_TCS Fig 7(OB-DHVC) + Fig 9(NOB-RBC)를 multi-GPU로 재현** (§9b)

---

## 1. 설계 원칙

| 원칙 | 내용 |
|---|---|
| 자료구조 = 클래스 | `Grid`, `Field`, `Bands`, `System` 등은 데이터+단순 접근자만 |
| 연산 = 자유함수 | `this` 없는 자유함수, 데이터 전부 인자로 → 호출부에서 흐름이 보임. 거대 `MomentumEquation` 해체 |
| **CPU/GPU = 분리 타입 (C1=b)** | device 데이터(`Field`,`Bands`,`System`)는 **`Cpu*`/`Gpu*` 두 벌**. host 메타(`Grid`,`MpiTopology`,`Boundary`,`Config`)는 단일. virtual `Backend` 폐기 |
| 백엔드 명시 | 모든 연산 함수에 `_cpu`/`_gpu` 접미사 (alias 금지). 한 번 정하면 끝까지 고정 |
| **CUDA-aware MPI 필수 · 1 rank = 1 GPU** | halo·transpose는 device-to-device **CUDA-aware MPI**. ⚠ **하나의 GPU에 여러 rank 금지** — `cudaSetDevice(local_rank)`로 1:1 매핑 (과거 실수 재발 방지) |
| solve 분리·재사용 | 선형솔버(`solve/`)는 공용. equation은 A·b 조립 후 `solve/` 호출 |
| BC는 input 주도·BC-agnostic | 임의 BC(channel·cavity·DHVC·RBC)에서 동일 코드, 입력 BC만 다름. sweep/cyclic/ghost/행보정/Poisson변환을 BC에서 도출 → §9c |
| 분기 없음 · 쓰는 것만 할당 | `if(solve_thermal)` 제거. 안 풀 물리는 **호출 생략 + 자료 선언조차 안 함** → 연산·메모리 0. 중앙 레지스트리 미사용. momentum은 **T 비의존** → §7,§9 |

---

## 2. 아키텍처 — 3섹션 × equation별 × CPU/GPU

```
[물리 모디파이어/PHYSICS]  (선택적, main이 명시 호출 — 호출=on, 생략=off)
   properties(μ(T)…)·les(ν_t) → effective 계수 (assemble 전)
        │
[전처리/ASSEMBLE]  equation별·문제별 (확장 지점)
   A(Bands)·b(RHS)·BC 처리
        │  ← buoyancy(b 가산)·ibm(A·b 수정) : physics에서 정의, assemble 직후 main이 호출
[통신/HALO]  exchange_halo_{cpu,gpu}  (명시적 자유함수, solve 직전)
        │
[풀이/SOLVE]  공용·implicit 고정
   solve/ 선형솔버로 x = A⁻¹b
   (banded TDMA/PTDMA[+rank간 통신], Poisson FFT/DCT/TDMA)
        │
[후처리/POST]
   statistics · diagnostics(Re_δ*,Θ_c,Nu) · io/restart
```
매 스텝 = `(physics) → assemble → (buoyancy/ibm) → halo → solve`를 방정식마다.

---

## 3. 디렉토리 구조 (최종)

```
src/
├ core/                       # 백엔드 무관 메타(host 단일) + device 자료(cpu/gpu 두 벌)
│   grid.{hpp,cpp}            # 격자·메트릭 (host 단일)
│   mpi_topology.{hpp,cpp}    # MPI 분해 + rank↔GPU 1:1 매핑 + CUDA-aware halo API (host 단일)
│   subdomain / config        # (host 단일)
│   boundary.{hpp,cpp}        # BC 자료: input에서 6면×변수별 kind/value/ghost 로드 (host 단일, §9c)
│   field_cpu.hpp / field_gpu.hpp   # CpuField(host배열) / GpuField(device배열)  ← C1=(b)
│   bands.hpp                 # Bands 논리레이아웃 (3→5중대각); cpu/gpu 저장 분리
│   system.hpp                # Scalar/Momentum/PressureSystem (cpu/gpu)
│
├ solve/                      # [풀이] 선형솔버 (공용)
│   banded_solver.hpp   cpu/ banded_solver_cpu.cpp   gpu/ banded_solver_gpu.cu   # (P)TDMA, rank간 통신
│   poisson_solver.hpp  cpu/ poisson_solver_cpu.cpp  gpu/ poisson_solver_gpu.cu  # FFT/DCT/TDMA 변환을 BC서 선택 (§9c)
│
├ equation/                   # equation별 assemble + solve
│   momentum/  assemble.hpp solve.hpp  + cpu/ gpu/   # solve_momentum이 U,V,W,dU,dV,dW+블록커플링 한 번에 (M2, MPM-STD fortran 구조)
│   scalar/    assemble.hpp solve.hpp  + cpu/ gpu/
│   pressure/  assemble.hpp solve.hpp  + cpu/ gpu/   # solve=poisson_solver 호출
│
├ physics/                    # [모디파이어] 선택적 물리 (전부 cpu/+gpu/) — 파라미터 타입도 여기서 정의
│   buoyancy/    buoyancy.hpp    cpu/ gpu/   # OB/NOB Boussinesq + BuoyancyParams
│   properties/  properties.hpp cpu/ gpu/   # NOB μ(T),1/ρ,κ,1/(ρCp) + PropertyModel
│   forcing/     forcing.hpp     cpu/ gpu/   # 채널 dPdx body force + mass-flow 보정 (U7)
│   ibm/         ibm.hpp         cpu/ gpu/   # 마스크 H + IB forcing + IbmMask
│   les/         les.hpp         cpu/ gpu/   # SGS ν_t
│
├ post/                       # [후처리] — Stats/Io 타입도 여기서 정의
│   statistics.{hpp} cpu/ gpu/   # Stats (전역 nx*ny 정규화!)
│   diagnostics.{hpp} cpu/ gpu/  # Re_δ*, Θ_c, Nu
│   io.{hpp} cpu/ gpu/           # Io, restart (기존 포맷 호환 목표 — §10)
│
├ driver/                     # 케이스 공용 헬퍼 (중복 방지, M5) — main이 조립해서 씀
│   cfl.hpp  restart.hpp  monitor.hpp  (cpu/gpu)
│
apps/                         # 사용자 코드 (libmpmstd 링크) — BC는 입력만 다름
├ channel/ main.cpp  cavity/ main.cpp  dhvc/ main.cpp  rbc/ main.cpp
│
tests/                        # 별도 디렉토리에서 실험·검증 (M3, 기존 방식 유지)
├ unit/         (test_tdma, test_grid, test_bc, test_stencil … 신구조로 이식)
├ integration/  (단일 방정식·소형)
└ regression/   (golden: Re_tau=180 통계, cavity Ghia, Fig7/9)
Makefile / Makefile.inc       # libmpmstd(core+solve+equation+physics+post+driver, cpu+gpu) + apps + tests
```
규칙: 같은 기능 = 같은 이름 + `_cpu`/`_gpu`, `cpu/`·`gpu/` 폴더 분리.
**모든 신규 코드는 6계층 중 하나에 귀속**: `core`·`solve`·`equation`·`physics`·`post`·`driver`. 추가 시 "어느 계층/폴더에 `_cpu`/`_gpu`로 둘지" 먼저 결정.

---

## 4. 핵심 자료구조 (core/)

> **구조 재설계 반영(2026-06-09, commit 8d9db67).** time loop의 12-인자/`*tdma`/혼재 스칼라 호출을 정돈된 묶음 객체로 압축. 분석=`docs/STRUCTURE_REDESIGN.md`.

```cpp
// C1=(b): device 데이터는 cpu/gpu 두 벌. 논리 레이아웃 동일, 메모리 위치만 다름.
class CpuField { /* host 배열 (정렬 메모리) */ };   // field_cpu.hpp
class GpuField { /* device 배열 (cudaMalloc)  */ };  // field_gpu.hpp
//  Grid / MpiTopology / Subdomain / Config 는 host 단일 타입 (백엔드 무관 메타).

// ── 묶음 1: Domain = "어디서/어떻게" (격자·MPI·서브도메인·TDMA) ──  domain.hpp
struct Domain {
  const Grid& grid; const MpiTopology& topo; const Subdomain& sub;
  linear_solver::tdma::TdmaRegistry& tdma;   // ⚠ 참조 — main에서 `auto& tdma=*owner;` 한 번 (호출부에 `*tdma` 없음)
};

// ── 묶음 2: BoundaryCondition = BC (Domain과 분리) ──  boundary.hpp
using BoundaryCondition = boundary::Problem;   // [topology] + 면별 FieldBoundary(U,V,W,P,T)

// ── 묶음 3: Fields = 쓰는 물리변수 + 상수, typed-key 컨테이너 ──  variables.hpp
enum class Var   { U, V, W, P, T, Count };     // 공간변수(Field)
enum class Const { nu, alpha_T, Count };       // 스칼라(Constant) — 향후 nu→mu(T)
struct Constant { real_t value; };
template<class FieldT> class FieldStore {      // fields[Var::U] ; fields.constant(Const::nu)
  FieldT& add(Var, const Subdomain&);  void add_constant(Const, real_t);   // 루프 전 등록 (쓰는 것만)
};
using CpuFields = FieldStore<CpuField>;  using GpuFields = FieldStore<GpuField>;

// A 밴드 (3중대각 now, 5중대각 대비). cpu/gpu 저장 분리(device 배열).
struct Bands { int bandwidth=1;  /* lo2,lo1,diag,up1,up2, rhs (host or device) */ };

// 한 방정식의 Ax=b 묶음 (cpu/gpu). MomentumSystem은 증분(dU,dV,dW)을 내부 소유.
struct ScalarSystem   { Bands x,y,z; /* 방향 메타 */ };
template<class FieldT> struct MomentumSystemT {
  FieldT dU, dV, dW;                 // ⚠ 증분은 momentum이 소유(사용자가 안 만듦); rhs/stage/A,B,C,D
};
using CpuMomentumSystem = MomentumSystemT<CpuField>;  using GpuMomentumSystem = MomentumSystemT<GpuField>;
struct PressureSystem { /* rhs + 파수(wavenumber) + 방향별 변환종류 + engine(shared_ptr) */ };

// 파라미터/후처리 타입은 각 소속 계층에서 정의:
//   physics/: PropertyModel(properties), BuoyancyParams(buoyancy), IbmMask(ibm)
//   post/   : Stats(statistics), Io(io)
```

**GPU-resident 모델(사용자 최우선):** GPU 풀이 시 모든 변수를 초기에 `GpuFields`/`GpuMomentumSystem`으로 device 선언 → 모든 계산을 GPU에서만 → **fileout 때만 device→host 복사**. `FieldStore`가 field 타입에 템플릿이라 같은 레시피가 CPU/GPU 양쪽에 성립.

---

## 5. 자유함수 네이밍 규약 (명시적 `_cpu`/`_gpu`; 아래는 cpu, gpu는 동일 시그니처)

> **구조 재설계로 실현된 인자 규약:** 모든 연산은 **`op(domain, [bc,] fields, system, dt)`** 형태 — 개별 `CpuField` 인자 나열 대신 §4의 묶음을 받음. 아래 스케치의 의도(자유함수·`_cpu`/`_gpu`·const/var 점성 분리·M2)는 그대로, **인자 그룹만** 바뀜. 줄임말 금지 — `domain/fields/field/grid/momentum/pressure` 풀네임. 실현된 시그니처 예:
> ```cpp
> void assemble_momentum_const_visc_cpu(const Domain& domain, CpuFields& fields, CpuMomentumSystem& momentum, real_t dt);
> void solve_momentum_cpu(const Domain& domain, const BoundaryCondition& bc, CpuFields& fields, CpuMomentumSystem& momentum, real_t dt);
> void update_velocity_cpu(CpuFields& fields, const CpuMomentumSystem& momentum);
> void solve_pressure_cpu(const Domain& domain, const BoundaryCondition& bc, CpuFields& fields, PressureSystem& pressure, real_t dt);
> real_t compute_cfl_dt_cpu(const Domain& domain, const CpuFields& fields, real_t max_cfl, real_t dt_cap);
> void   sync_field_cpu(CpuFields& fields, Var, const BoundaryCondition& bc, const Subdomain& sub);  // halo+ghost 한 번에
> ```
> **추가 힘은 main 합성(사용자 지시):** buoyancy 등은 main에서 사용자가 명시 호출(call=on/omit=off), solve/assemble 내부에 박지 않음. momentum assemble = 점성+대류만(force-agnostic). 채널 forcing(dPdx·mass-flow)도 main 루프 호출.

```cpp
// ── 통신 (C3): solve 직전 명시 호출 (Field 단위 primitive; 컨테이너용 sync_field_cpu는 위 참조) ──
void exchange_halo_cpu(CpuField&, const Subdomain&);   // 고스트셀 교환 (gpu는 CUDA-aware MPI)

// ── equation/scalar ──
void assemble_scalar_system_cpu (ScalarSystem&, const CpuField& T, const CpuField& U,V,W,
                                 const CpuField& kappa, const Grid&, const Boundary&, real_t dt); // κ: NOB는 필드, OB는 상수필드
void solve_scalar_cpu (ScalarSystem&, CpuField& T, const MpiTopology&);                          // 내부 ADI sweep + rank간 통신

// ── equation/momentum (T 비의존; const/var 분리) ──
void assemble_momentum_const_visc_cpu(MomentumSystem&, const CpuField& U,V,W,P,
                                      real_t nu, const Grid&, const Boundary&, real_t dt);     // cross-stress 없음
void assemble_momentum_var_visc_cpu  (MomentumSystem&, const CpuField& U,V,W,P,
                                      const CpuField& mu, const Grid&, const Boundary&, real_t dt); // cross-stress 포함
//  M2: solve_momentum이 U,V,W,dU,dV,dW를 "한 번에" — 3성분 ADI + 블록 하삼각 커플링까지 내부 처리
//      (MPM-STD fortran core_momentum: solvedU/V/W + blockLdV/U 와 동일 구조). couple 별도 호출 없음.
void solve_momentum_cpu (MomentumSystem&, const CpuField& U,V,W, CpuField& dU,dV,dW, const MpiTopology&);
void update_velocity_cpu(CpuField& U,V,W, const CpuField& dU,dV,dW);                            // pseudoupdate U+=dU

// ── equation/pressure ──
void assemble_pressure_system_cpu(PressureSystem&, const CpuField& U,V,W, const Grid&, real_t dt);
void solve_pressure_cpu          (PressureSystem&, CpuField& dP, const MpiTopology&);           // poisson_solver 호출
void project_velocity_cpu        (CpuField& U,V,W,P, const CpuField& dP, const Grid&, real_t dt); // 무조건 수행 + P 갱신(U7)

// ── solve (공용 선형솔버) ──
void solve_banded_adi_cpu (ScalarSystem&,   CpuField& x,  const MpiTopology&);   // (P)TDMA, rank간 통신 내부
void solve_poisson_cpu    (PressureSystem&, CpuField& dP, const MpiTopology&);   // FFT/DCT/TDMA를 BC서 선택 (§9c)

// ── physics (호출=on, 생략=off) ──
void update_properties_cpu     (CpuField& mu, CpuField& irho, CpuField& kappa, CpuField& irhocp,
                                const CpuField& T, const PropertyModel&);                       // NOB
void add_buoyancy_force_cpu    (MomentumSystem&, const CpuField& T, const BuoyancyParams&, real_t dt); // OB/NOB, assemble 직후
void apply_pressure_gradient_cpu(MomentumSystem&, real_t dpdx, real_t dt);                       // 채널 forcing(U7)
void apply_mass_flow_correction_cpu(CpuField& U, real_t Ub_target, const MpiTopology&, real_t dt);// 채널 forcing(U7)
void compute_sgs_viscosity_cpu (CpuField& nu_t, const CpuField& U,V,W, const Grid&);            // LES
void apply_ibm_forcing_cpu     (MomentumSystem&, const IbmMask&);                               // IBM

// ── post ──
void accumulate_statistics_cpu (Stats&, const CpuField& U,V,W,P);   // ⚠ 전역(global) nx*ny 정규화
void compute_Re_delta_star_cpu (real_t&, const CpuField& U, const Grid&, const MpiTopology&);   // Fig 7
void compute_center_temp_cpu   (real_t&, const CpuField& T, const Grid&, const MpiTopology&);   // Fig 9
void write_restart_cpu / read_restart_cpu (...);                                                // 기존 포맷 호환 목표

// ── driver (케이스 공용) ──
real_t compute_cfl_dt_cpu(const CpuField& U,V,W, const Grid&, const MpiTopology&, real_t cap);
```
(GPU: 전부 `GpuField`+`_gpu`. 시그니처 동일)

---

## 6. 빌드

- 산출물 **`libmpmstd`** (정적): core+solve+equation+physics+post+driver 의 **cpu·gpu 모두**.
- `.cpp`→mpicxx, `.cu`→nvhpc(nvc++ `-cuda`), 링크 통합. **GPU 빌드는 CUDA-aware MPI 라이브러리(nvhpc hpcx)에 링크.**
- `make`=cpu+gpu+apps+tests 동시. 분리: `make cpu`(CUDA 의존 X) / `make gpu`.
- 사용자는 `apps/<case>/main.cpp`에서 링크 후 `_cpu` 또는 `_gpu` 직접 호출 (혼용 X).
- **실행: 1 rank = 1 GPU.** launcher가 `local_rank → cudaSetDevice` 보장 (run 스크립트·`mpi_topology` 초기화에 명시).

---

## 7. main 가독성 목표 (분기 없음 · 쓰는 것만 선언)

```cpp
// ── 사례 A: 등온 채널 (실현된 레시피, apps/channel/main.cpp) ──
Domain domain{grid, topo, sub, tdma};          // 어디서/어떻게 (tdma는 ref)
BoundaryCondition bc = load_problem_from_config(cfg);
CpuFields fields;                              // 쓰는 것만 등록 (T·mu 없음)
fields.add(Var::U, sub); fields.add(Var::V, sub); fields.add(Var::W, sub); fields.add(Var::P, sub);
fields.add_constant(Const::nu, nu);
CpuMomentumSystem momentum(sub);  PressureSystem pressure;  Stats stats;   // momentum이 dU,dV,dW 소유
while (time < t_end) {
    const real_t dt = compute_cfl_dt_cpu(domain, fields, max_cfl, dt_cap);
    assemble_momentum_const_visc_cpu(domain, fields, momentum, dt);        // 상수점성, T 비의존, cross-stress X
    solve_momentum_cpu              (domain, bc, fields, momentum, dt);    // 3성분 ADI+블록커플링 한 번에(M2)
    update_velocity_cpu             (fields, momentum);
    sync_field_cpu(fields, Var::V, bc, sub);  sync_field_cpu(fields, Var::W, bc, sub);
    apply_body_force_cpu            (fields, dpdx, dt);                     // ← 추가 힘: main 합성(physics/forcing)
    apply_mass_flow_correction_cpu  (domain, fields, 1.0, dt, total_vol, dpdx);  // Ub=1 유지
    sync_field_cpu(fields, Var::U, bc, sub);
    solve_pressure_cpu              (domain, bc, fields, pressure, dt);     // div+Poisson+project 묶음
    accumulate_statistics_cpu       (stats, domain, fields);
}

// ── 사례 B: NOB RBC (T·물성·부력을 "추가"; P6/P7에서 실현) ──  apps/rbc/main.cpp
CpuFields fields; /* U,V,W,P + */ fields.add(Var::T, sub);  // 추가 등록 (NOB일 때만)
ScalarSystem Tsys; /* ... */
while (time < t_end) {
    const real_t dt = compute_cfl_dt_cpu(domain, fields, max_cfl, dt_cap);
    update_properties_cpu        (domain, fields, model);                  // physics: μ(T)…
    assemble_scalar_system_cpu   (domain, bc, fields, Tsys, dt);  solve_scalar_cpu(domain, fields, Tsys);
    assemble_momentum_var_visc_cpu(domain, fields, momentum, dt);          // 변동점성(cross-stress)
    add_buoyancy_force_cpu       (momentum, fields[Var::T], buoy, dt);     // ← 추가 힘: main 합성(physics/buoyancy)
    solve_momentum_cpu           (domain, bc, fields, momentum, dt);  update_velocity_cpu(fields, momentum);
    solve_pressure_cpu           (domain, bc, fields, pressure, dt);
    accumulate_statistics_cpu    (stats, domain, fields);
}
```
→ thermal/부력/forcing/LES/IBM = **줄 넣으면 on, 빼면 off**(main 합성). 등온 main은 `Var::T`/`mu` **등록조차 안 함** → 0 메모리·0 연산. 고스트셀 sync는 `sync_field_cpu`(halo+ghost)로 묶어 solve 둘레에 명시(C3). 함수 이름만으로 흐름·물리옵션·cpu/gpu 명확.

---

## 8. 단계별 마일스톤

| 단계 | 내용 | DoD / 검증 |
|---|---|---|
| **P-0.5** | **인터페이스 스파이크**(U4): `CpuField`/`GpuField`·`Bands`/`System`·`exchange_halo`·`MpiTopology(rank↔GPU)` API만 먼저 확정. 작은 PoC(필드 1개 halo, 1방정식 solve)로 시그니처 검증 | API 동결, 2-process halo 동작 |
| **P0** | 전체 스켈레톤 일괄 생성 (디렉토리·헤더·스텁·빌드·tests 골격) | `make`로 libmpmstd+빈 apps+tests 빌드 성공 |
| **P1** | CPU 구현 신구조 이식 (virtual Backend 제거, 자유함수화). **단위테스트 이식하며 모듈별 진행** | 기존 단위테스트 통과 + Re_tau=180 통계가 **기존 코드와 일치** |
| **P2** | channel main 가독화 (§7 레시피) | ✅ 완료 (`099a04c`) |
| **구조 재설계** | §4 묶음 타입(Domain/BoundaryCondition/Fields{Field+Constant}/MomentumSystem dU 소유) + 통일 시그니처 `op(domain,[bc,]fields,system,dt)` + 풀네임 + GPU-resident | ✅ 완료 (`8d9db67`); Re_tau=180 **비트-동일**, 듀얼빌드 그린 |
| **P3** | solve 일반화: 5중대각 hook; **BC-agnostic**(sweep/cyclic/ghost/행보정/Poisson변환을 BC서 도출). sweep order는 BC 자동(z-wall→x,y,z; y-wall→x,z,y) | 3중대각·channel 결과 불변 |
| ~~**P3b**~~ | ~~cavity~~ — ❌ **SKIP 확정(2026-06-09, §8b)**: 전-Neumann Poisson 신설은 DHVC/RBC가 미사용. 대신 **P6 DHVC 먼저** | — |
| **P4** | **GPU core + multi-GPU 통신**: `GpuField`, **CUDA-aware MPI halo(device-to-device)**, **1 rank=1 GPU(cudaSetDevice)**, 동시 빌드 | 2-GPU device halo·PTDMA 동작, `*_gpu` 실행 |
| **P5** | GPU 커널 (assemble/solve/post `_gpu`) — 기존 `kernels_cuda.cu` 활용 | GPU가 **기존/CPU와 통계 일치** (점단위 1e-10 폐기, §10) |
| **P6** | 에너지(T)+OB 부력 → **DHVC** (Re_δ* 진단, tanh stretch, Ra 범위, 상수물성) | **Fig 7 재현** (CPU→multi-GPU) |
| **P7** | NOB 변동물성 + **풀 변동점성 모멘텀(cross-stress)** + 변동 κ 에너지 → **RBC 글리세롤** | **Fig 9 재현** (CPU→multi-GPU) |
| **P8 (최종)** | **Multi-GPU 검증 스위트**: Fig 7 + Fig 9 재현 | §9b 정량 목표 충족 |
| **P9** | LES (`physics/les/`, ν_t) | 난류 LES 검증 |
| **P10** | IBM (`physics/ibm/`, 마스크+forcing) | 복잡형상 검증 |

> ⚠ Fig 9(NOB)가 **변동점성**을 요구 → 상수점성 채널에서 미뤄둔 **cross-stress 모멘텀이 P7에서 필수**. 상수점성 채널(Re_tau=180, 검증완료)이 토대.

---

## 8b. cavity(P3b) 구현비용 분석 — 보류 검토 (2026-06-09)

**질문(사용자):** cavity 구현이 오래 걸리면 굳이 안 하고 RBC·DHVC로 직행.

**핵심 = Poisson 솔버.** 현 압력 엔진은 **X·Y 주기를 하드 요구**:
```
pressure_engine_cpu.cpp:41  if(!is_periodic(X)||!is_periodic(Y)) throw "X,Y must be periodic — use DctPressureSolver";
```
(x: batched r2c FFT, y: FFT, z: 분산 TDMA — pencil-FFT). **`DctPressureSolver`는 아직 없음.**

| 사례 | 주기축 | Poisson | 신규 작업 |
|---|---|---|---|
| channel(검증완료) | x,y | 기존 FFT 엔진 | — |
| **DHVC**(Fig7) | **2방향(x,y)** + z벽 | **기존 FFT 엔진 그대로** | scalar(T)+OB부력+Re_δ* (목표 경로) |
| **RBC**(Fig9) | **수평 2방향** + z벽 | **기존 FFT 엔진 그대로** | NOB물성+cross-stress+부력 (목표 경로) |
| cavity | **없음(전벽)** | ❌ 전-Neumann 신설 필요 | **`DctPressureSolver` 신규** |

**cavity 비용(= 다른 데 안 쓰이는 신규 인프라):**
1. `DctPressureSolver` 신설 — 기존 `PressureSolver`(엔진 200+줄: FFTW 플랜·수정파수·펜슬 transpose)와 맞먹는 규모. REDFT10/01 코사인 변환 플랜 + Neumann 수정파수 + **null-space(전-Neumann ⇒ 특이행렬 → 평균압력 고정)**.
2. ⚠ **DCT는 균일격자에서만 라플라시안 대각화** → Ghia가 흔히 쓰는 벽근방 stretch 격자에선 대각화 실패 → **반복법(CG/멀티그리드) 필요 = 대형 작업**. 균일격자로 한정하면 DCT+TDMA 가능하나 검증 가치 축소.
3. 이동벽 BC(소), cavity main·Ghia 비교(소).

**결론 = cavity SKIP 권고.** cavity의 큰 비용(전-Neumann Poisson)은 **최종 목표(Fig7/9)가 안 씀** — DHVC/RBC는 z벽+2주기로 기존 FFT 재사용. cavity가 주려던 "임의 BC·Poisson 일반화" 검증은 우리가 안 쓸 능력에 대한 것. **P3b 건너뛰고 P6(DHVC)→P7(RBC) 직행.** BC-agnostic(sweep/ghost/행보정 BC도출, P3 경량부)은 DHVC/RBC가 다른 벽배치로 자연히 행사 → 별도 cavity 앱 불필요.

**권고 순서:** P3 경량(BC도출 검증, 5중대각 hook은 NOB 전엔 불요) → **P6 DHVC**(가장 쉬움: OB·상수물성·기존 Poisson, T수송+부력만 추가) → **P7 RBC**(NOB+cross-stress) → P8 multi-GPU.

### 확정 (2026-06-09)
- **cavity SKIP.** 계산량 적고 빠른 것부터 검증 → **DHVC 먼저**(OB·Pr=0.7·저Ra 작은격자 = RBC의 NOB+Pr=2547+128³보다 훨씬 가볍고 새 코드 적음).
- **CPU 먼저 → GPU.** DHVC **Ra=10⁵ 작은 격자**를 CPU(login01)에서 검증(채널 회귀보다 가벼움; 저Ra 정상상태 빨리 도달) → 물리(부력·T수송) 확정 후 GPU 커널(P5)+대규모를 multi-GPU(P8). ⚠ 고Ra 스윕(10⁹–10¹⁰)·RBC 수렴은 GPU 전용(CPU로 돌리지 말 것).

---

## 9. 물리/확장 모듈 설계 (모디파이어 플러그인)

**원칙**: 새 물리는 거대 클래스/상속 아님 → `physics/<name>/{cpu,gpu}/` **자유함수** + 그 파라미터 타입(PropertyModel/BuoyancyParams/IbmMask)도 같은 폴더에서 정의. main이 assemble 전후 **명시 호출**(빼면 off). 자료는 `Field`로, A·b는 `System`/`Bands`로 수정.

| 모듈 | 위치 | 어디에 끼나 | 대표 함수 |
|---|---|---|---|
| buoyancy | `physics/buoyancy/` | momentum b 가산 (assemble 직후) | `add_buoyancy_force_{cpu,gpu}` |
| properties(NOB) | `physics/properties/` | T→μ,ρ,Cp,κ 필드 갱신 (assemble 전) | `update_properties_{cpu,gpu}` |
| forcing(채널) | `physics/forcing/` | dPdx body force + mass-flow 보정 | `apply_pressure_gradient_*`, `apply_mass_flow_correction_*` |
| IBM | `physics/ibm/` | 마스크+forcing로 A·b 수정 (assemble 후) | `build_ibm_mask_*`, `apply_ibm_forcing_*` |
| LES | `physics/les/` | ν_t → μ_eff (assemble 전) | `compute_sgs_viscosity_*` |

설계 포인트:
- **momentum assemble은 T 비의존.** T-결합은 ① 부력(b 가산) ② μ(T) 필드(properties가 생성)로 분리. 등온이면 둘 다 생략.
- **const/var 점성 함수 분리**: 등온·OB=const(스칼라 ν, cross-stress X), NOB·LES=var(μ 필드, cross-stress).
- **M2 — `solve_momentum`의 내부 구조 (MPM-STD fortran과 동일)**:
  ```
  solve_momentum(mom, U,V,W, dU,dV,dW, mpi):
     solve dU : 3-sweep ADI (sweep order=BC도출) over assembled bands   # solvedU + rank통신
     solve dV : 동일                                                     # solvedV (inline로 dU 사용 가능)
     solve dW : 동일                                                     # solvedW
     block-couple: dV -= dt·M23·dW ;  dU -= dt·(M12·dV+M13·dW)           # blockLdV, blockLdU (하삼각)
  ```
  → couple를 main에서 따로 안 부름. 한 함수가 fortran `core_momentum`에 대응.
- **IBM 마스크**(`is_fluid_face`→정상, 아니면 RHS=0·대각=1)은 이미 MPM-STD Amatrix에 존재 → `apply_ibm_forcing`으로 분리.
- 각 모듈 cpu/gpu 동일 시그니처.

---

## 9b. 최종 Validation 설계 (PaScaL_TCS 논문 Fig 7 · Fig 9)

### Fig 7 — OB 자연대류 (DHVC) → `apps/dhvc/`
- OB(상수물성), 공기 **Pr=0.7**, Boussinesq 부력(중력=streamwise, 벽법선에 수직).
- 차등가열 수직채널. no-slip; θ=±0.5 양 벽; 나머지 2방향 주기. **Lx=8H, Lz=4H**.
- **Ra=10⁵~10¹⁰**(6). tanh stretch, 벌크 Grötzbach + 경계층 Shishkina.
- 진단: `Re_δ* = U_max·δ*/ν`, `δ*=∫₀^δmax(1−U/U_max)dy`.
- 목표: `Re_δ*~0.23·Ra^0.28`, Ng(2017)와 Ra≤10⁹ 일치.
- 필요: scalar(T)+OB buoyancy+Re_δ* 진단+stretch. **상수점성(cross-stress X)**.

### Fig 9 — NOB RBC (글리세롤) → `apps/rbc/`
- NOB 온도의존 ρ,Cp,κ,ν. **Pr=2547, Ra=10⁶**.
- RBC. 벽법선=중력(연직). hot 바닥/cold 천장; 수평 주기; 정사각 **Γ=6**; **128³**.
- **Δ=20,40,60K**. 물성 다항식 Pan et al.
- 진단: 중심온도편차 `(Θ_c−Θ_m)` vs Δ (OB→0, NOB→>0).
- 목표: Zhang(1997)·Ahlers(2006)·Horn(2017) 일치.
- 필요: properties + **풀 변동점성 모멘텀(cross-stress)** + 변동 κ 에너지 + buoyancy.

### 공통 인프라
- **방향 불가지론**: channel(z벽,중력X)·DHVC(수평벽,streamwise중력)·RBC(연직벽=중력) → sweep·BC·중력축을 설정서 도출.
- tanh stretch + 경계층 해상도 기준.
- **multi-GPU 필수**: 고-Ra(10¹⁰)·NOB 대규모 → CUDA-aware MPI, 1 rank=1 GPU.

---

## 9c. 경계조건 일반화 — input 주도 · BC-agnostic (임의 BC, 예: cavity)

같은 코드가 입력 BC만 바꿔 channel·cavity·DHVC·RBC를 풂. **입력 형식 기존 유지.**
```toml
[topology]   x="wall"  y="wall"  z="wall"          # periodic/wall/inflow/outflow
[bc.z.plus.U]  kind="dirichlet" value=1.0 ghost="antisymmetric"   # cavity lid
[bc.x.minus.P] kind="neumann"   value=0.0                          # ∂p/∂n=0
```
**솔버가 BC서 자동 도출:**
| 항목 | 위치 | 규칙 |
|---|---|---|
| sweep order | core(topology) | 주기 먼저, wall 나중 → z-wall이면 **x,y,z**; y-wall이면 **x,z,y** (자동, 기존 `sweep_order()` 활용) |
| TDMA cyclic vs solve | solve/banded | 주기축→cyclic, else 일반+행보정 |
| ghost fill | equation assemble | 면별 ghost(zero/antisym/neumann) |
| 행렬 행 보정 | equation assemble | 면별 kind(dirichlet/neumann/…) |
| Poisson 방향별 변환 | solve/poisson | 주기→FFT, Neumann/wall+균일격자→DCT, 나머지축→TDMA |

**Poisson 일반화 (cavity)**: 주기축 없음 → 전면 Neumann 압력 → **DCT(균일축)+TDMA(나머지)+null-space 제거(평균압력 고정)**. 완전 비균일·전벽은 반복법/멀티그리드 폴백 옵션.
**검증**: `apps/cavity/` lid-driven cavity vs Ghia et al. (P3b).

---

## 10. 회귀·검증 전략

- **P-0.5/P0 직전 기준선 동결**: 검증된 Re_tau=180 통계(`stats_00040000.dat`)·난류 restart 보관.
- **CPU 리팩토링**: 단계마다 **기존 코드와 비교** + 기존 단위테스트(`tests/unit`) 통과.
- **GPU(M4)**: 점단위 1e-10 **폐기**. 대신 (1) 단일스텝 선형솔버는 상대오차 수렴 확인, (2) 누적/난류는 **통계량 일치**(u_rms⁺·Re_tau·Nu·Θ_c가 통계 오차 내), CPU/기존과 비교.
- **restart(U6)**: 기능 필수. **기존 포맷 호환을 목표** (global C-order double; 과거 X-fastest 변환 이력 참조). 호환이 과한 작업이면 신포맷으로 전환(호환 포기) — 단 restart 자체는 반드시 동작.
- ⚠ 통계·진단은 **전역 nx·ny 정규화** (과거 16배 버그 재발 방지).

---

## 11. 결정 완료 사항 요약

1. 산출물 `libmpmstd`(cpu+gpu 한 번에), main이 링크해 `_cpu`/`_gpu` 직접 호출
2. 자료구조=클래스, 연산=자유함수
3. **CPU/GPU 분리 타입**(C1=b): `Cpu*`/`Gpu*` (Field·Bands·System); host 메타는 단일
4. **CUDA-aware MPI 필수 · 1 rank=1 GPU**(C2; cudaSetDevice, 다중 rank/GPU 금지)
5. **halo는 명시적 자유함수** `exchange_halo_{cpu,gpu}`, solve 둘레에 노출(C3)
6. A=`Bands`(3→5중대각), 방정식=`*System`; equation별 assemble+solve, solve는 `solve/` 사용
7. 백엔드 `_cpu`/`_gpu` 접미사 명시, 끝까지 고정
8. **분기 없음·쓰는 것만 할당**; momentum T 비의존, const/var 점성 분리
9. **`solve_momentum`이 U,V,W,dU,dV,dW+블록커플링 한 번에**(M2, MPM-STD fortran 구조)
10. **BC input 주도·BC-agnostic**(§9c); Poisson 변환 BC서 선택; cavity 검증(P3b); sweep order 자동
11. 확장 물리·forcing은 `physics/` 모디파이어; Stats/Io는 `post/`; 공용 드라이버는 `driver/`
12. tests는 별도 디렉토리(M3); GPU 정합은 통계 기준(점단위 1e-10 폐기, M4)
13. P-0.5 인터페이스 스파이크 후 P0 일괄 스켈레톤
14. **최종 = P8: Fig 7(OB-DHVC)+Fig 9(NOB-RBC) multi-GPU 재현**
15. **구조 재설계(완료)**: time loop을 §4 묶음으로 — `Domain`(격자·MPI·sub·**tdma 참조**), `BoundaryCondition`(분리), `Fields`(typed-key `Var`/`Const`; **Field=공간변수, Constant=스칼라**, 쓰는 것만 등록), `MomentumSystem`이 증분 `dU,dV,dW` 소유. 모든 연산 `op(domain,[bc,]fields,system,dt)`. **추가 힘은 main 합성**(assemble은 force-agnostic). **GPU-resident**(전부 device, fileout만 host). 줄임말 금지(풀네임).
