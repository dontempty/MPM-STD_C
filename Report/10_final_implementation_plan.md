# MPM-STD (C++) 최종 구현 계획서

> 목적: [01](01_CaNS_analysis.md)~[09](09_pressure_solver_design.md) 보고서의 결정 사항을 종합하여, *실제로 코드를 작성한다는 가정* 하에 구현 순서·디렉토리 구조·각 영역의 책임·연결 관계를 명세
> 1 차 목표: PaScaL_TCS 의 NOB Rayleigh-Bénard (Ra=100, Pr=1, 512×128×256, 10 step) 결과 재현
> 2 차 목표: 동일 라이브러리로 Channel forced convection 풀이
> 확장 슬롯: LES / IBM / 다종 스칼라 / Filtered_TDMA 백엔드 / GPU 백엔드 / pybind11

---

## 0. 핵심 설계 결정 요약 (보고서 1–9 통합)

| # | 결정 | 출처 |
|---|---|---|
| 1 | **라이브러리 + 케이스별 main.cpp** (단일 Solver 클래스 폐기) | [08](08_design_revision_v2.md) §1 |
| 2 | **BC = Periodic/Dirichlet/Neumann** 3 종 1 차 지원, Wall/Inflow/Outflow enum 슬롯만 | [05](05_BC_design.md) §1 |
| 3 | **`Problem` 객체** 가 BC + topology 묶음, RBC 기본값 자동 (z=wall) | [05](05_BC_design.md) §3 |
| 4 | **`VectorField` (face) + `ScalarField` (cell)** 컨벤션, StagLocation enum 폐기 | [08](08_design_revision_v2.md) §2 |
| 5 | **`stencil/` 자유함수** 가 인덱스 산술 캡슐화 | [08](08_design_revision_v2.md) §2.2 |
| 6 | **Crank-Nicolson + ADI 단일 시간적분기**, Strategy 폐기 | [08](08_design_revision_v2.md) §4 |
| 7 | **PropertyPolicy + SourceTerm 합성** — RBC/Channel 운동량 단일화 | [07](07_momentum_unification.md) §C |
| 8 | **Plugin (Phase enum)** — LES/IBM/Stats/Probe 통합 | [08](08_design_revision_v2.md) §3.1 |
| 9 | **TdmaSolver 추상 + PascalTdmaBackend** 1 차 구현 (FilteredTdmaBackend 미래) | [08](08_design_revision_v2.md) §B.7 |
| 10 | **BC-aware FFT/DCT/DST 자동 선택** — `Problem` 에서 도출 | [09](09_pressure_solver_design.md) §2 |
| 11 | **sweep order 자동** — periodic 먼저, wall 마지막 | [05](05_BC_design.md) §6 |
| 12 | **`dPhat = 2·δP − δP_prev` 외삽** (NOB 정확도) | [04](04_PaScaL_TCS_analysis.md) §4.3 |
| 13 | **검증** = PaScaL_TCS golden 회귀 (Ra=100, 10 step, L∞<1e-10) | [04](04_PaScaL_TCS_analysis.md) §11 |
| 14 | **의존성** = MPI + FFTW3 + PaScaL_TDMA_C (1 차) | 사용자 결정 |
| 15 | **빌드** = Makefile (CMake 안 씀, HDF5 안 씀) | 사용자 결정 |

---

## 1. 전체 디렉토리 구조

```
MPM-STD(C++)/
├── Makefile                              ← 라이브러리 + 모든 apps 빌드
├── Makefile.inc                          ← 컴파일러·플래그·외부 경로
├── README.md
├── Report/                               ← 본 보고서 시리즈
│
├── include/mpmstd/                       ← 공개 헤더 (라이브러리 API)
│   ├── all.hpp                          ← 사용자 main 이 한 줄로 include
│   ├── types.hpp                        ← real_t, int_t typedef
│   ├── direction.hpp                    ← Direction/Side/Component enum
│   │
│   ├── runtime/
│   │   ├── mpi_context.hpp              ← MPI_Init/Finalize wrapper
│   │   ├── mpi_topology.hpp             ← 3D Cart + 1D/2D sub-comm
│   │   ├── subdomain.hpp                ← 인덱스 범위, halo DDT, 전치 DDT
│   │   ├── backend.hpp                  ← Backend 추상 (CPU/CUDA)
│   │   ├── cpu_backend.hpp              ← 1차 구현
│   │   └── logger.hpp                   ← rank-aware logging
│   │
│   ├── config.hpp                        ← TOML/INI 파서
│   │
│   ├── grid.hpp                          ← x, dx, dmx (축별), tanh stretching
│   │
│   ├── field/
│   │   ├── scalar_field.hpp             ← cell-centered, halo 1, host+device
│   │   ├── vector_field.hpp             ← 3 컴포넌트 (FaceX/Y/Z)
│   │   └── field_registry.hpp           ← name → field 룩업, 동적 alloc
│   │
│   ├── stencil/                          ← 인덱스 산술 캡슐화 (Part 2.2 [08])
│   │   ├── staggered.hpp                ← face↔cell 보간, gradient, divergence
│   │   └── viscous.hpp                  ← μ·∇²u 의 face-staggered harmonic mean
│   │
│   ├── boundary/
│   │   ├── bc_kind.hpp                  ← BcKind enum (3 종 + 슬롯)
│   │   ├── face_bc.hpp                  ← FaceBc + BcValueFn
│   │   ├── field_boundary.hpp           ← 6 면 컨테이너
│   │   ├── domain_topology.hpp          ← axis topology + sweep_order
│   │   ├── problem.hpp                  ← Problem (RBC 기본값 자동)
│   │   └── boundary_applier.hpp         ← apply_ghost + modify_tdma_row
│   │
│   ├── numerics/
│   │   ├── tdma_solver.hpp              ← 추상 인터페이스
│   │   ├── pascal_tdma_backend.hpp      ← PaScaL_TDMA_C 래퍼
│   │   ├── filtered_tdma_backend.hpp    ← 미래 (헤더만 1차)
│   │   ├── tdma_registry.hpp            ← 축별 백엔드 보관
│   │   ├── eigenvalues.hpp              ← R2C/DCT/DST 별 λ_k (Part §3 [09])
│   │   ├── fft_planner.hpp              ← BC-aware 변환 dispatch
│   │   └── transpose_plan.hpp           ← MPI_Alltoallw + DDT
│   │
│   ├── physics/
│   │   ├── property_policy.hpp          ← 추상
│   │   ├── constant_properties.hpp      ← Channel 용 (μ=1, 1/ρ=1)
│   │   ├── nob_properties.hpp           ← RBC 용 (T-의존 다항식)
│   │   ├── les_properties.hpp           ← 미래 (μ_eff = μ + ρν_t)
│   │   ├── source_term.hpp              ← 추상
│   │   ├── nob_buoyancy.hpp             ← 비선형 (PaScaL_TCS 식)
│   │   ├── boussinesq_buoyancy.hpp      ← 선형 (MPM-STD Fortran 식)
│   │   └── bulk_forcing.hpp             ← Channel bulk dp/dx
│   │
│   ├── momentum/
│   │   └── rhs_builders.hpp             ← convection/viscous/pgrad 자유함수 (Part 4.4 [08])
│   │
│   ├── equations/
│   │   ├── momentum_equation.hpp        ← CN+ADI: predict + block_couple + pseudo_update
│   │   ├── pressure_equation.hpp        ← compute_rhs + solve + project + dPhat
│   │   └── scalar_equation.hpp          ← 온도/스칼라 공통 (NameTrait)
│   │
│   ├── plugins/
│   │   ├── plugin.hpp                   ← 추상 + Phase enum
│   │   ├── statistics_plugin.hpp        ← mean·rms·Nusselt 누적
│   │   ├── probe_plugin.hpp             ← 점 단위 출력
│   │   ├── ibm_plugin.hpp               ← 슬롯만 (1차 throw stub)
│   │   └── les_plugin.hpp               ← 슬롯만
│   │
│   ├── utilities/
│   │   ├── cfl_controller.hpp           ← dt 적응
│   │   ├── diagnostics.hpp              ← div check, CFL monitor, Nu 등
│   │   └── timer.hpp                    ← 프로파일링
│   │
│   └── io/
│       ├── restart_io.hpp               ← MPI-IO binary 체크포인트
│       ├── instant_io.hpp               ← 순간 필드 dump
│       └── stats_io.hpp                 ← 통계 dump
│
├── src/                                  ← 라이브러리 구현 (.cpp)
│   └── (각 .hpp 와 1:1 대응)
│
├── apps/                                 ← 케이스별 실행파일
│   ├── rbc/
│   │   ├── main.cpp                     ← RBC 조립 + 시간루프
│   │   ├── input.toml
│   │   └── Makefile                     ← 라이브러리에 링크
│   ├── channel/
│   │   ├── main.cpp                     ← Channel 조립
│   │   ├── input.toml
│   │   └── Makefile
│   ├── thermal_only_check/              ← 검증용 (M2)
│   │   └── main.cpp                     ← 속도 동결 + manufactured solution
│   └── poisson_only_check/              ← 검증용 (M4)
│       └── main.cpp
│
├── tests/
│   ├── unit/                             ← 라이브러리 단위 테스트
│   │   ├── test_grid.cpp
│   │   ├── test_field.cpp
│   │   ├── test_stencil.cpp
│   │   ├── test_problem_defaults.cpp
│   │   ├── test_sweep_order.cpp
│   │   ├── test_bc_apply.cpp
│   │   ├── test_tdma_backend.cpp
│   │   ├── test_eigenvalues.cpp
│   │   ├── test_fft_planner.cpp
│   │   └── test_transpose_plan.cpp
│   ├── integration/
│   │   ├── test_thermal_manufactured.cpp
│   │   ├── test_poisson_analytic.cpp
│   │   └── test_momentum_one_step.cpp
│   └── regression/
│       ├── golden/                       ← PaScaL_TCS / MPM-STD Fortran 골든
│       │   ├── pascal_tcs_Ra100_Pr1_10step.bin
│       │   └── mpm_std_fortran_channel.bin
│       └── compare.py                   ← reassemble + L∞ 비교
│
├── python/                               ← pybind11 바인딩 (미래)
│   ├── pyproject.toml
│   └── bindings.cpp
│
├── run/                                  ← 실행 스크립트
│   ├── rbc.sh                           ← SLURM batch (4-rank 등)
│   └── channel.sh
│
└── build/                                ← 빌드 산출물 (gitignore)
    ├── obj/
    └── bin/                              ← rbc, channel, *_check 실행파일
```

---

## 2. 디렉토리별 책임·연결 관계

### 2.1 `runtime/` — 인프라 계층 (case 무관)

| 파일 | 책임 | 의존 | 의존받음 |
|---|---|---|---|
| `mpi_context.hpp` | MPI_Init/Finalize, world rank/size | MPI | mpi_topology |
| `mpi_topology.hpp` | 3D Cart 생성, 1D sub-comm × 3, 2D sub-comm (FFT 전치용) | mpi_context | subdomain, tdma, fft, transpose |
| `subdomain.hpp` | local 인덱스 범위, halo DDT, C↔I↔K 전치 DDT, `exchange_halo()` | mpi_topology, grid | 거의 모든 equation |
| `backend.hpp` | 추상 인터페이스: `alloc`, `to_device`, `to_host`, kernel dispatch | — | field, equations |
| `cpu_backend.hpp` | 단순 std::vector 기반 구현 | backend | (위와 동일) |
| `logger.hpp` | rank 0 만 출력, level 별 분기 | — | (모든 곳에서 호출 가능) |

**핵심 원칙**: MPI 호출은 이 계층 안에서만. `equations/`, `physics/` 는 MPI 를 *직접 호출하지 않음*.

### 2.2 `config.hpp` — 입력 파싱

- TOML 파일 → `Config` 객체
- `cfg.get<double>("Ra"), cfg.get<int>("nx")` 등 타입 안전 getter
- 모든 의존자가 const ref 로 받음 (글로벌 싱글톤 금지)

### 2.3 `grid.hpp` — 좌표·메트릭

- 축별 `x[d]`, `dx[d]`, `dmx[d]` 배열 (cell center + face)
- tanh stretching (`gamma1/2/3` 파라미터)
- 비-uniform wall 축 지원

**연결**:
- `Subdomain` 이 local extent 결정에 사용
- `ScalarField`/`VectorField` 가 alloc 크기 결정에 사용
- `stencil/` 함수가 dx, dmx 인자로 받음
- `FftPlanner` 가 eigenvalue 계산에 사용

### 2.4 `field/` — 데이터 저장

| 파일 | 책임 |
|---|---|
| `scalar_field.hpp` | cell-centered 3D 배열 + halo 1 + host/device 포인터 |
| `vector_field.hpp` | 3 개의 ScalarField, 각자 다른 face 위치, 다른 크기 |
| `field_registry.hpp` | 이름→필드 매핑, 동적 add/lookup |

**핵심**:
- `StagLocation` enum **없음** (사용자 결정)
- `VectorField::x()` 가 FaceX, `.y()` FaceY, `.z()` FaceZ — 컨벤션 고정
- 인덱스 안전은 `stencil/` 가 담당, 타입 시스템 아님

**연결**:
- `Subdomain` 의 local extent 사용
- `Backend` 가 alloc 담당
- `equations/`, `plugins/` 가 read/write

### 2.5 `stencil/` — 인덱스 산술 캡슐화

| 함수 | 의미 |
|---|---|
| `dudx_at_cell(Ux, g, i, j, k)` | face-centered U → cell 의 ∂U/∂x |
| `dpdx_at_face_x(P, g, i, j, k)` | cell-centered P → +x face 의 ∂P/∂x |
| `dudy_at_face_y(Ux, g, i, j, k)` | cross derivative |
| `divergence_at_cell(U, g, i, j, k)` | ∇·u |
| `harmonic_mean_face(μ, g, i, j, k, axis)` | viscosity face-staggered 보간 |
| `laplacian_u_at_face_x(Ux, μ, g, i, j, k)` | viscous Laplacian |

**모든 인덱스 산술이 이 헤더에만 존재**. 단위 테스트 1 회로 모든 호출자 보호.

**연결**:
- `equations/momentum_equation.cpp` 가 RHS·LHS 빌드에 사용
- `equations/scalar_equation.cpp` 가 convection/diffusion 에 사용
- `equations/pressure_equation.cpp` 가 발산·gradient 계산에 사용
- `momentum/rhs_builders.hpp` 가 호출

### 2.6 `boundary/` — BC 시스템 ([05](05_BC_design.md))

```
BcKind (enum) ← FaceBc ← FieldBoundary ← Problem ← BoundaryApplier
                                            ↑
                                      DomainTopology
```

| 파일 | 책임 |
|---|---|
| `bc_kind.hpp` | `enum { Periodic, Dirichlet, Neumann, Wall, Inflow, Outflow }` (뒤 3 개는 슬롯) |
| `face_bc.hpp` | `{ BcKind, BcValueFn }` + 정적 생성자 (`FaceBc::dirichlet(v)` 등) |
| `field_boundary.hpp` | 6 면 컨테이너, `face(d, side)` |
| `domain_topology.hpp` | 축별 periodic 여부, `sweep_order()`, `wall_axis()` |
| `problem.hpp` | `Problem` 클래스. 생성자가 RBC 기본값 (z=wall) 자동. `validate()`, `disable_thermal()`. |
| `boundary_applier.hpp` | `apply_ghost(Field, FieldBoundary, t)`, `modify_tdma_row(d, fbc, A,B,C,D, ...)` |

**연결**:
- `Problem` 은 거의 모든 `equations/` 가 const ref 로 받음
- `FftPlanner` 가 `Problem` 에서 변환 종류 도출
- `MomentumEquation`, `ScalarEquation` 의 ADI sweep order 가 `topology.sweep_order()` 에서
- `BoundaryApplier` 는 ghost exchange 후 매번 호출

### 2.7 `numerics/` — 선형 해법 도구

| 파일 | 책임 |
|---|---|
| `tdma_solver.hpp` | 추상: `solve_many`, `solve_many_cyclic` |
| `pascal_tdma_backend.hpp` | PaScaL_TDMA_C 래핑. plan 캐시 (n_sys 별) |
| `tdma_registry.hpp` | 축별 백엔드 보관, `get(Direction)` |
| `eigenvalues.hpp` | 자유함수: `eigvals_periodic`, `eigvals_neumann`, `eigvals_dirichlet` |
| `fft_planner.hpp` | `Problem` 받아 축별 변환 종류 결정, FFTW3 plan 생성, eigenvalues 보관 |
| `transpose_plan.hpp` | C↔I, I↔K, K↔I, I↔C. `MPI_Alltoallw` + DDT |

**연결**:
- `TdmaRegistry` 는 `MomentumEquation`, `ScalarEquation`, `PressureEquation` 모두 사용
- `FftPlanner` 는 `PressureEquation` 만 사용
- `TransposePlan` 도 `PressureEquation` 만

### 2.8 `physics/` — Policy 클래스 ([07](07_momentum_unification.md) §C)

| 파일 | 책임 |
|---|---|
| `property_policy.hpp` | 추상: `update(fr)`, `mu()`, `invRho()` |
| `constant_properties.hpp` | μ=1, 1/ρ=1. `update()` no-op |
| `nob_properties.hpp` | T 의존 다항식. 매 step `update(T)` |
| `les_properties.hpp` | 미래 — base 위에 ν_t 누적 |
| `source_term.hpp` | 추상: `target_component()`, `add_to_rhs(c, fr, props, RHS, idx)` |
| `nob_buoyancy.hpp` | `Cmt·(Tc + a12·Tc²·ΔT)·invRho` (PaScaL_TCS) |
| `boussinesq_buoyancy.hpp` | `Cmt·(Tc − Tg)·invRho` (MPM-STD Fortran) |
| `bulk_forcing.hpp` | `-presgrad` (Channel) |

**연결**:
- `MomentumEquation` 이 `PropertyPolicy&` + `std::vector<SourceTerm*>` 를 생성자로 받음
- 매 step 시작 시 `props.update(fr)` 호출 (T 의존이면 μ 갱신)
- RHS 빌드 시 모든 SourceTerm 의 `add_to_rhs` 누적

### 2.9 `momentum/` + `equations/` — 운동량 방정식 ([07](07_momentum_unification.md))

```
momentum/rhs_builders.hpp (자유함수)
       ↓ 호출
equations/momentum_equation.hpp (캡슐화된 시간 적분기)
```

| 파일 | 책임 |
|---|---|
| `momentum/rhs_builders.hpp` | `assemble_convection_rhs_at(c, U, g, i,j,k)`, `assemble_viscous_rhs_at(c, U, props, g, i,j,k)`, `assemble_pressure_gradient_rhs_at(c, P, props, g, i,j,k)` — 자유함수. 시간 스킴 무관. |
| `equations/momentum_equation.hpp` | CN+ADI 시간 적분기. `compute_coeffi`, `predict(c, dt)`, `block_couple_V/U`, `pseudo_update`. PaScaL_TCS `solvedU/V/W + blockLdU/V + pseudoupdate` 와 1:1 대응. |

**연결**:
- `MomentumEquation` ← Grid, Subdomain, FieldRegistry, Problem, TdmaRegistry, BoundaryApplier, PropertyPolicy, std::vector<SourceTerm*>
- `MomentumEquation` → `momentum/rhs_builders` 호출 → `stencil/` 호출
- 내부에서 ADI sweep order = `problem.topology.sweep_order()`
- 각 stage 의 TDMA 는 `tdma.get(d).solve_many[_cyclic]` 호출

### 2.10 `equations/pressure_equation.hpp` — 압력 ([09](09_pressure_solver_design.md))

| 메서드 | 책임 |
|---|---|
| `compute_rhs(fr, dt)` | `PRHS = (1/Δt)·∇·u* + NOB 보정` |
| `solve(fr)` | C→I 전치, x 변환, I→K 전치, y 변환, 모드별 TDMA, 역변환, 역전치, 평균 제거 |
| `project(fr, dt)` | `u^{n+1} = u* − Δt·(1/ρ)·∇δP`, `P += δP`, `dP_prev = δP` |
| `mean_remove_if_singular(dP)` | 모든 면 Periodic/Neumann 이면 평균 0 강제 |

**연결**:
- `PressureEquation` ← Grid, Subdomain, FieldRegistry, Problem, TdmaRegistry, FftPlanner, BoundaryApplier
- `FftPlanner` 는 `Problem` 에서 자동으로 R2C/DCT/DST 결정
- `dPhat` 외삽은 `project` 안에서 `2·δP − δP_prev` 로

### 2.11 `equations/scalar_equation.hpp` — 온도·스칼라

- `ScalarTraits { name, diffusivity, source_fn? }` 받아 생성
- 3-stage ADI (운동량과 동일 구조, 단순함)
- 다종 스칼라는 `ScalarEquation` 인스턴스 여러 개

### 2.12 `plugins/` — 확장 인터페이스 ([08](08_design_revision_v2.md) §3.1)

```cpp
enum class Phase { Setup, Finalise, PreStep, BeforeMomentum, BeforePressure, PostStep };
class Plugin {
  virtual Phase phase() const = 0;
  virtual void call(/*ctx*/) {}
  virtual void setup(/*ctx*/) {}
  virtual void finalise(/*ctx*/) {}
  int every_nsteps = 1;
};
```

main.cpp 가 phase 별로 직접 호출 (라이브러리가 자동 호출 안 함).

| 플러그인 | Phase | 책임 |
|---|---|---|
| `StatisticsPlugin` | PostStep | mean·rms·Nu 누적 |
| `ProbePlugin` | PostStep | 점 단위 dump |
| `IbmPlugin` | BeforePressure | 슬롯만 (throw stub) |
| `LesPlugin` | PreStep | 슬롯만 |

### 2.13 `utilities/` + `io/` — 보조 도구

| 파일 | 책임 |
|---|---|
| `cfl_controller.hpp` | `next_dt(fr, g)` — max(\|u\|/dx) 기반 |
| `diagnostics.hpp` | `check_divergence`, `monitor_print`, `nusselt` |
| `timer.hpp` | section 별 wallclock |
| `restart_io.hpp` | MPI-IO binary read/write |
| `instant_io.hpp` | 순간 필드 dump |
| `stats_io.hpp` | StatisticsPlugin 출력 |

### 2.14 `apps/<case>/main.cpp` — 케이스별 솔버

[08](08_design_revision_v2.md) §1.4-1.5 참고. RBC 와 Channel main 의 차이는 다음 표:

| 항목 | RBC | Channel |
|---|---|---|
| 활성 필드 | U, dU, P, dP, **T** | U, dU, P, dP |
| PropertyPolicy | `NobProperties` | `ConstantProperties` |
| SourceTerm | `NobBuoyancy(W)` | `BulkForcing(U)` |
| Equations | thermal + momentum + pressure | momentum + pressure |
| Time loop | thermal → momentum → pressure | momentum → pressure |

---

## 3. 의존성 그래프

```
                          types, direction
                                 │
                  ┌──────────────┼──────────────────┐
                  │              │                  │
              config           mpi_context        logger
                  │              │
                  │         mpi_topology
                  │              │
                  │         subdomain
                  │              │
                  │              ├─→ backend ─→ cpu_backend
                  ├──────────────┤
                  │              ├─→ grid
                  │              │     │
                  │              │     ├─→ stencil/staggered
                  │              │     │
                  │              │     ├─→ field/scalar_field
                  │              │     ├─→ field/vector_field
                  │              │     └─→ field/field_registry
                  │              │           │
                  │              │           │
                  ├──────────────┴───────────┤
                  │                          │
              boundary/                  numerics/
                  │                          │
              bc_kind                    tdma_solver
                  ↓                          ↓
              face_bc                    pascal_tdma_backend
                  ↓                          ↓
              field_boundary             tdma_registry
                  ↓                          ↓
              domain_topology            eigenvalues
                  ↓                          ↓
              problem ─────────┐         fft_planner
                  ↓            │             ↓
              boundary_applier │         transpose_plan
                       │       │             │
                       │       │             │
                       ↓       ↓             ↓
                ┌──── physics/  ──── momentum/rhs_builders ───┐
                │                                              │
        property_policy            source_term                │
        constant/nob/les           nob/boussinesq/bulk        │
                │                       │                      │
                └───────┬───────────────┘                      │
                        │                                      │
                        ↓                                      ↓
                ┌── equations/ ────────────────────────────────┤
                │                                              │
        scalar_equation       momentum_equation       pressure_equation
                │                     │                     │
                └─────────────────────┴─────────────────────┘
                                      │
                       ┌──────────────┼──────────────┐
                       ↓              ↓              ↓
                  plugins/        utilities/        io/
                  (Phase enum)    (cfl, diag)       (restart, instant, stats)
                       │              │              │
                       └──────────────┴──────────────┘
                                      │
                                      ↓
                          apps/<case>/main.cpp
```

→ 위쪽이 아래쪽에 의존. 화살표 역방향 호출 금지.

---

## 4. 구현 순서 (M0 → M8)

각 마일스톤은 **테스트 통과** 가 DoD. 다음 단계로 진행 전에 unit/integration 테스트가 모두 green 이어야 함.

### M0 — 인프라 + 데이터 모델 (2 주 예상)

**파일** (생성 순서):
```
include/mpmstd/types.hpp
include/mpmstd/direction.hpp
include/mpmstd/config.hpp                       + src/config.cpp
include/mpmstd/runtime/logger.hpp               + src/logger.cpp
include/mpmstd/runtime/mpi_context.hpp          + src/mpi_context.cpp
include/mpmstd/runtime/mpi_topology.hpp         + src/mpi_topology.cpp
include/mpmstd/runtime/backend.hpp
include/mpmstd/runtime/cpu_backend.hpp          + src/cpu_backend.cpp
include/mpmstd/runtime/subdomain.hpp            + src/subdomain.cpp
include/mpmstd/grid.hpp                         + src/grid.cpp
include/mpmstd/field/scalar_field.hpp           + src/scalar_field.cpp
include/mpmstd/field/vector_field.hpp           + src/vector_field.cpp
include/mpmstd/field/field_registry.hpp         + src/field_registry.cpp
include/mpmstd/stencil/staggered.hpp            (header-only)
include/mpmstd/io/restart_io.hpp                + src/restart_io.cpp
Makefile + Makefile.inc
tests/unit/test_grid.cpp
tests/unit/test_field.cpp
tests/unit/test_stencil.cpp
tests/unit/test_subdomain_halo.cpp
```

**DoD**:
- 8 랭크 (2×2×2) 실행, TOML 입력 파싱
- `ScalarField`, `VectorField` alloc + halo exchange 검증
- `stencil::divergence_at_cell` 단위 테스트 EOC=2
- restart write/read 왕복 검증
- ghost cell exchange unit test 통과

### M1 — BC + TDMA (2 주)

**파일**:
```
include/mpmstd/boundary/bc_kind.hpp
include/mpmstd/boundary/face_bc.hpp             + src/face_bc.cpp
include/mpmstd/boundary/field_boundary.hpp
include/mpmstd/boundary/domain_topology.hpp     + src/domain_topology.cpp
include/mpmstd/boundary/problem.hpp             + src/problem.cpp
include/mpmstd/boundary/boundary_applier.hpp    + src/boundary_applier.cpp
include/mpmstd/numerics/tdma_solver.hpp
include/mpmstd/numerics/pascal_tdma_backend.hpp + src/pascal_tdma_backend.cpp
include/mpmstd/numerics/tdma_registry.hpp       + src/tdma_registry.cpp
tests/unit/test_problem_defaults.cpp
tests/unit/test_sweep_order.cpp
tests/unit/test_bc_apply.cpp
tests/unit/test_tdma_backend.cpp
```

**DoD**:
- `Problem p;` 한 줄로 RBC 기본 (z=wall) 자동 채움
- `sweep_order()` 가 RBC/PaScaL_TCS/Cavity 모두 올바른 순서 반환
- PaScaL_TDMA_C wrap 의 cyclic / non-cyclic 1D 시스템이 알려진 해와 일치
- 면별 ghost 채움이 Periodic/Dirichlet/Neumann 모두 정확

### M2 — 스칼라 (열) 방정식 (2 주)

**파일**:
```
include/mpmstd/equations/scalar_equation.hpp    + src/scalar_equation.cpp
include/mpmstd/physics/property_policy.hpp
include/mpmstd/physics/constant_properties.hpp  + src/constant_properties.cpp
include/mpmstd/physics/nob_properties.hpp       + src/nob_properties.cpp
apps/thermal_only_check/main.cpp                ← 검증용 앱
tests/integration/test_thermal_manufactured.cpp
```

**DoD**:
- 속도 동결 + manufactured solution
- 시·공간 EOC = 2
- 1, 4, 16 랭크 결과 bit-equivalent (또는 1e-12 이내)
- PaScaL_TCS 의 `mpi_thermal_solver` 와 1 step 후 L∞ < 1e-12

### M3 — 운동량 방정식 (3 주)

**파일**:
```
include/mpmstd/momentum/rhs_builders.hpp        + src/rhs_builders.cpp
include/mpmstd/physics/source_term.hpp
include/mpmstd/physics/nob_buoyancy.hpp         + src/nob_buoyancy.cpp
include/mpmstd/physics/boussinesq_buoyancy.hpp  + src/boussinesq_buoyancy.cpp
include/mpmstd/physics/bulk_forcing.hpp         + src/bulk_forcing.cpp
include/mpmstd/equations/momentum_equation.hpp  + src/momentum_equation.cpp
tests/integration/test_momentum_one_step.cpp
```

**DoD**:
- `predict(c, dt)` × 3 + `block_couple_V/U` + `pseudo_update` 의 1 step 결과
- 같은 입력으로 PaScaL_TCS 의 `solvedU/V/W + blockLdU/V + pseudoupdate` 와 L∞ < 1e-12
- RBC (NobProperties + NobBuoyancy) / Channel (ConstantProperties + BulkForcing) 둘 다 정상 작동

### M4 — 압력 Poisson (3 주, 가장 어려움)

**파일** (M4.1 → M4.7 [09](09_pressure_solver_design.md) §11 참고):
```
include/mpmstd/numerics/eigenvalues.hpp         (header-only)
include/mpmstd/numerics/fft_planner.hpp         + src/fft_planner.cpp
include/mpmstd/numerics/transpose_plan.hpp      + src/transpose_plan.cpp
include/mpmstd/equations/pressure_equation.hpp  + src/pressure_equation.cpp
apps/poisson_only_check/main.cpp
tests/unit/test_eigenvalues.cpp
tests/unit/test_fft_planner.cpp
tests/unit/test_transpose_plan.cpp
tests/integration/test_poisson_analytic.cpp
```

**DoD** (단계별 검증):
- M4.1: `eigenvalues.hpp` 각 BC 별 EOC = 2
- M4.2: FFT/DCT/DST forward·backward identity
- M4.3: TransposePlan 역전치 = identity
- M4.4: PRHS = ∇·u* 가 알려진 비-발산장에서 0
- M4.5: 해석해 Poisson 에서 L∞ < 1e-10
- M4.6: 모든 면 Neumann 케이스 평균 = 0
- M4.7: `project` 후 div(U) < 1e-12, dP 가 PaScaL_TCS 와 1e-10 이내

### M5 — RBC 통합 (2 주)

**파일**:
```
include/mpmstd/utilities/cfl_controller.hpp     + src/cfl_controller.cpp
include/mpmstd/utilities/diagnostics.hpp        + src/diagnostics.cpp
include/mpmstd/utilities/timer.hpp              + src/timer.cpp
include/mpmstd/plugins/plugin.hpp
include/mpmstd/plugins/statistics_plugin.hpp    + src/statistics_plugin.cpp
include/mpmstd/plugins/probe_plugin.hpp         + src/probe_plugin.cpp
include/mpmstd/io/instant_io.hpp                + src/instant_io.cpp
include/mpmstd/io/stats_io.hpp                  + src/stats_io.cpp
apps/rbc/main.cpp
apps/rbc/input.toml
apps/rbc/Makefile
tests/regression/compare.py
tests/regression/golden/pascal_tcs_Ra100_Pr1_10step.bin  ← PaScaL_TCS 로 생성
```

**DoD**:
- RBC 앱이 10 step 실행 완료
- PaScaL_TCS 골든과 L∞(T,U,V,W) < 1e-10, L∞(P) < 1e-9
- 1/4/16 랭크 모두 일치
- restart write/read 왕복 → 1 step 진행 → 동일 결과

### M6 — Channel 통합 (1 주)

**파일**:
```
apps/channel/main.cpp
apps/channel/input.toml
apps/channel/Makefile
```

**DoD**:
- forced Channel Re_τ=180 표준 케이스
- 평균 속도 프로파일이 Kim-Moin-Moser 데이터와 일치
- 1 라이브러리 코드 그대로, main.cpp 만 다름

### M7 — Plugin 슬롯 + IBM 골격 (1 주)

**파일**:
```
include/mpmstd/plugins/ibm_plugin.hpp           + src/ibm_plugin.cpp (1차 throw)
include/mpmstd/plugins/les_plugin.hpp           + src/les_plugin.cpp (1차 throw)
include/mpmstd/physics/les_properties.hpp       (헤더만)
```

**DoD**:
- Plugin 인터페이스 안정성 확인 (인자 변경 없이 미래 구현 가능)
- IBM/LES 호출 시 throw 메시지 명확

### M8 — 미래 (선택)

- `FilteredTdmaBackend` (백엔드 교체 가능성 검증)
- `CudaBackend` (GPU)
- `pybind11` 바인딩
- 다종 스칼라 (실은 `ScalarEquation` 추가 인스턴스라 거의 자동)

---

## 5. 외부 의존성과 빌드

### 5.1 의존성

| 의존 | 버전 | 용도 |
|---|---|---|
| MPI | OpenMPI ≥ 4.0 / MPICH ≥ 3.4 | 도메인 분할 + 통신 |
| FFTW3 | ≥ 3.3.10 | R2C FFT + DCT-II (`FFTW_REDFT10`) + DST-II (`FFTW_RODFT10`) |
| PaScaL_TDMA_C | 동봉 (`/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/`) | 분산 TDMA |
| C++ 표준 | C++17 | std::optional, std::variant, [[nodiscard]] 등 |
| 컴파일러 | g++ ≥ 9 / clang++ ≥ 12 / icpx | — |

### 5.2 `Makefile.inc`

```make
CXX        := mpicxx
CXXSTD     := -std=c++17
OPT        := -O3 -march=native -fno-fast-math -fno-associative-math
WARN       := -Wall -Wextra -Wpedantic

PASCAL_TDMA_DIR := /shared/home/wel1come1234/workspace/PaScaL_TDMA_C
FFTW_DIR        := /usr

INCS  := -Iinclude -I$(PASCAL_TDMA_DIR)/src -I$(FFTW_DIR)/include
LIBS  := -L$(PASCAL_TDMA_DIR)/build -lpascal_tdma \
         -L$(FFTW_DIR)/lib -lfftw3 -lm

DEPFLAGS = -MMD -MP
```

### 5.3 `Makefile` 의 역할

```
make lib            # libmpmstd.a 빌드
make app-rbc        # apps/rbc/ 빌드 (라이브러리에 링크)
make app-channel    # apps/channel/ 빌드
make tests          # 모든 unit + integration 빌드 + 실행
make regression     # PaScaL_TCS golden 비교 실행
make clean
```

각 app/ 디렉토리는 자기 Makefile 을 갖고 라이브러리에 링크.

---

## 6. 검증 전략

### 6.1 3 계층 검증

| 계층 | 무엇 | 기준 |
|---|---|---|
| **Unit** | 각 헤더 1 개씩 격리 테스트 | 알려진 입력에 대한 정확한 출력 |
| **Integration** | 여러 모듈 결합 (예: thermal 단독, poisson 단독) | manufactured solution EOC = 2 |
| **Regression** | 전체 앱 (RBC, Channel) | PaScaL_TCS / MPM-STD Fortran golden 과 L∞ |

### 6.2 Golden 생성

- PaScaL_TCS 를 RBC 케이스 (Ra=100, Pr=1, 512×128×256, 10 step) 로 실행
- 결과 `cont_U.bin, cont_V.bin, cont_W.bin, cont_P.bin, cont_THETA.bin` 을 `tests/regression/golden/` 에 보관
- 동일 컴파일 옵션 (`-O3 -fno-fast-math`) 으로 일관성

### 6.3 비교 스크립트 `compare.py`

```python
def compare(a_path, b_path, tol):
    a = read_field_mpi_reassembled(a_path)
    b = read_field_mpi_reassembled(b_path)
    diff = np.abs(a - b)
    print(f"L∞ = {diff.max():.3e}, L2 = {np.linalg.norm(diff):.3e}")
    assert diff.max() < tol
```

### 6.4 Stage-level dump (디버깅용)

`MPMSTD_DUMP_STAGES=1` 빌드 플래그로:
- thermal → momentum predict (U/V/W) → block_couple → pseudo_update → pressure_rhs → pressure_solve → project 각 직후에 필드 dump
- PaScaL_TCS 측에도 동일 지점 패치 → 첫 발산 지점 localize

---

## 7. 위험 영역과 완화

| # | 위험 | 완화 |
|---|---|---|
| 1 | PaScaL_TDMA_C 의 plan 캐시가 leak / 잘못 재사용 | M1 단위 테스트 + 메모리 분석 (Valgrind/ASan) |
| 2 | DCT-II / DST-II 정규화 상수 오류 | M4.1 단위 테스트: forward-backward identity 후 정규화 인자 검증 |
| 3 | C↔I↔K 전치 DDT 의 오프셋 실수 | M4.3 단위 테스트: 역전치 = identity. 4 랭크에서 검증 |
| 4 | NOB 의 `dPhat = 2δP − δP_prev` 첫 step (이전 없음) | 첫 step 은 `dP_prev = 0` 으로 초기화 ([04](04_PaScaL_TCS_analysis.md) §4.3 참고) |
| 5 | wall ghost 의 zero-vs-antisymmetric 정책 | [feedback_wall_bc_zero_ghost.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_wall_bc_zero_ghost.md) 의 zero-ghost + flag-drop 채택 — `BoundaryApplier::modify_tdma_row` 가 처리 |
| 6 | cross-direction matrix 의 자기-미분 오염 | [feedback_bw_cross_direction.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_bw_cross_direction.md): cross-direction 에 `0.5·∂q/∂d` 포함 금지. `momentum/rhs_builders.hpp` 의 자유함수에서 명시 |
| 7 | RBC 의 buoyancy component 가 U vs W 혼동 | `NobBuoyancy(Component::W)` 명시. PaScaL_TCS 가 U 인 이유는 좌표축 컨벤션 — 우리는 z=wall 이므로 W. golden 비교 시 좌표축 회전 고려 |
| 8 | 평균 제거를 잊었을 때 압력 drift | `PressureEquation::solve` 끝에 `mean_remove_if_singular` 무조건 호출 |
| 9 | restart write 중 일부 랭크만 진행 | MPI-IO collective write 사용 |
| 10 | TDMA registry 의 축별 백엔드 미설정 | `TdmaRegistry` 생성자에서 모든 축 PascalTdmaBackend 디폴트 설정 |

---

## 8. 일정 추정 (혼자 작업 기준)

| 마일스톤 | 기간 | 누적 |
|---|---|---|
| M0 인프라 + 데이터 | 2 주 | 2 주 |
| M1 BC + TDMA | 2 주 | 4 주 |
| M2 ScalarEquation | 2 주 | 6 주 |
| M3 MomentumEquation | 3 주 | 9 주 |
| M4 PressureEquation | 3 주 | 12 주 |
| M5 RBC 통합 + 검증 | 2 주 | 14 주 |
| M6 Channel 검증 | 1 주 | 15 주 |
| M7 Plugin 슬롯 + 안정성 | 1 주 | 16 주 |
| **합계** | **약 4 개월** | |

→ M0–M5 가 핵심 (12–14 주). M6 는 main.cpp 작성 + 검증만이라 짧음. M7 는 슬롯만.

---

## 9. 작업 단위 — "한 번에 한 파일"

각 마일스톤 내에서도 **한 헤더 + 그 .cpp + 그 단위 테스트** 를 하나의 commit 단위로:

```
M1 의 예:
  commit 1: bc_kind.hpp
  commit 2: face_bc.hpp + face_bc.cpp + test
  commit 3: field_boundary.hpp + test
  commit 4: domain_topology.hpp/.cpp + test_sweep_order
  commit 5: problem.hpp/.cpp + test_problem_defaults
  commit 6: boundary_applier.hpp/.cpp + test_bc_apply
  commit 7: tdma_solver.hpp (인터페이스만)
  commit 8: pascal_tdma_backend.hpp/.cpp + test_tdma_backend
  commit 9: tdma_registry.hpp/.cpp
```

→ 각 commit 이 독립적으로 빌드 + 테스트 통과. 이전 단계 회귀 안 함.

[memory: feedback_make_rm_before_sbatch.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_make_rm_before_sbatch.md) 의 교훈: 회귀 테스트 전 stale output 정리 (`make rm`).

---

## 10. 결론 — 이 문서의 사용법

본 문서는 **단독 실행 계획**. 코드를 작성할 때:

1. **시작 전**: §1 디렉토리 구조 생성 (`mkdir -p include/mpmstd/{runtime,field,...}`)
2. **각 마일스톤 시작 시**: §4 의 해당 마일스톤 파일 목록 확인 → 순서대로 헤더 작성 → .cpp 구현 → 단위 테스트 작성·통과
3. **마일스톤 종료 시**: §4 의 DoD 확인 + §6 의 회귀 비교
4. **막힐 때**: §7 위험 영역 표 확인 + 해당 출처 보고서 (07, 09 등) 의 디테일 참조
5. **새 case 추가 시**: §2.14 의 RBC/Channel 차이 표 따라 `apps/<new_case>/main.cpp` 작성. 라이브러리 미변경.

이 문서로 작업 시작 가능. 추가 디테일이 필요한 부분은 해당 출처 보고서 (01–09) 참조.

---

## 부록 — 보고서 cross-reference

| 주제 | 깊이 | 보고서 |
|---|---|---|
| CaNS 의 데이터 모델 | 중 | [01](01_CaNS_analysis.md) |
| PyFR 의 Backend/Plugin 패턴 | 중 | [02](02_PyFR_analysis.md) |
| MPM-STD Fortran 의 구조 | 깊음 | [03](03_MPM-STD_Fortran_analysis.md) |
| PaScaL_TCS 의 알고리듬 디테일 | 깊음 | [04](04_PaScaL_TCS_analysis.md) |
| BC 시스템 설계 | 깊음 | [05](05_BC_design.md) |
| 단일 Solver god-class 거부 + Strategy/PropertyPolicy 도입 | 중 | [06](06_design_critique_and_revision.md) |
| 운동량 RBC/Channel 통합 | 깊음 | [07](07_momentum_unification.md) |
| 라이브러리 + apps/main.cpp + Field 단순화 + CN 단일 | 깊음 | [08](08_design_revision_v2.md) |
| PressureSolver BC-aware FFT/DCT | 깊음 | [09](09_pressure_solver_design.md) |
| **종합 구현 계획 (본 문서)** | — | **[10](10_final_implementation_plan.md)** |
