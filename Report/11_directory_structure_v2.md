# 디렉토리 구조 v2 — 폴더별 `main` 패턴

> 사용자 결정 사항 반영:
> 1. **MPI / CUDA / Backend 분리** — 여러 하위 폴더 가능
> 2. **field, boundary, linear_solver, physics, equation, integrator, post, stat 각 카테고리 폴더**
> 3. **physics 안에 LES, IBM 전용 하위 폴더**
> 4. **equation 안에 momentum, pressure, scalar 등 항목별 하위 폴더**
> 5. **post (instant·restart) 와 stat (통계) 분리**
> 6. **각 폴더에 `main.hpp` (+ 필요시 `main.cpp`)** — 폴더 안 작은 파일들을 모아 고수준 인터페이스 제공
> 7. **hpp + cpp 같은 폴더 (Convention B)**
> 8. **apps 가 라이브러리의 `main` 헬퍼들을 조립**하여 케이스별 솔버 작성. **중복 무관**.
> 9. **test, run, build 폴더 분리**
> + 보충 추가 (필자 판단): `common/`, `config/`, `grid/`, `utilities/`, `stencil/`

> 이 보고서가 [10_final_implementation_plan.md](10_final_implementation_plan.md) §1 의 디렉토리 구조를 **대체**.

---

## 1. 전체 디렉토리 트리

```
MPM-STD(C++)/
├── Makefile
├── Makefile.inc
├── README.md
├── Report/                              ← 분석·설계 보고서 (본 시리즈)
│
├── src/                                 ← 라이브러리 본체. hpp + cpp 같은 폴더.
│   │
│   ├── common/                          ← 공통 typedef·enum
│   │   ├── main.hpp                    ← facade
│   │   ├── types.hpp                   ← real_t, int_t
│   │   └── direction.hpp               ← Direction/Side/Component enum
│   │
│   ├── parallel/                        ← MPI + CUDA 관리 (사용자 요구 #1)
│   │   ├── main.hpp
│   │   ├── mpi/
│   │   │   ├── main.hpp
│   │   │   ├── mpi_context.hpp + .cpp
│   │   │   ├── mpi_topology.hpp + .cpp
│   │   │   └── subdomain.hpp + .cpp
│   │   ├── backend/
│   │   │   ├── main.hpp
│   │   │   ├── backend.hpp             ← 추상
│   │   │   ├── cpu_backend.hpp + .cpp
│   │   │   └── cuda_backend.hpp + .cpp ← 미래 (1차에 스텁만)
│   │   └── cuda/
│   │       ├── main.hpp                 ← 1차에 비어 있음
│   │       └── (미래: 메모리 풀, stream 등)
│   │
│   ├── config/                          ← 입력 파싱·로깅
│   │   ├── main.hpp
│   │   ├── config.hpp + .cpp           ← TOML/INI 파서
│   │   └── logger.hpp + .cpp           ← rank-aware logging
│   │
│   ├── grid/                            ← 격자 좌표 + 메트릭
│   │   ├── main.hpp
│   │   ├── grid.hpp + .cpp             ← x, dx, dmx (축별)
│   │   └── stretching.hpp + .cpp       ← tanh 등 stretching 함수
│   │
│   ├── field/                           ← 변수 정의 (사용자 요구 #2)
│   │   ├── main.hpp
│   │   ├── scalar_field.hpp + .cpp     ← cell-centered
│   │   ├── vector_field.hpp + .cpp     ← FaceX/Y/Z
│   │   ├── field_registry.hpp + .cpp
│   │   └── stencil/                    ← 인덱스 산술 캡슐화
│   │       ├── main.hpp
│   │       ├── staggered.hpp           ← inline (gradient, divergence)
│   │       └── viscous.hpp             ← inline (harmonic mean μ)
│   │
│   ├── boundary/                        ← 경계조건 (사용자 요구 #3)
│   │   ├── main.hpp
│   │   ├── bc_kind.hpp                 ← enum (Periodic/Dirichlet/Neumann + 슬롯)
│   │   ├── face_bc.hpp + .cpp
│   │   ├── field_boundary.hpp
│   │   ├── domain_topology.hpp + .cpp  ← sweep_order 자동
│   │   ├── problem.hpp + .cpp          ← Problem (RBC 기본값 z=wall)
│   │   └── boundary_applier.hpp + .cpp ← apply_ghost, modify_tdma_row
│   │
│   ├── linear_solver/                   ← 선형 해법 (사용자 요구 #4)
│   │   ├── main.hpp
│   │   ├── tdma/
│   │   │   ├── main.hpp
│   │   │   ├── tdma_solver.hpp         ← 추상
│   │   │   ├── pascal_tdma_backend.hpp + .cpp
│   │   │   ├── filtered_tdma_backend.hpp + .cpp ← 미래
│   │   │   └── tdma_registry.hpp + .cpp
│   │   └── fft/
│   │       ├── main.hpp
│   │       ├── fft_planner.hpp + .cpp  ← BC-aware (R2C/DCT/DST)
│   │       ├── eigenvalues.hpp          ← inline
│   │       └── transpose_plan.hpp + .cpp
│   │
│   ├── equation/                        ← 핵심 방정식 (사용자 요구 #12, 항목별 하위 폴더)
│   │   ├── main.hpp
│   │   ├── momentum/
│   │   │   ├── main.hpp                ← 빌더 (`make_rbc_momentum`, `make_channel_momentum`)
│   │   │   ├── momentum_equation.hpp + .cpp   ← CN+ADI orchestrator
│   │   │   ├── rhs_builders.hpp                ← inline
│   │   │   ├── property_policy.hpp             ← 추상
│   │   │   ├── constant_properties.hpp + .cpp
│   │   │   ├── nob_properties.hpp + .cpp
│   │   │   ├── source_term.hpp                 ← 추상
│   │   │   ├── nob_buoyancy.hpp + .cpp
│   │   │   ├── boussinesq_buoyancy.hpp + .cpp
│   │   │   └── bulk_forcing.hpp + .cpp
│   │   ├── pressure/
│   │   │   ├── main.hpp
│   │   │   ├── pressure_equation.hpp + .cpp    ← rhs + solve + project orchestrator
│   │   │   ├── rhs_assembler.hpp + .cpp        ← div(u*) + NOB 보정
│   │   │   └── projection.hpp + .cpp           ← u, P 갱신 + dPhat 외삽
│   │   └── scalar/
│   │       ├── main.hpp
│   │       ├── scalar_equation.hpp + .cpp      ← 온도·다종 스칼라 공통
│   │       └── scalar_traits.hpp
│   │
│   ├── physics/                         ← LES + IBM (사용자 요구 #5)
│   │   ├── main.hpp
│   │   ├── plugin.hpp                  ← 공통 Plugin 추상 + Phase enum
│   │   ├── les/
│   │   │   ├── main.hpp
│   │   │   ├── smagorinsky.hpp + .cpp
│   │   │   ├── les_properties.hpp + .cpp        ← PropertyPolicy 데코레이터
│   │   │   └── les_plugin.hpp + .cpp
│   │   └── ibm/
│   │       ├── main.hpp
│   │       ├── ibm_mask.hpp + .cpp              ← H 필드, cell tagging
│   │       ├── cell_classification.hpp + .cpp
│   │       └── ibm_plugin.hpp + .cpp            ← matrix row 치환 hook
│   │
│   ├── integrator/                      ← 시간 update (사용자 요구 #7)
│   │   ├── main.hpp
│   │   ├── time_stepper.hpp + .cpp
│   │   └── cfl_controller.hpp + .cpp
│   │
│   ├── post/                            ← 순간 필드·restart·probe (사용자 요구 #6)
│   │   ├── main.hpp
│   │   ├── instant_io.hpp + .cpp
│   │   ├── restart_io.hpp + .cpp
│   │   └── probe.hpp + .cpp
│   │
│   ├── stat/                            ← 통계 누적 (사용자 요구 #6)
│   │   ├── main.hpp
│   │   ├── statistics_accumulator.hpp + .cpp
│   │   ├── statistics_plugin.hpp + .cpp        ← physics::Plugin 상속
│   │   └── stat_io.hpp + .cpp
│   │
│   └── utilities/                       ← cross-cutting
│       ├── main.hpp
│       ├── diagnostics.hpp + .cpp      ← div check, monitor
│       └── timer.hpp + .cpp            ← 프로파일링
│
├── apps/                                ← 케이스별 솔버 (사용자 요구 #8, 중복 무관)
│   ├── rbc/
│   │   ├── main.cpp                    ← 라이브러리 main 들 조립
│   │   ├── input.toml
│   │   └── Makefile
│   ├── channel/
│   │   ├── main.cpp
│   │   ├── input.toml
│   │   └── Makefile
│   ├── thermal_only_check/             ← 검증용 (M2)
│   │   └── main.cpp
│   └── poisson_only_check/             ← 검증용 (M4)
│       └── main.cpp
│
├── test/                                ← 기능 검증 (사용자 요구 #9)
│   ├── unit/
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
│       ├── golden/                      ← PaScaL_TCS / MPM-STD Fortran 골든
│       │   ├── pascal_tcs_Ra100_Pr1_10step.bin
│       │   └── mpm_std_fortran_channel.bin
│       └── compare.py
│
├── run/                                 ← 배치 + 입력 (사용자 요구 #10)
│   ├── rbc.sh                          ← SLURM
│   ├── channel.sh
│   └── inputs/
│       ├── rbc_Ra100.toml
│       └── channel_Re180.toml
│
└── build/                               ← 빌드 산출물 (사용자 요구 #11, gitignore)
    ├── obj/                            ← .o 파일
    └── bin/
        ├── rbc                         ← 실행파일
        ├── channel
        ├── thermal_only_check
        └── tests/
```

---

## 2. 폴더별 책임 표

| 폴더 | 책임 | 사용자 요구 # | 의존 (위→아래) |
|---|---|---|---|
| `common/` | 공통 typedef, enum, 매크로 | (필자 추가) | — |
| `parallel/mpi/` | MPI Cart + sub-comm + halo DDT | 1 | common |
| `parallel/backend/` | CPU/CUDA 추상 + 구현 | 1 | common |
| `parallel/cuda/` | CUDA 직접 사용 헬퍼 | 1 | common, backend |
| `config/` | TOML 파싱, logger | (필자 추가) | common |
| `grid/` | 격자 좌표 + 메트릭 + stretching | (필자 추가) | common, config, parallel |
| `field/` | ScalarField, VectorField, Registry | 2 | grid, parallel, common |
| `field/stencil/` | 인덱스 산술 캡슐화 (inline) | (필자 추가) | grid, field |
| `boundary/` | BC 시스템 (Problem, BoundaryApplier) | 3 | field, grid |
| `linear_solver/tdma/` | TDMA 추상 + PaScaL_TDMA wrap | 4 | parallel, common |
| `linear_solver/fft/` | BC-aware FFT/DCT/DST + 전치 | 4 | parallel, grid, boundary |
| `equation/momentum/` | CN+ADI 운동량 + PropertyPolicy + SourceTerm | 12 | field, boundary, linear_solver |
| `equation/pressure/` | rhs + Poisson solve + projection | 12 | field, boundary, linear_solver |
| `equation/scalar/` | 온도 + 다종 스칼라 ADI | 12 | field, boundary, linear_solver |
| `physics/` | 공통 Plugin 추상 + Phase enum | 5 | field, boundary |
| `physics/les/` | Smagorinsky, ν_t 갱신 | 5 | physics 베이스, equation/momentum |
| `physics/ibm/` | H 마스크, matrix row 치환 | 5 | physics 베이스, equation/momentum |
| `integrator/` | time_stepper, CFL controller | 7 | equation, utilities |
| `post/` | instant 필드 dump, restart, probe | 6 | field, parallel |
| `stat/` | 통계 누적, stat dump | 6 | field, physics (Plugin 상속) |
| `utilities/` | diagnostics, timer | (필자 추가) | field |
| `apps/<case>/` | 케이스별 main.cpp | 8 | (모든 라이브러리) |
| `test/` | 단위·통합·회귀 | 9 | (라이브러리) |
| `run/` | 배치 스크립트, 입력 | 10 | — |
| `build/` | 빌드 산출물 | 11 | — |

---

## 3. `main` 패턴 — 각 폴더의 facade

### 3.1 두 가지 형태

| 형태 | 내용 | 사용처 |
|---|---|---|
| **(A) 단순 aggregator** | `main.hpp` 가 폴더 안 모든 헤더를 `#include`. `main.cpp` 없음. | 작은 폴더 (common, stencil) |
| **(B) 헬퍼·빌더 제공** | `main.hpp` 가 high-level 함수 선언, `main.cpp` 가 구현. apps 가 한 줄 호출로 setup. | 큰 폴더 (linear_solver, equation/momentum) |

### 3.2 형태 A 예시 — `field/main.hpp`

```cpp
// src/field/main.hpp
#pragma once
#include "field/scalar_field.hpp"
#include "field/vector_field.hpp"
#include "field/field_registry.hpp"
#include "field/stencil/main.hpp"
```

→ 사용자: `#include "field/main.hpp"` 한 줄.

### 3.3 형태 B 예시 — `linear_solver/main.hpp`

```cpp
// src/linear_solver/main.hpp
#pragma once
#include "linear_solver/tdma/main.hpp"
#include "linear_solver/fft/main.hpp"

namespace mpmstd::linear_solver {

struct LinearSolverSet {
  std::unique_ptr<tdma::TdmaRegistry> tdma;
  std::unique_ptr<fft::FftPlanner>    fft;
};

// 한 줄로 모든 선형해법 도구 setup
LinearSolverSet setup(const config::Config&    cfg,
                       const boundary::Problem& problem,
                       const grid::Grid&        grid,
                       const parallel::mpi::Subdomain& sub,
                       parallel::backend::Backend& backend);

}
```

```cpp
// src/linear_solver/main.cpp
#include "linear_solver/main.hpp"

namespace mpmstd::linear_solver {

LinearSolverSet setup(const config::Config& cfg,
                       const boundary::Problem& problem,
                       const grid::Grid& g,
                       const parallel::mpi::Subdomain& sub,
                       parallel::backend::Backend& backend) {
  LinearSolverSet s;
  s.tdma = std::make_unique<tdma::TdmaRegistry>(cfg, sub.topology());
  s.fft  = std::make_unique<fft::FftPlanner>(problem, g, sub, backend);
  return s;
}

}
```

### 3.4 형태 B 의 깊은 예 — `equation/momentum/main.hpp`

```cpp
// src/equation/momentum/main.hpp
#pragma once
#include "equation/momentum/momentum_equation.hpp"
#include "equation/momentum/property_policy.hpp"
#include "equation/momentum/constant_properties.hpp"
#include "equation/momentum/nob_properties.hpp"
#include "equation/momentum/source_term.hpp"
#include "equation/momentum/nob_buoyancy.hpp"
#include "equation/momentum/boussinesq_buoyancy.hpp"
#include "equation/momentum/bulk_forcing.hpp"

namespace mpmstd::equation::momentum {

// RBC 용 — NobProperties + NobBuoyancy(W) 자동 합성
std::unique_ptr<MomentumEquation> make_rbc(
    const config::Config& cfg,
    grid::Grid& g, parallel::mpi::Subdomain& sub,
    field::FieldRegistry& fr, boundary::Problem& p,
    linear_solver::tdma::TdmaRegistry& tdma,
    boundary::BoundaryApplier& bc);

// Channel forced — ConstantProperties + BulkForcing(U)
std::unique_ptr<MomentumEquation> make_channel(
    const config::Config& cfg,
    grid::Grid& g, parallel::mpi::Subdomain& sub,
    field::FieldRegistry& fr, boundary::Problem& p,
    linear_solver::tdma::TdmaRegistry& tdma,
    boundary::BoundaryApplier& bc);

}
```

```cpp
// src/equation/momentum/main.cpp
#include "equation/momentum/main.hpp"

namespace mpmstd::equation::momentum {

std::unique_ptr<MomentumEquation> make_rbc(/*...*/) {
  auto props = std::make_unique<NobProperties>(g, /*coefs from cfg*/);
  auto buoy  = std::make_unique<NobBuoyancy>(Component::W,
                                              cfg.get<double>("Cmt"),
                                              cfg.get<double>("a12pera11"),
                                              cfg.get<double>("DeltaT"));
  std::vector<SourceTerm*> sources = { buoy.get() };
  return std::make_unique<MomentumEquation>(g, sub, fr, p, tdma, bc,
                                              std::move(props), std::move(sources));
}

std::unique_ptr<MomentumEquation> make_channel(/*...*/) {
  auto props = std::make_unique<ConstantProperties>(g);
  auto force = std::make_unique<BulkForcing>(Component::U,
                                              cfg.get<double>("presgrad1"));
  std::vector<SourceTerm*> sources = { force.get() };
  return std::make_unique<MomentumEquation>(g, sub, fr, p, tdma, bc,
                                              std::move(props), std::move(sources));
}

}
```

→ apps/rbc/main.cpp 는 **한 줄로 RBC momentum 완성**.

### 3.5 최상위 `src/main.hpp` (선택)

```cpp
// src/main.hpp — 사용자가 라이브러리 전체 한 줄 import
#pragma once
#include "common/main.hpp"
#include "parallel/main.hpp"
#include "config/main.hpp"
#include "grid/main.hpp"
#include "field/main.hpp"
#include "boundary/main.hpp"
#include "linear_solver/main.hpp"
#include "equation/main.hpp"
#include "physics/main.hpp"
#include "integrator/main.hpp"
#include "post/main.hpp"
#include "stat/main.hpp"
#include "utilities/main.hpp"
```

---

## 4. 폴더별 `main` 의 형태 분류

| 폴더 | 형태 | 핵심 함수/클래스 |
|---|---|---|
| `common/` | A | (aggregator) |
| `parallel/mpi/` | B | `make_context`, `make_topology`, `make_subdomain` |
| `parallel/backend/` | B | `make_cpu_backend`, `make_cuda_backend` (미래) |
| `parallel/cuda/` | A | (1차 비어 있음) |
| `parallel/` | B | `make_parallel(cfg, argc, argv)` → struct {context, topology, subdomain, backend} |
| `config/` | B | `Config::load(path)`, `Logger::init(rank)` |
| `grid/` | B | `make_grid(cfg, subdomain)` |
| `field/` | B | `make_rbc_fields`, `make_channel_fields` |
| `field/stencil/` | A | (inline aggregator) |
| `boundary/` | B | `make_problem_rbc`, `make_problem_channel`, `make_applier` |
| `linear_solver/tdma/` | B | `make_registry(cfg, topo)` |
| `linear_solver/fft/` | B | `make_planner(problem, grid, sub, be)` |
| `linear_solver/` | B | `setup(cfg, problem, grid, sub, be)` → LinearSolverSet |
| `equation/momentum/` | B | `make_rbc`, `make_channel` |
| `equation/pressure/` | B | `make_pressure(...)` |
| `equation/scalar/` | B | `make_temperature(...)`, `make_passive_scalar(name, ...)` |
| `equation/` | A | (하위 폴더 aggregator) |
| `physics/` | A | (Plugin 추상만) |
| `physics/les/` | B | `make_smagorinsky_plugin(cfg)` |
| `physics/ibm/` | B | `make_ibm_plugin(cfg, geometry_file)` |
| `integrator/` | B | `make_time_stepper(cfg)`, `make_cfl_controller(cfg)` |
| `post/` | B | `make_instant_io(cfg, fr)`, `make_restart_io(cfg, fr)`, `make_probe(cfg, fr)` |
| `stat/` | B | `make_statistics_plugin(cfg, fr)`, `make_stat_io(cfg, fr)` |
| `utilities/` | B | `make_diagnostics(fr, mpi)`, `make_timer()` |

---

## 5. apps 가 조립하는 방식

apps/main.cpp 는 거의 100% "각 폴더 main 함수 호출" 만으로 구성됨.

### 5.1 RBC 앱 — `apps/rbc/main.cpp`

```cpp
#include "main.hpp"             // 라이브러리 전체
using namespace mpmstd;

int main(int argc, char** argv) {
  // ---- 1. parallel + config ----
  auto par = parallel::make(argc, argv);       // mpi + backend 한 번에
  auto cfg = config::Config::load("input.toml");
  config::Logger::init(par.mpi->rank());

  // ---- 2. grid ----
  auto grid = grid::make_grid(cfg, *par.subdomain);

  // ---- 3. problem (BC) ----
  auto problem = boundary::make_problem_rbc(cfg);    // 기본 RBC + 옵션 덮어쓰기

  // ---- 4. field ----
  auto fields = field::make_rbc_fields(*grid, *par.backend);

  // ---- 5. linear solver ----
  auto solvers = linear_solver::setup(cfg, *problem, *grid, *par.subdomain, *par.backend);

  // ---- 6. boundary applier ----
  auto bc = boundary::make_applier(*problem);

  // ---- 7. equations ----
  auto thermal  = equation::scalar  ::make_temperature(cfg, *grid, *par.subdomain, *fields,
                                                        *problem, *solvers.tdma, *bc);
  auto momentum = equation::momentum::make_rbc       (cfg, *grid, *par.subdomain, *fields,
                                                        *problem, *solvers.tdma, *bc);
  auto pressure = equation::pressure::make           (cfg, *grid, *par.subdomain, *fields,
                                                        *problem, *solvers.tdma, *solvers.fft, *bc);

  // ---- 8. plugins ----
  std::vector<std::unique_ptr<physics::Plugin>> plugins;
  plugins.push_back(stat::make_statistics_plugin(cfg, *fields));
  plugins.push_back(post::make_probe(cfg, *fields));

  // ---- 9. integrator / IO / diagnostics ----
  auto stepper = integrator::make_time_stepper(cfg);
  auto cfl     = integrator::make_cfl_controller(cfg);
  auto restart = post::make_restart_io(cfg, *fields);
  auto instant = post::make_instant_io(cfg, *fields);
  auto stat_io = stat::make_stat_io(cfg, *fields);
  auto diag    = utilities::make_diagnostics(*fields, *par.mpi);

  restart->read_if_continue();
  for (auto& p : plugins) p->setup({/*ctx*/});

  // ---- 10. 시간 루프 (RBC 전용 순서) ----
  while (stepper->time() < cfg.get<double>("t_end")) {
    thermal ->step(*fields, stepper->dt());
    par.subdomain->exchange_halo(fields->scalar("T"));

    momentum->step(*fields, stepper->dt());      // predict + block_couple + pseudo_update
    par.subdomain->exchange_halo(fields->vector("U"));

    pressure->step(*fields, stepper->dt());      // rhs + solve + project

    for (auto& p : plugins) p->call_if_phase(physics::Phase::PostStep, {});
    diag->check_and_monitor(stepper->step(), stepper->time(), stepper->dt());

    stepper->advance(cfl->next_dt(*fields, *grid));
    if (instant->should_write(stepper->step())) instant->write(*fields, stepper->time());
    if (restart->should_write(stepper->step())) restart->write(*fields, stepper->time());
  }
  for (auto& p : plugins) p->finalise({});
  return 0;
}
```

### 5.2 Channel 앱 — `apps/channel/main.cpp`

차이점만:
- `field::make_channel_fields` (T 없음)
- `boundary::make_problem_channel` (T BC 미설정)
- `equation::scalar::make_temperature` 호출 **안 함**
- `equation::momentum::make_channel` (ConstantProperties + BulkForcing)
- 시간 루프에서 thermal step 단계 **제거**

→ **중복 코드 있음. 사용자 결정 #8 에 따라 허용.** RBC main 과 Channel main 의 차이가 눈에 명확히 보이는 게 오히려 유지보수 장점.

---

## 6. include path 와 빌드

### 6.1 include path

```
-Isrc
```

한 줄. 모든 라이브러리 헤더는 `src/` 아래에 있으므로:
```cpp
#include "field/main.hpp"
#include "equation/momentum/main.hpp"
```

같은 식.

### 6.2 Makefile 구조

```make
# Makefile.inc
CXX        := mpicxx
CXXSTD     := -std=c++17
OPT        := -O3 -march=native -fno-fast-math -fno-associative-math
WARN       := -Wall -Wextra -Wpedantic
INCS       := -Isrc -I$(PASCAL_TDMA_DIR)/src -I$(FFTW_DIR)/include
LIBS       := -L$(PASCAL_TDMA_DIR)/build -lpascal_tdma -lfftw3 -lm

# Makefile (루트)
.PHONY: all lib apps tests clean

all: lib apps

lib:
	$(MAKE) -C src           # libmpmstd.a → build/lib/

apps: lib
	$(MAKE) -C apps/rbc      # → build/bin/rbc
	$(MAKE) -C apps/channel  # → build/bin/channel

tests: lib
	$(MAKE) -C test          # → build/bin/tests/

clean:
	rm -rf build/
```

각 폴더 (apps/rbc/, apps/channel/, test/) 안에 자체 Makefile. 라이브러리에 링크.

### 6.3 src/ 의 빌드

`src/` 안에서는 모든 .cpp 를 한 번에 컴파일 → `libmpmstd.a` 1 개 정적 라이브러리. apps 가 이걸 링크.

```make
# src/Makefile
SRCS := $(shell find . -name "*.cpp")
OBJS := $(SRCS:%.cpp=../build/obj/%.o)

../build/lib/libmpmstd.a: $(OBJS)
	ar rcs $@ $^

../build/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(INCS) -MMD -MP -c $< -o $@

-include $(OBJS:.o=.d)
```

---

## 7. 변경 요약 (v1 = report 10 → v2 = 본 보고서)

| 항목 | v1 | v2 |
|---|---|---|
| 헤더 위치 | `include/mpmstd/` + `src/` 미러링 | **`src/` 한 곳, hpp+cpp 같이** |
| MPI/CUDA | `runtime/` 단일 폴더 | **`parallel/{mpi, backend, cuda}` 분리** |
| equation | 단일 폴더 | **하위 폴더 (`momentum`, `pressure`, `scalar`)** |
| physics | 추상 1 폴더 | **`les/`, `ibm/` 전용 하위 폴더** |
| io / 통계 | `io/` 단일 | **`post/` + `stat/` 분리** |
| 각 폴더 facade | 없음 (외부에서 직접 헤더 include) | **`main.hpp` (+ `main.cpp` 빌더)** |
| apps 의 boilerplate | 길었음 (~100 줄) | **짧음 (~40 줄, main 호출만)** |
| include path | `-Iinclude` | **`-Isrc`** |
| 새 카테고리 (필자 추가) | — | **`common/`, `config/`, `grid/`, `utilities/`, `field/stencil/`** |

---

## 8. 사용자 결정과 매핑

| 사용자 요구 # | 매핑된 폴더 |
|---|---|
| 1. MPI/CUDA 관리 (여러 폴더 OK) | `parallel/mpi/`, `parallel/backend/`, `parallel/cuda/` |
| 2. field | `field/` (+ `field/stencil/`) |
| 3. boundary | `boundary/` |
| 4. linear solver | `linear_solver/tdma/`, `linear_solver/fft/` |
| 5. physics (LES, IBM 하위) | `physics/les/`, `physics/ibm/` |
| 6. post, stat (여러개 가능) | `post/`, `stat/` |
| 7. integrator | `integrator/` |
| 8. apps (사용자 조립, 중복 OK) | `apps/<case>/` |
| 9. test | `test/{unit, integration, regression}/` |
| 10. run | `run/` |
| 11. 빌드 산출물 | `build/{obj, bin}/` |
| 12. equation (항목별 하위) | `equation/{momentum, pressure, scalar}/` |
| **추가**: 각 폴더에 `main` | 모든 폴더에 `main.hpp` (+ 큰 폴더는 `main.cpp`) |
| **추가**: hpp+cpp 같이 | `src/` 한 곳에 통합 |

### 추가 항목 (필자 판단)

| 신규 폴더 | 이유 |
|---|---|
| `common/` | types, direction enum — 모든 곳이 의존. 따로 두면 순환 의존 회피 |
| `config/` | TOML 파싱·logger 가 cross-cutting. 다른 폴더에 박으면 부적합 |
| `grid/` | 격자는 field 와 별개. tanh stretching 등 충분한 로직 보유 |
| `utilities/` | diagnostics, timer 가 cross-cutting |
| `field/stencil/` | field 와 깊게 연결되지만 stencil 인덱스 캡슐화는 별개 책임 |

---

## 9. 다음 단계 — 실제 구현 시작

1. **`mkdir -p` 로 디렉토리 골격 생성**:
   ```bash
   cd /shared/home/wel1come1234/workspace/MPM-STD\(C++\)
   mkdir -p src/{common,parallel/{mpi,backend,cuda},config,grid,field/stencil,\
                boundary,linear_solver/{tdma,fft},\
                equation/{momentum,pressure,scalar},\
                physics/{les,ibm},integrator,post,stat,utilities}
   mkdir -p apps/{rbc,channel,thermal_only_check,poisson_only_check}
   mkdir -p test/{unit,integration,regression/golden}
   mkdir -p run/inputs
   mkdir -p build/{obj,bin/tests}
   ```

2. **각 폴더에 빈 `main.hpp` 생성** (스켈레톤):
   ```bash
   for dir in $(find src -type d); do
     touch "$dir/main.hpp"
   done
   ```

3. **M0 부터 시작** — [10_final_implementation_plan.md](10_final_implementation_plan.md) §4 의 마일스톤을 본 디렉토리 구조에 맞춰 진행:
   - M0: `common/`, `parallel/`, `config/`, `grid/`, `field/`, `field/stencil/`, `post/restart_io`, Makefile 골격
   - M1: `boundary/`, `linear_solver/tdma/`
   - M2: `equation/scalar/` + property 일부, `apps/thermal_only_check/`
   - M3: `equation/momentum/` 전체
   - M4: `linear_solver/fft/`, `equation/pressure/`
   - M5: `integrator/`, `stat/`, `utilities/`, `apps/rbc/`
   - M6: `apps/channel/`
   - M7: `physics/les/`, `physics/ibm/` 슬롯

---

## 10. 결론

본 디렉토리 구조 v2 는 사용자 요구 12 가지 + 필자 추가 5 가지를 모두 반영한 **최종 작업 골격**.

핵심:
- **hpp + cpp 같은 폴더** → 편집 효율 우선 (1인 학술 프로젝트)
- **각 폴더 `main`** → apps 가 라이브러리 조립을 한 줄씩 호출
- **항목별 하위 폴더** (`equation/momentum`, `physics/les` 등) → 모듈 응집도 + 새 기능 추가 시 본체 미변경
- **`-Isrc` 한 줄** → 빌드 설정 단순

→ 이제 `mkdir -p` 부터 시작해도 무방. 보고서 [10_final_implementation_plan.md](10_final_implementation_plan.md) §4 의 마일스톤·파일 목록을 본 구조에 맞춰 작업.
