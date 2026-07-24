# 설계 수정안 v2 — 사용자 피드백 반영

> [06_design_critique_and_revision.md](06_design_critique_and_revision.md) 에 대한 사용자 추가 피드백 5 가지 반영:
> 1. **Solver 단일화 거부** — 라이브러리 + 케이스별 main.cpp 패턴 채택
> 2. **sweep order / FFT / TDMA 자동 도출** — 유지
> 3. **Plugin 패턴** — 유지
> 4. **Field 의 StagLocation enum 제거** — VectorField(face) + ScalarField(cell) 컨벤션 고정 + **인덱스 실수 방지는 stencil 자유함수 캡슐화로 분리**
> 5. **Time integrator = Crank-Nicolson 단일 채택** — explicit/고차 스킴은 미래에도 통합 시도 안 함 (코드 흐름이 본질적으로 다름)

---

## Part 1. 라이브러리 + 케이스별 Solver 스크립트 패턴 (06 의 Solver 단일화 폐기)

### 1.1 사용자 피드백 요지

> "결국 solver 코드는 다 따로 만들거야, 그게 유지보수가 좋을거 같거든. 소스 코드에서는 둘 모두를 고려한 뒤 각 문제에 맞는 것을 내가 가져다가 조립해서 푸는 것으로 할거임. 그래서 problem 으로 분기를 만드는 것이 아니라, 그냥 솔버 코드로 다 분기를 만들 것임"

→ 타당. 사실상 OpenFOAM 의 `icoFoam` / `pisoFoam` 분리, PyFR 의 app-per-system 패턴과 동일. **단일 `Solver` god-class 가 모든 case 를 알아야 하는 부담** 을 케이스별 작은 main.cpp 가 흡수.

### 1.2 디렉토리 구조 — 라이브러리와 앱 분리

```
MPM-STD(C++)/
├── include/mpmstd/                    ← 라이브러리 (모든 케이스가 공유)
│   ├── runtime/  (MpiContext, Backend, Subdomain)
│   ├── grid.hpp, field.hpp, ...
│   ├── boundary/ (BcKind, FaceBc, FieldBoundary, BoundaryApplier)
│   ├── numerics/ (TdmaSolver+백엔드, FftPlanner, TransposePlan)
│   ├── equations/
│   │   ├── momentum_equation.hpp
│   │   ├── pressure_equation.hpp
│   │   └── scalar_equation.hpp
│   ├── physics/
│   │   ├── property_policy.hpp       (ConstantProperties, NobProperties, LesProperties)
│   │   └── source_term.hpp           (NobBuoyancy, BoussinesqBuoyancy, BulkForcing)
│   ├── plugins/
│   │   ├── plugin.hpp                (abstract)
│   │   ├── ibm_plugin.hpp            (미래)
│   │   ├── les_plugin.hpp            (미래)
│   │   ├── statistics_plugin.hpp
│   │   └── probe_plugin.hpp
│   ├── io/                           (RestartIo, InstantIo, StatsIo)
│   └── utilities/                    (CflController, Diagnostics, TimeLoopRunner)
│
├── src/                              ← 라이브러리 구현
│   └── (헤더 대응 .cpp)
│
├── apps/                             ← 케이스별 솔버. 사용자가 main 을 직접 작성.
│   ├── rbc/
│   │   ├── main.cpp                 ← RBC 전용. 컴포넌트 조립 + 시간루프.
│   │   ├── input.toml
│   │   └── Makefile
│   ├── channel/
│   │   ├── main.cpp                 ← Channel 전용. 다른 조립, 다른 시간루프.
│   │   ├── input.toml
│   │   └── Makefile
│   └── (미래: rbc_les/, channel_ibm/, ...)
│
├── tests/
└── Makefile                          ← 라이브러리 빌드 + 모든 apps 빌드
```

### 1.3 핵심 원칙

| 원칙 | 의미 |
|---|---|
| **라이브러리는 case-agnostic** | `mpmstd::MomentumEquation` 안에 `if (case == "rbc")` 같은 분기 0 개 |
| **분기는 main.cpp 에서** | "RBC 면 thermal + buoyancy 추가, Channel 이면 forcing 추가" 같은 결정은 사용자 main 의 책임 |
| **공유는 함수/클래스 단위로** | 시간 루프의 *조각* 들 (predict / block_couple / project / cfl_update) 은 라이브러리에. 그 조각을 어떤 순서로 호출할지는 main |
| **빌드 단위 분리** | 각 app 은 독립 실행파일. 라이브러리 한 번 빌드 후 app 마다 링크 |

### 1.4 RBC 앱 — `apps/rbc/main.cpp` 예시

```cpp
#include <mpmstd/all.hpp>
using namespace mpmstd;

int main(int argc, char** argv) {
  // ---------- (1) Runtime ----------
  MpiContext   mpi(argc, argv);
  Config       cfg("input.toml");
  CpuBackend   backend(mpi, cfg);
  Logger::init(mpi.rank());

  // ---------- (2) Topology + Grid ----------
  MpiTopology  topo(cfg, mpi);
  Subdomain    sub(topo, cfg);
  Grid         grid(cfg, sub);

  // ---------- (3) Problem (BC + topology) ----------
  Problem problem;                          // 기본값: z=wall RBC (T BCs 자동 포함)
  // 필요시 일부 면만 덮어쓰기:
  problem.T.face(Z, Minus) = FaceBc::dirichlet(cfg.get<double>("T_hot"));
  problem.T.face(Z, Plus)  = FaceBc::dirichlet(cfg.get<double>("T_cold"));
  problem.validate();

  // ---------- (4) Fields ----------
  FieldRegistry fields(grid, backend);
  fields.add_vector("U");                   // face-staggered
  fields.add_vector("dU");                  // increment, 같은 stag
  fields.add_scalar("P");                   // cell center
  fields.add_scalar("dP");
  fields.add_scalar("T");

  // ---------- (5) Numerics ----------
  TdmaRegistry  tdma(cfg, topo);            // 축별 PaScaL_TDMA 백엔드 등록
  FftPlanner    fft  (problem, grid, backend);   // BC 에서 R2C / DCT 자동 선택
  BoundaryApplier bc (problem);

  // ---------- (6) Physics policies ----------
  NobProperties      props(grid, backend, cfg.nob_coeffs());
  NobBuoyancy        buoy (Component::W, cfg.Cmt(), cfg.a12pera11(), cfg.DeltaT());

  // ---------- (7) Equations ----------
  ScalarEquation     thermal ({.name="T", .diffusivity=cfg.Ct()},
                              grid, sub, fields, problem, tdma, bc);
  MomentumEquation   momentum(grid, sub, fields, problem, tdma, bc, props, {&buoy});
  PressureEquation   pressure(grid, sub, fields, problem, tdma, fft, bc);

  // ---------- (8) Plugins ----------
  std::vector<std::unique_ptr<Plugin>> plugins;
  plugins.emplace_back(std::make_unique<StatisticsPlugin>(cfg, fields));
  plugins.emplace_back(std::make_unique<ProbePlugin>(cfg, fields));

  // ---------- (9) IO + Diagnostics ----------
  RestartIo    restart  (cfg, fields);
  InstantIo    instant  (cfg, fields);
  StatsIo      stats    (cfg, fields);
  CflController cfl     (cfg);
  Diagnostics  diag     (fields, mpi);

  restart.read_if_continue();
  for (auto& p : plugins) p->setup({/*solver context*/});

  // ---------- (10) Time loop (RBC 전용 순서) ----------
  double time = cfg.t_start();
  double dt   = cfg.dt_start();
  int    step = 0;
  while (time < cfg.t_end() && step < cfg.max_steps()) {
    // [B1-B2] Thermal
    thermal.compute_coeffi(fields);
    thermal.step(fields, dt);
    sub.exchange_halo(fields.scalar("T"));

    // [B3] Property update (T-dependent μ, 1/ρ)
    props.update(fields);

    // [B4-B6] Momentum predict (U, V, W)
    for (Component c : {U, V, W}) {
      momentum.predict(c, dt);
      sub.exchange_halo(fields.vector("dU").component(c));
    }
    // [B7-B8] Block coupling
    momentum.block_couple_V();   sub.exchange_halo(fields.vector("dU").y());
    momentum.block_couple_U();
    // [B9] Pseudo update
    momentum.pseudo_update();
    sub.exchange_halo(fields.vector("U"));

    // [B10-B12] Pressure RHS + Poisson + Projection
    pressure.compute_rhs(fields);
    pressure.solve(fields);                 // FFT/DCT/TDMA 모두 problem 에서 자동 도출
    sub.exchange_halo(fields.scalar("dP"));
    pressure.project(fields, dt);
    sub.exchange_halo(fields.vector("U"));
    sub.exchange_halo(fields.scalar("P"));

    // [B13] Plugins (Stats, Probes, ...)
    for (auto& p : plugins) p->call_if_phase(Phase::PostStep, /*ctx*/);

    // Diagnostics + CFL
    diag.check_divergence(fields);
    diag.monitor(step, time, dt);
    dt = cfl.next_dt(fields, grid);

    // IO
    if (instant.should_write(step)) instant.write(step, time, fields);
    if (restart.should_write(step)) restart.write(step, time, fields);

    time += dt;
    ++step;
  }

  for (auto& p : plugins) p->finalise({});
  return 0;
}
```

### 1.5 Channel 앱 — `apps/channel/main.cpp` 예시

```cpp
#include <mpmstd/all.hpp>
using namespace mpmstd;

int main(int argc, char** argv) {
  // ---------- (1) Runtime / (2) Topology / (3) Problem ----------
  MpiContext   mpi(argc, argv);
  Config       cfg("input.toml");
  CpuBackend   backend(mpi, cfg);
  MpiTopology  topo(cfg, mpi);
  Subdomain    sub(topo, cfg);
  Grid         grid(cfg, sub);

  Problem problem;                          // 기본 z=wall (RBC 기본값과 같음)
  problem.disable_thermal();                // T 면 BC 는 갖되, T 방정식 안 만들면 됨
  problem.validate();

  // ---------- (4) Fields (T 없음) ----------
  FieldRegistry fields(grid, backend);
  fields.add_vector("U");
  fields.add_vector("dU");
  fields.add_scalar("P");
  fields.add_scalar("dP");
  // ← fields.add_scalar("T") 호출하지 않음

  // ---------- (5) Numerics ----------
  TdmaRegistry  tdma(cfg, topo);
  FftPlanner    fft (problem, grid, backend);
  BoundaryApplier bc(problem);

  // ---------- (6) Physics policies (상수 μ, ρ) ----------
  ConstantProperties props(grid, backend);            // μ = 1, 1/ρ = 1
  BulkForcing        forcing(Component::U,           // streamwise
                              cfg.get<double>("presgrad1"));

  // ---------- (7) Equations (thermal 없음) ----------
  MomentumEquation   momentum(grid, sub, fields, problem, tdma, bc, props, {&forcing});
  PressureEquation   pressure(grid, sub, fields, problem, tdma, fft, bc);

  // ---------- (8) Plugins ----------
  std::vector<std::unique_ptr<Plugin>> plugins;
  plugins.emplace_back(std::make_unique<StatisticsPlugin>(cfg, fields));
  // 미래: ibm, les, wall_function 추가는 여기에 한 줄

  // ---------- (9) IO ----------
  RestartIo restart(cfg, fields);
  InstantIo instant(cfg, fields);
  StatsIo   stats  (cfg, fields);
  CflController cfl(cfg);
  Diagnostics diag (fields, mpi);
  restart.read_if_continue();
  for (auto& p : plugins) p->setup({});

  // ---------- (10) Time loop (Channel 전용 순서 — thermal 단계 없음) ----------
  double time = cfg.t_start(), dt = cfg.dt_start();
  int    step = 0;
  while (time < cfg.t_end() && step < cfg.max_steps()) {
    // [B3] Property update (no-op for constant, future LES Plugin 가 끼어들 수 있음)
    props.update(fields);
    for (auto& p : plugins) p->call_if_phase(Phase::PreStep, {});

    // [B4-B9] Momentum (RBC 와 동일 라이브러리 호출)
    for (Component c : {U, V, W}) {
      momentum.predict(c, dt);
      sub.exchange_halo(fields.vector("dU").component(c));
    }
    momentum.block_couple_V();   sub.exchange_halo(fields.vector("dU").y());
    momentum.block_couple_U();
    momentum.pseudo_update();
    sub.exchange_halo(fields.vector("U"));

    // [B10-B12] Pressure (동일)
    pressure.compute_rhs(fields);
    pressure.solve(fields);
    sub.exchange_halo(fields.scalar("dP"));
    pressure.project(fields, dt);
    sub.exchange_halo(fields.vector("U"));
    sub.exchange_halo(fields.scalar("P"));

    for (auto& p : plugins) p->call_if_phase(Phase::PostStep, {});

    diag.check_divergence(fields);
    diag.monitor(step, time, dt);
    dt = cfl.next_dt(fields, grid);

    if (instant.should_write(step)) instant.write(step, time, fields);
    if (restart.should_write(step)) restart.write(step, time, fields);

    time += dt;
    ++step;
  }
  for (auto& p : plugins) p->finalise({});
  return 0;
}
```

### 1.6 두 main.cpp 의 차이 분석

| 부분 | RBC | Channel |
|---|---|---|
| Field set | + T | T 제외 |
| Property | NobProperties | ConstantProperties |
| Source 주입 | NobBuoyancy | BulkForcing |
| Equation 생성 | + ScalarEquation(T) | thermal 없음 |
| Time loop 단계 | + thermal + ghost exchange T + props.update(fields) | property 만 (T 의존 아님) |

**그 외는 모두 동일** (라이브러리 호출 패턴). → **중복 부담은 그리 크지 않다**.

### 1.7 중복 줄이기 — 헬퍼 유틸리티 (선택)

만일 시간 루프 중복이 거슬리면 라이브러리에 옵션 헬퍼:

```cpp
// include/mpmstd/utilities/standard_time_loop.hpp
class StandardTimeLoop {
public:
  void add_step_action(Phase phase, std::function<void()> action);
  void run(double t_end, int max_steps, CflController& cfl);
};

// RBC main 에서:
StandardTimeLoop loop;
loop.add_step_action(Phase::Thermal, [&]{ thermal.step(fields, dt); ... });
loop.add_step_action(Phase::Momentum, [&]{ momentum.predict(...); ... });
loop.add_step_action(Phase::Pressure, [&]{ pressure.solve(...); ... });
loop.run(cfg.t_end(), cfg.max_steps(), cfl);
```

→ 라이브러리는 **제공만**. 사용자가 쓰고 싶으면 쓰고, 직접 루프 짜고 싶으면 안 써도 됨. **강제하지 않음**.

---

## Part 2. Field — StagLocation enum 제거 + 인덱스 실수 방지 전략

### 2.1 사용자 피드백 1 — 컨벤션으로 통일

> "그냥 벡터 변수 (U, V, W)는 cell면에, scalar는 cell 중앙에 저장하는 것으로 통일하면 되는거 아닌가?"

→ **타당**. 구조 격자 MAC 컨벤션은 universal: vector = face, scalar = cell. enum 으로 런타임 분기할 일이 없음.

### 2.1' 사용자 피드백 2 — 타입이 인덱스 실수를 막아주지 않는다

> "어떻게 인덱스 실수를 방지할 수 있다는거야?"

→ **정직하게 인정**: `ScalarField` vs `VectorField` 타입 구분은 **인덱스 실수를 막아주지 못한다**. 06 보고서에서 "type-safe staggering" 이라고 쓴 부분은 과장. 실제로 타입이 보장하는 것과 보장하지 못하는 것을 명확히 구분해야 한다.

#### 타입이 *실제로* 보장하는 것

| 보장 | 어떻게 |
|---|---|
| 배열 크기 자동 결정 | `U.x()` 가 `(n1+1, n2, n3)`, `P` 가 `(n1, n2, n3)` 으로 alloc 됨 |
| halo 메모리 자동 분리 | 컴포넌트마다 독립 alloc |
| 변수 혼동 컴파일 차단 | `divergence(P)` 같이 잘못된 인자 타입 → 컴파일 에러 |
| 자기 문서화 | `U.x()` 가 x-방향 velocity 임은 이름으로 명시 |

#### 타입이 *막지 못하는* 것

```cpp
// 둘 다 컴파일 통과 — 컴파일러는 어느 게 맞는지 모름
double dudx_correct = (U.x()(i+1,j,k) - U.x()(i,j,k)) / dx[i];   // 정답 (cell center)
double dudx_wrong   = (U.x()(i,j,k)   - U.x()(i-1,j,k)) / dx[i]; // 틀림 (face i 위치)
```

`U.x()(i,j,k)` 가 face x=(i−½)·dx 인지 face x=i·dx 인지는 **컨벤션** 일 뿐. 컴파일러에겐 그저 `double*` 의 인덱스 접근.

### 2.2 진짜 해법 — stencil 자유함수 캡슐화

인덱스 산술을 **사용자 코드에서 분리** 해 한 곳에만 두기.

```cpp
// include/mpmstd/stencil/staggered.hpp
namespace mpmstd::stencil {

// face-centered U → cell center 에서의 ∂U/∂x
inline double dudx_at_cell(const ScalarField& Ux, const Grid& g,
                            int i, int j, int k) {
  return (Ux(i+1, j, k) - Ux(i, j, k)) / g.dx1(i);
}

// cell-centered P → face x=(i−½) 에서의 ∂P/∂x
inline double dpdx_at_face_x(const ScalarField& P, const Grid& g,
                              int i, int j, int k) {
  return (P(i, j, k) - P(i-1, j, k)) / g.dmx1(i);
}

// face-centered U → face y=(j−½) 에서의 ∂U/∂y (cross derivative)
inline double dudy_at_face_y(const ScalarField& Ux, const Grid& g,
                              int i, int j, int k) {
  return (Ux(i, j, k) - Ux(i, j-1, k)) / g.dmx2(j);
}

// 발산 한 번에
inline double divergence_at_cell(const VectorField& U, const Grid& g,
                                  int i, int j, int k) {
  return (U.x()(i+1,j,k) - U.x()(i,j,k)) / g.dx1(i)
       + (U.y()(i,j+1,k) - U.y()(i,j,k)) / g.dx2(j)
       + (U.z()(i,j,k+1) - U.z()(i,j,k)) / g.dx3(k);
}

// face x 에서의 viscous Laplacian 등 자주 쓰는 패턴 모두 헬퍼화
inline double laplacian_u_at_face_x(const ScalarField& Ux, const ScalarField& Mu,
                                     const Grid& g, int i, int j, int k);
// ... 등등

}  // namespace mpmstd::stencil
```

#### 사용 측은 raw 인덱스 산술 없음

```cpp
// MomentumEquation::predict 내부
for (int k=1; k<=n3m; ++k)
for (int j=1; j<=n2m; ++j)
for (int i=1; i<=n1m; ++i) {
  double dpdx = stencil::dpdx_at_face_x(P, grid, i, j, k);
  double div  = stencil::divergence_at_cell(U, grid, i, j, k);
  // 이름이 곧 위치를 보증. 인덱스 산술은 stencil/ 헤더 한 곳에만.
}
```

→ stencil 의 인덱스 산술이 **한 곳에만 존재**. 단위 테스트 한 번 작성하면 모든 호출자가 검증됨.

#### 단위 테스트로 보장

```cpp
// tests/unit/test_stencil.cpp
TEST(Stencil, DudxAtCellIsSecondOrder) {
  // U.x()(i,j,k) = sin(x_face(i)) 로 채움
  // dudx_at_cell(...) 결과를 cos(x_center(i)) 와 비교
  // 격자 4 배 미세화 후 EOC = 2 확인
}
TEST(Stencil, DivergenceAnnihilatesSolenoidal) {
  // U = (∂ψ/∂y, -∂ψ/∂x, 0) (2D 비회전) 으로 채움 → divergence ≈ 0
}
```

→ 한 번 검증된 stencil 헬퍼는 어디서 호출하든 안전.

### 2.3 책임 분담 — 정직한 정리

| 책임 | 누가 |
|---|---|
| 배열 크기·halo 정합성 | `ScalarField` / `VectorField` 타입 |
| 변수 혼동 방지 | `ScalarField` vs `VectorField` 타입 |
| **인덱스 산술 정확성** | **`stencil/` 자유함수 + 단위 테스트** |
| 자기 문서화 | 컴포넌트 이름 (`U.x()`, `P`) + stencil 헬퍼 이름 |

→ 타입에 기대지 말고 **stencil 캡슐화** 로 인덱스 실수를 막는다.

### 2.4 참고 — PaScaL_TCS·MPM-STD Fortran 의 약점이 바로 이 부분

[04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md) §10 의 "고쳐야 할 약점" 에서 지적한 항목:
- `mpi_momentum_solvedU` (340 LOC) 한 do-loop 안에 모든 stencil 인라인
- dU/dV/dW 가 3× 중복 (~1200 LOC) — 인덱스 실수 발견·수정이 3 군데
- CaNS 가 `mod_mom` 의 `momx_a, momy_a, momz_a` 로 분리한 것이 더 나은 접근

→ C++ 에서는 **stencil 헬퍼 헤더** + dU/dV/dW 통합 (PaScaL_TCS 의 3× 중복 제거) 로 두 약점을 동시에 개선.

### 2.5 단순화된 Field 클래스

```cpp
// include/mpmstd/field.hpp
namespace mpmstd {

// 모든 ScalarField 는 cell center
class ScalarField {
public:
  ScalarField(const Grid& g, Backend& be, std::string name);

  // 인덱스 접근 (host)
  double&       operator()(int i, int j, int k);
  double        operator()(int i, int j, int k) const;

  // raw pointer (host / device)
  double*       host_ptr();
  double*       device_ptr();
  void          to_device();
  void          to_host();

  int           n1sub() const;
  int           n2sub() const;
  int           n3sub() const;
  static constexpr int halo_width() { return 1; }
  const std::string& name() const { return name_; }

private:
  std::vector<double> host_;
  double*  device_ = nullptr;
  int      n1_, n2_, n3_;
  std::string name_;
  Backend& backend_;
};

// VectorField = 3 개의 ScalarField, 각각 다른 face 에 위치
class VectorField {
public:
  VectorField(const Grid& g, Backend& be, std::string name);

  // 컴포넌트별 접근
  ScalarField&       x()         { return x_; }   // FaceX
  ScalarField&       y()         { return y_; }   // FaceY
  ScalarField&       z()         { return z_; }   // FaceZ
  const ScalarField& x()  const  { return x_; }
  const ScalarField& y()  const  { return y_; }
  const ScalarField& z()  const  { return z_; }

  // enum-인덱싱
  ScalarField&       component(Component c);
  const ScalarField& component(Component c) const;

  const std::string& name() const { return name_; }

private:
  ScalarField x_, y_, z_;       // 각자 다른 크기 (face 위치에 따라)
  std::string name_;
};

}  // namespace mpmstd
```

### 2.6 face 별 크기 차이 — 내부에서만 알아서

```cpp
// VectorField 생성자 내부
VectorField::VectorField(const Grid& g, Backend& be, std::string n)
  : x_(g.size_at_face(Direction::X), be, n + ".x"),    // (n1+1, n2,   n3)
    y_(g.size_at_face(Direction::Y), be, n + ".y"),    // (n1,   n2+1, n3)
    z_(g.size_at_face(Direction::Z), be, n + ".z"),    // (n1,   n2,   n3+1)
    name_(std::move(n)) {}
```

→ 사용자는 `U.x()` 가 어떤 크기인지 신경 안 씀. `ScalarField` API 가 동일.

### 2.7 사용 예 — 깔끔

```cpp
// 발산 계산 (cell center 에 저장)
void compute_divergence(const VectorField& U, ScalarField& div, const Grid& g) {
  for (int k=1; k<=g.n3m(); ++k)
  for (int j=1; j<=g.n2m(); ++j)
  for (int i=1; i<=g.n1m(); ++i) {
    div(i,j,k) = (U.x()(i+1,j,k) - U.x()(i,j,k)) / g.dx1(i)
               + (U.y()(i,j+1,k) - U.y()(i,j,k)) / g.dx2(j)
               + (U.z()(i,j,k+1) - U.z()(i,j,k)) / g.dx3(k);
  }
}
```

→ **`StagLocation::FaceX` 같은 태그 확인 없이도 인덱스가 직관적**. `U.x()(i+1,j,k)` 이 +x face 라는 사실은 컴포넌트 이름이 보증.

### 2.8 사용자 의문 해소

| 의문 | 답 |
|---|---|
| **VectorField 의 3 컴포넌트 크기가 달라도 되나?** | Yes. 각 컴포넌트가 독립 `ScalarField`. 내부 메모리 alloc 도 독립. |
| **halo 폭은 같나?** | 1로 통일. face vs cell 의 halo 의미 차이는 stencil 작성 시 자연스럽게 해결. |
| **dU, dV, dW 같은 increment 도 같은 컨벤션?** | Yes. `VectorField dU;` 가 `dU.x() (FaceX), dU.y() (FaceY), dU.z() (FaceZ)`. |
| **H (forcing source) 는?** | `VectorField H;` — momentum 각 component 와 같은 face 에 위치. |
| **μ, ν_t, p, dP, T 는?** | 모두 `ScalarField` — cell center. |
| **새로 추가될 scalar (다종 스칼라, IBM mask 등) 는?** | 모두 `ScalarField`. 예외 없음. |

### 2.9 StagLocation 가 필요 없는 이유 — 정직한 정리

| 잠재 사용처 | 필요한가? | 대안 |
|---|---|---|
| 컴파일타임 안전성 | No, 타입 (`ScalarField` vs `VectorField`) 이 이미 보장 |
| 런타임 분기 | No, hot loop 에서 절대 분기 안 함 |
| 자기-문서화 | 컴포넌트 이름 (`U.x()`, `P`) 이 이미 표현 |
| Generic operator | Vector vs Scalar 만 구분하면 충분 |
| 새 stag (예: cell vertex) 미래 추가 | 거의 없음. 필요하면 그때 enum 도입 |

→ **단순함이 안전함**. 사용자 제안 채택.

---

## Part 3. Plugin / sweep_order / FFT 자동 도출 — 유지

[06_design_critique_and_revision.md](06_design_critique_and_revision.md) 의 §B.5, §B.6 그대로 유지. 추가 코멘트만:

### 3.1 Plugin 의 호출 시점은 main.cpp 가 결정

기존 06 안은 `Solver` 가 Plugin 을 자동 호출. 본 v2 안에서는 **main 이 직접 호출**:

```cpp
// main.cpp 안에서
for (auto& p : plugins) p->call_if_phase(Phase::PostStep, /*ctx*/);
```

→ Plugin 인터페이스는 동일 (Phase enum + `call/setup/finalise`). 다만 *언제 호출할지* 가 라이브러리(`Solver`) 가 아니라 *사용자 main* 의 결정.

장점: Channel 에서는 `Phase::Thermal` 같은 단계를 그냥 건너뛰면 됨 (해당 phase 의 plugin 이 있어도 호출 안 함). RBC 와 다른 phase 호출 순서를 갖는 자유로움.

### 3.2 sweep_order 자동 도출 — 라이브러리 책임 유지

```cpp
// MomentumEquation::predict 내부 (라이브러리)
auto sweep = problem_.topology.sweep_order();  // periodic 먼저, wall 마지막
for (Direction d : sweep) {
  // ... TDMA cycle vs non-cycle 자동 선택
}
```

→ main.cpp 가 *어떤 BC 인지* 만 정하면 라이브러리가 *어떻게 푸는지* 책임. 분리 양호.

### 3.3 FFT 종류 자동 도출 — 라이브러리 책임 유지

`FftPlanner` 가 `Problem` 의 P face BC 를 보고 R2C / DCT-II / DST-II 자동 선택. main 은 종류 지정 안 함.

---

## Part 4. 시간 적분기 — Crank-Nicolson + ADI 단일 채택

### 4.1 사용자 결정

> "시간 적분기를 explicit 으로 처리하는 것은 너무 코드가 달라지니까 포기하자. 그냥 crank nicolson 으로 통일하는 것으로."

→ **단일 스킴 채택**. `Scheme` Strategy 폐기. `MomentumEquation` 등 클래스 이름에 `CrankNicolsonAdi` prefix 도 굳이 안 붙임 (다른 게 없으므로).

### 4.2 왜 explicit / 고차 통합을 포기하는가 — 정직한 이유

시간 스킴이 바뀌면 *알고리듬 자료 흐름 자체가 달라짐*. 통합의 비용이 이득을 초과:

| 스킴 | TDMA 사용 | block_couple | RHS 구조 | dt 제한 | stencil 폭 |
|---|---|---|---|---|---|
| **CN + ADI (채택)** | ✅ 방향마다 | ✅ cross-velocity 결합 | implicit 좌변 + explicit 우변 누적 | 큼 (viscous CFL 제거) | 3-pt |
| Explicit RK | ❌ 안 씀 | ❌ 불필요 | 순수 RHS = N+L+source | 작음 (viscous CFL) | 3-pt |
| IMEX | 일부만 | 부분 | 분할 (N explicit, L implicit) | 중간 | 3-pt |
| 고차 컴팩트 | ❌ → block-pentadiag | 다름 | 사전·사후 derivative 필터 | 큼 | 5–7-pt + 보조계 |
| Pseudo-spectral | ❌ | ❌ | FFT 기반, stencil 없음 | 다름 | 전역 |

→ TDMA·block_couple·RHS 어셈블리·halo 폭 모두 달라지므로 **공통 인터페이스로 묶어도 빈 메서드와 if-else 분기만 남음**. 차라리 정직하게 한 가지만.

### 4.3 라이브러리에서의 표현

`Scheme` 추상 클래스 없음. `MomentumEquation` 자체가 CN+ADI 구현체:

```cpp
class MomentumEquation {        // 이름에 CN 안 붙임 — 라이브러리에 한 가지뿐
public:
  void compute_coeffi();
  void predict(Component c, double dt);   // CN+ADI: 3-stage TDMA sweep
  void block_couple_V();                  // cross-velocity coupling correction
  void block_couple_U();                  //   "
  void pseudo_update();                   // U^* = U + dU
};
```

→ "CN+ADI 가 아니라 다른 걸 쓰고 싶다" 면 이 클래스 자체를 새로 작성. 1 차 프로젝트 범위 밖.

### 4.4 미래에 다른 스킴이 정말 필요해질 경우 — 추출 친화적 작성

지금 통합은 안 하지만, **미래 추출이 쉽도록** 내부 구현 시 다음을 지킴:

1. **convection / viscous / pressure-gradient RHS 어셈블리** 를 별도 자유함수로 (Part 2 의 stencil 헬퍼 사용).
2. **CN+ADI 고유 로직** (좌변 implicit band 빌드, TDMA 호출, block_couple) 만 `MomentumEquation` 안에.
3. 미래에 explicit RK 가 필요하면 위 1 번 함수를 그대로 재사용해 새 `ExplicitRk3Momentum` 작성.

```cpp
// include/mpmstd/momentum/rhs_builders.hpp
namespace mpmstd::momentum {

// CN-ADI 와 미래 explicit RK 둘 다에서 호출 가능
double assemble_convection_rhs_at(Component c, const VectorField& U,
                                   const Grid& g, int i, int j, int k);

double assemble_viscous_rhs_at(Component c, const VectorField& U,
                                const PropertyPolicy& props,
                                const Grid& g, int i, int j, int k);

double assemble_pressure_gradient_rhs_at(Component c, const ScalarField& P,
                                          const PropertyPolicy& props,
                                          const Grid& g, int i, int j, int k);

}  // namespace mpmstd::momentum
```

→ **CN-ADI 고유 코드** (좌변 band, TDMA, block_couple) 는 `MomentumEquation` 내부. **재사용 가능한 RHS 어셈블리** 는 자유함수 헤더. 미래 추출 시 마찰 최소.

### 4.5 결론

- 시간 적분기 추상화 폐기 (1 차에 단일 스킴).
- `MomentumEquation` = CN+ADI 구현체 (이름에 명시 안 함).
- RHS 어셈블리 자유함수 분리 → 미래에 다른 스킴 추가 시 재사용 친화적.

---

## Part 5. 수정된 클래스 구조 요약

```cpp
namespace mpmstd {

// === Runtime ===
class MpiContext;
class Backend;          // abstract
class CpuBackend;       // 1차 구현
class CudaBackend;      // 미래 (인터페이스만)
class Logger;

// === Topology / Grid ===
class MpiTopology;
class Subdomain;
class Grid;

// === Field ===
class ScalarField;      // cell center
class VectorField;      // FaceX/FaceY/FaceZ
class FieldRegistry;

// === Boundary ===
enum class BcKind { Periodic, Dirichlet, Neumann, /*future*/ Wall, Inflow, Outflow };
struct FaceBc;
class FieldBoundary;
struct DomainTopology;
class Problem;
class BoundaryApplier;

// === Numerics ===
class TdmaSolver;       // abstract
class PascalTdmaBackend;
class FilteredTdmaBackend;  // 미래
class TdmaRegistry;
class FftPlanner;       // BC-aware
class TransposePlan;

// === Equations (case-agnostic) ===
class MomentumEquation;
class PressureEquation;
class ScalarEquation;

// === Physics policies (case-specific 주입) ===
class PropertyPolicy;   // abstract
class ConstantProperties;
class NobProperties;
class LesProperties;    // 미래
class SourceTerm;       // abstract
class NobBuoyancy;
class BoussinesqBuoyancy;
class BulkForcing;

// === Plugins (extension hooks) ===
class Plugin;           // abstract
class StatisticsPlugin;
class ProbePlugin;
class IbmPlugin;        // 미래
class LesPlugin;        // 미래

// === Utilities (case-specific main 이 사용) ===
class CflController;
class Diagnostics;
class StandardTimeLoop; // 선택. 사용 안 해도 됨.

// === IO ===
class RestartIo;
class InstantIo;
class StatsIo;

}  // namespace mpmstd
```

**없어진 것들** (06 대비):
- ❌ `Solver` god-class — 폐기. main.cpp 가 그 역할.
- ❌ `StagLocation` enum — 폐기. ScalarField/VectorField 가 컨벤션 보장.
- ❌ `BaseSolver`, `ChannelSolver`, `RBCSolver` 계층 (06 의 critique 에서 이미 제거됐고 v2 에서도 부활 안 함)
- ❌ `Scheme` Strategy / `CrankNicolsonAdi` 추상 — 시간 적분기는 **Crank-Nicolson + ADI 단일 채택**. Strategy 패턴 자체를 폐기. (이유는 Part 4.1 참조)

**추가된 것들**:
- ✅ `apps/<case>/main.cpp` 패턴 — 케이스별 솔버 분리
- ✅ `Problem::disable_thermal()` 같은 명시적 비활성 helper — 사용자가 main 에서 호출
- ✅ `stencil/` 자유함수 헤더 — 인덱스 산술 캡슐화 (Part 2.2)
- ✅ `momentum/rhs_builders.hpp` 자유함수 — 미래 시간 스킴 추출 친화적 (Part 4.4)

---

## Part 6. v2 의 검증 전략

[06_design_critique_and_revision.md](06_design_critique_and_revision.md) 의 검증 전략 (PaScaL_TCS golden 회귀) 은 그대로 유지하되, **케이스별로** 실행:

```
tests/
├── unit/                              ← 라이브러리 단위 테스트
│   ├── test_problem_defaults.cpp
│   ├── test_sweep_order.cpp
│   ├── test_field_vector_scalar.cpp
│   ├── test_tdma_backend.cpp
│   └── test_bc_apply.cpp
├── apps/                              ← 앱별 회귀 테스트
│   ├── rbc/
│   │   ├── golden_pascal_tcs_Ra100_Pr1.bin
│   │   └── compare.py
│   └── channel/
│       ├── golden_mpm_std_fortran.bin
│       └── compare.py
└── integration/
    ├── test_thermal_only.cpp          ← 라이브러리 단위
    ├── test_momentum_only.cpp
    └── test_poisson_only.cpp
```

**핵심 보장**: 라이브러리 단위 테스트는 케이스 비의존. 앱별 회귀는 RBC↔PaScaL_TCS, Channel↔MPM-STD Fortran 각각 비교.

---

## Part 7. 단계별 구현 로드맵 (v2)

| 마일스톤 | 라이브러리 추가 | 앱 추가 | DoD |
|---|---|---|---|
| **M0** | Runtime, MpiTopology, Subdomain, Grid, ScalarField, VectorField, FieldRegistry, Backend(Cpu), Io | — | 단위 테스트 통과 |
| **M1** | BcKind/FaceBc/FieldBoundary/Problem/BoundaryApplier, TdmaSolver+PascalTdmaBackend | — | RBC 기본값 자동, sweep_order 도출, TDMA wrap |
| **M2** | ScalarEquation (3-stage ADI) | `apps/thermal_only_check/main.cpp` (속도 동결) | 제조해 2 차 수렴, 1/4/16 랭크 일치 |
| **M3** | MomentumEquation (predict+block_couple+pseudo_update), PropertyPolicy (`ConstantProperties`, `NobProperties`), SourceTerm (`BulkForcing`, `NobBuoyancy`, `BoussinesqBuoyancy`) | — | 1 step 중간속도 PaScaL_TCS 와 1e-12 이내 |
| **M4** | FftPlanner (R2C+DCT), TransposePlan, PressureEquation | — | div(U)<1e-12, 1 step dP PaScaL_TCS 와 1e-10 |
| **M5** | CflController, Diagnostics, StatisticsPlugin, RestartIo/InstantIo/StatsIo | **`apps/rbc/main.cpp`** | RBC Ra=100, Pr=1, 512×128×256, 10 step PaScaL_TCS golden L∞<1e-10 |
| **M6** | (라이브러리 추가 없음) | **`apps/channel/main.cpp`** | forced channel Re_τ=180 표준값 일치 |
| **M7** | `IbmPlugin` slot 마련 (구현은 비워둠) | `apps/channel_ibm/main.cpp` 스켈레톤 | Plugin 인터페이스 안정성 |
| **M8** | `LesPlugin`, `LesProperties` | `apps/channel_les/main.cpp` | LES Smagorinsky 표준값 |
| **M9** | `FilteredTdmaBackend`, `CudaBackend` | (앱은 그대로 동작) | TDMA 백엔드 교체 가능 검증 |

---

## Part 8. 결론

v1 (06 보고서) 의 단일 `Solver` god-class 패턴을 폐기하고, **라이브러리 + 케이스별 main.cpp** 패턴으로 전환. 이는:

1. **사용자 컨트롤 강화** — 시간 루프의 단계 순서를 사용자가 명시적으로 결정
2. **유지보수성** — 새 case 추가는 새 `apps/<case>/main.cpp` 작성. 라이브러리 미변경
3. **검증 단순성** — 각 case 가 독립 실행파일이라 회귀 비교가 명확
4. **OpenFOAM·PyFR 패턴 부합** — 검증된 산업 표준

**Field 추상화** — `StagLocation` enum 폐기. `VectorField` (face) + `ScalarField` (cell) 컨벤션 으로 표현. 단, **타입이 인덱스 실수를 막아주지 못함을 정직히 인정**. 대신 **`stencil/` 자유함수 캡슐화 + 단위 테스트** 로 인덱스 산술 안전성 확보.

**시간 적분기** — Crank-Nicolson + ADI 단일 채택. Strategy 패턴 폐기. Explicit/고차/Spectral 통합 시도하지 않음 (자료 흐름 자체가 달라 통합 비용 > 이득). 단 RHS 어셈블리는 자유함수로 분리하여 미래 추출이 쉽도록.

남은 v1 의 가치 있는 부분:
- `Problem` 객체 ([05_BC_design.md](05_BC_design.md))
- sweep_order 자동 도출
- Plugin 인터페이스 + Phase enum
- PropertyPolicy + SourceTerm 분리 ([07_momentum_unification.md](07_momentum_unification.md))
- TDMA 백엔드 추상화
- Backend 추상화
- 검증 전략

→ 이 7 가지는 v2 에서도 그대로. 단 *호출 주체* 가 라이브러리의 `Solver` 가 아니라 *사용자 main.cpp* 라는 점만 다름.

---

## v1 → v2 변경 요약표

| 항목 | v1 (06) | v2 (08) |
|---|---|---|
| 케이스 분기 | `Solver` 클래스 + `Problem` | **`apps/<case>/main.cpp` 분리** |
| Field 타입 안전 | `StagLocation` enum | **`VectorField` / `ScalarField` 컨벤션만** |
| 인덱스 실수 방지 | 타입 시스템 (과장) | **`stencil/` 자유함수 + 단위 테스트** |
| 시간 적분기 | `Scheme` Strategy + `CrankNicolsonAdi` | **CN+ADI 단일, 추상화 없음** |
| 시간 루프 호출 주체 | `Solver::run()` | **`main.cpp` 가 직접** |
| Plugin 인터페이스 | 유지 | 유지 (호출은 main 이) |
| `Problem` BC 객체 | 유지 | 유지 |
| sweep_order 자동 | 유지 | 유지 |
| PropertyPolicy / SourceTerm | 유지 | 유지 |
| TDMA 백엔드 추상 | 유지 | 유지 |
