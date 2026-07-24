# 기존 설계서 비판적 분석 + 수정안

> 대상 문서: [cfd_solver_architecture_gpu_mpi_design_report.md](cfd_solver_architecture_gpu_mpi_design_report.md)
> 목적: 다른 AI 가 작성한 C++ CFD 솔버 설계서를 비판적으로 검토하고, 본 프로젝트의 실제 요구사항 (PaScaL_TCS / MPM-STD Fortran 알고리듬 + BC 설계서 + RBC/Channel 통합 + 미래 IBM/LES 확장) 에 맞춘 수정안 제시
> 참고: [01_CaNS_analysis.md](01_CaNS_analysis.md), [02_PyFR_analysis.md](02_PyFR_analysis.md), [03_MPM-STD_Fortran_analysis.md](03_MPM-STD_Fortran_analysis.md), [04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md), [05_BC_design.md](05_BC_design.md)

---

## Part A. 비판적 분석

### A.1 잘한 점 (보존할 가치)

| # | 항목 | 평가 |
|---|---|---|
| 1 | Field / Model / Equation / Solver / Kernel / MPI / GPU **7 계층 분리** | 관심사 분리 원칙 양호. PyFR 패턴과 부합. |
| 2 | `FieldRegistry` 동적 생성 | 사용하지 않는 변수 alloc 방지. RBC/Channel 둘 다에 유효. |
| 3 | GPU host/device 포인터 분리 | MPM-STD Fortran 의 `u`/`u_d` 패턴과 일치. |
| 4 | "GPU kernel은 절차적, 상위는 OO" 명시 | 핵심 통찰. PyFR 의 Backend 패턴과 부합. |
| 5 | `HaloExchange` 가 Field 의존성에서 분리 | CaNS 의 `mod_bound` 패턴. |
| 6 | `BuoyancyModel`, `LESModel`, `PropertyModel` 별도 분리 | 옵션 기능을 Plugin-like 로 떼어둠. |
| 7 | `main.cpp` 의 단순함 | 좋은 관행. |

### A.2 결정적 문제 13 가지

| # | 문제 | 영향 |
|---|---|---|
| **1** | **`ChannelSolver` / `RBCSolver` 클래스 분리** | 가장 큰 설계 오류. RBC·Channel 은 *같은 NS 알고리듬*. 차이는 T 방정식 활성화 + buoyancy + BC 값뿐. 클래스 분리하면 알고리듬 코드가 **중복 작성**되고 두 케이스의 회귀 검증이 어려워짐. |
| **2** | **`ChannelMomentumEquation` / `RBCMomentumEquation` 분리** | 1 과 동일한 문제. 운동량 방정식은 동일. buoyancy 는 *추가 source term* 일 뿐. 분리는 부적절. |
| **3** | **MPM-STD 알고리듬 디테일 부재** | ADI 3-stage sweep, blockLdU/V 보정, `dPhat=2dP−dP_prev` 외삽, BC-aware FFT/DCT 자동 선택 — 네 분석 보고서가 발견한 *핵심 알고리듬 구조*가 설계서에 한 줄도 없음. |
| **4** | **TDMA 가 `LinearSolver` 의 sibling 으로 `KrylovSolver` 와 나란히** | TDMA 는 ADI sweep 의 *기본 도구*. Krylov 의 대체재가 아님. 같은 부모 밑에 둬 추상화가 부적절. **Krylov 는 본 프로젝트 범위 밖** — 제거 권장. |
| **5** | **`Mesh / GlobalMesh / LocalMesh / HaloInfo` 4 계층** | 구조 격자엔 과한 추상화. CaNS·PaScaL_TCS·MPM-STD 모두 축별 1D 배열 (`x[3]`, `dx[3]`, `dmx[3]`) 로 충분. `Mesh` 추상화는 PyFR (unstructured) 영향이지만 본 프로젝트엔 부적절. |
| **6** | **MAC 스태거드 격자 표현 부재** | U, V, W 가 *서로 다른 face center* 에 있는 점이 `VectorField` 라는 단일 컨테이너로 압축됨. 스태거 위치 (cell, faceX, faceY, faceZ) 가 *타입* 또는 *런타임 태그* 로 표현되어야 함. 이게 빠지면 stencil 작성 시 인덱스 오류 다발. |
| **7** | **`bc.add("U", NoSlip, YMinus)` string-keyed API** | 1차에서 string lookup, runtime 에러 (오타). [05_BC_design.md](05_BC_design.md) 의 `enum BcKind` + 타입 안전 API 와 모순. 또한 설계서 자체가 "GPU kernel 에 unordered_map lookup 금지" 라고 명시하면서 BC 는 string-keyed — 불일치. |
| **8** | **wall-normal 축이 y (`YMinus/YPlus`) 로 암시** | [05_BC_design.md](05_BC_design.md) 에서 사용자가 명시한 **z = wall-normal (RBC 기본)** 과 정면 충돌. PaScaL_TCS 컨벤션 (y) 을 무비판적으로 차용한 흔적. |
| **9** | **PaScaL_TDMA 백엔드 / 백엔드 교체 가능성 미언급** | 사용자가 명시: PaScaL_TDMA → Filtered_TDMA 교체 가능해야 함. 설계서엔 단순 `TDMA` 클래스만 존재. |
| **10** | **확장 (IBM, LES, Inflow/Outflow) 의 Plugin slot 부재** | LES 와 Buoyancy 는 별도 클래스로 분리되어 있지만 *어떻게 등록·호출* 되는지 미정의. IBM 은 언급 없음. 각각이 ad-hoc 으로 붙으면 코드 부패. 통합된 Plugin 인터페이스 필요. |
| **11** | **검증 / 회귀 전략 부재** | PaScaL_TCS 의 golden output 과 어떻게 일치 확인? Channel↔RBC 결과의 상호 일관성 어떻게 보장? "Solver 별 분리" 설계로는 검증 자체가 어려움. |
| **12** | **메모리 레이아웃 / halo 정책 미정의** | row-major? halo 폭? AoS vs SoA? GPU 정렬? — 한 줄도 없음. 본 프로젝트의 hot path 성능을 결정짓는 사안. |
| **13** | **Time integrator 미설계** | "Time loop" 가 의사코드로만 있고, *어떤 시간 적분 스킴* (Crank-Nicolson + ADI? RK?), *어떻게 추상화* 할지 (Strategy 패턴?) 가 부재. PyFR 의 Integrator 계층 같은 명시적 클래스 필요. |

### A.3 약한 점 (개선 권장)

| # | 항목 | 보완 |
|---|---|---|
| 14 | `GPUManager` 가 막연 | Backend 추상화 (PyFR 스타일) 로 명시. Memory pool, stream, device id 책임 명확화. |
| 15 | `Config` 가 모든 곳에 reference 로 전달 | OK 이나, **`Problem` 객체** ([05_BC_design.md](05_BC_design.md)) 가 BC + topology + 활성 방정식 셋 을 묶어 전달하는 게 더 깔끔. |
| 16 | "MomentumEquation::solve()" 의 의미 모호 | 한 step 풀이? 한 stage 풀이? ADI 세 sweep 중 하나? 인터페이스 시그니처가 알고리듬 의도를 못 담음. |
| 17 | initialize() 의 단계 분리는 좋으나 *순서* 가 약함 | 예: Field allocate 이전에 BC 가 와야 wall 위치를 알고 staggered face 결정 가능. |
| 18 | Logger 만 있고 Diagnostics / Statistics 누락 | CFL, divergence, Nusselt, energy 등 진단은 필수. |
| 19 | IO 가 `initializeIO()` 한 줄 | restart write/read, instantaneous snapshot, statistics dump — 모두 인터페이스 명시 필요. |

---

## Part B. 수정 설계안

기존 설계서의 7 계층 골격은 유지하되, **위 13 + 6 문제를 모두 해결** 하는 수정안.

### B.1 핵심 변경 7 가지

| 변경 | Before | After |
|---|---|---|
| **1. Solver 단일화** | `BaseSolver` + `ChannelSolver` + `RBCSolver` | 단일 `Solver` + `Problem` 객체로 분기 |
| **2. Equation 단일화** | `Channel*Equation` / `RBC*Equation` 분리 | 단일 `MomentumEquation` 등 + Source term 주입 |
| **3. BC** | string-keyed `bc.add("U", NoSlip, ...)` | [05_BC_design.md](05_BC_design.md) 의 enum + `FieldBoundary` 타입 안전 API |
| **4. wall-normal** | YMinus/YPlus 암시 | **z 기본 (RBC), `Problem` 에서 자동 도출** |
| **5. Plugin 계층** | 없음 | `Plugin` 인터페이스 (PyFR 스타일). LES, Buoyancy, IBM, Stats 모두 Plugin. |
| **6. TDMA 백엔드** | 단일 `TDMA` 클래스 | `TdmaSolver` 추상 + `PascalTdmaBackend` / `FilteredTdmaBackend` |
| **7. Mesh 간소화** | 4 계층 (`GlobalMesh/LocalMesh/HaloInfo`) | `Grid` 단일 클래스 (CaNS·PaScaL_TCS 패턴) |

### B.2 클래스 트리 (수정안)

```
mpmstd/
├─ main.cpp                          ← 얇음 유지 (기존 OK)
│
├─ Config                            ← INI/TOML 파서. const ref 전달.
├─ Problem                           ← [05_BC_design.md] 의 BC + topology + 활성 방정식 세트
│
├─ Runtime
│  ├─ MpiTopology                    ← 3D Cart + 1D sub-comm × 3 + 2D sub-comm (FFT)
│  ├─ Subdomain                      ← 인덱스 범위, halo DDT, 전치 DDT
│  ├─ Backend (abstract)             ← CPU/CUDA 추상. Memory + Stream + Kernel dispatch.
│  │  ├─ CpuBackend
│  │  └─ CudaBackend                 ← 미래 (인터페이스만 1차에 마련)
│  └─ Logger / Diagnostics
│
├─ Grid                              ← 축별 x, dx, dmx (단일 클래스, Mesh 4 계층 폐기)
│
├─ Field
│  ├─ StagLocation enum              ← Cell, FaceX, FaceY, FaceZ
│  ├─ Field<T>                       ← host + device 포인터, halo, stag-tagged
│  └─ FieldRegistry                  ← 동적 생성 (기존 OK)
│
├─ Boundary
│  ├─ BcKind, FaceBc, FieldBoundary  ← [05_BC_design.md]
│  ├─ DomainTopology
│  └─ BoundaryApplier                ← apply_ghost, modify_tdma_row
│
├─ Numerics
│  ├─ TdmaSolver (abstract)
│  │  ├─ PascalTdmaBackend
│  │  └─ FilteredTdmaBackend         ← 미래
│  ├─ FftPlanner                     ← FFTW3 (Periodic→R2C, Neumann→DCT-II, Dirichlet→DST)
│  └─ TransposePlan                  ← MPI_Alltoallw + DDT (C↔I, I↔K)
│
├─ Equation                          ← 단일 클래스 (Channel/RBC 분기 없음)
│  ├─ MomentumEquation               ← predict(c, T) + block_couple + pseudo_update
│  ├─ PressureEquation               ← RHS + BC-aware FFT + projection + dPhat
│  └─ ScalarEquation                 ← 온도, 다종 스칼라 공통 (NameTrait 로 의미 부여)
│
├─ Plugin (abstract)                 ← PyFR 영감. 매 step / setup / finalise 훅.
│  ├─ BuoyancyPlugin                 ← T → momentum source
│  ├─ LesPlugin                      ← Smagorinsky 등 → ν_t field 갱신
│  ├─ IbmPlugin                      ← 미래 IBM 추가 slot
│  ├─ ForcingPlugin                  ← bulk pressure gradient (Channel)
│  ├─ StatisticsPlugin               ← 통계 누적
│  └─ ProbePlugin                    ← 점 단위 출력
│
├─ TimeStepper
│  ├─ Scheme (abstract)              ← Strategy
│  │  └─ CrankNicolsonAdi            ← 1차 유일
│  └─ CflController                  ← dt 갱신
│
├─ Io
│  ├─ RestartIo                      ← MPI-IO binary
│  ├─ InstantIo
│  └─ StatsIo
│
└─ Solver                            ← orchestrator. **단 하나의 클래스**.
   - initialize() / step_once() / run() / finalize()
   - Plugin 리스트 보유, time loop 가 Plugin 호출 시점 결정
```

### B.3 단일 Solver + Problem 객체 — Channel/RBC 통합

```cpp
class Solver {
public:
  Solver(const Config& cfg, const Problem& problem,
         MpiTopology& mpi, Backend& be);

  void initialize();
  void run();
  void finalize();

private:
  void step_once();

  const Config&    cfg_;
  Problem          problem_;        // BC + topology + active equations
  MpiTopology&     mpi_;
  Backend&         backend_;

  Grid             grid_;
  Subdomain        sub_;
  FieldRegistry    fields_;
  BoundaryApplier  bc_;

  std::unique_ptr<MomentumEquation> momentum_;
  std::unique_ptr<PressureEquation> pressure_;
  std::optional<ScalarEquation>     thermal_;   // RBC 면 활성, Channel-only 면 미생성

  TdmaRegistry      tdma_;
  FftPlanner        fft_;
  std::unique_ptr<Scheme> scheme_;
  CflController     cfl_;

  std::vector<std::unique_ptr<Plugin>> plugins_;   // Buoyancy/LES/IBM/Forcing/Stats
  Diagnostics       diag_;
  std::unique_ptr<RestartIo>  restart_;
  std::unique_ptr<InstantIo>  instant_;
  std::unique_ptr<StatsIo>    stats_;

  double time_ = 0, dt_ = 0;
  int    step_ = 0;
};
```

`Problem` 이 RBC 인지 Channel 인지 결정하므로 **솔버 클래스 분기 불필요**:

```cpp
// main.cpp
Problem problem;  // 생성자가 RBC 기본값 (z=wall-normal) 자동 채움
if (cfg.case_type == "channel") {
  problem.T_disable();              // ThermalSolver 비활성
  plugins.push_back(std::make_unique<ForcingPlugin>(cfg));  // bulk dp/dx 강제
}
// rbc 는 추가 작업 없음 (Problem 기본값이 RBC)

Solver solver(cfg, problem, mpi, backend);
solver.run();
```

### B.4 step_once() 의 골격 (Plugin 통합)

```cpp
void Solver::step_once() {
  // 1) Plugin pre-step (예: LES ν_t 갱신)
  for (auto& p : plugins_) if (p->phase() == Phase::PreStep) p->call(*this);

  // 2) Thermal (활성화 시)
  if (thermal_) {
    thermal_->compute_coeffi(fields_.scalar("T"));
    thermal_->step(fields_, dt_);
    sub_.exchange_halo(fields_.scalar("T"));
  }

  // 3) Plugin between-thermal-momentum (예: Buoyancy → momentum source)
  for (auto& p : plugins_) if (p->phase() == Phase::BeforeMomentum) p->call(*this);

  // 4) Momentum predict + block couple + pseudoupdate
  momentum_->compute_coeffi(fields_);
  for (Component c : {U, V, W}) {
    momentum_->predict(c, fields_, dt_);
    sub_.exchange_halo(fields_.increment(c));
  }
  momentum_->block_couple_V(fields_);  sub_.exchange_halo(fields_.dV());
  momentum_->block_couple_U(fields_);
  momentum_->pseudo_update(fields_);
  for (auto v : {U,V,W}) sub_.exchange_halo(fields_.vector(v));

  // 5) Plugin between-momentum-pressure (예: IBM 셀 재구성)
  for (auto& p : plugins_) if (p->phase() == Phase::BeforePressure) p->call(*this);

  // 6) Pressure
  pressure_->compute_rhs(fields_);
  pressure_->solve(fields_, problem_.topology, fft_, tdma_);
  sub_.exchange_halo(fields_.scalar("dP"));
  pressure_->project(fields_, dt_);
  for (auto v : {U,V,W,"P"}) sub_.exchange_halo(fields_.get(v));

  // 7) Plugin post-step (Stats, Probes)
  for (auto& p : plugins_) if (p->phase() == Phase::PostStep) p->call(*this);

  // 8) Diagnostics + dt 갱신
  diag_.check_divergence(fields_);
  dt_ = cfl_.next_dt(fields_, grid_);

  time_ += dt_; ++step_;
}
```

→ **알고리듬은 PaScaL_TCS / MPM-STD Fortran 의 13 단계 1:1**. 단, Plugin 슬롯이 단계 사이에 끼어들 수 있어 IBM 등 미래 확장이 본체 수정 없이 가능.

### B.5 BC 기반 sweep order / FFT / TDMA 자동 도출

[05_BC_design.md](05_BC_design.md) 의 `Problem` 객체가 모든 분기를 흡수:

```cpp
// Momentum ADI 한 stage
for (Direction d : problem_.topology.sweep_order()) {  // periodic 먼저, wall 마지막
  build_bands_along(d, ...);
  if (problem_.topology.is_periodic(d))
    tdma_.get(d).solve_many_cyclic(A, B, C, D, n_sys, n_row);
  else {
    bc_.modify_tdma_row(d, fbc_for_field, A,B,C,D, n_sys, n_row);
    tdma_.get(d).solve_many(A,B,C,D, n_sys, n_row);
  }
}

// Pressure FFT — BC 종류에 따라 자동
for (int a = 0; a < 3; ++a) {
  auto d = Direction(a);
  if (a == int(problem_.topology.wall_axis().value())) continue;   // wall 축은 TDMA
  if (problem_.topology.is_periodic(d)) plans[a] = R2C_FFT;
  else if (P.face(d, -1).kind == Neumann) plans[a] = DCT_II;
  else                                    plans[a] = DST_II;
}
```

→ PaScaL_TCS (wall=y), MPM-STD Fortran (wall=z), 미래 IO-Channel (wall=x,z) 가 **같은 코드** 로 처리됨. 기존 설계서의 `ChannelSolver/RBCSolver` 분기는 원천 불필요해짐.

### B.6 Plugin 인터페이스 (IBM·LES·Buoyancy·Forcing 통합)

```cpp
enum class Phase {
  Setup, Finalise,
  PreStep, BeforeMomentum, BeforePressure, PostStep
};

class Plugin {
public:
  virtual ~Plugin() = default;
  virtual Phase phase() const = 0;
  virtual void  call(Solver& s) {}
  virtual void  setup(Solver& s) {}
  virtual void  finalise(Solver& s) {}
  std::string name;
  int  every_nsteps = 1;
};

// 구현 예
class BuoyancyPlugin : public Plugin {
  Phase phase() const override { return Phase::BeforeMomentum; }
  void call(Solver& s) override {
    // T → momentum source term H 에 누적
    accumulate_buoyancy(s.fields().scalar("T"),
                        s.fields().vector("H"),
                        cfg_);
  }
};

class IbmPlugin : public Plugin {           // ← 1차에 인터페이스만, 미래 구현
  Phase phase() const override { return Phase::BeforePressure; }
  void call(Solver& s) override {
    // forcing field 적용 + 셀 재구성
  }
};

class LesPlugin : public Plugin {
  Phase phase() const override { return Phase::PreStep; }
  void call(Solver& s) override {
    // ν_t 갱신
    update_eddy_viscosity_smagorinsky(
      s.fields().vector("U"), s.fields().scalar("nu_t"), cfg_);
  }
};

class ForcingPlugin : public Plugin {       // Channel 전용 bulk dp/dx
  Phase phase() const override { return Phase::BeforeMomentum; }
  void call(Solver& s) override {
    add_constant_pressure_gradient(s.fields().vector("H"), cfg_);
  }
};

class StatisticsPlugin : public Plugin {
  Phase phase() const override { return Phase::PostStep; }
  void call(Solver& s) override {
    if (s.step() >= cfg_.t_start_stat) accumulate_means(s.fields(), means_);
  }
};
```

→ **새 물리/모델 추가 = Plugin 자식 클래스 1 개**. 본체 (`Solver::step_once`) 는 손대지 않음. IBM 추가가 이미 마련된 slot 에 들어감.

### B.7 TDMA 백엔드 추상화

```cpp
class TdmaSolver {
public:
  virtual void solve_many       (double* A,double* B,double* C,double* D,
                                 int n_sys,int n_row) = 0;
  virtual void solve_many_cyclic(double* A,double* B,double* C,double* D,
                                 int n_sys,int n_row) = 0;
};

class PascalTdmaBackend : public TdmaSolver { /* PaScaL_TDMA_C 래핑 */ };
class FilteredTdmaBackend : public TdmaSolver { /* 미래 */ };

// Config 에서 축별 백엔드 선택
// tdma.backend.x1 = pascal
// tdma.backend.x2 = pascal
// tdma.backend.x3 = pascal      ← 후일 filtered 로 교체 가능
class TdmaRegistry {
  std::array<std::unique_ptr<TdmaSolver>, 3> backends_;
public:
  TdmaSolver& get(Direction d) { return *backends_[int(d)]; }
};
```

### B.8 Field — MAC stagger 와 host/device 통합

```cpp
enum class StagLocation { Cell, FaceX, FaceY, FaceZ };

template<typename T = double>
class Field {
public:
  Field(int n1sub, int n2sub, int n3sub, StagLocation stag, Backend& be);

  T*       host_ptr();
  T*       device_ptr();      // 1차 미할당 가능, Backend 가 결정
  void     to_device();
  void     to_host();

  StagLocation stag() const { return stag_; }
  std::string  name() const { return name_; }

  // 인덱스 접근 (host only)
  T& operator()(int i, int j, int k);

private:
  std::vector<T> host_;
  T*  device_ = nullptr;
  int n1_, n2_, n3_;
  static constexpr int HW = 1;
  StagLocation stag_;
  std::string  name_;
  Backend&     backend_;
};
```

→ U 의 stag = `FaceX`, V = `FaceY`, W = `FaceZ`, P/T = `Cell`. stencil 작성 시 어떤 face 인덱스인지 타입으로 명시 — 인덱싱 실수 차단.

### B.9 ScalarEquation 단일화 (T, 다종 스칼라 공통)

```cpp
struct ScalarTraits {
  std::string name;                // "T", "Y_O2" 등
  double      diffusivity;
  bool        has_source = false;
  std::function<void(Field&, const Solver&)> source_fn;
};

class ScalarEquation {
public:
  ScalarEquation(ScalarTraits traits, Backend& be);
  void compute_coeffi(const Field& phi);
  void step(FieldRegistry& fields, double dt);
};
```

→ Temperature·passive scalar 가 **같은 구현**. 다종 스칼라 확장 비용 0.

### B.10 검증 / 회귀 전략

기존 설계서는 부재. 본 수정안의 핵심 추가 항목:

```
tests/
├── unit/
│   ├── test_tdma_backend.cpp           ← PaScaL_TDMA wrapper 정확성
│   ├── test_grid_stretch.cpp           ← tanh stretching 메트릭
│   ├── test_bc_apply.cpp               ← 면별 ghost 채움
│   ├── test_problem_defaults.cpp       ← Problem 생성자 RBC 기본값
│   └── test_sweep_order.cpp            ← topology → sweep_order 자동 도출
├── integration/
│   ├── test_thermal_only.cpp           ← 속도 동결 + manufactured solution
│   ├── test_poisson_only.cpp           ← Poisson 단독 + 알려진 해
│   └── test_momentum_only.cpp          ← 압력 없이 운동량 1 step
└── regression/
    ├── golden_pascal_tcs/              ← PaScaL_TCS 10-step golden
    │   ├── Ra100_Pr1_512x128x256.bin
    │   └── compare.py
    └── golden_mpm_std_fortran/         ← Fortran MPM-STD 케이스
        └── ...
```

**검증 기준**: PaScaL_TCS Ra=100, Pr=1, 512×128×256, 10 step 결과 와 L∞ < 1e-10 일치 (동일 FP 플래그). [04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md) §11 참고.

---

## Part C. 마이그레이션 가이드 — 기존 설계서에서 무엇을 옮기고 무엇을 폐기

### C.1 그대로 가져갈 부분

- `main.cpp` 의 얇음 (단, `ChannelSolver/RBCSolver` 분기는 제거)
- 7 계층 분리 철학 (Field/Equation/Model/Solver/Kernel/MPI/GPU)
- `FieldRegistry` 동적 생성
- GPU host/device 포인터 분리
- "상위 OO, 하위 절차적 kernel" 원칙
- `HaloExchange` 가 Field 와 분리
- `Logger`

### C.2 폐기해야 할 부분

| 폐기 항목 | 이유 |
|---|---|
| `ChannelSolver`, `RBCSolver` 클래스 분리 | Problem 객체로 흡수 |
| `ChannelMomentumEquation`, `RBCMomentumEquation` | Source term Plugin 으로 흡수 |
| string-keyed `bc.add(...)` | enum + `FieldBoundary` ([05_BC_design.md](05_BC_design.md)) |
| `YMinus/YPlus` 컨벤션 | z=wall (RBC 기본), 축은 BC 에서 자동 도출 |
| `KrylovSolver` | 본 프로젝트 범위 밖. FFT-TDMA 로 충분 |
| `Mesh / GlobalMesh / LocalMesh / HaloInfo` 4 계층 | 단일 `Grid` |
| `GPUManager` 단일 클래스 | `Backend` 추상화로 대체 |

### C.3 새로 도입해야 할 부분

| 신규 항목 | 출처 |
|---|---|
| **`Problem` 객체** | [05_BC_design.md](05_BC_design.md) |
| `BoundaryApplier` + `BcKind` enum | [05_BC_design.md](05_BC_design.md) |
| `Plugin` 인터페이스 + Phase enum | PyFR ([02_PyFR_analysis.md](02_PyFR_analysis.md)) |
| `TdmaSolver` 추상 + 백엔드 등록 | 사용자 요구사항 (PaScaL_TDMA → Filtered_TDMA) |
| `Backend` 추상 (`CpuBackend` / 미래 `CudaBackend`) | PyFR |
| `StagLocation` enum + `Field<T>` 타입 안전 | CaNS [01_CaNS_analysis.md](01_CaNS_analysis.md) |
| `Scheme (CrankNicolsonAdi)` Strategy | PyFR 영감 |
| 알고리듬 디테일: blockLdU/V, dPhat 외삽, BC-aware FFT/DCT | [04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md), [03_MPM-STD_Fortran_analysis.md](03_MPM-STD_Fortran_analysis.md) |
| 회귀 테스트 + golden output 비교 | [04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md) §11 |
| `Diagnostics` 클래스 (CFL, div, Nu, energy) | PaScaL_TCS `mod_post` |
| `RestartIo` / `InstantIo` / `StatsIo` 별도 클래스 | MPM-STD Fortran `cuda_post_*` |

---

## Part D. 단계별 구현 로드맵 (수정안 기준)

| 마일스톤 | 범위 | DoD |
|---|---|---|
| **M0** | `Config`, `MpiTopology`, `Subdomain`, `Grid`, `Field`, `FieldRegistry`, `Backend(Cpu)`, `Io` 스캐폴드 | 8 랭크 실행, namelist 파싱, 필드 alloc, restart write/read, ghost-exchange unit test 통과 |
| **M1** | `BcKind`, `FaceBc`, `FieldBoundary`, `Problem`, `BoundaryApplier`, `TdmaSolver`+`PascalTdmaBackend` | RBC 기본값 자동 셋업 단위 테스트, sweep_order 자동 도출, TDMA wrap unit test |
| **M2** | `ScalarEquation` (T) — 속도 동결 ADI 3-stage | 제조해 검증, 시·공간 2 차 수렴, 1/4/16 랭크 일치 |
| **M3** | `MomentumEquation` (predict + block_couple + pseudoupdate), Buoyancy/Forcing Plugin | 1 step 중간속도가 PaScaL_TCS 와 1e-12 이내 |
| **M4** | `FftPlanner` (R2C + DCT-II), `TransposePlan`, `PressureEquation` (BC-aware FFT + wall TDMA) | div(U) < 1e-12, 1 step dP 가 PaScaL_TCS 와 1e-10 이내 |
| **M5** | 전체 결합, `CrankNicolsonAdi` Scheme, `CflController`, `Diagnostics`, `StatisticsPlugin` | RBC Ra=100, Pr=1, 512×128×256, 10 step ↔ PaScaL_TCS golden L∞ < 1e-10 |
| **M6** | Channel forced 케이스 (T 비활성, `ForcingPlugin`), 검증 | 알려진 Re_τ vs 표준값 일치 |
| **M7** | `IbmPlugin` (slot 만 마련, 1차 구현은 사용자 요청 시) | Plugin 인터페이스 안정성 검증 |
| **M8 (미래)** | `LesPlugin`, `FilteredTdmaBackend`, `CudaBackend` | 옵션 |

---

## Part E. 핵심 요약

기존 설계서의 가장 큰 오류는 **"케이스(Channel/RBC) = 솔버 클래스"** 의 등가치환. 본 수정안은 **"케이스 = `Problem` 데이터 객체 + Plugin 조합"** 으로 재구성하여:

1. **RBC ↔ Channel 알고리듬 코드 1 본** (중복 없음)
2. **wall-normal 축은 BC 에서 자동 도출** (z 하드코딩 / y 하드코딩 모두 회피)
3. **새 물리 (IBM, LES, 다종 스칼라) = Plugin 한 개** (본체 손대지 않음)
4. **TDMA 백엔드 교체 가능** (PaScaL → Filtered)
5. **알고리듬 디테일 명시** (blockLdU/V, dPhat 외삽, BC-aware FFT/DCT, ADI sweep)
6. **검증 전략 내장** (PaScaL_TCS golden output 회귀)

기존 설계서의 7 계층 분리는 좋은 출발점이었으나, **MPM-STD 의 실제 알고리듬 디테일** 과 **본 프로젝트의 BC/확장 요구사항** 을 반영하지 못함. 본 수정안은 다섯 분석 보고서 ([01](01_CaNS_analysis.md), [02](02_PyFR_analysis.md), [03](03_MPM-STD_Fortran_analysis.md), [04](04_PaScaL_TCS_analysis.md), [05](05_BC_design.md)) 의 결론을 통합한 결과물.
