# PyFR 구조 분석 보고서

> 대상: [/shared/home/wel1come1234/workspace/PyFR](../../PyFR/)
> 목적: C++ 유동 솔버 설계를 위한 OO 패턴·추상화 계층 참고.
> 주의: PyFR 은 **고차 Flux Reconstruction + GPU**. 우리 MPM-STD (구조 FDM 채널) 와 도메인은 다르지만 **설계 패턴**은 직접 차용 가능.

---

## 1. 코드 베이스 개요

- 언어: Python (수치 hot kernel 은 Mako 템플릿으로 CUDA/HIP/Metal/OpenCL/OpenMP/x86 SIMD 코드 생성)
- 본체 경로: [pyfr/](../../PyFR/pyfr/)
- 엔트리: [pyfr/__main__.py](../../PyFR/pyfr/__main__.py)
- 설정: `.ini` (configparser)
- 메시: 자체 HDF5 포맷 (`.pyfrm`)

### 디렉토리 구조

| 패키지 | 역할 |
|---|---|
| `backends/` | **하드웨어 추상 계층** (cuda, hip, metal, opencl, openmp, base) |
| `solvers/` | 물리 시스템 (Euler, NS, advection-diffusion 베이스) |
| `integrators/` | 시간적분 (explicit RK, implicit, controller) |
| `readers/` | 메시 입력 (Gmsh, native) |
| `writers/` | 출력 (VTK, native HDF5) |
| `partitioners/` | 도메인 분할 (METIS, Scotch) |
| `plugins/` | 런타임 확장 (모니터링, 후처리, source term) |
| `quadrules/` | 요소별 quadrature 데이터 |
| `resamplers/` | 메시 리파인 시 보간 |
| `inifile.py` | 설정 파싱 |
| `mpiutil.py` | MPI 래퍼 |

**핵심 인사이트**: 디렉토리 분류 자체가 클래스 계층의 1:1 반영. **"수평 분리 (계층) + 수직 분리 (백엔드)"** 가 직교한다.

---

## 2. 시작 → 종료 실행 흐름

`pyfr run case.pyfrm case.ini` 호출 시:

```python
# __main__.py::main()
ArgumentParser → subcommand 'run' → process_run(args)

# process_run → _process_common(args, None, Inifile.load(args.cfg))
init_mpi()
NativeReader(args.mesh)                  # 메시 로드 (HDF5)
backend = get_backend(args.backend, cfg) # cuda/hip/openmp 중 선택
                                         #   subclass_where(BaseBackend, name=...)
solver = get_solver(backend, mesh, soln, cfg)
    systemcls = subclass_where(BaseSystem, name=cfg['solver']['system'])
    integrator = get_integrator(backend, systemcls, mesh, initsoln, cfg)
        # ★ 동적 type 합성:
        controller_cls = ...   # 예: SSPK54Controller
        stepper_cls    = ...   # 예: SSPRK43Stepper
        integrator_cls = type('Comp', (controller_cls, stepper_cls), {})
        return integrator_cls(backend, systemcls, mesh, initsoln, cfg)
optionally wrap with ProgressBar plugin
solver.run()    # → for t in tlist: advance_to(t)
```

**한 줄 요약**: 솔버는 단일 클래스가 아니라 **(controller + stepper) 의 동적 합성 타입** 이다.

---

## 3. 핵심 클래스 계층

### 3.1 Backend 계층 — 하드웨어 추상화

```python
# pyfr/backends/base/backend.py
class BaseBackend:
    name = None                   # 'cuda', 'openmp', ...
    has_double = True
    def __init__(self, cfg):
        self.fpdtype = float32/64
        self.ixdtype = int32/64
    def malloc(self, obj, extent): ...
    def commit(self): ...
    def kernel(self, name, *args, **kwargs):   # 핵심: 이름으로 커널 dispatch
        for prov in self._providers:
            try: return prov.kernel_for(name, ...)
            except NotSuitableError: continue
    def const_matrix(self, initval, ...): ...
    def matrix(self, ioshape, ...): ...
    def view(self, matmap, rmap, cmap, ...): ...
    def run_kernels(self, kernels): ...
    def run_graph(self, graph): ...           # GPU DAG 최적화

class CUDABackend(BaseBackend):    name = 'cuda'
class OpenMPBackend(BaseBackend):  name = 'openmp'
class HIPBackend(BaseBackend):     name = 'hip'
class MetalBackend(BaseBackend):   name = 'metal'
class OpenCLBackend(BaseBackend):  name = 'opencl'
```

**설계 원리**:
- 백엔드는 **kernel provider 의 컬렉션**.
- `kernel(name, ...)` 가 provider 들을 순회하며 `name` 을 처리할 수 있는 것을 찾음.
- Provider 예: `PointwiseKernels`, `BlasExtKernels`, `PackingKernels`, `xsmm` (CPU 전용 BLAS).
- **autotuning**: 여러 provider 가 가능하면 실측 후 가장 빠른 것 선택.

### 3.2 System / Elements / Inters — 물리·기하·인터페이스 분리

```python
# pyfr/solvers/base/system.py
class BaseSystem:
    elementscls = None     # 요소 타입 클래스 (예: EulerElements)
    intinterscls = None    # 내부 인터페이스
    mpiinterscls = None    # MPI 인터페이스
    bbcinterscls = None    # 경계 인터페이스
    def __init__(self, backend, mesh, initsoln, registers, cfg, serialiser):
        ...
        eles, elemap, ics = self._load_eles(...)
        self._int_inters = self._load_int_inters(...)
        self._mpi_inters = self._load_mpi_inters(...)
        self._bc_inters, self._bc_prefns = self._load_bc_inters(...)
        self._alloc_register_banks(registers, eles, ics)
    # 물리 hook
    convars(ndims, cfg) -> [str]    # 보존변수 이름 (rho, rhoE, ...)
    privars(ndims, cfg) -> [str]    # 원시변수 이름

# 계층:
BaseSystem
  └─ BaseAdvectionSystem
       ├─ EulerSystem
       └─ BaseAdvectionDiffusionSystem
            └─ NavierStokesSystem
```

```python
# pyfr/solvers/base/elements.py
class BaseElements:
    def __init__(self, basiscls, eles, cfg):
        self.basis = basiscls(nspts, cfg)
        self.nupts, self.nfpts, self.nqpts = basis...
        self.kernels = {}                 # 이름→커널 테이블

# pyfr/solvers/base/inters.py
class BaseInters:
    def __init__(self, be, lhs, elemap, cfg):
        self.elemap = elemap
        self.kernels = {}
        self.mpireqs = {}
        self._external_args = {}; self._external_vals = {}
```

**핵심**:
- `BaseSystem` = **기하 + 레지스터 (메모리) 관리** — 물리 모름.
- `EulerSystem` 같은 자식 = **물리 (flux, BC)**.
- `Elements` / `Inters` = **커널 테이블만 보유** — 비즈니스 로직 없음.
- 새 물리 (NS, MHD) 추가 = `BaseSystem` 자식 + 새 커널 템플릿. 기하·메모리 코드 재사용.

### 3.3 Integrator 계층 — 동적 합성

```python
# pyfr/integrators/base.py
class BaseIntegrator(metaclass=RegisterMeta):
    def __init__(self, backend, systemcls, mesh, initsoln, cfg):
        self.dt = cfg.getfloat('solver-time-integrator', 'dt')
        self.plugins = []
        self.nrhsevals = 0

class BaseExplicitIntegrator(BaseIntegrator): ...
class BaseExplicitController(BaseExplicitIntegrator): ...     # 스텝 크기 제어
class BaseExplicitStepper(BaseExplicitIntegrator):    ...     # Butcher tableau

# get_integrator() 에서:
IntegCls = type(name, (ControllerCls, StepperCls), {'name': name})
return IntegCls(backend, systemcls, mesh, initsoln, cfg)
```

**왜 이 패턴이 우월한가**:
- N 개 controller × M 개 stepper = N+M 클래스 (선언). 사용 조합만 동적 합성.
- 단일 상속 + Cartesian product = N×M 클래스 폭발을 피함.
- 새 controller 추가 시 기존 stepper 와 자동 조합.

### 3.4 Plugin 시스템 — 옵저버 패턴

```python
# pyfr/plugins/base.py
class BasePlugin:
    name = None
    systems = None      # 정규식: 어떤 시스템과 호환?
    dimensions = None
    nsteps = None       # N 스텝마다 호출
    trigger = None      # 조건식
    def __init__(self, *, cfg=None, cfgsect=None, ndims=None): ...
    def __call__(self, intg): pass           # 매 스텝
    def setup(self, sdata, prevcfg, ser): pass    # 1회
    def finalise(self, intg): pass                 # 종료
```

예: `ResidualPlugin`, `SourcePlugin`, `FWHPlugin` (음향 후처리).

**핵심 디자인**:
- 플러그인은 **솔버의 public 인터페이스만 본다** (`intg.tcurr`, `intg.system.ele_banks`).
- 솔버는 플러그인 리스트를 갖고 매 스텝 호출만 함 — **양방향 의존성 없음**.
- 새 기능 (난류 모델, source term, 통계) = 플러그인 추가. 본체 수정 X.

---

## 4. 커널 생성·디스패치 전략

### 4.1 Mako 템플릿 기반 코드 생성

```
pyfr/backends/base/kernels/packing.mako        # 공통
pyfr/backends/cuda/kernels/axnpby.mako         # CUDA y = αx + βy
pyfr/backends/cuda/kernels/reduction.mako
pyfr/solvers/euler/kernels/...                 # 물리 (flux, RHS)
pyfr/integrators/explicit/kernels/...          # 시간적분 (register update)
```

예 — `axnpby.mako` 가 렌더 시점에 다음을 주입:
- `fpdtype`, `ixdtype` (정밀도/인덱스 타입)
- `ncola`, `nv` (변수 개수)
- in/out scale factor

→ 같은 템플릿에서 `axnpby<float32, int32, 5, 4>` 같은 **특화 인스턴스** 들이 생성.

### 4.2 Provider 패턴

```python
def kernel(self, name, *args, **kwargs):
    best = None
    for prov in self._providers:        # [Pointwise, BlasExt, Packing, xsmm]
        meth = getattr(prov, name, None)
        if meth:
            try: kern = meth(*args, **kwargs)
            except NotSuitableError: continue
            if best is None or kern.dt < ifac * best.dt:
                best = kern
    return best
```

- **Pointwise**: Mako 템플릿 커널 (flux, residual).
- **BlasExt**: BLAS 래퍼 (padding 포함 행렬곱).
- **Packing**: AoS↔SoA 변환.
- **xsmm**: libxsmm 특화 (CPU SIMD).

### 4.3 물리 → 커널 디스패치 예 (Euler RHS)

```python
# 1. 내부 flux
self.backend.kernel('tflux_int_flux', ele_bank, ...).run()
# 2. 인터페이스 flux (Riemann)
self.backend.kernel('tinter_comm_flux', interface_bank, ...).run()
# 3. divergence → RHS
self.backend.kernel('tdiv', ele_bank, ...).run()
```

→ 솔버 코드는 **커널 이름** 만 알면 됨. 백엔드가 적절한 구현 선택.

---

## 5. 설정 관리

**INI 형식**:
```ini
[solver]
system = euler
order  = 3

[solver-time-integrator]
tstart     = 0.0
tend       = 1.0
dt         = 0.001
controller = sspk54
scheme     = ssp-rk43

[backend]
precision    = double
memory-model = normal
```

**Inifile 클래스**:
```python
class Inifile:
    def get(section, option, default=_sentinel): ...
    def getint / getfloat / getbool(section, option, default): ...
    def getexpr(section, option, subs={}): ...     # 수식 평가
    def items_as(section, type, prefix=''): ...
```

**전파 방식**: **글로벌 싱글톤 아님**. 생성자 체인을 통해 const reference 로 전달. → 테스트·재현성·병렬 sweep 친화적.

---

## 6. C++ 솔버에 직접 적용 가능한 패턴

### 6.1 Backend 추상화

```cpp
class Backend {
public:
  virtual ~Backend() = default;
  virtual Kernel* get_kernel(std::string name, const KernelArgs&) = 0;
  virtual void   run_kernels(std::span<Kernel*>) = 0;
  virtual Buffer alloc(std::size_t bytes) = 0;
};

class CPUBackend : public Backend { ... };
class CUDABackend : public Backend { ... };  // 미래

// Provider 패턴
class KernelProvider {
public:
  virtual std::optional<std::unique_ptr<Kernel>>
    try_get(std::string name, const KernelArgs&) = 0;
};
```

→ 우리 MPM-STD 1단계는 `CPUBackend` 만, 인터페이스만 미리 잡아두면 GPU 확장 비용이 낮다.

### 6.2 System/Solver 계층 분리

```cpp
// 기하·메모리 (물리 모름)
class ComputeSystem {
protected:
  std::vector<Register> registers_;
public:
  virtual void allocate_registers(int nstages);
  virtual void set_mesh(const Grid&);
};

// 물리
class IncompressibleNS : public ComputeSystem {
public:
  virtual void rhs(const VectorView& soln, VectorView& rhs);
  virtual std::unique_ptr<Kernel> flux_kernel(const Block&);
};

class NobRayleighBenard : public IncompressibleNS {
  // NOB 보정 + 온도 결합만 override
};
```

→ 우리 MPM-STD 의 `MomentumSolver`, `ThermalSolver` 도 같은 패턴으로 베이스→파생 으로 펼쳐도 좋다.

### 6.3 Strategy: Controller × Scheme

PyFR 의 동적 합성을 C++ 에서는 **Strategy 패턴**으로:

```cpp
class IntegrationScheme {
public:
  virtual void step(Integrator& intg, double dt) = 0;
};
class SSPRk43 : public IntegrationScheme { ... };
class CrankNicolson : public IntegrationScheme { ... };

class StepController {
public:
  virtual double next_dt(const StepInfo&) = 0;
};
class CFLController : public StepController { ... };

class Integrator {
  std::unique_ptr<IntegrationScheme> scheme_;
  std::unique_ptr<StepController>    controller_;
  void advance() {
    double dt = controller_->next_dt(...);
    scheme_->step(*this, dt);
  }
};
```

→ N controller + M scheme = N+M 클래스. 우리 MPM-STD 가 1단계엔 CN+ADI 하나뿐이지만, 이 인터페이스를 미리 잡으면 RK 추가가 후일 무비용.

### 6.4 Plugin 시스템

```cpp
class Plugin {
public:
  virtual ~Plugin() = default;
  virtual void on_step(Integrator&) {}
  virtual void on_setup(Integrator&) {}
  virtual void on_finalise(Integrator&) {}
  int  nsteps = 1;        // 호출 간격
};

class Integrator {
  std::vector<std::unique_ptr<Plugin>> plugins_;
  void step() {
    for (auto& p : plugins_)
      if (step_count_ % p->nsteps == 0) p->on_step(*this);
  }
};

// 응용
class StatsPlugin : public Plugin {
  void on_step(Integrator& i) override {
    if (i.tcurr() > t_start_stats_) accumulate_means(i);
  }
};
class ProbePlugin : public Plugin { ... };
class NusseltPlugin : public Plugin { ... };
```

→ 통계, 모니터, source term, 후처리, dump 를 모두 본체 손대지 않고 추가 가능.

### 6.5 Config = passed-by-reference data, not singleton

```cpp
class Config {
public:
  double  get_float(std::string sec, std::string key) const;
  int     get_int  (std::string sec, std::string key) const;
  bool    get_bool (std::string sec, std::string key) const;
  std::string get(std::string sec, std::string key, std::string def="") const;
};

// 생성자 체인으로 전달
Backend::Backend(const Config& cfg);
Solver::Solver(Backend& be, const Mesh& mesh, const Config& cfg);
```

→ **글로벌 싱글톤 금지**. 테스트·sweep·재현성 모두 향상.

---

## 7. 가져오지 말 것 (Anti-patterns)

| 안티패턴 | PyFR 에서 왜 정당화되는가 | 우리 구조 FDM 채널엔 왜 과한가 |
|---|---|---|
| **물리 커널까지 런타임 코드 생성** | 고차 FR + 다 백엔드 (CUDA/HIP/Metal/...) 필요 | 우리 1 백엔드(CPU), 1 stencil. 컴파일 타임 템플릿이면 충분 |
| **Mako 같은 별도 DSL** | 6 종 GPU/CPU 백엔드 단일 소스 유지 | C++ 템플릿·`constexpr` 로 해결 가능 |
| **메시 partitioner 통합** | unstructured | 구조격자는 MPI Cart + 1D/2D pencil 로 충분 |
| **요소(Elements)·인터페이스(Inters) 분리** | 고차 FR 에서 face flux 가 핵심 | 구조 FDM 은 stencil 직접 적용. 인터페이스 객체 불필요 |
| **다단계 metaclass (`RegisterMeta` 등)** | 자식 클래스 자동 등록 | C++ 에서는 등록 함수 / static initializer 면 충분 |

---

## 8. PyFR vs CaNS — 우리 C++ 솔버 관점에서의 비교

| 항목 | CaNS | PyFR | 우리 채택 |
|---|---|---|---|
| 언어 | Fortran 2003 | Python + Mako | **C++17/20** |
| OO 정도 | type + module | 깊은 상속 계층 | **얕은 상속 + Strategy + Plugin** |
| 백엔드 추상화 | 조건부 컴파일 (OpenACC) | 런타임 dispatch | **CPU 만 1단계, 인터페이스만 미리** |
| Solver 객체 | 없음 (절차적) | 동적 합성 type | **단일 `Solver` orchestrator + Strategy** |
| 설정 | namelist | INI passed-by-ref | **INI 또는 TOML, passed-by-ref** |
| Plugin 시스템 | 없음 | 1급 시민 | **1급 시민으로 채택** |
| 커널 dispatch | 명시적 호출 | provider/autotuning | **명시적 호출** (구조격자 1 stencil) |
| MPI 격리 | 잘 됨 (2decomp 위임) | 잘 됨 (mpiutil + mpi_inters) | **격리 (`MpiTopology`, halo 함수)** |

---

## 9. 결론 — 우리 MPM-STD (C++) 가 PyFR 에서 가져갈 것

직접 채택:

1. **Backend 추상 인터페이스** — 1단계 CPU 만 구현하더라도, 미래 GPU 를 위해 인터페이스 선언.
2. **Strategy 패턴 (Controller × Scheme)** — 시간적분을 단일 클래스로 만들지 말 것.
3. **Plugin 1급 시민화** — 통계, 모니터, source term, 후처리, dump 를 모두 플러그인으로.
4. **Config = const ref** — 글로벌 싱글톤 금지.
5. **계층 분리 (Backend / System / Integrator / Plugin / IO)** — 디렉토리 구조 자체가 계층을 반영.
6. **Plugin 호환성을 정규식/태그로 선언** — 어떤 시스템·차원에서 동작하는지 자기 선언.

채택하지 않을 것:

1. 물리 커널 런타임 코드 생성 (우리는 C++ template + `constexpr` 로 충분).
2. Mako 같은 DSL (C++ 자체 메타프로그래밍).
3. 요소/인터페이스 객체 (구조격자엔 과함).
4. metaclass 패턴 (C++ 에서는 abstract base + factory 등록).

→ **요약**: CaNS 가 "데이터 구조와 모듈 분리" 의 모범이라면, PyFR 은 "계층 분리와 확장 인터페이스" 의 모범이다. **두 코드의 장점을 합치면** 우리 MPM-STD(C++) 의 청사진이 된다. CaNS 의 `Field`/`Grid`/`Domain` 모델 + PyFR 의 `Backend`/`Integrator`/`Plugin` 패턴 = 1단계 단순함과 미래 확장성을 동시에 확보.
