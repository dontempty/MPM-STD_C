# CaNS 구조 분석 보고서

> 대상: [/shared/home/wel1come1234/workspace/CaNS](../../CaNS/)
> 목적: C++ 유동 솔버 설계를 위한 참고자료. 코드 흐름·모듈화 패턴·객체지향 요소 분석.

---

## 1. 코드 베이스 개요

**CaNS (Canonical Navier-Stokes)** 는 비압축성 NS 채널/덕트/캐비티 솔버. Fortran 2003/2008, 30 개 모듈 약 12,700 LOC. MPI + OpenMP + OpenACC(GPU) 멀티 백엔드.

### 디렉토리 구조

| 디렉토리 | 역할 |
|---|---|
| [src/](../../CaNS/src/) | 30 개 `.f90` 모듈 (전체 솔버 본체) |
| [configs/](../../CaNS/configs/) | 컴파일러·플래그·라이브러리 빌드 설정 (`compilers.mk`, `flags.mk`, `libs.mk`) |
| [examples/](../../CaNS/examples/) | 20+ 케이스 (채널, 덕트, TGV, 캐비티 등 namelist 셋) |
| [dependencies/](../../CaNS/dependencies/) | submodules: **2decomp&fft** (CPU pencil decomp + FFT), **cuDecomp** (NVIDIA), **diezDecomp** (AMD) |
| [utils/](../../CaNS/utils/) | 후처리 스크립트 |
| [tests/](../../CaNS/tests/) | 회귀·검증 테스트 |
| [Makefile](../../CaNS/Makefile) | AWK (`gen-deps.awk`) 로 자동 의존성 추출, MPI/OpenMP/OpenACC 빌드, FP32/FP64 토글 |

**핵심 인사이트**: 분할(decomposition)·FFT·GPU 통신을 모두 **외부 라이브러리에 위임**. CaNS 본체는 물리·수치만 작성.

---

## 2. 시작 → 종료 실행 흐름

### 2.1 초기화 (main.f90:149–397)

```
MPI_INIT
  → read_input(namelist input.nml)        ! mod_param
  → initmpi (decomp2d/cuDecomp init, pencil decomp, GPU device)
  → allocate u,v,w,p,pp [0:n+1] (1-width halos)
  → initgrid (z-stretching, x,y uniform)
  → initsolver (FFT plans + tridiag matrix)
  → initflow (initial condition by 'inivel' string)
  → bounduvw / boundp (halo + BC)
```

### 2.2 시간 루프 — 3-stage low-storage RK3 (main.f90:427–642)

```fortran
do irk = 1, 3
  ! (1) 스칼라 (옵션)
  call rk_scal()                   ! convection + (옵션) implicit diffusion
  if (is_impdiff) call solve_helmholtz()

  ! (2) 운동량 RHS
  call rk()                        ! convection + pressure-grad + viscous + forcing + buoyancy
                                   ! viscous 는 explicit 또는 solve_helmholtz()

  ! (3) 압력 보정 (분리법)
  call bounduvw()
  call fillps()                    ! RHS = div(u*)
  call updt_rhs_b()                ! BC 효과를 RHS 에 흡수
  call solver()                    ! ∇²p = RHS  via  2D-FFT(x,y) + TDMA(z)
  call boundp()
  call correc()                    ! u ← u − dt·∇p
  call updatep()                   ! p ← p + p_rk
end do
```

### 2.3 종료 (main.f90:643–658)

```
stop criterion (nstep / time_max / wallclock)
  → I/O (0D/1D/2D/3D 통계, snapshot, checkpoint)
  → fftend()
  → decomp_2d_finalize() / cudecomp_finalize()
  → MPI_FINALIZE
```

---

## 3. 모듈 인벤토리

| 분류 | 모듈 | LOC | 책임 |
|---|---|---|---|
| **Infra** | `mod_types` | 20 | 정밀도 (sp/dp) |
| | `mod_param` | 393 | 전역 파라미터, namelist 파싱 |
| | `mod_common_mpi` | 16 | MPI 글로벌 (myid, halo) |
| | `mod_common_cudecomp` | — | GPU decomp 핸들 |
| | `mod_timer` | 385 | wallclock 타이머 |
| **Grid & Domain** | `mod_initgrid` | 265 | z 비균일 격자 |
| | `mod_initmpi` | 298 | decomp2d 초기화, pencil 방향 선택 |
| | `mod_initsolver` | 233 | FFT 계획 + 삼중대각 계수 |
| **Numerics: Time** | `mod_rk` | 578 | RK3 + 운동량 RHS |
| | `mod_chkdt` | — | CFL |
| | `mod_chkdiv` | — | 발산 노름 |
| **Numerics: P-V Coupling** | `mod_fillps` | — | Poisson RHS = div(u*) |
| | `mod_solver` | 637 | CPU FFT+TDMA Poisson |
| | `mod_solver_gpu` | 1356 | GPU cuFFT+PTDMA |
| | `mod_solve_helmholtz` | — | Helmholtz wrapper |
| | `mod_correc` / `mod_updatep` | — | velocity 보정 / pressure 누적 |
| **Momentum & Flux** | `mod_mom` | 1468 | convection·diffusion 커널 (`momx_a`, `momx_d`, …) |
| **BC** | `mod_bound` | 632 | halo 교환, BC 적용 |
| **Scalar** | `mod_scal` | 352 | 패시브 스칼라 transport |
| **FFT** | `mod_fft` | 762 | FFTW/cuFFT 래퍼 |
| **IO** | `mod_load` | 1875 | HDF5/ADIOS2 체크포인트 |
| | `mod_output` | 530 | 통계/스냅샷 (`out0d/1d/2d/3d`) |
| | `mod_post` | 172 | vorticity 등 후처리 |
| **Util** | `mod_initflow` | 500 | 초기조건 |
| | `mod_sanity` | 418 | 입력 검증 |
| | `mod_utils` | — | bulk_mean 등 helper |
| **GPU** | `mod_workspaces` | 164 | cuFFT 작업버퍼 |

---

## 4. 객체지향·모듈화 패턴

Fortran 은 OO 가 아니지만 CaNS 는 **derived type** + **module-level procedure** 로 객체지향에 가깝게 작성.

### 4.1 `type(scalar)` — 데이터+행동 캡슐화의 대표 예

[src/scal.f90](../../CaNS/src/scal.f90) (lines 17–38):
```fortran
type scalar
  real(rp), allocatable :: val(:,:,:)        ! 필드 값
  real(rp), allocatable :: dsdtrko(:,:,:)    ! RK 저장
  real(rp) :: alpha                          ! 확산계수
  character(len=100) :: ini                  ! 초기조건 이름
  character(len=1), dimension(0:1,3) :: cbc  ! 면별 BC 종류 (P/D/N)
  real(rp), dimension(0:1,3) :: bc           ! 면별 BC 값
  real(rp) :: source, f, scalf               ! forcing
  logical :: is_forced
  real(rp), dimension(0:1,3) :: fluxo        ! 경계 flux
  type(C_PTR), dimension(2,2) :: arrplan     ! FFT 계획 (C 포인터)
  real(rp), allocatable :: lambdaxy(:,:)     ! FFT 고유값
  real(rp), allocatable :: a(:), b(:), c(:)  ! TDMA 계수
  real(rp) :: normfft
  type(rhs_bound) :: rhsb                    ! 경계 RHS
end type scalar
```

사용:
```fortran
type(scalar), allocatable :: scalars(:)
type(scalar), pointer    :: s
allocate(scalars(nscal))
do iscal = 1, nscal
  s => scalars(iscal)
  call rk_scal(..., s%val, s%alpha, ...)
  call solve_helmholtz(..., s%arrplan, s%lambdaxy, ..., s%val)
end do
```

→ **C++ 의 `class` 멤버 + 메서드 패턴과 동치**. 다만 method 는 자유함수로 분리 (Fortran 관행).

### 4.2 작은 부 타입 `rhs_bound`

```fortran
type rhs_bound
  real(rp), allocatable :: x(:,:,:), y(:,:,:), z(:,:,:)
end type
```

면별 RHS 보정값을 한 묶음으로. **POD struct** 패턴.

### 4.3 포인터 배열을 통한 generic I/O

```fortran
type arr_ptr
  real(rp), pointer, contiguous :: arr(:,:,:)
end type
type(arr_ptr), allocatable :: io_vars(:)
io_vars(1)%arr => u
io_vars(2)%arr => v
```

→ **type erasure** 흉내. C++ 에서는 `std::span<double>` 또는 추상 `Field*` 로 같은 효과.

### 4.4 "Solver 객체" 부재 — 절차적 분해

CaNS 는 모든 걸 묶는 `Solver` type 이 **없다**. 대신:
- 전역 module-level 변수 (mod_param 등) 가 설정값 보관
- 큰 배열 (`u,v,w,p`) 은 `main` 의 local 로 alloc 후 인자 전달
- 라이브러리 (FFT, decomp) 는 `type(C_PTR)` 핸들로 보관

→ **C++ 포팅 시 가장 큰 개선 여지**. `class Solver` 가 위 상태를 다 가지면 테스트·재현성이 향상.

### 4.5 연산자(div, grad, lap)는 자유 서브루틴

OpenFOAM 처럼 `fvc::div(U)` 가 아니라 `call momx_a(...)`, `call fillps(...)` 같은 **명시적 호출**. 추상도는 낮지만 성능 튜닝 (loop fusion) 이 쉬움.

---

## 5. C++ 포팅에 직접 매핑되는 추상화

### 5.1 Grid

- 스태거드 MAC: `u,v,w` face center, `p` cell center
- 인덱싱: `(0:n(dir)+1, ...)` halo width 1
- z 만 stretching, x·y uniform
- pre-computed inverse metrics: `dxi, dyi, dzci, dzfi`

```cpp
class Grid {
  int nx, ny, nz;
  double dx, dy;
  std::vector<double> z_c, z_f, dz_c, dz_f;
  std::vector<double> dxi, dyi, dzci, dzfi;
};
```

### 5.2 Field

- 1-width halo, contiguous 단일 할당
- Fortran column-major → C++ 에서는 row-major 변환 또는 명시적 strided index

```cpp
class Field {
  std::vector<double> data;
  int nx, ny, nz;
  static constexpr int HW = 1;
  double& operator()(int i,int j,int k) {
    return data[((i+HW)*(ny+2*HW) + (j+HW))*(nz+2*HW) + (k+HW)];
  }
};
```

### 5.3 BoundaryCondition — char 코드 기반

CaNS 의 `cbc(0:1, 3)` 는 면별 문자 코드 ('P'/'D'/'N'). **데이터-구동 BC** 의 좋은 예.

```cpp
enum class BCType { Periodic, Dirichlet, Neumann };
struct FaceBC { BCType type; double value; };
struct BoundaryConditions { FaceBC face[6]; /* 6 faces */ };
```

### 5.4 PoissonSolver — FFT(x,y) + TDMA(z)

```cpp
class PoissonSolver {
  std::vector<std::vector<double>> lambdaxy; // 고유값 (사전계산)
  std::vector<double> a, b, c;               // TDMA 계수
  double norm_fft;
  void solve(Field& rhs, Field& sol, const BoundaryConditions& bc) {
    fft_forward_xy(rhs);
    for (int ij : modes) tdma_z(sol_ij, lambdaxy[ij] + b, a, c, rhs_ij);
    fft_inverse_xy(sol);
  }
};
```

→ **PaScaL_TCS 와 동일한 구조** (FFT 면 직교화 + 1D TDMA). 우리 MPM-STD 의 PressureSolver 와 1:1 대응.

### 5.5 DomainDecomposition — 2decomp 추상화

```cpp
class DomainDecomposition {
  MPI_Comm cart_comm;
  int dims[3];
  std::vector<int> local_range[3];
  void halo_exchange(Field& f, int dir);
  void transpose_x_to_y(Field& px, Field& py);
};
```

→ MPI 를 numerics·physics 코드로부터 격리하는 패턴은 **CaNS 의 최대 강점**.

### 5.6 TimeIntegrator — RK3 coefficient 테이블

```cpp
class TimeIntegrator {
  // CaNS: rkcoeff(0:1, 3) = ((32/60,0), (25/60,−17/60), (45/60,−25/60))
  void step(FieldStorage& fs, double dt, MomentumSolver& mom, PoissonSolver& pp) {
    for (int s = 0; s < 3; ++s) {
      double dtrk = (a[s] + b[s]) * dt;
      mom.compute_rhs(fs, rhs, s);
      fs.U.axpy(a[s]*dt, rhs.U);
      // pressure correction ...
    }
  }
};
```

---

## 6. C++ 클래스로 직역 가능한 모듈 매핑

| Fortran 모듈/타입 | 제안 C++ 클래스 |
|---|---|
| `mod_param` | `SimulationConfig` |
| `type(scalar)` | `PassiveScalarField : public Field` |
| `mod_initmpi` + 2decomp | `DomainDecomposition` |
| `mod_initgrid` | `Grid` |
| `mod_initsolver` + `mod_solver` | `PoissonSolver` |
| `mod_rk` | `TimeIntegrator` |
| `mod_bound` | `BoundaryConditionManager` |
| `mod_mom` | `MomentumEquation` (또는 free `ConvectionKernel`/`DiffusionKernel`) |
| `mod_correc` + `mod_fillps` | `PressureCorrection` |
| `mod_scal` | `ScalarTransport` |
| `main` 루프 | `FlowSolver` (전체 orchestrator) |

---

## 7. PaScaL_TCS 와의 차이 (C++ 설계 시사점)

| 항목 | CaNS | PaScaL_TCS |
|---|---|---|
| **도메인** | 채널/덕트/캐비티/triperiodic 등 일반 | 채널 중심 (NOB Rayleigh-Bénard) |
| **Decomp** | 2decomp&fft / cuDecomp (pencil) | 자체 3D Cart + 1D/2D sub-comm |
| **격자** | z만 stretching, x·y uniform | 3축 독립 tanh stretching 가능 |
| **시간적분** | RK3 (low-storage, 3-stage) | Crank-Nicolson + ADI 3-sweep |
| **점성** | explicit / implicit (Helmholtz) 선택 | 모두 implicit (ADI) |
| **압력 해법** | FFT(x,y) + TDMA(z) | 전치 2회 + 2D-FFT + TDMA(wall axis) |
| **GPU** | OpenACC + cuDecomp / diezDecomp | (대부분 CPU) |
| **IO** | HDF5, ADIOS2, binary | binary MPI-IO |
| **물성** | 상수 또는 Boussinesq | NOB (T-dependent ρ, μ, κ) |

### C++ 솔버 설계에 가져올 점

1. **외부 라이브러리 위임** — pencil 통신·FFT 는 검증된 라이브러리에 맡겨라 (`2decomp` ↔ `PaScaL_TDMA_C` + `FFTW`).
2. **BC 의 데이터-구동** — `char` 코드 또는 `enum` + value 로 면별 BC. 코드 분기 없음.
3. **물성·forcing 의 캡슐화** — `type(scalar)` 처럼 한 변수에 필요한 모든 상태 (val, dt 저장, BC, FFT plan, TDMA 계수) 를 묶기.
4. **사전계산 메트릭** — `dxi`, `dzfi` 처럼 자주 쓰는 inverse 는 한 번 계산해 보관.
5. **명시적 procedure 호출** — operator overloading 으로 추상화 올리지 말고, hot loop 은 명시적 커널 호출 (성능).
6. **Sanity 모듈 분리** — `mod_sanity` 처럼 입력 검증을 한 곳에 모으기.

### 가져오지 말 것

- **글로벌 module-level 변수** — `Solver` 클래스로 묶어야 테스트·재현성 확보.
- **Fortran C-pointer 직접 노출** — C++ 에서는 RAII 로 FFTW plan 등 핸들 관리.
- **포인터 배열 트릭** — `std::variant`, `std::span`, 또는 다형 인터페이스로 깔끔하게 처리.

---

## 8. 결론

CaNS 는 **관심사 분리 (separation of concerns)** 의 모범:

1. **물리 커널** 은 순수 compute 루프 (`mod_mom`).
2. **분할·통신** 은 외부 라이브러리 (`2decomp`, `cuDecomp`) 로 격리.
3. **BC** 는 character 코드 + value 의 데이터-구동 설정.
4. **FFT·TDMA** 는 얇은 인터페이스 (`solve_helmholtz`, `solver`) 로 백엔드 교체 가능.
5. **GPU·CPU** 는 조건부 컴파일 + OpenACC 단일 소스.

C++ MPM-STD 가 직접 채택할 패턴:
- `Grid`, `Field`, `DomainDecomposition`, `BoundaryConditions` 같은 핵심 데이터 객체
- `MomentumEquation`, `PoissonSolver`, `ScalarTransport` 의 컴포저블 클래스
- `TimeIntegrator` 가 RK stage + pressure correction 을 orchestrate
- `SimulationConfig`, `FieldWriter` 가 IO/설정 담당
- **MPI 호출은 numerics 레벨까지만; physics 코드에는 절대 노출하지 않음**

→ 이 구조 그대로 가져오면 PaScaL_TCS 의 NOB 알고리듬을 C++ 로 옮기면서도 미래의 다른 BC·물리·GPU 확장이 자연스러워진다.
