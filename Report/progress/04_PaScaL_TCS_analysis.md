# PaScaL_TCS (Fortran) 구조 분석 보고서

> 대상: [/shared/home/wel1come1234/workspace/PaScaL_TCS](../../PaScaL_TCS/)
> 목적: C++ MPM-STD 의 **2차 레퍼런스**. Fortran MPM-STD 와 동일한 알고리듬 (block LU 분리 + ADI + FFT-TDMA Poisson) 의 CPU/원전 버전.
> 컨벤션: **y = wall-normal** (MPM-STD 의 z 와 다름. 메모리 [project_filtered_tdma_channel.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/project_filtered_tdma_channel.md) 확인).

---

## 1. 코드 베이스 개요

**PaScaL_TCS (Parallel and Scalable Library for Turbulent Channel-flow Simulation)**. 비-Oberbeck-Boussinesq Rayleigh-Bénard 자연대류 채널 솔버. Fortran 90/95, 8 개 `.f90` 파일 약 **5,678 LOC**. MPI + PaScaL_TDMA 라이브러리.

### 디렉토리 구조

| 디렉토리/파일 | 역할 |
|---|---|
| [src/](../../PaScaL_TCS/src/) | 본체 (`main.f90` + 7 모듈) |
| [PaScaL_TDMA/](../../PaScaL_TCS/PaScaL_TDMA/) | 병렬 TDMA 라이브러리 (정적 lib 으로 빌드) |
| [run/](../../PaScaL_TCS/run/) | `PARA_INPUT.dat` 네임리스트 |
| [Makefile](../../PaScaL_TCS/Makefile) | `make lib && make exe` 의 2-step 빌드 |
| `Makefile.inc` | 컴파일러·플래그 |
| `LICENSE`, `README.md` | 메타 |

### 빌드 단계

```
make lib   → PaScaL_TDMA/src/*.f90 → libpascal_tdma.a
make exe   → src/*.f90 → src/obj/*.o → 실행파일
make all   → 둘 다
make clean → 정리
```

### 외부 의존성

- **PaScaL_TDMA**: 동봉된 in-house TDMA 라이브러리 (MPI 분산 + cyclic/non-cyclic)
- **MPI**: 필수
- **FFT**: 없음 (in-house cosine/sine 변환 기반)

→ **Fortran MPM-STD 와 달리 GPU/cuFFT 의존성 없음**. 순수 CPU MPI 코드.

---

## 2. 시작 → 종료 실행 흐름

### 2.1 진입점 [main.f90](../../PaScaL_TCS/src/main.f90)

```fortran
program main
  call MPI_Init(...)
  ! 타이머 18 개 등록 (프로파일링)
  ! 출력 디렉토리 생성 (rank 0)
  call global_inputpara()           ! namelist 파싱
  call mpi_topology_make()
  call mpi_subdomain_make()
  call mpi_subdomain_mesh()
  call mpi_subdomain_ghostcell_setup()
  call mpi_thermal_allocation()
  call mpi_momentum_allocation()
  call mpi_pressure_allocation()
  call mpi_thermal_initial();    call mpi_thermal_boundary()
  call mpi_momentum_initial();   call mpi_momentum_boundary()
  call mpi_pressure_initial()
  call mpi_pressure_wave_number()
  ! 초기 ghost 교환 (T, U, V, W, P)

  do TimeStep = 1, Timestepmax
    !  --- time loop body (§2.3) ---
  end do

  ! restart 저장
  ! 모든 deallocate
  call MPI_Finalize()
end program
```

### 2.2 초기화 (main.f90:48–127)

| # | 블록 | 호출 | 산출물 |
|---|---|---|---|
| A1 | MPI 시작 | `MPI_Init` | `myrank`, `nprocs` |
| A2 | 타이머 18 개 | (프로파일링) | per-phase timer slot |
| A3 | 출력 디렉토리 | rank 0 만 | `./data/1_continue/`, `./data/2_instanfield/` |
| A4 | namelist 파싱 | `global_inputpara()` ([module_global.f90:75](../../PaScaL_TCS/src/module_global.f90)) | `Ra, Pr, n*m, np*, pbc*, Gamma*, DeltaT, MaxCFL`, NOB 다항식 계수 |
| A5 | Cart + sub-comm | `mpi_topology_make()` ([module_mpi_topology.f90:66](../../PaScaL_TCS/src/module_mpi_topology.f90)) | `comm_1d_x1/2/3`, `comm_1d_x1n2` (FFT 전용 2D) |
| A6 | 서브도메인 인덱스 | `mpi_subdomain_make()` | `ista..iend`, `n*sub`, `n*msub` |
| A7 | 격자·메트릭 | `mpi_subdomain_mesh()` | `x*_sub, dx*_sub, dmx*_sub` (uniform 또는 tanh) |
| A8 | Ghost DDT | `mpi_subdomain_ghostcell_setup()` | 면별 derived datatype |
| A9 | 필드 alloc | `mpi_*_allocation` | `T, U, V, W, P` 모두 `(0:n1sub, 0:n2sub, 0:n3sub)` |
| A10 | 초기조건 + BC | `mpi_*_initial`, `mpi_*_boundary` | 필드 초기화, `TBCup/bt`, `UBCup/bt` 등 |
| A11 | 파수 사전계산 | `mpi_pressure_wave_number()` | `dxk1(i), dzk(k)` |
| A12 | 초기 ghost 교환 | T, U, V, W, P | halo 채움 |

### 2.3 시간 루프 — 13 단계 (main.f90:129–223)

```
[B1]  mpi_thermal_coeffi()              ! κ(T), dκ/dT, 1/(ρCp) at T^n
      ↓
[B2]  mpi_thermal_solver(U,V,W)         ! T^{n+1}: ADI 3-stage  z→x→y
                                        ! 각 stage: build (a,b,c,d) → TDMA solve
      ↓ ghost T
[B3]  mpi_momentum_coeffi(T^{n+1})      ! μ(T), 1/ρ(T)
      ↓
[B4]  mpi_momentum_solvedU(T) → dU      ! ADI 3-stage z→x→y (TDMA cyclic z/x, non-cyclic y)
      ↓ ghost dU
[B5]  mpi_momentum_solvedV(T) → dV      ! 동일 구조
      ↓ ghost dV
[B6]  mpi_momentum_solvedW(T) → dW      ! 동일 구조
      ↓ ghost dW
[B7]  mpi_momentum_blockLdV(T) → dV 보정 ! V 의 cross-direction 결합
      ↓ ghost dV
[B8]  mpi_momentum_blockLdU(T) → dU 보정 ! U 의 결합 (dV 사용)
      ↓
[B9]  mpi_momentum_pseudoupdateUVW()    ! U* = U+dU, V*, W*;  dU/dV/dW deallocate
      ↓ ghost U, V, W
[B10] mpi_pressure_RHS(...)             ! div(U*) + NOB extra term
      ↓
[B11] mpi_pressure_Poisson_FFT2(...)    ! 전치 C→I, FFTx, 전치 I→K, FFTz,
                                        ! wall-축 TDMA (real+imag), 역FFT, 역전치
      ↓ ghost dP
[B12] mpi_pressure_Projection(...)      ! U -= dt·∇dP/ρ;  P += dP;  dPhat = 2dP - dP_prev
      ↓ ghost U, V, W, P, dPhat
[B13] mpi_Post_Div / MonitorOut / CFL   ! div check (>1e-3 면 abort), CFL 기반 dt 갱신
      ↓
      time += dt
```

→ **Fortran MPM-STD 와 동일한 MPM-STD 알고리듬 구조**. 차이는 (i) wall-축 y, (ii) sweep z→x→y, (iii) GPU 없음, (iv) IBM/LES/XSEM 없음, (v) NOB Rayleigh-Bénard 전용.

### 2.4 종료 (main.f90:224–242)

- restart 저장 (`mpi_Post_FileOut_Continue_Post_Reassembly_IO`)
- 모든 모듈 deallocate, `MPI_Finalize`

---

## 3. 모듈 인벤토리 (8 파일 / 5,678 LOC)

| 모듈 | LOC | 책임 |
|---|---|---|
| [main.f90](../../PaScaL_TCS/src/main.f90) | 245 | program main: init → time loop → finalize |
| [module_global.f90](../../PaScaL_TCS/src/module_global.f90) | 168 | 전역 파라미터, namelist 파싱, NOB 다항식 계수 (a10–a15, b10–b15, …) |
| [module_mpi_topology.f90](../../PaScaL_TCS/src/module_mpi_topology.f90) | 124 | 3D Cart + 1D sub-comm × 3 + 2D sub-comm (FFT 전치용) |
| [module_mpi_subdomain.f90](../../PaScaL_TCS/src/module_mpi_subdomain.f90) | 815 | 서브도메인 인덱스, 격자 메트릭, ghost DDT, FFT 전치 DDT |
| [module_solve_thermal.f90](../../PaScaL_TCS/src/module_solve_thermal.f90) | 534 | 에너지 방정식: 계수 (κ, 1/(ρCp)) + ADI 3-stage 솔버 |
| [module_solve_momentum.f90](../../PaScaL_TCS/src/module_solve_momentum.f90) | 1,461 | 운동량 U/V/W: dU/dV/dW ADI + blockLdU/V + pseudoupdate |
| [module_solve_pressure.f90](../../PaScaL_TCS/src/module_solve_pressure.f90) | 876 | 압력: RHS + FFT2 Poisson (전치 2회) + projection + dPhat 외삽 |
| [module_post.f90](../../PaScaL_TCS/src/module_post.f90) | 1,455 | 발산 체크, CFL, 모니터, restart IO (multiple 방식) |

### 분류

| 카테고리 | 모듈 |
|---|---|
| Infrastructure | main, module_global, module_mpi_topology, module_mpi_subdomain |
| Physics | module_solve_thermal, module_solve_momentum, module_solve_pressure |
| IO / Diagnostics | module_post |

→ **8 개 파일에 모든 것이 압축**. Fortran 채널 코드의 전형. CaNS (30 파일) 의 1/4, MPM-STD Fortran (~25 파일) 의 1/3. **모듈 분리가 거칠어 C++ 포팅 시 펼쳐야 함**.

---

## 4. 알고리듬 세부 — ADI 3-stage 표준

### 4.1 운동량 (module_solve_momentum.f90)

#### `mpi_momentum_solvedU` (line 225)

3-stage ADI for `dU`:

```
Stage 1 (z-direction, lines 281–416):
  - Skew-symmetric convection 빌드 (u1 = 0.5*(U(im,j,k)+U(i,j,k)))
  - Viscous: Mu(i)·dudx2 − Mu(im)·dudx1, cross-grad viscous_u12/13
  - LHS: ac = mAC·dt + 1.0, ap = mAP·dt, am = mAM·dt
  - PaScaL_TDMA_many_solve_cycle (line 420), batched (n1msub·n2msub)
Stage 2 (x-direction, lines 425–541):
  - 전치 → (j,k,i)
  - PaScaL_TDMA_many_solve_cycle (line 476), batched (n2msub·n3msub)
Stage 3 (y-direction, lines 550–541):
  - 전치 → (i,k,j)
  - PaScaL_TDMA_many_solve (line 541), non-cyclic (wall), (n3msub·n1msub)
```

#### Cross-direction matrix 의 함정

[feedback_bw_cross_direction.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_bw_cross_direction.md) 의 규칙: cross-direction matrix 에는 `0.5·∂q/∂d` 자기미분을 **포함하지 않음**. own-direction `M_d` 에만 들어감. 포함시 Poiseuille 평형이 깨짐.

#### 부력 / 압력

- Pressure: `-Cmp·invRhoc·(P(i,j,k) − P(im,j,k))/dmx1(i)` (RHS 에)
- Buoyancy: `Cmt·(Tc + a12pera11·Tc²·DeltaT)·invRhoc` (RHS 에 명시적, NOB 보정)

### 4.2 열 (module_solve_thermal.f90)

`mpi_thermal_solver` (line 175) — 운동량과 **동일한 z→x→y ADI 패턴**. 차이:
- 변동성 열전도도 `KPP(i,j,k)` 와 그 미분 `dKPP`
- Harmonic mean: `kPP1 = 0.5/dmx1(i)·(dx1(i)·KPP(im,j,k) + dx1(im)·KPP(i,j,k))`
- BC: Dirichlet (T 고정) at y-wall, periodic x/z

### 4.3 압력 (module_solve_pressure.f90)

#### RHS (`mpi_pressure_RHS`, line 140)

```
DivUm = (U(ip,j,k)-U(i,j,k))/dx1(i)
      + (V(i,jp,k)-V(i,j,k))/dx2(j)
      + (W(i,j,kp)-W(i,j,k))/dx3(k)
ExtraTerm = NOB 보정 ((1-invRho2)·ddpdx2 − ...) / dx_i
```

#### Poisson 해법 (`mpi_pressure_Poisson_FFT2`, line 489)

**2D FFT + 1D TDMA 의 핵심 데이터 이동**:

```
Phase 1: Forward
  C → I 전치 (line 558–566)     ← MPI_Alltoallw with DDT
  Forward FFT in x
  I → K 전치 (line 579–581)
  Forward FFT in z

Phase 2: Wall-axis TDMA
  Coefficient: Ac_r(k,i,j) = fft_acj − dxk1(i) − dzk(k)
  PaScaL_TDMA_many_solve (real, line 638)
  PaScaL_TDMA_many_solve (imag, line 639)
  Boundary: fft_amj=0 at j=1, fft_apj=0 at j=n2msub (Neumann)

Phase 3: Backward
  Inverse FFT z, then x
  K → I → C 역전치
  ∫P = 0 평균 제거 (line 702–714)

dPhat = 2·dP − dP_prev   ! 다음 step 외삽
```

#### Projection (`mpi_pressure_Projection`, line 756)

```
U(i,j,k) -= dt·Cmp·invRhoc·(dP(i,j,k) − dP(im,j,k))/dmx1(i)
V, W 동일 (NOB 보정 dPhat 사용)
P(i,j,k) += dP(i,j,k)
```

### 4.4 TDMA 호출 통계 — 13 회/step

| 단계 | 방향 | 종류 | n_sys |
|---|---|---|---|
| Thermal z / x / y | z, x, y | cycle, cycle, solve | (n1·n2), (n2·n3), (n3·n1) |
| dU z / x / y | 동일 | cycle, cycle, solve | 동일 |
| dV z / x / y | 동일 | cycle, cycle, solve | 동일 |
| dW z / x / y | 동일 | cycle, cycle, solve | 동일 |
| Pressure wall-축 | 1 회 (real + imag = 2 call) | solve | (n3·n1) on FFT-축 |

→ **9 (운동량) + 3 (열) + 2 (압력 real+imag) = 14 TDMA call / step**.

---

## 5. 메모리 / 상태 (인벤토리)

### 5.1 필드 (allocation은 각 물리 모듈이 담당)

| 필드 | shape | 위치 | 모듈 |
|---|---|---|---|
| `U, V, W` | `(0:n1sub, 0:n2sub, 0:n3sub)` | host | module_solve_momentum (line 58–59) |
| `P` | 동일 | host | module_solve_momentum |
| `dU, dV, dW` | 동일 | host (스텝 중 일시 alloc/dealloc) | module_solve_momentum (lines 548, 907, 1257) |
| `T` | 동일 | host | module_solve_thermal (line 49) |
| `μ, 1/ρ, κ, 1/(ρCp)` | 동일 | host | 각 솔버의 coeffi 단계에서 alloc, clean 에서 dealloc |

→ **모든 필드 host only** (GPU 없음).
→ index `0` 과 `n*sub` 가 ghost layer.

### 5.2 BC 배열

`UBCup_sub, UBCbt_sub`, `VBCup_sub/bt`, `WBCup_sub/bt`, `TBCup_sub/bt` 모두 `(0:n1sub, 0:n3sub)` 모양 (y wall 양면). 인덱스 마스크 `iC_BC, jC_BC, kC_BC` (interior=1, ghost=0) + 스태거드용 `iS_BC, jS_BC, kS_BC`.

### 5.3 격자

`mpi_subdomain` 모듈:
- `x1_sub(0:n1sub), x2_sub(0:n2sub), x3_sub(0:n3sub)`: cell center
- `dx1_sub, dx2_sub, dx3_sub`: cell spacing
- `dmx1_sub, dmx2_sub, dmx3_sub`: half-cell (face-to-face)
- Stretching: hyperbolic tangent `x = L/2·(1 + tanh(0.5·γ·(2ξ/Nm − 1))/tanh(0.5·γ))`

### 5.4 MPI 상태

```fortran
mpi_world_cart           ! 3D Cartesian
comm_1d_x1, _x2, _x3     ! 축별 1D sub-comm
comm_1d_x1n2             ! 압력 FFT 전용 2D sub-comm (x-y plane)
```

각 comm 구조는 `myrank, nprocs, west_rank, east_rank, mpi_comm` 캡슐화.

### 5.5 시간 적분 상태

- `time, dt`: 현재 시각과 시간폭
- `TimeStep`: 카운터 (1..Timestepmax)
- `dt 갱신`: `mpi_Post_CFL()` (module_post.f90:168–229), `newdt = MaxCFL·min(dx)/max(|U|/dx)`
- `dPhat = 2·dP − dP_prev`: 다음 step 압력 외삽 → 정확도 향상

### 5.6 입력 namelist 전체 (`PARA_INPUT.dat`)

```
&sim_continue        ContinueFilein/out, dir_*
&meshes              n1m, n2m, n3m
&MPI_procs           np1, np2, np3
&periodic_boundary   pbc1, pbc2, pbc3   ! 일반적으로 (T, F, T)
&uniform_mesh        uniform1/2/3       ! 1=uniform, 0=stretched
&mesh_stretch        gamma1/2/3
&aspect_ratio        Aspect1, H, Aspect3
&sim_parameter       Ra, Pr, DeltaT, MaxCFL
&sim_control         dtStart, tStart, Timestepmax, print_*
```

---

## 6. 모듈화 / OO 패턴

PaScaL_TCS 는 **derived type 거의 없음** + 모듈-수준 변수 사용. CaNS·MPM-STD Fortran 과 같은 절차적 스타일이지만 더 미니멀.

### 6.1 캡슐화 단위 = 모듈

각 물리 모듈이 자신의 필드 + 계수 + alloc/dealloc/init/solver/clean 을 보유. 외부는 `mpi_*_solver(...)` 같은 진입점만 호출.

```fortran
module mpi_solve_momentum
  real(rp), allocatable :: U(:,:,:), V(:,:,:), W(:,:,:), P(:,:,:)
  real(rp), allocatable :: Mu(:,:,:), invRho(:,:,:)  ! 계수
  ! ...
contains
  subroutine mpi_momentum_allocation();  end subroutine
  subroutine mpi_momentum_initial();     end subroutine
  subroutine mpi_momentum_boundary();    end subroutine
  subroutine mpi_momentum_coeffi(T);     end subroutine
  subroutine mpi_momentum_coeffi_clean(); end subroutine
  subroutine mpi_momentum_solvedU(T);    end subroutine
  ! ...
end module
```

### 6.2 작은 derived type

`mpi_topology` 의 `comm_1d` 구조체 (rank, nprocs, mpi_comm 등 묶음) 정도. 그 외에는 거의 없음.

### 6.3 dU/dV/dW 의 alloc-on-stage 패턴

`solvedU` 진입 시 `dU` 를 alloc, exit 직전 `pseudoupdateUVW` 에서 사용 후 dealloc. **스텝 사이에는 메모리 차지하지 않음**. 메모리 절약 ↔ alloc 오버헤드의 trade-off.

→ C++ 에서는 RAII + 재사용 버퍼 (`std::vector` reserve once) 로 둘 다 해결.

### 6.4 코드 중복

dU/dV/dW solver 가 거의 동일한 구조의 ~400 LOC 씩 (총 1,200 LOC) **3 번 복사**. Fortran MPM-STD 는 single `solvedU` + flag('U'/'V'/'W') 로 압축. PaScaL_TCS 는 분리되어 있음.

→ **C++ 에서 가장 직접적인 개선 포인트**: `predict(Component c, ...)` 단일 함수 + Direction template.

### 6.5 "Solver 객체" 부재 — main 이 모든 걸 호출

`main.f90` 이 모든 모듈의 진입점을 직접 호출하는 절차적 컨트롤러. 모듈 간 데이터는 module-scope 전역 변수 (`use module_global` 등) 로 공유.

→ C++ 에서는 `Solver` orchestrator + 명시적 의존 주입으로 개선.

---

## 7. C++ 포팅을 위한 핵심 추상화

### 7.1 Grid

```cpp
class Grid {
  std::array<int, 3>     n;          // n1m, n2m, n3m
  std::array<double, 3>  L;          // domain lengths
  std::array<std::vector<double>, 3> x, dx, dmx;  // metrics
  std::array<bool, 3>    periodic;
  std::array<bool, 3>    uniform;
  std::array<double, 3>  gamma;      // tanh stretch
};
```

### 7.2 Field (1단계 host only)

```cpp
class Field {
  std::vector<double> data;
  int n1sub, n2sub, n3sub;
  static constexpr int HW = 1;        // halo width
  StagLocation stag;                  // Cell, FaceX, FaceY, FaceZ
  double& operator()(int i,int j,int k);
};
```

### 7.3 BoundaryConditions — 면별 자유 설정

PaScaL_TCS 는 y-wall 만 지원. C++ 에서는 일반화:

```cpp
enum class BCKind { Periodic, Dirichlet, Neumann, Wall };
struct FaceBC { BCKind kind; double value; };
class BoundaryConditions {
  std::array<FaceBC, 6> faces;
public:
  Direction wall_axis() const;           // 자동 추론
  std::array<Direction, 3> sweep_order() const;  // periodic 먼저, wall 마지막
  bool is_periodic_axis(Direction d) const;
};
```

`sweep_order` 가 PaScaL_TCS (wall=y → z→x→y) 와 MPM-STD Fortran (wall=z → x→y→z) 을 같은 코드로 처리.

### 7.4 TDMA 추상 인터페이스

```cpp
class TdmaSolver {
public:
  virtual void solve_many       (double* A,double* B,double* C,double* D, int n_sys,int n_row) = 0;
  virtual void solve_many_cyclic(double* A,double* B,double* C,double* D, int n_sys,int n_row) = 0;
};
class PascalTdmaBackend : public TdmaSolver { /* PaScaL_TDMA_C 래핑 */ };
class FilteredTdmaBackend : public TdmaSolver { /* 미래 */ };
```

### 7.5 ADI sweep — direction-agnostic

```cpp
for (Direction d : bc.sweep_order()) {
  auto [n_sys, n_row] = subdomain.lines_along(d);
  fill_bands_along(d, phi, coeffs, A, B, C, D);
  if (bc.is_periodic_axis(d))
    tdma.solve_many_cyclic(A,B,C,D, n_sys, n_row);
  else {
    bc.modify_tdma_row_at_walls(d, A,B,C,D, n_sys, n_row);
    tdma.solve_many(A,B,C,D, n_sys, n_row);
  }
}
```

→ PaScaL_TCS 의 z→x→y 하드코딩을 BC 기반 자동 결정으로 대체.

### 7.6 Pressure Solver — 전치 2 회 + FFT + TDMA

```cpp
class PressureSolver {
  TransposePlan c_to_i, i_to_k;       // MPI Alltoallw + DDT
  FftPlan1D     fft_x, fft_z;         // FFTW3
  TdmaSolver&   tdma_wall;
  std::vector<double> dxk1, dzk;      // 사전계산 파수
public:
  void solve(const Field& rhs, Field& dp, const BoundaryConditions& bc);
};
```

### 7.7 Momentum predictor + block coupling

```cpp
class MomentumSolver {
  void predict(Component c, const Field& T);          // dU, dV, dW (단일 함수, c=U/V/W)
  void block_couple_V();                              ! blockLdV
  void block_couple_U();                              // blockLdU (dV 사용)
  void pseudo_update();                               // U* = U + dU
};
```

→ PaScaL_TCS 의 3 × 400 LOC 중복을 단일 함수로 축약.

### 7.8 Thermal Solver

```cpp
class ThermalSolver {
  void compute_coeffi(const Field& T);                // KPP, dKPP, 1/(ρCp)
  void step(const Field& U, const Field& V, const Field& W, double dt);
};
```

### 7.9 Projection + dPhat 외삽

```cpp
class Projection {
  Field dP_prev;                                       // 이전 dP 보관
public:
  void apply(FieldStorage& fs, double dt);             // V* -= dt·∇dPhat 등
};
```

---

## 8. PaScaL_TCS vs MPM-STD Fortran — 직접 비교

| 항목 | PaScaL_TCS | MPM-STD (Fortran) |
|---|---|---|
| **wall-normal** | **y** | **z** |
| **Sweep 순서** | z → x → y | x → y → z |
| **LOC** | 5,678 (8 파일) | ~25 파일 |
| **GPU** | 없음 | 전면 (CUDA Fortran + OpenACC + cuFFT) |
| **dU/dV/dW solver** | **분리 (3× 중복)** | **단일 함수 + flag** |
| **압력 FFT** | 2D FFT + 전치 2 회 | BC-aware FFT/DCT 자동 선택 |
| **Wall function** | 없음 (단순 Dirichlet 만) | 있음 (log-law) |
| **LES** | 없음 | 있음 |
| **IBM** | 없음 | 있음 |
| **Inflow generator** | 없음 | XSEM |
| **Species (다종 스칼라)** | 없음 | 있음 |
| **NOB** | **풀스택** (T 의존 ρ, μ, κ) | 동일 |
| **Python 바인딩** | 없음 | F2Py |
| **TDMA** | PaScaL_TDMA (CPU) | PaScaL_TDMA (CUDA) |
| **Restart IO** | MPI-IO multiple 방식 | 동일 + binary 병렬 |
| **모듈 분리** | 매우 거침 (8 파일) | 잘 분리 (~25 파일) |

→ **PaScaL_TCS** = NOB-RB 의 원전 (clean reference), **MPM-STD Fortran** = 확장 (production system).
→ C++ 포팅 시 **알고리듬은 PaScaL_TCS 그대로, 구조는 MPM-STD Fortran 의 모듈 분리, 확장은 PyFR 의 Plugin 시스템**.

---

## 9. 보존할 인터페이스 (C++ 가져갈 시그니처)

### 9.1 Thermal solver

```fortran
call mpi_thermal_solver(U, V, W)
```
→
```cpp
thermal.step(fs.U, fs.V, fs.W, dt);
```

### 9.2 Momentum (분리된 형식)

```fortran
call mpi_momentum_solvedU(T)     ! → dU
call mpi_momentum_solvedV(T)     ! → dV
call mpi_momentum_solvedW(T)     ! → dW
call mpi_momentum_blockLdV(T)    ! dV 보정
call mpi_momentum_blockLdU(T)    ! dU 보정 (dV 사용)
call mpi_momentum_pseudoupdateUVW()
```
→ (C++ 권장: 통합)
```cpp
for (Component c : {U, V, W}) momentum.predict(c, fs.T);
momentum.block_couple_V(fs.T);
momentum.block_couple_U(fs.T);
momentum.pseudo_update();
```

### 9.3 Pressure Poisson

```fortran
call mpi_pressure_RHS(...)
call mpi_pressure_Poisson_FFT2(dx2_sub, dmx2_sub)
call mpi_pressure_Projection(...)
```
→
```cpp
pressure.compute_rhs(fs, prhs);
pressure.solve(prhs, dp, bc);
projection.apply(fs, dp, dt);
```

### 9.4 Ghost 교환

```fortran
call mpi_subdomain_ghostcell_update(U)
```
→
```cpp
subdomain.exchange_halo(field);
```

### 9.5 진단

```fortran
call mpi_Post_Div(...)             ! max|div(U)| > 1e-3 → abort
call mpi_Post_CFL(...)             ! dt 갱신
call mpi_Post_MonitorOut(...)      ! 표준출력
```
→
```cpp
diag.check_divergence(fs);
double new_dt = diag.cfl_dt(fs);
diag.print_monitor(step, time, dt, cfl, div);
```

---

## 10. C++ 포팅 교훈

### 보존해야 할 강점

1. **블록 LU 결합 보정** (`blockLdU/V`) — NOB 의 정확도 보존에 핵심
2. **`dPhat = 2·dP − dP_prev`** 외삽 — projection 정확도 향상
3. **3D MPI Cart + 1D sub-comm × 3 + 2D sub-comm** — 깔끔한 통신 계층
4. **NOB 다항식 계수** (a10–a15, b10–b15, c10–c15, d10–d15) — namelist 데이터로 명시적 관리
5. **압력 평균 제거** `∫P = 0` — Neumann BC 정합성
6. **사전계산 wave number** (`dxk1, dzk`) — 매 step 재계산 안 함
7. **CFL-adaptive dt** (`MaxCFL` 만 입력) — 안정성·효율 균형
8. **multiple restart IO 옵션** (rank 0 only, MPI-IO collective, aggregated, post-reassembly) — 코어 수에 따른 선택

### 개선해야 할 약점

1. **dU/dV/dW 의 3× 중복** → 단일 `predict(Component)` 로 압축
2. **wall-축 y 하드코딩** → BC 에서 `wall_axis()` 도출
3. **모듈 분리 거침 (8 파일에 5.7K LOC)** → MPM-STD Fortran 처럼 ~20 파일로 분할
4. **`Solver` 객체 부재** → orchestrator 클래스로 캡슐화
5. **alloc-on-stage 패턴** (`dU` 매 step alloc/dealloc) → 재사용 버퍼 (RAII + scratch pool)
6. **Stretching x-축 한정 미지원** → 3 축 독립 stretching 가능하게 (이미 가능하나 검증 부족)
7. **Wall function 없음** → C++ 에서는 `WallFunction` 추상화 미리
8. **GPU 미지원** → C++ 은 `Backend` 인터페이스 미리 마련 (1단계는 CPU)
9. **Python 바인딩 없음** → pybind11 로 추가

### MPM-STD Fortran 에서 가져올 점 (PaScaL_TCS 에는 없음)

- **단일 `solvedU` + flag** 패턴 (코드 중복 제거)
- **BC-aware FFT/DCT 자동 선택** (Neumann → DCT)
- **모듈을 ~20 개로 펼친 분리도**
- **Plugin 형 확장** (LES, IBM, Wall function, Inflow)
- **F2Py 의 dict-driven 케이스 생성**

---

## 11. C++ MPM-STD 의 PaScaL_TCS 검증 시나리오

C++ 코드의 1차 검증은 PaScaL_TCS 의 NOB-RB 결과 재현이다:

```
Input: Ra=100, Pr=1, n1m×n2m×n3m=512×128×256, Timestepmax=10
       np1×np2×np3 = 2×2×2 = 8 ranks
       pbc=(T,F,T)  ← y wall
       gamma2=2.6 (y stretching)
```

**Golden data**: PaScaL_TCS 의 `./data/1_continue/cont_U.bin, cont_V.bin, …` (10 step 후) → `tests/regression/golden/`.

**Comparator** (`tests/regression/compare_pascal_tcs.py`):
- 양쪽 모두 reassemble → L∞ 비교
- Pass: L∞(T, U, V, W) < 1e-10, L∞(P) < 1e-9 (동일 FP flag 컴파일 시)

**Stage-단위 dump**:
- C++ 측 `MPMSTD_DUMP_STAGES=1`
- PaScaL_TCS 측 1회용 패치 (`!! debug_dump_after_z_sweep` 등)
- 첫 발산 지점에서 버그 localize

---

## 12. 결론

PaScaL_TCS 는 **C++ MPM-STD 의 알고리듬 원전**. clean·minimal 한 NOB Rayleigh-Bénard 솔버로, **block LU 결합 + ADI 3-stage + FFT2 Poisson + dPhat 외삽** 의 정석 구현.

C++ 포팅 전략:
- **알고리듬은 PaScaL_TCS 를 1:1 직역** (먼저 NOB-RB 결과 재현)
- **모듈 분리는 MPM-STD Fortran 따라** (8 → ~20 파일)
- **확장 인터페이스는 PyFR 영감** (Plugin, Strategy, Backend)
- **데이터 모델은 CaNS 영감** (Grid/Field/BC 의 데이터-구동 추상화)
- **wall-축 / sweep 순서 / BC 종류는 모두 BC 설정에서 도출** (PaScaL_TCS 의 하드코딩 제거)

→ 네 보고서 (CaNS, PyFR, MPM-STD Fortran, PaScaL_TCS) 의 종합 = C++ MPM-STD 청사진 완성. 다음 단계는 **최종 설계 권고안** 문서로 통합.
