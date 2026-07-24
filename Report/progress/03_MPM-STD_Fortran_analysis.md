# MPM-STD (Fortran) 구조 분석 보고서

> 대상: [/shared/home/wel1come1234/workspace/MPM-STD_main/MPM-STD](../../MPM-STD_main/MPM-STD/)
> 목적: C++ MPM-STD 의 **1차 레퍼런스**. PaScaL_TCS·CaNS·PyFR 의 분석을 종합하기 전, 직접 포팅 대상이 되는 Fortran 코드 자체를 해부.
> 핵심 확인사항: **z = wall-normal** (PaScaL_TCS 는 y, 메모리에 기록된 컨벤션과 일치).

---

## 1. 디렉토리 구조 & 빌드

| 디렉토리/파일 | 역할 |
|---|---|
| [src/](../../MPM-STD_main/MPM-STD/src/) | Fortran 본체 — `core/`, `domain/`, `global/`, `pascal_tdma/`, `post/`, `debug/` 로 분리 |
| [run/](../../MPM-STD_main/MPM-STD/run/) | 실행 디렉토리, `PARA_INPUT.dat` |
| [python/](../../MPM-STD_main/MPM-STD/python/) | F2Py 바인딩 + 파라미터 생성기 + 후처리 |
| [examples/](../../MPM-STD_main/MPM-STD/examples/) | 케이스 참조 (POISSON_CHECK, AIJ_CaseH_H16, Channel_XSEM, RBC, Teheran, …) |
| [documents/](../../MPM-STD_main/MPM-STD/documents/) | 기술 문서 |
| [multiplerun/](../../MPM-STD_main/MPM-STD/multiplerun/) | 파라미터 sweep 배치 인프라 |
| `Makefile` / `Makefile.inc` | nvfortran (NVHPC SDK) 기반 빌드. debug `-O0 -g -traceback -C`, release `-O4 -tp=native -fast` |
| `f2py.sh`, `f2py_mpmc.sh`, `f2py_mpmc_4gpu.sh` | F2Py 빌드 (2-GPU / 4-GPU 설정) |
| `mpi.sh` | MPI 실행 래퍼 |
| `Teheran_*gpu_*.output`, `*.err` | GPU 실행 로그 (2-GPU/4-GPU 케이스 존재 확인) |

### GPU 지원

**전면적 GPU 활용 코드**:
- **CUDA Fortran**: `attributes(global)` 커널 + `<<<blocks, threads>>>` 호출
- **OpenACC**: `!$acc parallel loop`, `!$cuf kernel do` (단순 루프용)
- **cuFFT**: 압력 Poisson 해법
- **PaScaL_TDMA (MPI+CUDA)**: 분산 메모리 + GPU 배치 TDMA

### 외부 의존성

- **PaScaL_TDMA** (in-house, `src/pascal_tdma/`)
- **cuFFT**
- **MPI**
- **NVHPC SDK** (nvfortran, cuFFT, OpenACC)

---

## 2. 시작 → 종료 실행 흐름

### 2.1 메인 (`src/entrypoint.f90`)

```fortran
program mpm_std
  call mpi_init()
  call initial()
  call continue_fileIO('in')
  call run_simulation()
  call continue_fileIO('out')
  call clear()
  call mpi_finalize()
end program
```

### 2.2 초기화 `initial()` (src/submodule.f90:35–83)

```
1. MPI subdomain (Cart topology, halo width)
2. CUDA 디바이스 환경 (thread block sizing)
3. PARA_INPUT.dat 파싱 (global_input_parameter)
4. Inflow BC 컨트롤 셋업
5. 메모리 alloc (host + device)
6. PaScaL_TDMA 계획 생성
7. IBM 초기화
8. 물리 모듈 초기화 (energy, momentum, pressure, statistics)
9. Ghost cell 업데이트 (GPU)
10. Device → Host 복사 (체크용)
```

### 2.3 시간 루프 `run_simulation()` (src/submodule.f90:85–115)

```
[1]  coefficient        ! μ(T), 열 전도도 등
[2]  LES                ! 옵션 — Smagorinsky 등
[3]  inflow             ! XSEM 합성 난류 인렛 (옵션)
[4]  boundary           ! wall/farfield BC; XMBC_d/YMBC_d/ZMBC_d
[5]  core_energy        ! 온도 (옵션)
[6]  core_species       ! 스칼라 (옵션)
[7]  core_momentum:
       solvedU(dU)
       solvedU(dV)
       solvedU(dW)
       blockldU('dV')   ! V-방향 결합 보정
       blockldU('dU')   ! U-방향 결합 보정
       pseudoupdateUVW  ! 압력 그래디언트 외삽 추가
[8]  IBM                ! 셀 재구성, wall function
[9]  core_pressure:
       RHS 조립
       FFT (또는 Multigrid stub)
[10] core_projection:
       속도 보정
       ghost 교환
```

→ **PaScaL_TCS 와 동일한 MPM-STD 구조**: thermal → momentum (predictor + block coupling) → pressure projection. 차이는 wall-normal 축과 GPU·IBM·LES·XSEM 등 부가 기능.

### 2.4 좌표계 확인

`PARA_INPUT.dat` 의 `pbc1, pbc2, pbc3` 와 `master_settings.py` 의 `"pbc": (False, True, False)` →
- **x (pbc1=F)**: non-periodic (farfield 또는 inflow/outflow)
- **y (pbc2=T)**: **periodic**
- **z (pbc3=F)**: **wall-normal** (Neumann + 벽 모델)

벽 함수가 z 전용으로 하드코딩됨: `standard_wall_cell_center_z()` (src/core/core_wall.f90:163).

→ **메모리에 기록된 컨벤션 [project_filtered_tdma_channel.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/project_filtered_tdma_channel.md) 와 정확히 일치: z = wall-normal**.

---

## 3. 모듈/파일 인벤토리

### Core Physics

| 파일 | LOC | 역할 |
|---|---|---|
| [src/core/core_momentum.f90](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90) | 1826 | U/V/W predictor, block LU 결합 보정, ghost 교환 |
| [src/core/core_pressure.f90](../../MPM-STD_main/MPM-STD/src/core/core_pressure.f90) | 1268 | Poisson RHS, FFT 해법, 압력 projection, DCT/FFT 변환 |
| [src/core/core_energy.f90](../../MPM-STD_main/MPM-STD/src/core/core_energy.f90) | 517 | 온도 (TDMA 기반) |
| [src/core/core_species.f90](../../MPM-STD_main/MPM-STD/src/core/core_species.f90) | 548 | 다종 스칼라 transport |
| [src/core/core_boundary.f90](../../MPM-STD_main/MPM-STD/src/core/core_boundary.f90) | 1327 | BC 데이터 alloc, inflow/outflow, wall BC |
| [src/core/core_wall.f90](../../MPM-STD_main/MPM-STD/src/core/core_wall.f90) | 289 | 벽 함수 (log-law, 열전달) — **z 전용** |
| [src/core/core_LES.f90](../../MPM-STD_main/MPM-STD/src/core/core_LES.f90) | 528 | LES 모델, eddy viscosity |
| [src/core/core_IBM.f90](../../MPM-STD_main/MPM-STD/src/core/core_IBM.f90) | 1082 | Immersed Boundary Method (셀 분류, 재구성) |
| [src/core/core_inflow_XSEM.f90](../../MPM-STD_main/MPM-STD/src/core/core_inflow_XSEM.f90) | 875 | 합성 난류 인렛 (Spectral Synthesis) |

### Infra / Domain

| 파일 | 역할 |
|---|---|
| `src/entrypoint.f90` | program main (26 LOC) |
| `src/submodule.f90` | orchestrator (`initial`, `run_simulation`, `clear`, …) |
| `src/global/global.f90` | 전역 파라미터, namelist |
| `src/global/interface.f90` | generic `allocate_and_init` (1D~4D, real/int, host/device) |
| `src/global/nvtx.f90` | NVIDIA Nsight 프로파일링 훅 |
| `src/domain/mpi_topology.f90` | 3D Cart, rank mapping |
| `src/domain/mpi_subdomain.f90` | halo 교환, global↔subdomain index |
| `src/domain/cuda_subdomain.f90` | CUDA grid/block sizing, ghost 커널, host↔device 복사 |

### TDMA

| 파일 | 역할 |
|---|---|
| `src/pascal_tdma/cuda_pascal_tdma_wrapper.f90` | 고수준 plan 인터페이스 |
| `src/pascal_tdma/pascal_tdma_cuda.f90` | 저수준 TDMA 커널 (LU + 전·후방 대입) (870 LOC) |
| `src/pascal_tdma/tdmas_cuda.f90` | cyclic/periodic TDMA |
| `src/pascal_tdma/para_range.f90` | MPI work 분배 헬퍼 |

### Post-processing / Debug

| 파일 | 역할 |
|---|---|
| `src/post/cuda_post.f90` | 통계 누적, 발산 체크, CFL 모니터 |
| `src/post/cuda_post_stat.f90` | 1D/2D/3D 통계 출력 |
| `src/post/cuda_post_instant.f90` | 순간 필드 스냅샷 (binary 병렬 IO) |
| `src/post/cuda_post_surrogate.f90` | ML surrogate / 축소 모델 출력 |
| `src/debug/debug.f90`, `cuda_debug.f90` | 검증 루틴, GPU 메모리 체크 |

### 분류

| 카테고리 | 모듈 |
|---|---|
| Infrastructure | entrypoint, submodule, global, interface, nvtx, mpi_topology, mpi_subdomain, cuda_subdomain |
| Numerics core | pascal_tdma_cuda, tdmas_cuda, pascal_tdma_wrapper |
| Physics | core_momentum, core_pressure, core_energy, core_species |
| BC & Models | core_boundary, core_wall, core_inflow_XSEM, core_IBM, core_LES |
| IO/Stats | cuda_post, cuda_post_stat, cuda_post_instant, cuda_post_surrogate |
| Debug | debug, cuda_debug |

---

## 4. PaScaL_TCS / CaNS 와의 비교

### MPM-STD (Fortran) vs PaScaL_TCS

| 항목 | MPM-STD (Fortran) | PaScaL_TCS |
|---|---|---|
| **wall-normal** | **z** (pbc3=F) | **y** (pbc2=F) |
| **Predictor-corrector** | Block LU 분리 (`blockldU`로 dU/dV 보정) | 동일 (`mpi_momentum_blockLdU/V`) |
| **ADI sweep** | 방향별 TDMA (x→y→z), PaScaL_TDMA로 벡터화 | z→x→y |
| **Pressure Poisson** | **FFT/DCT 선택형** (BC 따라) + wall-축 TDMA | 2D FFT(x,z) + TDMA(y), C↔I↔K 두 번 전치 |
| **BC** | wall은 z만 하드코딩, periodic은 axis별 flag | 동일 (y만 wall) |
| **격자 stretching** | tanh + h-stretching 옵션 | tanh |
| **GPU** | **전면적 CUDA Fortran + OpenACC + cuFFT** | 거의 CPU |
| **IBM/LES/XSEM** | **있음** | 없음 |
| **다종 스칼라** | 있음 (`core_species`) | 단일 온도 |
| **시간 적분** | Crank-Nicolson + ADI | 동일 |
| **TDMA 라이브러리** | PaScaL_TDMA (CUDA 버전) | PaScaL_TDMA (CPU 버전) |
| **Python 바인딩** | F2Py 통한 케이스 자동 생성/실행 | 없음 |

### MPM-STD vs CaNS (간단)

CaNS 는 채널-only, RK3 + FFT(x,y)+TDMA(z) 의 미니멀 구조. MPM-STD 는 그 위에 **IBM/LES/XSEM/다종 스칼라/GPU/F2Py** 가 모두 얹힌 풀스택. 알고리듬 코어 (FFT Poisson + 방향별 TDMA) 는 동일하지만 MPM-STD 의 확장 범위가 훨씬 넓다.

### 핵심 아키텍처 차이

**MPM-STD 의 block factorization** = 각 방향 운동량을 분리 풀고 그 후 pressure 결합으로 근사 보정. PaScaL_TCS 도 같은 방식. **CaNS 는 RK3 의 각 stage 안에서 fractional step** 으로 다른 분해 사용.

→ C++ MPM-STD 는 PaScaL_TCS·이 Fortran MPM-STD 와 **동일한 block-coupling 구조** 를 채택해야 함.

---

## 5. 모듈화 / OO 패턴

### 5.1 상태 관리 — module-scope allocatable

PaScaL_TCS·CaNS 와 마찬가지로 derived type 거의 없음. 각 물리 모듈이 자체 module 변수로 보관:

```fortran
! src/core/core_momentum.f90:10–12
real(rp),         allocatable, dimension(:,:,:) ::    u,    v,    w,    p   ! host
real(rp), device, allocatable, dimension(:,:,:) ::  u_d,  v_d,  w_d,  p_d   ! device
real(rp), device, allocatable, dimension(:,:,:) :: du_d, dv_d, dw_d         ! work
```

### 5.2 할당 전략

```fortran
call cuda_momentum_variables_memory('allocate')   ! src/core/core_momentum.f90:52
call cuda_momentum_coeffi_memory   ('allocate')
```

→ "구역별 메모리 관리 서브루틴" 패턴. C++ 에서는 RAII 로 자연스러움.

### 5.3 Host vs Device 분리

- **host** 배열: 체크포인트·IO 시만 사용
- **device** 배열: 시간 루프 hot path
- 동기화: `cuda_subdomain_dtoh_total()` (src/submodule.f90:67) — **시작/체크포인트 시점에만**.

### 5.4 Generic interface 패턴 — `interface.f90`

```fortran
interface allocate_and_init
  module procedure allocate_and_init_1d_real_host
  module procedure allocate_and_init_3d_real_device
  ! ... 다양한 차원/타입/위치 조합
end interface
```

→ **C++ 의 함수 템플릿**으로 1:1 매핑 가능.

### 5.5 "Solver 객체" 부재

PaScaL_TCS·CaNS 와 동일. 전역 모듈 변수 + 자유 서브루틴 호출 구조.

→ **C++ 포팅 시 가장 큰 개선 여지**: `Solver`, `MomentumSolver`, `PressureSolver` 등의 클래스로 캡슐화.

### 5.6 격자 추상화

`cuda_subdomain` 모듈에 모두 보관:
- `x1_d, x2_d, x3_d`: cell-center 좌표
- `dx1_d, dx2_d, dx3_d`: cell width
- `dmx1_d, dmx2_d, dmx3_d`: face-to-face 거리

MAC stagger 는 **암묵적** (이름과 인덱스 컨벤션으로만). 타입 안전성 없음.

### 5.7 Ghost cell

`cuda_subdomain_ghostcell_update()` — 주요 solve 후 호출. halo width는 방향별 설정 가능.

---

## 6. C++ 포팅을 위한 핵심 추상화

### 6.1 Grid

```cpp
class Grid {
  std::array<int, 3>      n;             // n1m, n2m, n3m
  std::array<double, 3>   L;             // L1, L2, L3
  std::array<std::vector<double>, 3> x;  // cell centers
  std::array<std::vector<double>, 3> dx; // cell widths
  std::array<std::vector<double>, 3> dmx;// face-to-face
  std::array<bool, 3>     periodic;      // pbc1/2/3
  // tanh 또는 h-stretching 빌드 함수
};
```

### 6.2 Field

```cpp
class Field {
  std::vector<double> host;             // contiguous + halo
  double*             device = nullptr; // optional GPU mirror
  int nx, ny, nz;
  int halo = 1;
  StagLocation stag;                    // Cell, FaceX, FaceY, FaceZ
  void to_device(); void to_host();
};
```

### 6.3 BoundaryCondition — face별 configurable

**현 Fortran 의 한계**: wall이 z 전용. C++ 은 면별 자유 설정 필요.

```cpp
enum class BCKind { Periodic, Dirichlet, Neumann, Wall, Inflow, Outflow };
struct FaceBC { BCKind kind; std::function<double(double,double,double,double)> value; };
class BoundaryConditions {
  std::array<FaceBC, 6> face;  // -x,+x,-y,+y,-z,+z
public:
  Direction wall_axis() const;  // wall이 있는 축 자동 추론
  bool is_periodic(Direction d) const;
  void apply_ghost(Field&) const;
  void modify_tdma_row(Direction d, ... ) const;
};
```

### 6.4 PressureSolver — BC-aware FFT

Fortran 은 `Poisson_model = 'FFT'` 일 때 BCtype 에 따라 DCT/FFT 자동 선택. C++ 에서는:

```cpp
class PressureSolver {
  std::array<TransformKind, 3> tx;  // BC 에서 도출: DCT-II(N), DST-II(D), R2C(P)
  FFTPlan2D plan_xy_fwd, plan_xy_bwd;
  TdmaSolver& tdma_wall;            // wall 축에서 batched TDMA
public:
  void solve(const Field& rhs, Field& dp, const BoundaryConditions& bc);
};
```

### 6.5 TDMA — PaScaL_TDMA 래퍼

Fortran 인터페이스:
```fortran
! src/pascal_tdma/cuda_pascal_tdma_wrapper.f90:66
call cuda_ptdma_core(plan, A_d, B_d, C_d, D_d, pbc)
```

C++ 매핑:
```cpp
class TdmaSolver {
public:
  virtual void solve_many       (double* A, double* B, double* C, double* D, int n_sys, int n_row) = 0;
  virtual void solve_many_cyclic(double* A, double* B, double* C, double* D, int n_sys, int n_row) = 0;
};
class PascalTdmaBackend : public TdmaSolver { /* plan caching */ };
```

### 6.6 GPU 백엔드 — 1단계 보류, 인터페이스만 미리

PyFR 패턴 차용:
```cpp
class Backend { virtual Buffer alloc(size_t) = 0; virtual void run_kernel(Kernel&) = 0; };
class CPUBackend  : public Backend { ... };  // 1단계
class CUDABackend : public Backend { ... };  // 후일
```

### 6.7 시간 적분

Fortran 의 CN+ADI 만 사용. C++ 도 1단계는 단일 스킴, 인터페이스만 Strategy 패턴:
```cpp
class IntegrationScheme {
  virtual void step(Solver& s, double dt) = 0;
};
class CrankNicolsonADI : public IntegrationScheme { ... };
```

---

## 7. BC 철학 — 현 Fortran 의 한계와 C++ 의 처방

### 현 Fortran 의 하드코딩 지점

- **벽 함수**: z 전용 (`standard_wall_cell_center_z`, src/core/core_wall.f90:70–111)
- **Inflow (XSEM)**: x1 방향 전용 (src/core/core_inflow_XSEM.f90)
- **압력 BC**: `pbc1/2/3` 로 전역 결정 (src/core/core_pressure.f90:47–68)
- **per-face 다른 BC** (예: −z=Neumann, +z=Dirichlet) 지원 안 함

### C++ 의 처방

1. **`Direction wall_axis()`** 를 BC 설정에서 도출 (어떤 축이 wall인지 자동).
2. **Sweep order**: "periodic 축 먼저, wall 축 마지막" 규칙. PaScaL_TCS 가 z→x→y, Fortran MPM-STD 가 x→y→z 인 이유 = 각각 wall이 y, z 이기 때문.
3. **벽 함수**: `WallFunction` 추상 클래스, 어느 면에든 attach 가능하게.
4. **Inflow**: `InflowGenerator` 추상화, 면별 등록.
5. **TDMA periodicity**: `bc.is_periodic_axis(d)` 가 cyclic vs solve 변형 자동 선택.

---

## 8. python/ 디렉토리

### 역할

**F2Py 래퍼 + 케이스 자동 생성/실행 + 후처리**. Fortran 변경 없이 Python 으로 워크플로우 구성.

### 핵심 파일

| 파일 | 역할 |
|---|---|
| `python/entrypoint.py` | MPI-aware Python 스크립트. `mpm_std.submodule` 모듈 (F2Py 컴파일) 호출 |
| `python/src/master_settings.py` | 케이스별 사전정의 (POISSON_CHECK, AIJ_CaseH_H16, Channel_XSEM, RBC, Teheran, …) |
| `python/src/para_input_manager.py` | dict → Fortran namelist 변환 |
| `python/plt_ascii_to_binary.py` | ASCII → binary 변환 |
| `python/entrypoint_csv.py` | CSV-based 파라미터 입력 |

### 빌드

`make f2py` 또는 `f2py.sh` → `src/modules/mpm_std.so`.

### 호출 패턴

```python
mpm_std.submodule.initial()
mpm_std.submodule.run_simulation()
mpm_std.submodule.clear()
```

### C++ 포팅 시 고려

- F2Py 자리에 **pybind11** 사용 권장 (C++ 친화적, 더 단순)
- master_settings.py 의 dict-기반 케이스 정의는 **그대로 유용** — C++ 의 INI/TOML/YAML 입력 포맷에 매핑

---

## 9. 보존할 인터페이스 (C++ 에서 그대로 가져갈 시그니처)

### 9.1 Pressure Poisson

```fortran
! src/core/core_pressure.f90:349–359
select case(trim(Poisson_model))
case('FFT')
  call cuda_pressure_Poisson_FFT_1D(PRHS_d, dp_d)
end select
```
→
```cpp
void solve(const Field& rhs, Field& dp, const BoundaryConditions& bc);
```

### 9.2 TDMA

```fortran
call cuda_ptdma_core(plan, A_d, B_d, C_d, D_d, pbc)
```
→
```cpp
tdma.solve_many(A, B, C, D, n_sys, n_row);
tdma.solve_many_cyclic(...);
```

### 9.3 Momentum predictor-corrector

```fortran
call cuda_momentum_solvedU(T_d, H_d, 'U')   ! solvedU 가 첫 인자 flag로 U/V/W 모두 처리
call cuda_momentum_solvedU(T_d, H_d, 'V')
call cuda_momentum_solvedU(T_d, H_d, 'W')
call cuda_momentum_blockldU(T_d, H_d, 'dV')
call cuda_momentum_blockldU(T_d, H_d, 'dU')
call cuda_momentum_pseudoupdateUVW(H_d)
```
→
```cpp
class MomentumSolver {
  void predict(Component c, const Field& T, const Field& H);
  void block_couple(Component c, const Field& T, const Field& H);
  void pseudo_update(const Field& H);
};
```

→ **단일 `solvedU` 가 flag('U'/'V'/'W') 로 3 변수 처리하는 패턴**은 Fortran 코드의 중복 제거 노력이고, C++ 에서는 enum + template/runtime dispatch 로 자연스럽게 표현.

### 9.4 Ghost 교환

```fortran
call cuda_subdomain_ghostcell_update(U_d)
```
→
```cpp
domain.exchange_halo(field);
```

### 9.5 Boundary 설정

```fortran
call cuda_boundary_init()
call cuda_momentum_boundary_memory('allocate')
call core_boundary(U_d, V_d, W_d, T_d, C_d)
```
→
```cpp
boundaries.init(grid, config);
boundaries.apply(velocity, scalars);
```

---

## 10. C++ 포팅 교훈

### 보존해야 할 강점

1. **Block LU 분리**: dU/dV/dW 분리 후 압력으로 결합 보정. 메모리 대역폭·반복 효율 우수.
2. **BC-aware FFT/DCT 자동 선택**: Neumann → DCT, Periodic → FFT. 우아하고 정확.
3. **PaScaL_TDMA 의 batched TDMA**: 수많은 1D 시스템을 GPU 친화적으로 일괄 처리.
4. **모듈식 wall/inflow/LES/IBM**: 기능 toggle 이 쉬움. C++ 에서는 Plugin 패턴으로 자연스럽게.
5. **Host/Device 분리**: 시간 루프 hot path 는 device only. Sync 는 체크포인트에서만.
6. **Generic interface (`interface.f90`)**: 차원/타입/위치 조합을 한 이름으로 호출. C++ 의 함수 템플릿이 더 깔끔.

### 고쳐야 할 약점

1. **wall-normal 축 하드코딩**: z 전용. **C++ 은 BC 설정에서 wall 축을 도출**.
2. **per-face BC 불가**: 같은 축의 양면이 같은 BC. **C++ 은 6 면 독립 설정**.
3. **Solver 객체 부재**: 전역 module 변수가 솔버 상태를 분산 보유. **C++ 은 `Solver` 클래스로 캡슐화**.
4. **MAC stagger 가 암묵적**: 이름과 인덱스로만. **C++ 은 `StagLocation` enum 또는 `Field` 서브클래스로 타입 안전성** 확보.
5. **메모리 ownership 명시 부재**: `subroutine *_memory('allocate'/'deallocate')` 패턴. **C++ 은 RAII**.
6. **에러 처리**: `write(*,*)` 기반. **C++ 은 예외 + 구조화된 로깅**.
7. **Async 미사용**: GPU 와 MPI/IO 의 비동기 오버랩 미적용. **C++ 은 처음부터 stream-aware 설계 가능**.
8. **GPU 메모리 풀 없음**: 매 케이스 alloc/dealloc. **C++ 은 메모리 풀로 fragmentation 감소**.
9. **Python 바인딩 분산**: master_settings + para_input_manager 가 분리. **C++ 은 단일 config (INI/TOML/JSON) + pybind11 로 통합**.

---

## 11. C++ MPM-STD 권장 구조 (이 보고서의 결론)

```
MPM-STD(C++)/
├── include/mpmstd/
│   ├── config.hpp                       ← PARA_INPUT.dat + master_settings.py 통합
│   ├── direction.hpp, stag_location.hpp ← enum
│   ├── mpi_topology.hpp                 ← Fortran mpi_topology
│   ├── subdomain.hpp                    ← cuda_subdomain + mpi_subdomain
│   ├── grid.hpp                         ← x_d, dx_d, dmx_d
│   ├── field.hpp                        ← host+device, stag-aware
│   ├── boundary_conditions.hpp          ← 면별 자유 설정
│   ├── tdma_solver.hpp + backends       ← PaScaL_TDMA(C) 래퍼
│   ├── fft_planner.hpp                  ← FFTW3 (+ cuFFT 미래)
│   ├── backend.hpp                      ← CPU/CUDA 추상 (PyFR 영감)
│   ├── thermal_solver.hpp               ← core_energy
│   ├── momentum_solver.hpp              ← core_momentum (predict + blockLd + pseudoupdate)
│   ├── pressure_solver.hpp              ← core_pressure (BC-aware FFT)
│   ├── projection.hpp
│   ├── les_model.hpp     (Plugin)
│   ├── ibm.hpp           (Plugin)
│   ├── inflow_xsem.hpp   (Plugin)
│   ├── wall_function.hpp (Plugin)
│   ├── diagnostics.hpp
│   ├── plugin.hpp                        ← PyFR 스타일 Plugin base
│   ├── io.hpp                            ← cuda_post_* 통합
│   ├── time_stepper.hpp + scheme.hpp     ← Strategy
│   └── solver.hpp                        ← orchestrator
├── src/                                  ← 위 헤더의 .cpp 구현
├── python/                               ← pybind11 바인딩
├── run/                                  ← 입력 케이스
└── tests/
```

### 단계별 매핑

| Fortran MPM-STD 모듈 | C++ 클래스 |
|---|---|
| `submodule.f90 :: initial/run_simulation/clear` | `Solver` |
| `global.f90` | `Config` |
| `mpi_topology / mpi_subdomain / cuda_subdomain` | `MpiTopology`, `Subdomain` |
| `interface.f90` | C++ 함수 템플릿 |
| `core_momentum` | `MomentumSolver` (predict / block_couple / pseudo_update) |
| `core_pressure` | `PressureSolver` + `FftPlanner` |
| `core_energy` | `ThermalSolver` |
| `core_species` | `SpeciesSolver` (옵션 / Plugin) |
| `core_boundary` + `core_wall` | `BoundaryConditions` + `WallFunction` plugin |
| `core_LES` | `LESPlugin` |
| `core_IBM` | `IBMPlugin` |
| `core_inflow_XSEM` | `XsemInflowPlugin` |
| `pascal_tdma_*` | `PascalTdmaBackend : TdmaSolver` |
| `cuda_post_*` | `Diagnostics`, `Io` |
| `entrypoint.f90` | `main.cpp` |
| `python/` (F2Py) | pybind11 모듈 |

---

## 12. 결론

이 Fortran MPM-STD 는 **C++ 포팅의 가장 직접적인 reference**. 알고리듬 구조 (block LU 분리, BC-aware FFT Poisson, PaScaL_TDMA 활용) 는 그대로 가져가고, **다음 6 가지를 C++ 에서 개선**:

1. wall-normal 축의 **런타임 설정**
2. **per-face BC** 자유도
3. `Solver`·`MomentumSolver` 등 **OO 캡슐화**
4. **MAC stagger 의 타입 안전성**
5. **RAII 기반 메모리 관리**
6. **pybind11 단일 바인딩 + 통합 config**

→ PaScaL_TCS 분석 + CaNS 분석 + PyFR 분석 + 본 분석 = C++ MPM-STD 의 청사진 완성. 다음 단계는 네 보고서를 종합한 **최종 설계 권고안** 작성.
