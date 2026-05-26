# M0 + M1 코드 이해서

> **목적**: M0 (인프라 + Field + 빌드 시스템) 과 M1 (BC + TDMA) 까지 완성된 코드를 **나중에 다시 봐도** 머릿속에서 재구성할 수 있도록 정리.
> **읽는 순서**: Part I (빅 픽처) → Part II (실행 순서) → Part III (관습 / FAQ) → Part IV (M2 준비).
> 코드 발췌는 최소화. 본인이 코드를 직접 보면서 이 문서를 길잡이로 사용.

---

## Part I — 빅 픽처

### 1. 한 페이지 요약

**MPM-STD_C 는** 두 reference 코드를 C++ 하나로 합치는 프로젝트.
- **PaScaL_TCS** (Fortran, CPU, 검증된 알고리듬)
- **MPM-STD Fortran** (Fortran, GPU, 빠른 구현)
→ **C++ 라이브러리 + per-case `apps/<case>/main.cpp`** 형태.

**Option C 전략** (`Report/13_final_plan_option_C.md`):
- 폴더와 인터페이스는 **GPU-ready 골격** 으로 처음부터 만든다.
- 알고리듬은 **CPU 빌드로 먼저 완성** → PaScaL_TCS 와 bit-exact 검증.
- 그 다음 GPU 커널 채운다 (M5').

**M0 + M1 의 위치**:
- M0: "도메인을 자르고, 격자 만들고, Field 를 할당하고, restart 한다" 까지의 **인프라**.
- M1: "그 도메인에 BC 를 붙이고, TDMA 솔버를 호출할 준비" 까지의 **경계/선형대수 기초**.
- M2 부터는 **물리 방정식** 이 들어옴 (열, 운동량, 압력).

---

### 2. 레이어 다이어그램

코드는 7개 레이어. 위에서 아래로 흐름 (상위가 하위를 호출):

```
                    ┌─────────────────────────────┐
                    │   apps/<case>/main.cpp      │   ← 아직 없음 (M5)
                    │   (orchestrator)            │
                    └──────────────┬──────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
┌──────▼──────┐            ┌───────▼─────────┐         ┌───────▼──────┐
│  equation/  │            │   integrator/   │         │     post/    │
│  (M2+)      │            │   (M5)          │         │  restart_io  │
└──────┬──────┘            └─────────────────┘         └──────────────┘
       │
┌──────▼───────────────────────────────────────────────────────────┐
│  boundary/   ← M1                  linear_solver/tdma/   ← M1   │
│  Problem, FaceBc, FieldBoundary,   TdmaSolver,                  │
│  DomainTopology, boundary_applier  PascalTdmaCpuBackend,        │
│                                    TdmaRegistry                 │
└──────┬───────────────────────────────────────┬───────────────────┘
       │                                       │
┌──────▼─────────┐  ┌──────────────┐  ┌────────▼──────────┐
│   field/       │  │   grid/      │  │   config/         │
│   ScalarField  │  │   Grid       │  │   Config,         │
│   VectorField  │  │   stretching │  │   Logger          │
│   FieldRegistry│  │              │  │                   │
│   stencil/     │  │              │  │                   │
└──────┬─────────┘  └──────────────┘  └───────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│  parallel/                                                       │
│  ├── mpi/    MpiContext, MpiTopology, Subdomain,                │
│  │           cuda_aware_mpi                                      │
│  ├── backend/ Backend(abstract), CpuBackend, CudaBackend        │
│  └── cuda/   cuda_helpers namespace (cuda_runtime,              │
│              cuda_memory, cuda_stream, nvtx_range, ...)         │
└──────┬───────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│  common/   types(real_t), direction(Direction/Side/Component), │
│            macros(MPMSTD_HD, kHaloWidth, CUDA_CHECK)            │
└──────────────────────────────────────────────────────────────────┘
```

**규칙**: 위 레이어가 아래 레이어를 부른다. 아래가 위를 부르면 안 된다 (circular dependency 금지).

---

### 3. 핵심 멘탈 모델 7가지

이 7개를 기억하면 코드 대부분이 풀린다.

#### 모델 1 — "한 함수, 두 얼굴" (`#ifdef MPMSTD_BACKEND_CUDA`)

같은 함수 이름이 빌드 매크로에 따라 다른 구현. 예시:
- `cuda_helpers::device_alloc(bytes)`
  - CPU 빌드 → `posix_memalign`
  - CUDA 빌드 → `cudaMalloc`
- `cuda_helpers::copy_host_to_device(...)`
  - CPU 빌드 → `std::memcpy`
  - CUDA 빌드 → `cudaMemcpy`

→ **상위 코드는 빌드를 모름**. 한 코드로 두 빌드가 동작.

#### 모델 2 — "Backend = 메모리/동기화 추상, kernels = 핫루프 추상"

핫루프 (predict, project, FFT) 는 `Backend` 의 가상 메서드가 **아니다**. 핫루프는 빌드 시간에 `kernels_cpu.cpp` 또는 `kernels_cuda.cu` 중 하나가 컴파일됨. 이유: vtable 분기는 GPU 인라이닝을 깨고 성능을 죽임.

`Backend` 는 자주 안 불리는 것 (alloc, free, synchronize) 만 가짐. → 작게 유지.

#### 모델 3 — "RAII = 자원의 생애주기를 객체에 묶는다"

- `MpiContext` 생성 = `MPI_Init`, 소멸 = `MPI_Finalize`
- `Subdomain` 생성 = `MPI_Type_commit` × 6, 소멸 = `MPI_Type_free` × 6
- `MpiTopology` 생성 = `MPI_Cart_create` + 3 sub-comm, 소멸 = `MPI_Comm_free` × 4

→ 예외가 나도, 조기 return 이 있어도 자원 누수가 없음. **main 의 첫 줄에서 만들고 끝까지 살아있게** 한다.

#### 모델 4 — "row-major C 인덱싱: 마지막 축이 가장 빠르다"

`A[i][j][k]` 의 메모리 레이아웃:
```
stride_z = 1
stride_y = n_total[2]
stride_x = n_total[1] * n_total[2]
linear   = (i * n_total[1] + j) * n_total[2] + k
```

**Fortran (`A(i,j,k) = i + nx*j + nx*ny*k`) 와 반대**. PaScaL_TCS 를 포팅할 때 머릿속에서 인덱스 순서를 뒤집어야 함. GPU coalesced access 도 z-inner 가 자연스럽게 풀림.

#### 모델 5 — "도메인 분할은 PaScaL_TDMA 와 호환되어야 한다"

`compute_para_range` 의 분할 규칙 (앞 무거움: `base + (rank<rem ? 1 : 0)`) 은 **PaScaL_TDMA_C 의 `para_range.cpp` 와 같은 규칙**. 다르면 분산 TDMA 가 격자와 안 맞아서 망함. → 이 규칙은 **외부 라이브러리가 강제**.

#### 모델 6 — "Problem 객체 = BC + topology 의 한 묶음, 기본값은 RBC"

`Problem` 을 그냥 `new` 만 해도 RBC (z=wall, T 위 차갑·아래 따뜻) 완성. Channel 은 같은 default + T 무시. 사용자는 필요한 면만 override.

```cpp
Problem p;
p.T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet(2.5);   // 부분 override
```

→ "rare-case 만 사용자가 신경" — Convention over Configuration.

#### 모델 7 — "ADI sweep order 는 topology 가 정한다"

`sweep_order()`: 주기 축 먼저, 비주기(벽) 축 마지막. RBC (z=wall) → `(X, Y, Z)`. PaScaL_TCS (y=wall) → `(X, Z, Y)`. **사용자가 직접 정할 필요 없음**.

이유: 벽이 있는 축은 TDMA 행렬에 boundary 보정이 들어가는데, 마지막에 풀어야 다른 축의 결과가 안 오염됨.

---

### 4. 의존성 그래프 (누가 누구를 부르나)

핵심 의존만 발췌:

```
MpiContext ─────────────── (root)
    │
    └─→ MpiTopology ─────────── (root.next)
            │
            └─→ Subdomain ──┐
                            │
            ┌──→ Grid ──────┤
            │               │
            │               ▼
            │            ┌──────────┐
            │            │  Field   │ (ScalarField, VectorField)
            │            └────┬─────┘
            │                 │
            │                 ▼
            │            FieldRegistry (이름→Field map)
            │
            └─→ Problem ─────────────── (DomainTopology + FieldBoundary×5)
                    │
                    ▼
                BoundaryApplier ── (apply_ghost + modify_tdma_row)
                    │
                    ▼
                TdmaRegistry ── 축별 TdmaSolver 보관
                    │
                    ▼
                PascalTdmaCpuBackend ── PaScaL_TDMA_C wrap
```

**소유 관계**:
- `MpiContext`, `MpiTopology`, `Subdomain`, `Grid`, `Problem`, `TdmaRegistry` 는 모두 **`main` 이 직접 소유** (`unique_ptr` 또는 stack object).
- `Field` 들은 `FieldRegistry` 가 보관 — 이름으로 lookup.
- 다른 객체는 모두 **참조 (`const T&`)** 로 받는다 — shared_ptr 거의 안 씀.

---

### 5. 빌드 시스템 한눈에

```
  ┌─────────────────────────────────┐
  │   Makefile.inc   (BACKEND 선택)  |
  └────────────┬────────────────────┘
               │
        BACKEND=cpu  →  -DMPMSTD_BACKEND_CPU,  build/cpu/
        BACKEND=cuda →  -DMPMSTD_BACKEND_CUDA, build/cuda/
               │
               ▼
   src/Makefile 이 모든 *.cpp 와 *.cu 수집 →
     CPU 빌드: %_cuda.cpp 제외
     CUDA 빌드: %_cpu.cpp 제외 + *.cu 포함
               │
               ▼
   ../build/<backend>/lib/libmpmstd.a 생성
```

핵심 매크로:
- `MPMSTD_BACKEND_CPU` / `MPMSTD_BACKEND_CUDA` — 빌드 분기
- `MPMSTD_HD` — `inline` (CPU) 또는 `__host__ __device__ inline` (CUDA)
- `kHaloWidth` — 1 (현재). 5점 stencil 한계. 늘리면 9점도 가능.
- `MPMSTD_SINGLE_PRECISION` (옵션) — `real_t = float`. 기본은 `double`.

현재 M0+M1 상태에서는 **CUDA 빌드도 컴파일은 성공** 하지만 (`cuda_backend` 등이 throw stub), **실제 실행은 CPU 빌드만 동작**.

---

### 6. M0 / M1 에서 검증된 것

`test/unit/` 에 단위 테스트, `test/integration/` 에 통합 테스트:

| 테스트 | 검증 항목 | 마일스톤 |
|---|---|---|
| `test_grid.cpp` | Grid 좌표 + stretching | M0 |
| `test_grid_multi.cpp` | 멀티랭크 격자 | M0 |
| `test_field.cpp` | ScalarField, VectorField alloc/halo | M0 |
| `test_stencil.cpp` | stencil 자유함수 EOC=2 | M0 |
| `test_mpi_halo.cpp` | Subdomain halo 교환 | M0 |
| `test_restart_roundtrip.cpp` | restart 쓰기→읽기 동일 | M0 |
| `test_problem_defaults.cpp` | Problem() 기본값 = RBC | M1 |
| `test_sweep_order.cpp` | sweep_order() = (X,Y,Z) | M1 |
| `test_bc_apply.cpp` | apply_ghost 3종 BC | M1 |
| `test_bc_apply_multi.cpp` | 멀티랭크 BC | M1 |
| `test_tdma_backend_cpu.cpp` | PaScaL_TDMA wrap 1e-14 | M1 |
| `test_tdma_backend_cpu_multi.cpp` | 멀티랭크 TDMA | M1 |

→ M0+M1 의 모든 인프라가 **단위 단위로 검증됨**. M2 가 들어올 때 이 인프라를 안심하고 호출 가능.

---

### 7. M0/M1 에 **없는** 것 (의식적인 미구현)

| 항목 | 이유 |
|---|---|
| `equation/` 폴더 | M2 이후 |
| `physics/` 폴더 | M7 |
| `integrator/`, `stat/`, `utilities/` | M5 |
| `apps/rbc/main.cpp` | M5 |
| 실제 `CudaBackend::alloc` 동작 | M5' |
| `kernels_cuda.cu` 의 내용 | M5' |
| FFT 솔버 | M4 |

→ "골격은 깔려 있지만 내용은 비어 있음". `find src -type d` 해보면 M2~M7 폴더는 이미 존재하지만, 그 안의 `kernels_cuda.cu` 는 빈 stub 이거나 없음.

---

### 8. Part I 요약 — 이 7가지만 기억하면 됨

1. **레이어 7개**: common → parallel → grid/field/config → boundary/linear_solver → equation/integrator → post → apps
2. **한 함수 두 얼굴**: `#ifdef MPMSTD_BACKEND_CUDA` 가 핵심
3. **RAII**: MpiContext, Subdomain, MpiTopology 모두 생성/소멸이 MPI/CUDA 자원에 묶임
4. **row-major**: k 가 inner, i 가 outer (Fortran 과 반대)
5. **PaScaL_TDMA 가 분할 규칙 강제**
6. **Problem() 한 줄 = RBC 기본 완성**
7. **sweep_order 자동**: 비주기 축이 마지막

---

</content>
