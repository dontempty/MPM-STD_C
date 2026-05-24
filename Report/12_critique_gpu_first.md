# v2 구조 비판적 분석 — GPU 우선 구현을 위한 보완안

> 사용자 결정: **PaScaL_TCS 의 알고리듬을 GPU 로 구현하는 것이 1차 목표**
> 본 보고서: [11_directory_structure_v2.md](11_directory_structure_v2.md) 의 디렉토리 구조를 GPU 우선 관점에서 비판적으로 검토하고 보완안 제시
> 핵심 새 발견: `/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/` 에 이미 **C++ CUDA 포트가 존재**

---

## 0. 결정적 사실 — PaScaL_TDMA_C 의 GPU 버전 존재

[`PaScaL_TDMA_C/src/`](../../PaScaL_TDMA_C/src/):
```
pascal_tdma_many_cuda.cu    + .hpp     ← C++ CUDA TDMA (다중 시스템)
tdma_local_cuda.cu          + .cuh
pascal_tdma_many.cpp        + .hpp     ← CPU 버전 (참고)
tdma_local.cpp              + .hpp
```

[`PaScaL_TDMA_C/heat_gpu/`](../../PaScaL_TDMA_C/heat_gpu/):
```
solve_theta.cu              ← CUDA 커널 작성 예시
solve_theta.hpp
stencil_coeffs.hpp          ← stencil 패턴 헬퍼
gather_theta.hpp            ← halo gather 예시
mpi_subdomain.cpp/.hpp
mpi_topology.cpp/.hpp
main.cpp                    ← 전체 GPU 솔버 예시
Makefile
run_1gpu.sh, run_2gpu.sh    ← 배치 스크립트
```

→ **포팅이 아니라 *링크* 하면 됨**. 또한 `heat_gpu/` 가 **본 프로젝트의 거의 모든 GPU 기법을 작은 케이스로 시연** — heat (단일 스칼라) 솔버이지만 MPI 분할 + CUDA 커널 + TDMA 호출 + halo 교환 모두 포함. **레퍼런스 코드 1 순위**.

→ v2 의 `linear_solver/tdma/pascal_tdma_backend.hpp` 는 이 라이브러리를 **그대로 wrap**. CPU·GPU 백엔드 두 개 다 동일한 라이브러리에서 제공.

---

## 1. v2 구조의 GPU 측면 문제점 — 15 가지

### 1.1 Backend 추상화가 너무 얇음 (Critical)

**v2**: `CpuBackend` + `CudaBackend` 가 같은 추상 인터페이스 뒤. 미래 슬롯으로만 명시.

**문제**: GPU 가 1 차 목표가 되었으니 *진짜* 동작해야 함. 그런데 CPU stencil 함수 (header-only `inline`) 와 GPU 커널 (`__device__ inline`) 은 **선언이 다름**. 단일 추상 인터페이스 뒤에 같은 함수를 두 버전 두는 게 자연스럽지 않음.

**보완**:
- Backend 인터페이스는 **메모리·스트림·런치 관리** 만 추상화
- 실제 *연산* (stencil, kernel) 은 CPU 와 GPU 가 **별개 파일**. 사용자가 빌드 타임에 선택
- Compile-time switch: `MPMSTD_BACKEND=cuda` 또는 `cpu` 매크로로 어느 .cpp/.cu 가 빌드되는지 결정

### 1.2 Field 메모리 모델이 GPU 친화적이지 않음 (Critical)

**v2**: `ScalarField` 가 `host_ptr()` + `device_ptr()` 둘 다 보유. `to_device()`/`to_host()` 명시 호출.

**문제**:
- GPU 우선이면 **device 가 primary**. host 는 IO·체크포인트 시만 alloc 되어야 함 (메모리 절약).
- 현재 설계는 host 가 항상 alloc 됨 → 큰 격자 (1024³ 등) 에서 메모리 낭비.
- MPM-STD Fortran 도 `u` (host) + `u_d` (device) 둘 다 보유하지만, host 는 거의 안 씀.

**보완**:
- `ScalarField` 가 **device-primary**: 기본 alloc 은 device 만
- `mirror_to_host()` 호출 시에만 host 버퍼 alloc (체크포인트, IO 직전)
- 또는 `HostMirror` 별도 RAII 클래스: `auto h = field.host_mirror(); /* h 가 살아있는 동안만 host alloc */`

### 1.3 stencil 자유함수가 CPU 전용 (Critical)

**v2**: `field/stencil/staggered.hpp` 에 `inline double dpdx_at_face_x(...)` 같은 자유함수.

**문제**: GPU 커널 안에서 호출하려면 `__device__` 필요. v2 는 CPU 만 가정.

**보완**:
- 매크로 `MPMSTD_HD = __host__ __device__ inline`
- nvcc 빌드 시 `MPMSTD_HD = __host__ __device__ inline`
- 일반 g++ 빌드 시 `MPMSTD_HD = inline`
- 모든 stencil 헬퍼에 `MPMSTD_HD` 적용 → host·device 양쪽에서 호출 가능

```cpp
// field/stencil/staggered.hpp
#include "common/macros.hpp"     // MPMSTD_HD 정의
MPMSTD_HD double dpdx_at_face_x(const double* P, const double* dmx1,
                                  int i, int j, int k, int n1, int n2) {
  // device 에서도 호출 가능
}
```

→ stencil 자유함수가 **유일하게 host·device 공유 가능한 부분**. 그 외 큰 함수 (orchestrator) 는 host only.

### 1.4 GPU 커널이 들어갈 자리가 v2 에 없음 (Critical)

**v2**: `equation/momentum/momentum_equation.hpp + .cpp` 한 쌍.

**문제**: 실제 hot 계산 (~95% 시간) 은 CUDA 커널이어야 함. .cpp 안에 `__global__` 함수 두면 컴파일러가 처리 못 함 (nvcc 가 .cu 만 처리).

**보완**:
- 각 equation 하위 폴더에 **kernel 파일 추가**:
```
equation/momentum/
├── main.hpp
├── momentum_equation.hpp + .cpp     ← host 측 orchestrator
├── kernels.cu + kernels.cuh         ← __global__ + __device__ 커널들
├── rhs_builders.hpp                 ← MPMSTD_HD 자유함수 (cpu·gpu 공유)
└── (PropertyPolicy, SourceTerm 등은 그대로)
```

`.cu` 는 nvcc/nvc++ 로, `.cpp` 는 mpicxx 로 분리 빌드. 또는 NVHPC `nvc++` 로 통합 빌드.

### 1.5 `parallel/cuda/` 가 비어 있음 (Critical)

**v2**: "1차에 비어 있음" 으로 명시.

**문제**: GPU 우선이면 가장 핵심 폴더. 비워두면 모든 곳에서 ad-hoc CUDA 호출.

**보완**:
```
parallel/cuda/
├── main.hpp
├── cuda_runtime.hpp + .cpp          ← cudaSetDevice, error check, device init
├── cuda_memory.hpp + .cpp           ← alloc, free, copy (host↔device, device↔device)
├── cuda_stream.hpp + .cpp           ← stream pool, synchronize
├── cuda_launch.hpp                  ← grid/block sizing 헬퍼 (inline)
├── nvtx_range.hpp                   ← 프로파일링 RAII (Fortran MPM-STD 처럼 자주 사용)
├── shared_memory.hpp                ← shared memory 사이즈 계산
└── error_check.hpp                  ← CUDA_CHECK 매크로
```

MPM-STD Fortran 의 `cuda_subdomain` 모듈 패턴 그대로.

### 1.6 MPI 가 GPU-aware 인지 명시 안 됨 (Major)

**v2**: MPI 통신 = `Subdomain::exchange_halo(field)`. 내부 구현 미상.

**문제**: GPU 다중 노드에서 device-to-device 직접 전송 가능 (CUDA-aware OpenMPI / MVAPICH-GDR / NCCL). 안 쓰면 host 경유 → 성능 폭락.

**보완**:
- `Subdomain::exchange_halo(field)` 가 내부에서:
  ```cpp
  #ifdef MPMSTD_CUDA_AWARE_MPI
    MPI_Isend(field.device_ptr(), ...);   // 직접
  #else
    cudaMemcpy(host_buf, field.device_ptr(), ...);
    MPI_Isend(host_buf, ...);
    // ... 받은 후 다시 cudaMemcpy
  #endif
  ```
- 빌드 타임 매크로로 결정. 기본은 CUDA-aware (NVHPC SDK 의 OpenMPI 는 기본 지원).

### 1.7 FFT 백엔드가 FFTW3 단일 (Critical)

**v2 + 보고서 09**: `linear_solver/fft/fft_planner.hpp` 가 FFTW3 직접 호출.

**문제**: GPU 에서는 cuFFT 가 정답. cuFFT 는 DCT 직접 지원 안 함 → R2C + pre/post trick 필요 (MPM-STD Fortran 의 `cuda_pressure_DCT_f_pre/post` 패턴).

**보완**:
- `FftPlanner` 를 추상화. `CufftPlanner` + `FftwPlanner` 두 백엔드
- CPU 빌드 → FftwPlanner, GPU 빌드 → CufftPlanner
- DCT 는 cuFFT 의 R2C + 대칭 확장 trick. 헬퍼 함수로 캡슐화

```cpp
class FftPlanner {
public:
  virtual ~FftPlanner() = default;
  virtual void forward_x (double* data) = 0;     // in-place
  virtual void backward_x(double* data) = 0;
  // ...
};
class CufftPlanner : public FftPlanner { /* cuFFT + DCT trick */ };
class FftwPlanner  : public FftPlanner { /* FFTW3 r2r */ };
```

### 1.8 TDMA 백엔드의 GPU 버전 누락 명시 (Critical)

**v2**: `pascal_tdma_backend.hpp` 가 CPU PaScaL_TDMA_C 만 wrap.

**문제**: GPU 우선이면 `pascal_tdma_many_cuda.cu` 를 wrap 해야 함.

**보완**:
- `linear_solver/tdma/pascal_tdma_cuda_backend.hpp + .cpp` 추가
- 기존 `pascal_tdma_backend` (CPU) 와 함께 양립
- `TdmaRegistry::set_backend(d, /*type*/)` 으로 빌드 타임 또는 config 로 결정

```cpp
linear_solver/tdma/
├── main.hpp
├── tdma_solver.hpp                          ← 추상
├── pascal_tdma_cpu_backend.hpp + .cpp       ← PaScaL_TDMA_C (CPU)
├── pascal_tdma_cuda_backend.hpp + .cpp      ← PaScaL_TDMA_C (CUDA, .cu 호출)
├── filtered_tdma_backend.hpp + .cpp         ← 미래
└── tdma_registry.hpp + .cpp
```

### 1.9 빌드 시스템에 nvcc/nvc++ 가 없음 (Major)

**v2 Makefile.inc**: `mpicxx` 단일 컴파일러.

**문제**: `.cu` 파일을 컴파일하려면 nvcc 또는 nvc++ 필요. mpicxx 는 `.cu` 못 봄.

**보완**:
- NVHPC SDK 의 **`nvc++` (혹은 `mpic++` from NVHPC)** 단일 컴파일러 사용 — `.cpp`·`.cu` 모두 처리
- 또는 nvcc 따로: `.cu` → nvcc, `.cpp` → mpicxx, 링크 시 합침
- **권장**: NVHPC `nvc++` (CUDA Fortran 빌드와 일관) — MPM-STD Fortran 도 같은 환경

```make
# Makefile.inc (개정)
CXX        := mpic++                # NVHPC 의 MPI wrapper
CXXSTD     := -std=c++17
OPT        := -O3 -fast -gpu=cc80   # GPU arch (사용자 머신 확인 필요)
CUDA_FLAGS := -cuda                 # nvc++ 의 CUDA 모드
WARN       := -Wall

PASCAL_TDMA_DIR := /shared/home/wel1come1234/workspace/PaScaL_TDMA_C
CUDA_DIR        := /opt/nvidia/hpc_sdk/Linux_x86_64/24.1/cuda
INCS  := -Isrc -I$(PASCAL_TDMA_DIR)/src -I$(CUDA_DIR)/include
LIBS  := -L$(PASCAL_TDMA_DIR)/build -lpascal_tdma -lpascal_tdma_cuda \
         -lcufft -lcudart
```

→ NVHPC `nvc++ -cuda` 가 `.cu`·`.cpp` 모두 컴파일. `-gpu=cc80` 으로 아키텍처 지정.

### 1.10 정밀도 (single/double) 결정 미명시 (Medium)

**v2**: 모든 `double` 하드코딩.

**문제**: GPU 에서 single precision 이 2배 빠름. 사용자가 선택 가능해야 함. MPM-STD Fortran 의 `rp` 컨벤션과 부합.

**보완**:
- `common/types.hpp` 에 `using real_t = double;` (config 또는 빌드 매크로)
- 모든 `double` 을 `real_t` 로 치환
- 매크로 `MPMSTD_SINGLE_PRECISION` 으로 토글
- 1 차 구현은 double, 단 인터페이스는 `real_t` 로 통일

### 1.11 GPU shared memory 활용 패턴 누락 (Major)

**MPM-STD Fortran**: stencil 커널 안에서 `real, shared, dimension(blockdim%x+1, ...)` 로 U 를 thread block 단위 캐시 → 글로벌 메모리 접근 횟수 감소.

**문제**: v2 는 stencil 자유함수만 정의. shared memory 캐싱 패턴 부재.

**보완**:
- `parallel/cuda/shared_memory.hpp` 에 캐싱 도우미 (사이즈 계산, halo 로딩 helper)
- 각 kernels.cu 안에서 패턴 명시
- 1 차 구현은 단순 (no shared memory). M3 후반에 도입 (최적화 단계)

### 1.12 cuda_subdomain (MPM-STD 의 Fortran 모듈) 등가물 없음 (Major)

**MPM-STD Fortran**: `cuda_subdomain` 모듈이 `dx1_d, dx2_d, dx3_d` 등 격자 메트릭의 device 복사본 + ghost 커널 + 전치 커널 + block/thread sizing 보유.

**문제**: v2 의 `grid::Grid` 가 host array 만 가정.

**보완**:
- `grid/grid.hpp` 의 metrics 를 device 에도 복사 (`dx1_d` 등)
- `parallel/cuda/` 또는 `grid/` 안에 device-side 메트릭 보관 helper

### 1.13 nvtx 프로파일링이 누락 (Minor 이지만 중요)

**MPM-STD Fortran**: `nvtxStartRange("Amatrix"), nvtxEndRange` 가 매 단계에 삽입 → Nsight Systems 에서 한눈에 분석.

**v2**: `utilities/timer.hpp` 만 명시. nvtx 없음.

**보완**:
- `parallel/cuda/nvtx_range.hpp` 에 RAII wrapper
```cpp
class NvtxRange {
  NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
};
// 사용: { NvtxRange r("momentum.predict"); ... }
```

### 1.14 GPU 메모리 풀 누락 (Medium)

**문제**: v2 의 `dU/dV/dW` 가 매 step alloc/dealloc 패턴 (PaScaL_TCS 따라). 호스트는 std::vector 가 빠르지만 GPU 의 `cudaMalloc` 은 매우 느림 (~ms 단위).

**보완**:
- `parallel/cuda/cuda_memory.hpp` 에 메모리 풀
- `dU/dV/dW` 등 임시 버퍼는 풀에서 빌려서 사용 → step 끝나면 반환 (실제 dealloc 아님)
- 1 차 구현은 단순 (영구 alloc) — 매 step alloc/dealloc 피할 것

### 1.15 검증 데이터의 GPU·CPU 일치 보장 어려움 (Major)

**v2**: PaScaL_TCS (CPU) golden 과 L∞ < 1e-10 비교.

**문제**:
- GPU 의 `-fast` (NVHPC) 또는 fast math 가 reduction 순서를 바꿈 → bit-identical 안 됨
- 부동소수점 비결정성 (atomic_add 등) 도 영향
- 1e-10 tolerance 는 *너무 엄격* 할 수 있음

**보완**:
- GPU 빌드는 `-fno-fast-math -fno-fma -fno-associative-math` 사용 (성능 약간 손해)
- Tolerance 를 step 별로 약간 완화: 10 step 후 1e-8 정도
- 또는 PaScaL_TCS CPU 와 우리 C++ CPU 빌드 먼저 1e-10 검증 → GPU 빌드는 우리 C++ CPU 빌드와 1e-8 검증

---

## 2. 추가 구조적 제안

### 2.1 폴더 구조 보완 (v2 → v3)

**parallel/ 강화**:
```
parallel/
├── main.hpp
├── mpi/
│   ├── main.hpp
│   ├── mpi_context.hpp + .cpp
│   ├── mpi_topology.hpp + .cpp
│   ├── subdomain.hpp + .cpp
│   └── cuda_aware_mpi.hpp + .cpp     ← (추가) GPU-direct 통신 추상
├── backend/
│   ├── main.hpp
│   ├── backend.hpp                   ← 메모리·스트림 추상
│   ├── cpu_backend.hpp + .cpp
│   └── cuda_backend.hpp + .cpp       ← (강화) device init, stream pool, error check
└── cuda/                              ← (대폭 강화)
    ├── main.hpp
    ├── cuda_runtime.hpp + .cpp
    ├── cuda_memory.hpp + .cpp        ← 메모리 풀
    ├── cuda_stream.hpp + .cpp
    ├── cuda_launch.hpp               ← grid/block sizing
    ├── nvtx_range.hpp                ← 프로파일링
    ├── shared_memory.hpp
    └── error_check.hpp               ← CUDA_CHECK 매크로
```

**common/ 에 매크로 추가**:
```
common/
├── main.hpp
├── types.hpp                          ← real_t (double 또는 float)
├── direction.hpp
└── macros.hpp                         ← (추가) MPMSTD_HD, MPMSTD_INLINE, MPMSTD_RESTRICT
```

**각 equation 하위 폴더에 kernels.cu/cuh 추가**:
```
equation/momentum/
├── main.hpp
├── momentum_equation.hpp + .cpp      ← host orchestrator
├── kernels.cu + kernels.cuh          ← (추가) __global__ + __device__
├── rhs_builders.hpp                  ← MPMSTD_HD 자유함수 (host+device 공유)
├── property_policy.hpp
├── constant_properties.hpp + .cpp
├── nob_properties.hpp + .cpp + nob_properties_kernels.cu  ← (추가)
├── source_term.hpp
├── nob_buoyancy.hpp + .cpp + nob_buoyancy_kernels.cu      ← (추가)
├── boussinesq_buoyancy.hpp + .cpp
└── bulk_forcing.hpp + .cpp
```

**equation/pressure/ 도 동일**:
```
equation/pressure/
├── main.hpp
├── pressure_equation.hpp + .cpp
├── kernels.cu + kernels.cuh           ← RHS, projection 커널
├── rhs_assembler.hpp + .cpp
└── projection.hpp + .cpp
```

**linear_solver/ 강화**:
```
linear_solver/
├── main.hpp
├── tdma/
│   ├── main.hpp
│   ├── tdma_solver.hpp
│   ├── pascal_tdma_cpu_backend.hpp + .cpp      ← PaScaL_TDMA_C CPU
│   ├── pascal_tdma_cuda_backend.hpp + .cpp     ← PaScaL_TDMA_C CUDA wrap
│   ├── filtered_tdma_backend.hpp + .cpp        ← 미래
│   └── tdma_registry.hpp + .cpp
└── fft/
    ├── main.hpp
    ├── fft_planner.hpp                          ← 추상
    ├── fftw_planner.hpp + .cpp                  ← CPU 빌드 시
    ├── cufft_planner.hpp + .cpp + .cu          ← (추가) GPU 빌드 시
    ├── dct_helpers.hpp + .cu                    ← (추가) cuFFT R2C → DCT trick
    ├── eigenvalues.hpp                          ← MPMSTD_HD inline
    └── transpose_plan.hpp + .cpp + .cu          ← (추가) GPU 전치 커널
```

### 2.2 빌드 매트릭스

| 빌드 옵션 | 설명 |
|---|---|
| `MPMSTD_BACKEND=cpu` | mpicxx + FFTW3 + PaScaL_TDMA_C CPU |
| `MPMSTD_BACKEND=cuda` | NVHPC nvc++ -cuda + cuFFT + PaScaL_TDMA_CUDA |
| `MPMSTD_PRECISION=single` | float 사용 |
| `MPMSTD_PRECISION=double` | double 사용 (기본) |
| `MPMSTD_DEBUG=1` | `-O0 -g -G`, CUDA debug, more checks |
| `MPMSTD_CUDA_AWARE_MPI=1` | GPU-direct halo |

→ 같은 코드 base 가 두 가지 빌드 모두 지원. CPU 빌드는 검증용 (PaScaL_TCS golden bit-comparison), GPU 빌드는 production.

### 2.3 1 차 단순화 — CPU 빌드 우선 의도가 아니라 *함께* 빌드

GPU 가 우선이지만, **CPU 빌드를 동시에 유지** 하면:
1. PaScaL_TCS golden 과 bit-comparison 검증 가능
2. GPU 빌드의 부동소수점 비결정성 디버깅
3. GPU 가 없는 환경에서 코드 작성 가능
4. 단위 테스트 빠름

→ **양쪽 빌드를 같은 코드 base 로 유지** 를 목표로. M0–M2 는 CPU 빌드로 정확성 확보 → M3 부터 GPU 빌드 추가 → M5 에서 두 빌드 결과 일치 검증.

---

## 3. 구현 순서 재조정 — GPU 우선 반영

### 3.1 v2 (CPU 우선) vs v3 (GPU 우선)

| M | v2 (보고서 10) | v3 (본 보고서) |
|---|---|---|
| **M0** | infra + Field (CPU) | infra + Field **양쪽 빌드 가능한 골격** + CUDA toolchain check |
| **M1** | BC + TDMA (CPU) | BC + **TDMA CPU + CUDA 양쪽** wrap |
| **M2** | ScalarEquation | ScalarEquation **(CPU 우선, GPU kernels 동시 작성)** |
| **M3** | MomentumEquation | MomentumEquation **+ GPU kernels** |
| **M4** | PressureEquation | PressureEquation **+ cuFFT 통합** |
| **M5** | RBC 통합 (CPU vs PaScaL_TCS) | RBC 통합 **CPU 빌드 → PaScaL_TCS golden 일치** |
| **M5'** | (없음) | **GPU 빌드 → CPU 빌드와 일치 (1e-8 tol)** |
| **M6** | Channel | Channel (CPU+GPU 동시) |
| **M7** | LES/IBM 슬롯 | LES/IBM 슬롯 |

→ M5/M5' 가 가장 큰 위험 지점. 두 빌드의 결과를 좁은 tolerance 로 맞추는 게 어려울 수 있음. 기대 tolerance 를 미리 잡아야:
- CPU vs PaScaL_TCS: 1e-10 (compiler flag 통제 가능)
- CPU vs GPU 우리 코드: 1e-8 (CUDA reduce 순서 차이)
- GPU vs PaScaL_TCS 직접: 1e-8 (transitivity)

### 3.2 M0 세부 재조정

기존:
- common, parallel/mpi, config, grid, field, stencil, post/restart

추가:
- `parallel/cuda/` 의 핵심 (cuda_runtime, cuda_memory, error_check)
- `common/macros.hpp` 의 `MPMSTD_HD`
- Makefile 의 dual-build (CPU/GPU 선택)
- GPU 머신에서 `nvc++ -cuda -gpu=cc80 hello.cu` 가 빌드되는지 단위 테스트

### 3.3 M1 세부 재조정

- `pascal_tdma_cpu_backend` + `pascal_tdma_cuda_backend` 동시 작성
- 단위 테스트: 같은 입력에 두 백엔드 결과 비교 (1e-12)
- `linear_solver/tdma/` 의 빌드 매크로 정리

### 3.4 M3 (Momentum) 세부 재조정

기존: orchestrator + ADI sweep

추가:
- `equation/momentum/kernels.cu` 작성 (RHS·LHS 빌드 커널)
- shared memory 캐싱은 처음엔 안 함 (단순 글로벌 메모리 접근) → 검증 후 최적화
- nvtx 범위 추가
- CPU 빌드 결과 = GPU 빌드 결과 (1e-8) 검증

### 3.5 M4 (Pressure) 세부 재조정

기존: FFTW3 + 전치

추가:
- `cufft_planner` 와 `fftw_planner` 양쪽
- DCT R2C trick (MPM-STD Fortran 의 `cuda_pressure_DCT_f_pre/post` 패턴 직역)
- GPU 전치 커널 (`cuda_transpose_kernel` 류)

---

## 4. 의존성 정리 (GPU 우선 기준)

| 의존 | 사용처 | 비고 |
|---|---|---|
| **NVHPC SDK ≥ 24.1** | 컴파일러·cuFFT·CUDA runtime | 사용자가 MPM-STD Fortran 에서 이미 사용 |
| **MPI (OpenMPI from NVHPC)** | 통신, CUDA-aware | NVHPC 내장 |
| **cuFFT** | 압력 Poisson | NVHPC 내장 |
| **PaScaL_TDMA_C** | TDMA (CPU·CUDA 양쪽) | `/shared/home/wel1come1234/workspace/PaScaL_TDMA_C/` |
| **FFTW3** | CPU 빌드 시 압력 Poisson | 검증용 (선택) |
| **C++ 17** | 언어 표준 | nvc++ 지원 |

**핵심**: PaScaL_TDMA_C 의 GPU 포트가 이미 존재하므로 우리는 wrap 만 하면 됨. 가장 큰 알고리듬 위험 (분산 TDMA 의 GPU 구현) 이 *제거됨*.

---

## 5. heat_gpu 를 1 순위 reference 로 활용

[`PaScaL_TDMA_C/heat_gpu/`](../../PaScaL_TDMA_C/heat_gpu/) 가 우리 프로젝트의 *축소판*:

| heat_gpu 파일 | 우리 매핑 |
|---|---|
| `main.cpp` | `apps/rbc/main.cpp` 의 원형 |
| `mpi_subdomain.cpp/.hpp` | `parallel/mpi/subdomain.hpp` |
| `mpi_topology.cpp/.hpp` | `parallel/mpi/mpi_topology.hpp` |
| `solve_theta.cu/.hpp` | `equation/scalar/kernels.cu` |
| `stencil_coeffs.hpp` | `field/stencil/staggered.hpp` |
| `gather_theta.hpp` | `parallel/cuda/cuda_aware_mpi.hpp` 의 일부 |
| `Makefile` | 우리 `Makefile.inc` 의 base |
| `run_1gpu.sh, run_2gpu.sh` | `run/` 디렉토리 스크립트 |

→ **M0 첫 단계 = `heat_gpu/main.cpp` 빌드·실행 확인**. 환경 검증 + 의존 라이브러리 동작 확인.
→ **이후 우리 코드 작성 시 heat_gpu 파일들을 line-by-line 참조 가능**.

---

## 6. 위험 영역 추가 (v2 → v3)

[10_final_implementation_plan.md](10_final_implementation_plan.md) §7 의 위험 표에 추가:

| # | 위험 | 완화 |
|---|---|---|
| 11 | nvc++ vs nvcc 호환성 (C++ 17 의 일부 기능) | 1 차에는 보수적 C++ 17 만 사용. C++ 20 모듈, concept 사용 X |
| 12 | CUDA-aware MPI 가 사용자 클러스터에서 동작 안 함 | `MPMSTD_CUDA_AWARE_MPI=0` 으로 host 경유 fallback. 성능 저하 감수 |
| 13 | GPU memory 부족 (1024³ 큰 격자) | 메모리 풀 + 임시 버퍼 재사용. unified memory 는 *피함* (성능 안정성) |
| 14 | NVHPC 버전 차이로 빌드 실패 | 1 개 버전 (24.1 권장) 으로 픽스. Makefile.inc 에 경로 명시 |
| 15 | 부동소수점 비결정성으로 회귀 테스트 실패 | tolerance 단계별 완화. `-fno-fma` 등으로 일정 부분 통제 |
| 16 | shared memory 사용 시 bank conflict | 1 차에는 shared memory 안 씀. M3 후반에 도입 |
| 17 | atomic 연산이 결과 비결정 | 가능한 한 reduction 패턴으로 회피 |

---

## 7. 무엇을 v2 에서 가져갈 것인가 (그대로 유지)

v2 의 다음 결정은 GPU 우선 환경에서도 유효:

| 유지할 v2 결정 | 이유 |
|---|---|
| 폴더 카테고리 (parallel, field, boundary, equation 등) | GPU 와 무관, 모듈 응집도 |
| 각 폴더 `main.hpp` facade 패턴 | GPU 빌드도 facade 가 도움 |
| hpp + cpp 같은 폴더 | 편집 효율 |
| equation 하위 폴더 (momentum/pressure/scalar) | GPU 와 무관 |
| `Problem` 객체 + RBC 기본값 자동 | GPU 와 무관 |
| `BcKind` enum + 미래 슬롯 | GPU 와 무관 |
| sweep_order 자동 도출 | GPU 와 무관 |
| Plugin 인터페이스 | GPU 와 무관 |
| Crank-Nicolson + ADI 단일 | 시간 스킴은 GPU 와 무관 |
| apps/ 의 케이스별 main.cpp | GPU 와 무관 |
| 중복 코드 허용 | GPU 와 무관 |

→ 폴더 구조의 *틀* 은 그대로. **그 안에 GPU 파일 추가 + 빌드 매트릭스 강화** 가 본 보고서의 핵심 보완점.

---

## 8. 결론 — v3 의 정수

GPU 우선 으로 전환되면서 v2 에서 **결정적으로 보완해야 할 것 5 가지**:

1. **`parallel/cuda/` 를 비우지 말 것** — CUDA runtime, memory pool, stream, nvtx 모두 1 차에 작성
2. **각 equation 하위 폴더에 `kernels.cu` 추가** — host orchestrator (.cpp) 와 device kernel (.cu) 분리
3. **stencil 자유함수에 `MPMSTD_HD` 매크로** — host·device 양쪽 호출 가능하게
4. **TDMA·FFT 백엔드를 CPU/GPU 동시** — `pascal_tdma_cuda_backend`, `cufft_planner` 추가
5. **빌드 매트릭스** — `MPMSTD_BACKEND={cpu,cuda}` 토글로 같은 코드가 양쪽 빌드. CPU 빌드는 검증용, GPU 빌드는 production

**가장 큰 행운**: PaScaL_TDMA_C 에 이미 CUDA C++ 포트 존재 + heat_gpu 예제 코드 보유. **MPM-STD Fortran 의 CUDA Fortran 패턴을 C++ 로 직역하는 작업이 대부분 *참고 가능한 코드* 가 있음**. 알고리듬 위험은 거의 없고, 남는 건 *통합·관리* 위험.

→ **다음 작업**: 
1. `heat_gpu/` 를 실제로 빌드·실행해 사용자 GPU 머신 환경 검증 (1 일)
2. v3 구조에 맞춰 `mkdir -p` 골격 생성 + Makefile.inc 작성 (1 일)
3. M0 시작 — `parallel/cuda/cuda_runtime.hpp` 부터

이게 가장 효율적인 시작점.
