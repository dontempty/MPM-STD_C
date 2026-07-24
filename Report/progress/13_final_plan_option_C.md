# Option C 최종 구현 계획서 — GPU-ready 아키텍처 + CPU 우선 구현

> **방침**: GPU-ready 폴더·매크로·인터페이스 골격은 M0 부터 갖춰두되, **알고리듬 구현은 CPU 빌드로 먼저 완성** → PaScaL_TCS 와 bit-exact 검증 → 그 후 GPU 커널 채우기.
> 본 문서: 보고서 [01](01_CaNS_analysis.md)–[12](12_critique_gpu_first.md) 의 결정을 통합한 **최종 작업 매뉴얼**. M0 첫 commit 부터 M7 까지 단계별 파일·구현·검증을 명세.
> 1차 목표: PaScaL_TCS 의 NOB Rayleigh-Bénard (Ra=100, Pr=1, 512×128×256, 10 step) 결과 재현 — CPU 빌드 bit-exact, GPU 빌드 1e-8 일치.

---

## 0. 핵심 결정 (변경 없음, 재확인)

| # | 결정 | 출처 |
|---|---|---|
| 1 | 라이브러리 + 케이스별 `apps/<case>/main.cpp` | [08](08_design_revision_v2.md) |
| 2 | hpp + cpp 같은 폴더 (Convention B) | [11](11_directory_structure_v2.md) |
| 3 | 각 폴더 `main.hpp` (+ 큰 폴더 `main.cpp`) facade | [11](11_directory_structure_v2.md) |
| 4 | equation 항목별 하위 폴더 (momentum/pressure/scalar) | [11](11_directory_structure_v2.md) |
| 5 | physics 안에 LES/IBM 하위 폴더 | [11](11_directory_structure_v2.md) |
| 6 | BC: Periodic/Dirichlet/Neumann + 슬롯 (Wall/Inflow/Outflow) | [05](05_BC_design.md) |
| 7 | Problem 객체 + RBC 기본값 자동 (z=wall) | [05](05_BC_design.md) |
| 8 | VectorField (face) + ScalarField (cell) | [08](08_design_revision_v2.md) |
| 9 | stencil 자유함수가 인덱스 산술 캡슐화 | [08](08_design_revision_v2.md) |
| 10 | Crank-Nicolson + ADI 단일 시간적분기 | [08](08_design_revision_v2.md) |
| 11 | PropertyPolicy + SourceTerm — RBC/Channel 통합 | [07](07_momentum_unification.md) |
| 12 | Plugin (Phase enum) — Stats/Probe/LES/IBM | [08](08_design_revision_v2.md) |
| 13 | BC-aware FFT/DCT/DST 자동 선택 | [09](09_pressure_solver_design.md) |
| 14 | sweep_order 자동 도출 | [05](05_BC_design.md) |
| 15 | TDMA 백엔드 추상 (CPU + CUDA 동시) | [12](12_critique_gpu_first.md) |
| 16 | **GPU-ready 아키텍처 + CPU 우선 구현 (Option C)** | 본 문서 |
| 17 | PaScaL_TDMA_C 의 CPU·CUDA 백엔드 모두 wrap | [12](12_critique_gpu_first.md) |
| 18 | `heat_gpu/` 를 GPU 패턴 1 순위 reference 로 활용 | [12](12_critique_gpu_first.md) |

---

## 1. 최종 디렉토리 구조 (v3)

[11](11_directory_structure_v2.md) 의 v2 + [12](12_critique_gpu_first.md) 의 GPU 보완을 적용:

```
MPM-STD(C++)/
├── Makefile
├── Makefile.inc
├── README.md
├── Report/
│
├── src/                                       ← 라이브러리 본체 (hpp+cpp 같이)
│   │
│   ├── common/
│   │   ├── main.hpp
│   │   ├── types.hpp                         ← real_t (double/float toggle)
│   │   ├── direction.hpp                     ← Direction/Side/Component enum
│   │   └── macros.hpp                        ← MPMSTD_HD, MPMSTD_RESTRICT, CUDA_CHECK
│   │
│   ├── parallel/
│   │   ├── main.hpp
│   │   ├── mpi/
│   │   │   ├── main.hpp
│   │   │   ├── mpi_context.hpp + .cpp
│   │   │   ├── mpi_topology.hpp + .cpp
│   │   │   ├── subdomain.hpp + .cpp
│   │   │   └── cuda_aware_mpi.hpp + .cpp     ← GPU-direct halo (1차에 fallback path 도 포함)
│   │   ├── backend/
│   │   │   ├── main.hpp
│   │   │   ├── backend.hpp                   ← 추상 (메모리/스트림만)
│   │   │   ├── cpu_backend.hpp + .cpp        ← M0 구현
│   │   │   └── cuda_backend.hpp + .cpp       ← M0 stub → M5' 실제 구현
│   │   └── cuda/
│   │       ├── main.hpp
│   │       ├── cuda_runtime.hpp + .cpp       ← M0 stub → M5' 실구현
│   │       ├── cuda_memory.hpp + .cpp        ← M0 stub
│   │       ├── cuda_stream.hpp + .cpp        ← M0 stub
│   │       ├── cuda_launch.hpp               ← grid/block sizing helper (inline)
│   │       ├── nvtx_range.hpp                ← RAII (CPU 빌드 시 no-op)
│   │       ├── shared_memory.hpp             ← M5+
│   │       └── error_check.hpp               ← CUDA_CHECK 매크로
│   │
│   ├── config/
│   │   ├── main.hpp
│   │   ├── config.hpp + .cpp                 ← TOML 파서
│   │   └── logger.hpp + .cpp
│   │
│   ├── grid/
│   │   ├── main.hpp
│   │   ├── grid.hpp + .cpp                   ← x, dx, dmx (축별, host)
│   │   ├── grid_device.hpp + .cpp            ← device-side metrics mirror (GPU build)
│   │   └── stretching.hpp + .cpp             ← tanh
│   │
│   ├── field/
│   │   ├── main.hpp
│   │   ├── scalar_field.hpp + .cpp           ← host primary + optional device buffer
│   │   ├── vector_field.hpp + .cpp
│   │   ├── field_registry.hpp + .cpp
│   │   └── stencil/
│   │       ├── main.hpp
│   │       ├── staggered.hpp                 ← MPMSTD_HD inline (host+device 공유)
│   │       └── viscous.hpp                   ← MPMSTD_HD inline
│   │
│   ├── boundary/
│   │   ├── main.hpp
│   │   ├── bc_kind.hpp
│   │   ├── face_bc.hpp + .cpp
│   │   ├── field_boundary.hpp
│   │   ├── domain_topology.hpp + .cpp
│   │   ├── problem.hpp + .cpp                ← RBC 기본값 자동 (z=wall)
│   │   └── boundary_applier.hpp + .cpp
│   │
│   ├── linear_solver/
│   │   ├── main.hpp
│   │   ├── tdma/
│   │   │   ├── main.hpp
│   │   │   ├── tdma_solver.hpp               ← 추상
│   │   │   ├── pascal_tdma_cpu_backend.hpp + .cpp     ← M1 (PaScaL_TDMA_C CPU)
│   │   │   ├── pascal_tdma_cuda_backend.hpp + .cpp    ← M1 stub → M5' 실구현
│   │   │   ├── filtered_tdma_backend.hpp + .cpp       ← 미래
│   │   │   └── tdma_registry.hpp + .cpp
│   │   └── fft/
│   │       ├── main.hpp
│   │       ├── fft_planner.hpp               ← 추상
│   │       ├── fftw_planner.hpp + .cpp       ← M4 (CPU)
│   │       ├── cufft_planner.hpp + .cpp      ← M4 stub → M5'
│   │       ├── eigenvalues.hpp               ← MPMSTD_HD inline
│   │       ├── transpose_plan.hpp + .cpp     ← M4 (CPU MPI_Alltoallw)
│   │       └── kernels/                       ← M5+
│   │           ├── kernels.hpp                ← 선언
│   │           ├── kernels_cpu.cpp            ← M4 host 구현
│   │           └── kernels_cuda.cu            ← M5' device 구현
│   │
│   ├── equation/
│   │   ├── main.hpp
│   │   ├── momentum/
│   │   │   ├── main.hpp                      ← make_rbc, make_channel 빌더
│   │   │   ├── momentum_equation.hpp + .cpp  ← class + orchestrator
│   │   │   ├── rhs_builders.hpp              ← MPMSTD_HD inline 자유함수
│   │   │   ├── property_policy.hpp           ← 추상
│   │   │   ├── constant_properties.hpp + .cpp
│   │   │   ├── nob_properties.hpp + .cpp
│   │   │   ├── source_term.hpp               ← 추상
│   │   │   ├── nob_buoyancy.hpp + .cpp
│   │   │   ├── boussinesq_buoyancy.hpp + .cpp
│   │   │   ├── bulk_forcing.hpp + .cpp
│   │   │   └── kernels/                       ← 핵심: backend-specific 구현 격리
│   │   │       ├── kernels.hpp                ← 선언 (universal)
│   │   │       ├── kernels_cpu.cpp            ← host 구현 (CPU 빌드)
│   │   │       └── kernels_cuda.cu            ← device 구현 (CUDA 빌드)
│   │   ├── pressure/
│   │   │   ├── main.hpp
│   │   │   ├── pressure_equation.hpp + .cpp
│   │   │   ├── rhs_assembler.hpp + .cpp
│   │   │   ├── projection.hpp + .cpp
│   │   │   └── kernels/
│   │   │       ├── kernels.hpp
│   │   │       ├── kernels_cpu.cpp
│   │   │       └── kernels_cuda.cu
│   │   └── scalar/
│   │       ├── main.hpp
│   │       ├── scalar_equation.hpp + .cpp
│   │       ├── scalar_traits.hpp
│   │       └── kernels/
│   │           ├── kernels.hpp
│   │           ├── kernels_cpu.cpp
│   │           └── kernels_cuda.cu
│   │
│   ├── physics/
│   │   ├── main.hpp
│   │   ├── plugin.hpp                        ← 공통 Plugin + Phase enum
│   │   ├── les/                              ← M7+ 구현
│   │   │   ├── main.hpp
│   │   │   ├── smagorinsky.hpp + .cpp
│   │   │   ├── les_properties.hpp + .cpp
│   │   │   ├── les_plugin.hpp + .cpp
│   │   │   └── kernels/
│   │   │       ├── kernels.hpp
│   │   │       ├── kernels_cpu.cpp
│   │   │       └── kernels_cuda.cu
│   │   └── ibm/                              ← M7+ 구현
│   │       ├── main.hpp
│   │       ├── ibm_mask.hpp + .cpp
│   │       ├── cell_classification.hpp + .cpp
│   │       ├── ibm_plugin.hpp + .cpp
│   │       └── kernels/
│   │           ├── kernels.hpp
│   │           ├── kernels_cpu.cpp
│   │           └── kernels_cuda.cu
│   │
│   ├── integrator/
│   │   ├── main.hpp
│   │   ├── time_stepper.hpp + .cpp
│   │   └── cfl_controller.hpp + .cpp
│   │
│   ├── post/
│   │   ├── main.hpp
│   │   ├── instant_io.hpp + .cpp
│   │   ├── restart_io.hpp + .cpp
│   │   └── probe.hpp + .cpp
│   │
│   ├── stat/
│   │   ├── main.hpp
│   │   ├── statistics_accumulator.hpp + .cpp
│   │   ├── statistics_plugin.hpp + .cpp
│   │   └── stat_io.hpp + .cpp
│   │
│   └── utilities/
│       ├── main.hpp
│       ├── diagnostics.hpp + .cpp
│       └── timer.hpp + .cpp
│
├── apps/
│   ├── rbc/         { main.cpp, input.toml, Makefile }
│   ├── channel/     { main.cpp, input.toml, Makefile }
│   ├── thermal_only_check/   { main.cpp, Makefile }     ← M2 검증
│   └── poisson_only_check/   { main.cpp, Makefile }     ← M4 검증
│
├── test/
│   ├── unit/
│   ├── integration/
│   └── regression/
│       ├── golden/                            ← PaScaL_TCS 출력
│       └── compare.py
│
├── run/
│   ├── rbc_cpu.sh, rbc_gpu.sh
│   ├── channel_cpu.sh, channel_gpu.sh
│   └── inputs/
│
└── build/
    ├── cpu/                                   ← BACKEND=cpu 산출물
    │   ├── obj/
    │   ├── lib/libmpmstd_cpu.a
    │   └── bin/
    └── cuda/                                  ← BACKEND=cuda 산출물
        ├── obj/
        ├── lib/libmpmstd_cuda.a
        └── bin/
```

→ **핵심 패턴**: 각 equation 하위 폴더에 `kernels/` 서브폴더. 그 안에 `kernels.hpp` (선언), `kernels_cpu.cpp` (host 구현), `kernels_cuda.cu` (device 구현). 빌드 시스템이 BACKEND 에 따라 둘 중 하나만 컴파일.

---

## 2. 빌드 시스템

### 2.1 `Makefile.inc`

```make
# === Backend 선택 (필수) ===
BACKEND ?= cpu      # cpu 또는 cuda

# === 정밀도 (선택) ===
PRECISION ?= double # double 또는 single

# === 외부 의존성 경로 ===
PASCAL_TDMA_DIR := /shared/home/wel1come1234/workspace/PaScaL_TDMA_C

ifeq ($(BACKEND),cpu)
  CXX        := mpicxx
  CXXSTD     := -std=c++17
  OPT        := -O3 -march=native -fno-fast-math -fno-associative-math -fno-fma
  DEFINES    := -DMPMSTD_BACKEND_CPU
  EXT_LIBS   := -lfftw3 -lm \
                -L$(PASCAL_TDMA_DIR)/build/cpu/lib -lpascal_tdma
  BUILD_DIR  := build/cpu

else ifeq ($(BACKEND),cuda)
  CXX        := mpic++           # NVHPC 의 MPI wrapper
  CXXSTD     := -std=c++17
  OPT        := -O3 -fast -gpu=cc80      # GPU arch 는 사용자 머신에 맞게
  CUDA_FLAGS := -cuda                    # nvc++ CUDA 모드
  DEFINES    := -DMPMSTD_BACKEND_CUDA
  EXT_LIBS   := -cudalib=cufft \
                -L$(PASCAL_TDMA_DIR)/build/cuda/lib -lpascal_tdma_cuda
  BUILD_DIR  := build/cuda

else
  $(error Unknown BACKEND=$(BACKEND). Use cpu or cuda)
endif

ifeq ($(PRECISION),single)
  DEFINES += -DMPMSTD_SINGLE_PRECISION
endif

INCS  := -Isrc -I$(PASCAL_TDMA_DIR)/src
WARN  := -Wall
DEPFLAGS = -MMD -MP
```

### 2.2 루트 `Makefile`

```make
include Makefile.inc

.PHONY: all lib apps tests clean

all: lib apps tests

lib:
	$(MAKE) -C src

apps: lib
	$(MAKE) -C apps/rbc
	$(MAKE) -C apps/channel
	$(MAKE) -C apps/thermal_only_check
	$(MAKE) -C apps/poisson_only_check

tests: lib
	$(MAKE) -C test

clean:
	rm -rf build/

# 두 백엔드 모두 빌드
all-backends:
	$(MAKE) BACKEND=cpu  all
	$(MAKE) BACKEND=cuda all
```

### 2.3 `src/Makefile` — backend 별 파일 선택

```make
include ../Makefile.inc

# 모든 cpp 파일 수집 후 backend 와 매치되지 않는 것 제외
ALL_CPP := $(shell find . -name "*.cpp")
ALL_CU  := $(shell find . -name "*.cu")

ifeq ($(BACKEND),cpu)
  # CPU 빌드: .cu 제외, *_cuda.cpp 제외
  SRCS := $(filter-out %_cuda.cpp,$(ALL_CPP))
else
  # CUDA 빌드: *_cpu.cpp 제외 + .cu 포함
  SRCS := $(filter-out %_cpu.cpp,$(ALL_CPP)) $(ALL_CU)
endif

OBJS := $(patsubst %.cpp,../$(BUILD_DIR)/obj/%.o,$(filter %.cpp,$(SRCS))) \
        $(patsubst %.cu,../$(BUILD_DIR)/obj/%.o,$(filter %.cu,$(SRCS)))

LIB := ../$(BUILD_DIR)/lib/libmpmstd.a

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

../$(BUILD_DIR)/obj/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(DEFINES) $(INCS) $(DEPFLAGS) -c $< -o $@

../$(BUILD_DIR)/obj/%.o: %.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(OPT) $(CUDA_FLAGS) $(DEFINES) $(INCS) -c $< -o $@

-include $(OBJS:.o=.d)
```

→ **`kernels_cpu.cpp` 는 CPU 빌드에서만, `kernels_cuda.cu` 는 CUDA 빌드에서만 컴파일**. 같은 함수 시그니처가 `kernels.hpp` 에 선언되어 있어 orchestrator (`.cpp`) 는 양쪽 빌드 모두 동일.

### 2.4 빌드 명령어

```bash
make BACKEND=cpu  all        # CPU 빌드 → build/cpu/bin/
make BACKEND=cuda all        # GPU 빌드 → build/cuda/bin/
make BACKEND=cpu PRECISION=single all
make all-backends            # 양쪽 모두
make clean
```

---

## 3. 코드 컨벤션

### 3.1 `common/macros.hpp`

```cpp
#pragma once

// === host/device 통합 헬퍼 ===
#ifdef MPMSTD_BACKEND_CUDA
  #define MPMSTD_HD __host__ __device__ inline
  #define MPMSTD_DEVICE __device__ inline
  #define MPMSTD_RESTRICT __restrict__
#else
  #define MPMSTD_HD inline
  #define MPMSTD_DEVICE inline
  #define MPMSTD_RESTRICT __restrict__
#endif

// === CUDA 에러 체크 ===
#ifdef MPMSTD_BACKEND_CUDA
  #include <cuda_runtime.h>
  #define CUDA_CHECK(call) do {                                    \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
      std::abort();                                                 \
    }                                                               \
  } while(0)
#else
  #define CUDA_CHECK(call) ((void)0)
#endif

// === unused parameter (CPU 빌드의 stub) ===
#define MPMSTD_UNUSED(x) ((void)(x))
```

### 3.2 `common/types.hpp`

```cpp
#pragma once
namespace mpmstd {

#ifdef MPMSTD_SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

using int_t = int;

} // namespace mpmstd
```

### 3.3 `common/direction.hpp`

```cpp
#pragma once
namespace mpmstd {

enum class Direction : int { X = 0, Y = 1, Z = 2 };
enum class Side      : int { Minus = 0, Plus = 1 };
enum class Component : int { U = 0, V = 1, W = 2 };

constexpr int to_int(Direction d) { return static_cast<int>(d); }
constexpr int to_int(Side s)      { return static_cast<int>(s); }
constexpr int to_int(Component c) { return static_cast<int>(c); }

} // namespace mpmstd
```

### 3.4 stencil 자유함수 예 — `field/stencil/staggered.hpp`

```cpp
#pragma once
#include "common/macros.hpp"
#include "common/types.hpp"

namespace mpmstd::stencil {

MPMSTD_HD real_t dpdx_at_face_x(const real_t* MPMSTD_RESTRICT P,
                                  const real_t* MPMSTD_RESTRICT dmx1,
                                  int i, int j, int k,
                                  int n1, int n2, int n3) {
  const int idx_ic = i*n2*n3 + j*n3 + k;
  const int idx_im = (i-1)*n2*n3 + j*n3 + k;
  return (P[idx_ic] - P[idx_im]) / dmx1[i];
}

MPMSTD_HD real_t divergence_at_cell(const real_t* MPMSTD_RESTRICT Ux,
                                      const real_t* MPMSTD_RESTRICT Vy,
                                      const real_t* MPMSTD_RESTRICT Wz,
                                      const real_t* MPMSTD_RESTRICT dx1,
                                      const real_t* MPMSTD_RESTRICT dx2,
                                      const real_t* MPMSTD_RESTRICT dx3,
                                      int i, int j, int k, int n1, int n2, int n3) {
  // ... (raw pointer indexing for both host & device)
}

} // namespace mpmstd::stencil
```

→ **CPU 빌드**: `inline` 함수. **GPU 빌드**: `__host__ __device__ inline` — host 루프와 device 커널 모두에서 호출 가능.

### 3.5 kernels/ 패턴 — `equation/momentum/kernels/kernels.hpp`

```cpp
#pragma once
#include "common/types.hpp"
namespace mpmstd::equation::momentum {

// 1차 stage: z 방향 TDMA 행렬·RHS 빌드
void predict_build_z_stage(
    real_t* RHS, real_t* Am, real_t* Ac, real_t* Ap,
    const real_t* U, const real_t* V, const real_t* W, const real_t* P,
    const real_t* T, const real_t* Mu, const real_t* invRho,
    const real_t* dx1, const real_t* dx2, const real_t* dx3,
    const real_t* dmx1, const real_t* dmx2, const real_t* dmx3,
    int n1, int n2, int n3, real_t dt, Component c);

// block_couple, pseudo_update 등도 동일 패턴
void block_couple_v(/* ... */);
void block_couple_u(/* ... */);
void pseudo_update(/* ... */);

} // namespace
```

`kernels_cpu.cpp`:
```cpp
#include "kernels.hpp"
#include "field/stencil/staggered.hpp"
namespace mpmstd::equation::momentum {

void predict_build_z_stage(/*...*/) {
  for (int k = 1; k <= n3-1; ++k)
  for (int j = 1; j <= n2-1; ++j)
  for (int i = 1; i <= n1-1; ++i) {
    // stencil 자유함수 호출
    real_t rhs = /* ... */ ;
    // ... 행렬·RHS 빌드
  }
}

void block_couple_v(/*...*/) { /* host loop */ }
void block_couple_u(/*...*/) { /* host loop */ }
void pseudo_update (/*...*/) { /* host loop */ }

} // namespace
```

`kernels_cuda.cu`:
```cpp
#include "kernels.hpp"
#include "field/stencil/staggered.hpp"
#include "parallel/cuda/cuda_launch.hpp"
namespace mpmstd::equation::momentum {

__global__ void predict_build_z_stage_kernel(/*동일 시그니처*/) {
  int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z*blockDim.z + threadIdx.z + 1;
  if (i >= n1 || j >= n2 || k >= n3) return;
  // 동일한 stencil 자유함수 호출 (MPMSTD_HD)
}

// host-side launcher (kernels.hpp 의 함수 시그니처)
void predict_build_z_stage(/*...*/) {
  dim3 blocks(/*...*/), threads(/*...*/);
  predict_build_z_stage_kernel<<<blocks, threads>>>(/*...*/);
  CUDA_CHECK(cudaGetLastError());
}
// 동일 패턴으로 block_couple, pseudo_update launcher

} // namespace
```

→ `momentum_equation.cpp` (orchestrator) 는 `kernels::predict_build_z_stage(...)` 만 호출. **CPU/GPU 차이를 모름**. Backend 가 선택되면 자동으로 그 .cpp 또는 .cu 가 링크됨.

### 3.6 Field 메모리 모델 — Option C 친화

```cpp
// src/field/scalar_field.hpp
namespace mpmstd::field {

class ScalarField {
public:
  ScalarField(const Grid& g, std::string name);
  ~ScalarField();

  // host 접근 (양쪽 빌드 동일)
  real_t&  host(int i, int j, int k);
  real_t*  host_ptr();

  // device 접근 (CUDA 빌드에서만 의미)
  real_t*  device_ptr();

  // 동기화 (CPU 빌드에서 no-op)
  void     to_device();
  void     to_host();

  int n1sub() const; int n2sub() const; int n3sub() const;
  static constexpr int halo_width() { return 1; }
  const std::string& name() const { return name_; }

private:
  std::vector<real_t> host_buffer_;
#ifdef MPMSTD_BACKEND_CUDA
  real_t* device_buffer_ = nullptr;
#endif
  std::string name_;
  int n1_, n2_, n3_;
};

}
```

`.cpp` 구현에서:
- `to_device()`: GPU 빌드면 `cudaMemcpy`, CPU 빌드면 no-op
- `device_ptr()`: GPU 빌드면 device 포인터, CPU 빌드면 `host_ptr()` 와 동일 (즉 호출자는 차이 모름)

→ orchestrator 가 `field.device_ptr()` 를 kernels 에 넘기면 CPU 빌드에서는 host 포인터를 넘기는 효과. **같은 코드 두 빌드 동작**.

---

## 4. 마일스톤별 상세 계획

### M0 — 인프라 + Field + 빌드 시스템 (2 주)

#### 목표
- CPU 빌드가 동작하는 최소 골격
- GPU 빌드는 stub (.cu 파일 비어 있어도 컴파일 성공)
- 8 랭크 (2×2×2) MPI Cart 동작

#### 파일 (생성 순서)

```
src/common/                                  ← 모두 header-only
  main.hpp, types.hpp, direction.hpp, macros.hpp

src/parallel/mpi/
  main.hpp
  mpi_context.hpp + .cpp                    ← MPI_Init/Finalize
  mpi_topology.hpp + .cpp                   ← 3D Cart + 1D sub-comm
  subdomain.hpp + .cpp                      ← 인덱스 + ghost DDT
  cuda_aware_mpi.hpp + .cpp                 ← CPU 빌드: 단순 MPI 호출

src/parallel/backend/
  main.hpp
  backend.hpp                                ← 추상
  cpu_backend.hpp + .cpp                    ← 단순 malloc/free
  cuda_backend.hpp + .cpp                   ← stub (throw "not built")

src/parallel/cuda/
  main.hpp
  cuda_runtime.hpp + .cpp                   ← CPU 빌드: 모두 no-op
  cuda_memory.hpp + .cpp                    ← CPU 빌드: malloc/free
  cuda_stream.hpp + .cpp                    ← CPU 빌드: no-op
  cuda_launch.hpp                            ← grid/block helper (CPU에선 1)
  nvtx_range.hpp                             ← CPU 빌드: no-op RAII
  shared_memory.hpp                          ← M5+
  error_check.hpp                            ← 매크로

src/config/
  main.hpp, config.hpp + .cpp, logger.hpp + .cpp

src/grid/
  main.hpp, grid.hpp + .cpp, stretching.hpp + .cpp
  grid_device.hpp + .cpp                    ← CPU 빌드: no-op

src/field/
  main.hpp
  scalar_field.hpp + .cpp                   ← host primary
  vector_field.hpp + .cpp
  field_registry.hpp + .cpp
  stencil/
    main.hpp, staggered.hpp, viscous.hpp    ← MPMSTD_HD inline

src/post/
  main.hpp, restart_io.hpp + .cpp           ← MPI-IO binary

Makefile, Makefile.inc, src/Makefile

test/unit/
  test_grid.cpp
  test_field.cpp
  test_stencil.cpp
  test_mpi_halo.cpp
```

#### 구현 순서
1. 외부 의존성 빌드 확인:
   - `cd PaScaL_TDMA_C && make` → `build/cpu/lib/libpascal_tdma.a` 생성
   - `cd PaScaL_TDMA_C/heat_gpu && make && ./run_1gpu.sh` → 환경 동작 확인 (선택적 GPU 검증)
2. `common/` (header-only 4 개 파일) 작성
3. `parallel/mpi/` 4 개 파일 (Context → Topology → Subdomain → cuda_aware_mpi)
4. `parallel/backend/cpu_backend` + 추상 헤더
5. `parallel/cuda/` 의 stub 헤더 + .cpp (CPU 빌드에서 모두 no-op)
6. `config/` 와 `grid/`
7. `field/` (ScalarField → VectorField → FieldRegistry)
8. `field/stencil/staggered.hpp` (MPMSTD_HD 자유함수)
9. `post/restart_io`
10. Makefile 체계 완성 → `make BACKEND=cpu all` 통과
11. 단위 테스트 작성·통과

#### 검증 (M0 DoD)
- `make BACKEND=cpu all` 무 에러
- `make BACKEND=cuda all` 도 에러 없이 빌드 (실행은 안 해도 됨)
- `mpirun -np 8 build/cpu/bin/test_mpi_halo` 통과
- 격자 stretching 의 메트릭이 PaScaL_TCS 와 1e-15 일치
- ScalarField + VectorField alloc, halo exchange, restart 왕복 통과
- `stencil::divergence_at_cell` 단위 테스트 EOC = 2

---

### M1 — BC + TDMA (2 주)

#### 목표
- `Problem` 객체로 RBC/Channel BC 자동 셋업
- PaScaL_TDMA_C 의 CPU·CUDA 백엔드 모두 wrap (CUDA 는 stub)
- BC 적용 + matrix row 보정 검증

#### 파일

```
src/boundary/
  main.hpp
  bc_kind.hpp
  face_bc.hpp + .cpp
  field_boundary.hpp
  domain_topology.hpp + .cpp           ← sweep_order, wall_axis
  problem.hpp + .cpp                    ← RBC 기본값 자동
  boundary_applier.hpp + .cpp

src/linear_solver/tdma/
  main.hpp
  tdma_solver.hpp                       ← 추상
  pascal_tdma_cpu_backend.hpp + .cpp    ← PaScaL_TDMA_C CPU wrap
  pascal_tdma_cuda_backend.hpp + .cpp   ← stub (throw)
  tdma_registry.hpp + .cpp

test/unit/
  test_bc_kind.cpp
  test_problem_defaults.cpp
  test_sweep_order.cpp
  test_bc_apply.cpp
  test_tdma_backend_cpu.cpp
```

#### 구현 순서
1. `boundary/bc_kind.hpp` (enum)
2. `boundary/face_bc.hpp` (FaceBc + 정적 생성자)
3. `boundary/field_boundary.hpp` (6 면 컨테이너)
4. `boundary/domain_topology.cpp` (sweep_order, wall_axis 자동 도출)
5. `boundary/problem.cpp` — 생성자에서 RBC 기본 (z=wall, T BCs) 자동
6. `boundary/boundary_applier.cpp` — apply_ghost (3 종 BC), modify_tdma_row
7. `linear_solver/tdma/tdma_solver.hpp` (추상)
8. `linear_solver/tdma/pascal_tdma_cpu_backend.cpp` — PaScaL_TDMA_C 의 `pascal_tdma_many.cpp` wrap
9. `linear_solver/tdma/pascal_tdma_cuda_backend.cpp` — throw stub
10. `linear_solver/tdma/tdma_registry.cpp` — 축별 백엔드 보관

#### 검증
- `Problem p;` 한 줄로 RBC (z=wall) 완성 검증
- `sweep_order()` 가 RBC = (X, Y, Z) 반환 ([04](04_PaScaL_TCS_analysis.md): PaScaL_TCS 는 y=wall → (X, Z, Y) 반환)
- PaScaL_TDMA_C CPU 의 알려진 해 (cyclic + non-cyclic) 와 1e-14 일치
- BC apply_ghost: Periodic / Dirichlet / Neumann 모두 정확
- modify_tdma_row: Dirichlet/Neumann wall 면에서 행 변경 정확

---

### M2 — ScalarEquation (열) (2 주)

#### 목표
- 속도 동결 + manufactured solution 에서 시·공간 2 차 수렴
- PaScaL_TCS 의 `mpi_thermal_solver` 와 1 step L∞ < 1e-12

#### 파일

```
src/equation/scalar/
  main.hpp
  scalar_equation.hpp + .cpp                ← orchestrator
  scalar_traits.hpp
  kernels/
    kernels.hpp                              ← 선언
    kernels_cpu.cpp                          ← host 구현 (M2 작성)
    kernels_cuda.cu                          ← stub (빈 함수 또는 #if 0)

src/equation/momentum/                       ← PropertyPolicy 부분만 미리
  property_policy.hpp                        ← 추상
  constant_properties.hpp + .cpp             ← μ=1, 1/ρ=1
  nob_properties.hpp + .cpp                  ← T-의존 다항식

apps/thermal_only_check/
  main.cpp                                    ← 속도 동결 manufactured solution

test/integration/
  test_thermal_manufactured.cpp
```

#### 구현 순서
1. `scalar/scalar_traits.hpp` (name, diffusivity, source_fn?)
2. `scalar/kernels/kernels.hpp` — ADI 3-stage 함수 시그니처
3. `scalar/kernels/kernels_cpu.cpp` — host 루프
4. `scalar/kernels/kernels_cuda.cu` — 빈 stub (M5' 에서 채움)
5. `scalar/scalar_equation.cpp` — orchestrator (compute_coeffi + step)
6. `equation/momentum/constant_properties.cpp` (간단)
7. `equation/momentum/nob_properties.cpp` (다항식, PaScaL_TCS 의 `mpi_momentum_coeffi` 직역)
8. `apps/thermal_only_check/main.cpp`

#### 검증
- Manufactured solution: T(x,y,z,t) = sin(πx)·sin(πy)·sin(πz)·exp(-3π²t)
- 시간 격자 4 배 미세화 → L∞ 1/4 감소 (EOC = 2)
- 공간 격자 4 배 미세화 → L∞ 1/16 감소 (EOC = 2)
- 1, 4, 16 랭크 결과 1e-12 이내 일치
- PaScaL_TCS 의 `mpi_thermal_solver` 와 1 step 후 L∞ < 1e-12

---

### M3 — MomentumEquation (3 주, 가장 중요)

#### 목표
- predict + block_couple + pseudo_update 가 PaScaL_TCS 의 `solvedU/V/W + blockLdU/V + pseudoupdateUVW` 와 1 step L∞ < 1e-12
- 보고서 [07](07_momentum_unification.md) 의 PropertyPolicy + SourceTerm 합성 검증

#### 파일

```
src/equation/momentum/                       ← 완성
  main.hpp                                    ← make_rbc, make_channel 빌더
  momentum_equation.hpp + .cpp               ← class + orchestrator
  rhs_builders.hpp                            ← MPMSTD_HD 자유함수
  property_policy.hpp                         (이미 M2 에서 부분 작성)
  constant_properties.hpp + .cpp
  nob_properties.hpp + .cpp
  source_term.hpp                             ← 추상
  nob_buoyancy.hpp + .cpp
  boussinesq_buoyancy.hpp + .cpp
  bulk_forcing.hpp + .cpp
  kernels/
    kernels.hpp
    kernels_cpu.cpp                           ← M3 핵심 작업
    kernels_cuda.cu                           ← M5' 채움

test/integration/
  test_momentum_one_step.cpp                  ← PaScaL_TCS 와 1 step 비교
```

#### 구현 순서

**Step 1 — RHS 빌더** (rhs_builders.hpp, MPMSTD_HD):
- `assemble_convection_rhs_at(c, U, g, i, j, k)` — skew-symmetric (PaScaL_TCS line 295-302)
- `assemble_viscous_rhs_at(c, U, μ, 1/ρ, g, i, j, k)` — μ harmonic mean + cross derivatives (PaScaL_TCS line 317-341)
- `assemble_pressure_gradient_rhs_at(c, P, 1/ρ, g, i, j, k)` — `-Cmp·invRho·∂P/∂x` (line 344-345)

**Step 2 — kernels_cpu.cpp**:
- `predict_build_z_stage` (PaScaL_TCS solvedU line 281-416)
- `predict_build_x_stage` (line 425-541)
- `predict_build_y_stage` (line 550-541)
- `block_couple_v_apply` (mpi_momentum_blockLdV line 1278-1356)
- `block_couple_u_apply` (mpi_momentum_blockLdU line 1362-1460)
- `pseudo_update_apply` (mpi_momentum_pseudoupdateUVW line 178-208)

**Step 3 — SourceTerm 자식 클래스**:
- `nob_buoyancy.cpp`: `Cmt·(Tc + a12pera11·Tc²·ΔT)·invRho` (PaScaL_TCS line 351-352)
- `boussinesq_buoyancy.cpp`: `Cmt·(Tc - Tg)·invRho` (MPM-STD Fortran 식)
- `bulk_forcing.cpp`: `-presgrad` (MPM-STD Fortran line 875)

**Step 4 — momentum_equation.cpp** (orchestrator):
- `predict(c, dt)` = 3-stage ADI (sweep_order 따라)
- `block_couple_V`, `block_couple_U`, `pseudo_update`

**Step 5 — main.hpp 빌더**:
- `make_rbc(...)` = NobProperties + NobBuoyancy(W)
- `make_channel(...)` = ConstantProperties + BulkForcing(U)

#### 검증
- PaScaL_TCS 와 1 step 직접 비교 (속도·온도 동일 초기조건 → 1 step → L∞ < 1e-12)
- Stage 단위 dump (`MPMSTD_DUMP_STAGES`) — 각 ADI sweep 직후 PaScaL_TCS 와 일치 확인
- block_couple 단위로 분리 검증
- `make_rbc` / `make_channel` 동일 orchestrator 사용 확인

---

### M4 — PressureEquation (3 주, 두 번째로 어려움)

#### 목표
- BC-aware FFT/DCT/DST 자동 선택 ([09](09_pressure_solver_design.md))
- 해석해 Poisson L∞ < 1e-10
- `project` 후 div(U) < 1e-12
- PaScaL_TCS 와 1 step δP L∞ < 1e-10

#### 파일

```
src/linear_solver/fft/
  main.hpp
  fft_planner.hpp                             ← 추상
  fftw_planner.hpp + .cpp                    ← FFTW3 (M4 작성)
  cufft_planner.hpp + .cpp                   ← stub (M5' 채움)
  eigenvalues.hpp                             ← MPMSTD_HD inline
  transpose_plan.hpp + .cpp                  ← MPI_Alltoallw + DDT
  kernels/
    kernels.hpp
    kernels_cpu.cpp                           ← M4 작성
    kernels_cuda.cu                           ← M5' 채움

src/equation/pressure/
  main.hpp
  pressure_equation.hpp + .cpp                ← orchestrator
  rhs_assembler.hpp + .cpp                    ← compute_rhs
  projection.hpp + .cpp                       ← project + dPhat 외삽
  kernels/
    kernels.hpp
    kernels_cpu.cpp
    kernels_cuda.cu                           ← M5'

apps/poisson_only_check/
  main.cpp

test/unit/
  test_eigenvalues.cpp
  test_fftw_planner.cpp
  test_transpose_plan.cpp

test/integration/
  test_poisson_analytic.cpp
```

#### 구현 순서 ([09](09_pressure_solver_design.md) §11 의 M4.1–M4.7 그대로)

**M4.1 (3일)** — `eigenvalues.hpp` (MPMSTD_HD inline):
- `eigvals_periodic(N, dx)`, `eigvals_neumann(N, dx)`, `eigvals_dirichlet(N, dx)`
- 단위 테스트: 알려진 모드 입력 후 변환 결과 검증

**M4.2 (4일)** — `fftw_planner.cpp`:
- `Problem` 의 P face BC 로 R2C/DCT/DST 자동 선택
- FFTW3 의 `r2c`, `FFTW_REDFT10` (DCT-II), `FFTW_RODFT10` (DST-II) plan 생성
- forward·backward identity 단위 테스트

**M4.3 (4일)** — `transpose_plan.cpp`:
- C↔I↔K 전치 (MPI_Alltoallw + derived datatype)
- PaScaL_TCS [module_solve_pressure.f90:558-586](../../PaScaL_TCS/src/module_solve_pressure.f90) 직역
- 역전치 = identity 단위 테스트

**M4.4 (3일)** — `rhs_assembler.cpp`:
- `(1/Δt)·∇·u* + NOB 보정`
- 비-발산장 입력에서 0 검증

**M4.5 (4일)** — `pressure_equation::solve()`:
- 전치 + FFT + 모드별 TDMA + 역FFT + 역전치
- 해석해 비교 EOC = 2

**M4.6 (2일)** — `mean_remove_if_singular`:
- 모든 면 Neumann/Periodic 케이스 자동 판정
- 평균 = 0 검증

**M4.7 (3일)** — `projection.cpp` + dPhat:
- `u^{n+1} = u* − Δt·(1/ρ)·∇δP`
- `dP_prev` 보관, 다음 step `dPhat = 2δP − dP_prev`
- div(U) < 1e-12 + PaScaL_TCS 와 1 step δP L∞ < 1e-10

---

### M5 — RBC 통합 (CPU 빌드) (2 주)

#### 목표
- `apps/rbc/main.cpp` 완성
- PaScaL_TCS golden (Ra=100, Pr=1, 512×128×256, 10 step) 과 L∞ < 1e-10

#### 파일

```
src/integrator/
  main.hpp
  time_stepper.hpp + .cpp
  cfl_controller.hpp + .cpp

src/utilities/
  main.hpp
  diagnostics.hpp + .cpp                      ← divergence check, monitor
  timer.hpp + .cpp                            ← nvtx range (CPU 빌드 no-op)

src/post/
  instant_io.hpp + .cpp
  probe.hpp + .cpp

src/stat/
  main.hpp
  statistics_accumulator.hpp + .cpp
  statistics_plugin.hpp + .cpp                ← physics::Plugin 상속
  stat_io.hpp + .cpp

src/physics/
  main.hpp
  plugin.hpp                                  ← 추상 + Phase enum

apps/rbc/
  main.cpp                                    ← 통합
  input.toml
  Makefile

test/regression/
  golden/pascal_tcs_Ra100_Pr1_10step.bin     ← PaScaL_TCS 로 생성
  compare.py
```

#### 구현 순서
1. PaScaL_TCS 로 golden 생성 (Ra=100, Pr=1, 512×128×256, 10 step → 5 개 binary)
2. `integrator/` 와 `utilities/`
3. `post/instant_io`, `post/probe`
4. `physics/plugin.hpp` + `stat/statistics_plugin`
5. `apps/rbc/main.cpp` — 보고서 [11](11_directory_structure_v2.md) §5 의 코드 구조
6. `test/regression/compare.py` 작성
7. 회귀 실행 → L∞ < 1e-10 확인

#### 검증
- 10 step 후 PaScaL_TCS golden 과 L∞(T, U, V, W) < 1e-10, L∞(P) < 1e-9
- 1, 4, 8, 16 랭크 결과 모두 일치
- restart write → read → 1 step 추가 → 동일 결과

---

### M5' — GPU 빌드 활성화 (3 주, 가장 큰 위험)

#### 목표
- 모든 kernels/ 의 `kernels_cuda.cu` 채우기
- `parallel/cuda/` 실제 구현
- `pascal_tdma_cuda_backend` 와 `cufft_planner` 실구현
- GPU 빌드 결과가 CPU 빌드 결과와 L∞ < 1e-8 일치

#### 파일 (수정)

```
src/parallel/cuda/
  cuda_runtime.cpp                            ← cudaSetDevice + device info
  cuda_memory.cpp                             ← cudaMalloc/Memcpy
  cuda_stream.cpp                             ← stream pool
  nvtx_range.hpp                              ← nvtxRangePushA/Pop

src/parallel/backend/
  cuda_backend.cpp                            ← 실구현 (stub 교체)

src/parallel/mpi/
  cuda_aware_mpi.cpp                          ← GPU-direct Isend/Irecv

src/linear_solver/tdma/
  pascal_tdma_cuda_backend.cpp                ← 실구현 (stub 교체)

src/linear_solver/fft/
  cufft_planner.cpp                           ← 실구현
  kernels/kernels_cuda.cu                     ← DCT R2C trick (MPM-STD Fortran 패턴)

src/equation/momentum/kernels/kernels_cuda.cu        ← M3 의 host 루프를 __global__ 로
src/equation/pressure/kernels/kernels_cuda.cu        ← M4 의 host 루프를 __global__ 로
src/equation/scalar/kernels/kernels_cuda.cu          ← M2 의 host 루프를 __global__ 로

src/grid/grid_device.cpp                      ← device-side metrics mirror

src/field/
  scalar_field.cpp                            ← device buffer alloc/copy 추가
  vector_field.cpp                            ← 동일
```

#### 구현 순서
1. **환경 검증** — `heat_gpu` 빌드·실행 (사용자 GPU 머신에서)
2. **`parallel/cuda/` 실구현** — cuda_runtime, cuda_memory, stream, nvtx
3. **Field 의 device buffer** — `to_device()`/`to_host()` 실구현
4. **`pascal_tdma_cuda_backend.cpp`** — PaScaL_TDMA_C 의 `pascal_tdma_many_cuda.cu` wrap
5. **`scalar/kernels/kernels_cuda.cu`** — M2 의 host 루프를 `__global__` 로 변환. CPU 빌드 결과와 1e-8 비교
6. **`momentum/kernels/kernels_cuda.cu`** — M3 의 host 루프 변환. 비교.
7. **`cufft_planner.cpp` + `fft/kernels/kernels_cuda.cu`** — cuFFT plan + DCT pre/post (MPM-STD Fortran 의 `cuda_pressure_DCT_f_pre/post` 직역)
8. **`pressure/kernels/kernels_cuda.cu`** — M4 의 host 루프 변환
9. **`cuda_aware_mpi.cpp`** — `MPI_Isend(device_ptr, ...)` (CUDA-aware MPI)
10. **`apps/rbc/main.cpp`** — `make BACKEND=cuda` 로 GPU 실행
11. GPU vs CPU 결과 비교

#### 검증 — 단계별 일치

각 .cu 채울 때마다 그 함수만 CPU vs GPU 비교:
```
[M2 ScalarEquation 의 ADI 3-stage]:
  CPU 빌드 결과 vs GPU 빌드 결과 → L∞ < 1e-12 (single step) → 1e-10 (1000 step)

[M3 MomentumEquation 의 predict + block_couple]:
  CPU vs GPU → L∞ < 1e-10 (single step)

[M4 PressureEquation 의 solve]:
  CPU (FFTW) vs GPU (cuFFT) → L∞ < 1e-8 (DCT trick + reduce order)
  → 가장 큰 차이가 예상되는 부분

[M5 전체 통합]:
  CPU 10 step vs GPU 10 step → L∞ < 1e-8 (NOB Ra=100, 512×128×256)
  CPU 결과는 PaScaL_TCS 와 1e-10 (M5 에서 검증됨)
  → transitively, GPU 와 PaScaL_TCS 는 1e-8 거리
```

#### 위험
- cuFFT 의 DCT trick 이 정확하지 않으면 압력 결과 발산. MPM-STD Fortran 의 `cuda_pressure_DCT_f_pre` 와 `_post` 를 line-by-line 직역.
- shared memory 도입은 *최적화* 단계 — M5' 의 1차 구현은 글로벌 메모리 직접 접근 (느리지만 정확)
- CUDA-aware MPI 가 클러스터에서 안 되면 host-경유 fallback

---

### M6 — Channel forced (1 주)

#### 목표
- `apps/channel/main.cpp` 작성
- `make_channel` 빌더 사용
- forced Channel Re_τ=180 표준 케이스 통계가 Kim-Moin-Moser 데이터와 일치

#### 파일

```
apps/channel/
  main.cpp                                    ← M5 RBC main 의 변형
  input.toml
  Makefile
```

#### 구현 순서
1. `apps/rbc/main.cpp` 복사 → `apps/channel/main.cpp` 로 시작
2. Thermal 관련 코드 제거 (T 비활성)
3. `boundary::make_problem_channel(cfg)` 사용
4. `equation::momentum::make_channel(cfg, ...)` 사용
5. BulkForcing 활성

#### 검증
- CPU 빌드 결과가 Kim-Moin-Moser (1987) Re_τ=180 데이터 ⟨U⟩(y) 와 5% 이내
- GPU 빌드 결과가 CPU 빌드와 1e-8 일치

---

### M7 — Plugin 슬롯 + 안정성 (1 주)

#### 목표
- LES, IBM 플러그인의 인터페이스 안정성 확보 (구현은 throw)

#### 파일

```
src/physics/les/
  main.hpp
  smagorinsky.hpp + .cpp                      ← throw stub
  les_properties.hpp + .cpp                   ← PropertyPolicy 데코레이터 시그니처만
  les_plugin.hpp + .cpp                       ← throw stub
  kernels/                                    ← 빈 파일

src/physics/ibm/
  main.hpp
  ibm_mask.hpp + .cpp                         ← throw stub
  cell_classification.hpp + .cpp
  ibm_plugin.hpp + .cpp                       ← throw stub
  kernels/
```

#### 검증
- Plugin 인터페이스 (Phase enum, call/setup/finalise) 가 향후 변경 없이 IBM/LES 추가 가능
- apps/rbc/main.cpp 에 `plugins.push_back(make_les_plugin(...))` 추가 시 컴파일 통과 (실행 시 throw)

---

## 5. 검증 전략 종합

### 5.1 검증 단계 매트릭스

| 단계 | 비교 | Tolerance |
|---|---|---|
| **단위 테스트** (stencil, eigvals, …) | 알려진 해석해 | 1e-14 |
| **CPU 빌드 vs PaScaL_TCS 골든** (M5) | 10 step | L∞ < 1e-10 |
| **CPU 빌드 vs MPM-STD Fortran channel** (M6) | Re_τ 통계 | 5% |
| **GPU 빌드 vs CPU 빌드** (M5') | 10 step | L∞ < 1e-8 |
| **GPU 빌드 vs PaScaL_TCS 골든** (transitively) | 10 step | L∞ < 1e-8 |

### 5.2 단계별 dump 디버깅

`MPMSTD_DUMP_STAGES=1` 빌드 플래그:
- thermal step 직후
- momentum predict (U/V/W) 직후
- block_couple_V/U 직후
- pseudo_update 직후
- pressure compute_rhs 직후
- pressure solve 직후
- pressure project 직후

PaScaL_TCS 측에도 동일 지점에 1 회용 패치 → **첫 발산 지점 localize**.

### 5.3 회귀 자동화

```bash
# CPU 회귀
make BACKEND=cpu all
mpirun -np 8 build/cpu/bin/rbc run/inputs/rbc_Ra100.toml
python test/regression/compare.py build/cpu/output/ test/regression/golden/

# GPU 회귀  
make BACKEND=cuda all
mpirun -np 4 build/cuda/bin/rbc run/inputs/rbc_Ra100.toml
python test/regression/compare.py build/cuda/output/ build/cpu/output/
```

CI 가 가능하면 GitHub Actions 등에 자동화.

---

## 6. reference 코드 매핑

| 우리 파일 | reference |
|---|---|
| `parallel/mpi/subdomain.cpp` | `PaScaL_TDMA_C/heat_gpu/mpi_subdomain.cpp` + `PaScaL_TCS/src/module_mpi_subdomain.f90` |
| `parallel/mpi/mpi_topology.cpp` | `heat_gpu/mpi_topology.cpp` + `PaScaL_TCS/src/module_mpi_topology.f90` |
| `parallel/cuda/cuda_memory.cpp` | `MPM-STD_main/src/domain/cuda_subdomain.f90` |
| `field/stencil/staggered.hpp` | `heat_gpu/stencil_coeffs.hpp` |
| `boundary/problem.cpp` | [05_BC_design.md](05_BC_design.md) |
| `linear_solver/tdma/pascal_tdma_cpu_backend.cpp` | `PaScaL_TDMA_C/src/pascal_tdma_many.cpp` |
| `linear_solver/tdma/pascal_tdma_cuda_backend.cpp` | `PaScaL_TDMA_C/src/pascal_tdma_many_cuda.cu` |
| `linear_solver/fft/fftw_planner.cpp` | `PaScaL_TCS/src/module_solve_pressure.f90` (FFT 부분) |
| `linear_solver/fft/cufft_planner.cpp` | `MPM-STD_main/src/core/core_pressure.f90` (cuFFT 부분) |
| `linear_solver/fft/kernels/kernels_cuda.cu` | `MPM-STD_main/src/core/core_pressure.f90` (DCT pre/post) |
| `equation/scalar/kernels/kernels_cpu.cpp` | `PaScaL_TCS/src/module_solve_thermal.f90` |
| `equation/scalar/kernels/kernels_cuda.cu` | `MPM-STD_main/src/core/core_energy.f90` + `heat_gpu/solve_theta.cu` |
| `equation/momentum/kernels/kernels_cpu.cpp` | `PaScaL_TCS/src/module_solve_momentum.f90` |
| `equation/momentum/kernels/kernels_cuda.cu` | `MPM-STD_main/src/core/core_momentum.f90` |
| `equation/pressure/kernels/kernels_cpu.cpp` | `PaScaL_TCS/src/module_solve_pressure.f90` |
| `equation/pressure/kernels/kernels_cuda.cu` | `MPM-STD_main/src/core/core_pressure.f90` |
| `apps/rbc/main.cpp` | `PaScaL_TCS/src/main.f90` 의 program main + `heat_gpu/main.cpp` 의 GPU 구조 |
| `apps/channel/main.cpp` | `MPM-STD_main/src/entrypoint.f90` |
| `Makefile.inc` (GPU 빌드) | `MPM-STD_main/Makefile.inc` |

---

## 7. 일정 + 위험

### 7.1 일정 (혼자 작업)

| M | 범위 | 기간 | 누적 |
|---|---|---|---|
| M0 | 인프라 + Field + 빌드 시스템 | 2 주 | 2 주 |
| M1 | BC + TDMA (CPU 백엔드) | 2 주 | 4 주 |
| M2 | ScalarEquation (CPU) | 2 주 | 6 주 |
| M3 | MomentumEquation (CPU) | 3 주 | 9 주 |
| M4 | PressureEquation (CPU + FFTW) | 3 주 | 12 주 |
| **M5** | **RBC 통합 + PaScaL_TCS 검증** | **2 주** | **14 주** |
| **M5'** | **GPU 빌드 활성화** | **3 주** | **17 주** |
| M6 | Channel | 1 주 | 18 주 |
| M7 | LES/IBM 슬롯 | 1 주 | 19 주 |
| **합계** | | **약 5 개월** | |

→ Option C 는 약 5 개월. Option A (CPU then port) 와 비슷하지만 *포팅 rewrite 위험* 이 없음.

### 7.2 위험 표

| # | 위험 | 발생 단계 | 완화 |
|---|---|---|---|
| 1 | PaScaL_TDMA_C 의 CUDA wrap 인터페이스 변경 | M1 | `pascal_tdma_many_cuda.hpp` 의 시그니처를 1차 점검 시 고정 |
| 2 | FFTW3 의 DCT 정규화 상수 오류 | M4 | forward-backward identity 단위 테스트 |
| 3 | cuFFT 의 DCT R2C trick 부정확 | M5' | MPM-STD Fortran 의 pre/post 코드 line-by-line 직역 |
| 4 | CUDA-aware MPI 가 클러스터에서 안 됨 | M5' | host-경유 fallback path 유지 |
| 5 | CPU vs GPU 결과 차이가 1e-8 보다 큼 | M5' | `-fno-fma -fno-fast-math` 강제. atomic 회피 |
| 6 | Wall ghost 의 zero-vs-antisymmetric 정책 혼동 | M1, M3 | [feedback_wall_bc_zero_ghost.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_wall_bc_zero_ghost.md) 참고 |
| 7 | cross-direction matrix 의 자기-미분 오염 | M3 | [feedback_bw_cross_direction.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_bw_cross_direction.md) 참고 |
| 8 | dPhat 의 첫 step (이전 없음) | M4 | `dP_prev = 0` 초기화 |
| 9 | Singular Neumann 평균 제거 누락 | M4 | `solve` 끝에 무조건 호출 |
| 10 | `make_rm` 안 한 채로 회귀 실행 | 항상 | [feedback_make_rm_before_sbatch.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_make_rm_before_sbatch.md): 회귀 전 build/, output/ 정리 |
| 11 | NVHPC SDK 버전 차이 | M5' | 24.1 로 고정. Makefile.inc 에 명시 |
| 12 | GPU 메모리 부족 (1024³ 등) | 향후 | M5' 1차는 512³ 만, 메모리 풀 도입은 M5' 후반 |

---

## 8. 즉시 실행 가능한 첫 작업 (오늘 시작 가능)

```bash
cd /shared/home/wel1come1234/workspace/MPM-STD\(C++\)

# 1. 디렉토리 골격 생성
mkdir -p src/{common,parallel/{mpi,backend,cuda},config,grid,field/stencil,\
              boundary,linear_solver/{tdma,fft/kernels},\
              equation/{momentum/kernels,pressure/kernels,scalar/kernels},\
              physics/{les/kernels,ibm/kernels},\
              integrator,post,stat,utilities}
mkdir -p apps/{rbc,channel,thermal_only_check,poisson_only_check}
mkdir -p test/{unit,integration,regression/golden}
mkdir -p run/inputs
mkdir -p build/{cpu/{obj,lib,bin},cuda/{obj,lib,bin}}

# 2. PaScaL_TDMA_C 빌드 확인
cd ../PaScaL_TDMA_C
make
ls build/cpu/lib/libpascal_tdma.a    # 있는지 확인

# 3. (선택) heat_gpu 빌드·실행 — GPU 환경 검증
cd heat_gpu
make
./run_1gpu.sh    # 1 GPU 동작 확인

# 4. M0 첫 commit — common/
cd ../../MPM-STD\(C++\)
# src/common/types.hpp, direction.hpp, macros.hpp 작성
# git commit -m "M0: common types and macros"
```

→ 이후 [§4](#4-마일스톤별-상세-계획) M0 의 파일 목록을 순서대로 작성.

---

## 9. 결론

**Option C 의 5 가지 핵심 가치**:

1. **알고리듬 검증 우선**: M5 에서 PaScaL_TCS 와 bit-exact (1e-10) 검증 후 GPU 진입
2. **포팅 = 빈 슬롯 채우기**: M5' 가 rewrite 가 아니라 `kernels_cuda.cu` 작성 + Backend 실구현
3. **두 빌드 동시 유지**: CPU 빌드가 GPU 검증의 다리 (transitive bit-exact)
4. **PaScaL_TDMA_C 의 CPU/CUDA 동시 활용**: 가장 큰 알고리듬 위험 (분산 TDMA) 이 라이브러리로 *해결됨*
5. **5 개월 안에 RBC + Channel 두 케이스 + 미래 확장 슬롯**: 일정 안에 production 가능

**처음 시작점**: §8 의 명령어. M0 의 `common/` 부터 시작.

본 문서가 작업 매뉴얼. 각 단계에서 막히면 §4 의 마일스톤 상세 + §6 의 reference 매핑 + 출처 보고서 (07, 09 등) 참조.
