# `src/parallel/` — 처음 보는 사람을 위한 코드 워크스루

> **목적**: 이 코드를 한 번도 본 적 없는 사람이 **혼자 읽고 이해할 수 있도록** 배경 개념부터 시작해서 파일별·코드별로 풀어 설명한다.
> **읽기 전 알아야 할 것**: C++ 기초 (클래스, 네임스페이스, 포인터), MPI 가 뭔지 (대충 "여러 프로세스가 메시지 주고받는 라이브러리"), CUDA 가 뭔지 (대충 "GPU 에서 코드 돌리는 NVIDIA 도구") 정도면 충분.
> **연관 문서**: 큰 그림은 [01_m0_m1_understanding.md](01_m0_m1_understanding.md). 그쪽을 먼저 읽으면 이 문서가 훨씬 빠르게 들어옴.

---

## 0. 이 폴더가 풀려는 문제

CFD (전산유체역학) 코드는 보통 다음 둘 중 하나를 가짐:

- **A. 멀티프로세스 CPU 코드 (MPI)** — PaScaL_TCS 처럼
- **B. 단일 또는 멀티 GPU 코드 (CUDA)** — MPM-STD Fortran 처럼

이 프로젝트의 야망은 **둘 다 하나의 C++ 코드베이스로 표현하기**. 그러려면:

1. MPI 호출 (init, send/recv, finalize) 을 **C++ 객체 생애주기에 묶어서** 안전하게 관리해야 한다.
2. GPU 메모리 할당과 CPU 메모리 할당을 **같은 이름의 함수로 호출** 할 수 있어야 한다.
3. 빌드 시점에 "CPU 빌드냐 CUDA 빌드냐" 가 결정되면, **상위 코드는 그 차이를 모르고 동작** 해야 한다.

`src/parallel/` 폴더가 이 세 가지를 풀어준다. 위쪽 레이어 (격자, 필드, 방정식) 는 "랭크가 몇 개고, GPU 가 있는지" 를 신경 쓰지 않는다.

---

## 1. 폴더 지도

```
src/parallel/
├── main.hpp                         ← umbrella: 아래 3개를 한 번에 include
│
├── mpi/                             ← MPI 관련 (랭크, 통신, 도메인 분할)
│   ├── main.hpp
│   ├── mpi_context.hpp + .cpp       MPI_Init/Finalize 의 RAII 래퍼
│   ├── mpi_topology.hpp + .cpp      3D Cartesian 토폴로지 + 축별 sub-comm
│   ├── subdomain.hpp + .cpp         도메인 분할 + halo 교환
│   └── cuda_aware_mpi.hpp + .cpp    GPU 메모리 직접 통신 (M5' 자리)
│
├── backend/                         ← 메모리/동기화 추상 인터페이스
│   ├── main.hpp                     factory: make_default_backend()
│   ├── backend.hpp                  추상 클래스
│   ├── cpu_backend.hpp + .cpp       CPU 구현
│   └── cuda_backend.hpp + .cpp      CUDA 구현 (CPU 빌드면 throw)
│
└── cuda/                            ← CUDA 런타임 래퍼 (한 함수 두 얼굴)
    ├── main.hpp                     umbrella
    ├── error_check.hpp              CUDA_CHECK 가벼운 진입점
    ├── cuda_runtime.hpp + .cpp      initialize_device, sync, count
    ├── cuda_memory.hpp + .cpp       alloc/free/copy
    ├── cuda_stream.hpp + .cpp       Stream RAII + default_stream
    ├── cuda_launch.hpp              block/grid 사이즈 계산 (header-only)
    └── nvtx_range.hpp               NVTX 프로파일링 RAII (header-only)
```

세 하위 폴더는 역할이 명확히 분리되어 있다:

| 폴더 | 한 줄 요약 |
|---|---|
| `mpi/` | 프로세스간 통신 |
| `backend/` | "이 빌드의 메모리 정책" 추상 객체 |
| `cuda/` | CUDA API 자체의 얇은 래퍼 |

---

## 2. 본격 들어가기 전에 — 알아야 할 5가지 패턴

이 폴더의 코드 전체에 다음 5개 패턴이 반복된다. 미리 머릿속에 넣어두면 어떤 파일을 펼쳐도 빠르게 읽힌다.

### 패턴 1 — RAII (Resource Acquisition Is Initialization)

**문제**: C 의 MPI/CUDA API 는 손으로 짝을 맞춰야 한다. `MPI_Init` 했으면 `MPI_Finalize`, `cudaStreamCreate` 했으면 `cudaStreamDestroy`. 짝이 깨지면 누수 또는 행 (hang).

```cpp
// 위험한 C 스타일
MPI_Init(&argc, &argv);
if (something_failed) return 1;   // ← MPI_Finalize 안 부르고 종료. 문제.
MPI_Finalize();
```

**해법**: 자원의 생애주기를 C++ 객체의 생성/소멸에 묶는다.

```cpp
class MpiContext {
  MpiContext()  { MPI_Init(...); }
  ~MpiContext() { MPI_Finalize(); }
};

int main() {
  MpiContext mpi;          // ← 여기서 MPI_Init
  if (something_failed) return 1;   // ← 자동으로 MPI_Finalize 호출
}
```

예외든 조기 return 이든 소멸자는 반드시 불린다. **자원 누수가 컴파일 시간에 막힌다.**

이 폴더에서 RAII 객체: `MpiContext`, `MpiTopology`, `Subdomain`, `Stream`, `NvtxRange`.

### 패턴 2 — 한 함수, 두 얼굴 (`#ifdef` 분기)

**문제**: 같은 동작 (예: "메모리 할당") 을 CPU 빌드는 `malloc`, CUDA 빌드는 `cudaMalloc` 으로 해야 한다. 어떻게 호출자 코드가 한 줄로 끝나게 할까?

**해법**: 함수 이름은 같게, 내부 구현만 빌드 시점에 갈라지게.

```cpp
void* device_alloc(std::size_t bytes) {
#ifdef MPMSTD_BACKEND_CUDA
  void* p; cudaMalloc(&p, bytes); return p;
#else
  void* p; posix_memalign(&p, 64, bytes); return p;
#endif
}
```

호출자:
```cpp
void* buf = cuda_helpers::device_alloc(1024);  // 빌드를 모름
```

**왜 매크로가 아니라 함수?** 함수면 디버거에 보이고, 타입 체크 받고, 인라인 결정은 컴파일러에 맡길 수 있다. 매크로는 전처리 단계에서 사라져서 디버깅이 어려움.

### 패턴 3 — Umbrella 헤더 (`main.hpp`)

이 프로젝트의 모든 폴더에는 `main.hpp` 가 있고, **그 폴더의 public 헤더들을 한 번에 include** 한다.

```cpp
// src/parallel/mpi/main.hpp
#include "parallel/mpi/mpi_context.hpp"
#include "parallel/mpi/mpi_topology.hpp"
#include "parallel/mpi/subdomain.hpp"
#include "parallel/mpi/cuda_aware_mpi.hpp"
```

사용자는:
```cpp
#include "parallel/mpi/main.hpp"   // 한 줄로 MPI 관련 다 가져옴
```

**왜?**
- 사용자가 헤더 이름 하나하나 외울 필요 없음
- 폴더 내부에 파일 추가/삭제해도 사용자 코드 `#include` 안 바꿈
- "이 폴더가 제공하는 인터페이스의 목록" 역할

### 패턴 4 — Factory + Stub

**Factory** = 객체 생성 책임을 한 함수로 모은 것. 빌드별로 다른 객체를 반환:
```cpp
inline std::unique_ptr<Backend> make_default_backend() {
#ifdef MPMSTD_BACKEND_CUDA
  return std::make_unique<CudaBackend>();
#else
  return std::make_unique<CpuBackend>();
#endif
}
```

**Stub** = 빈 인터페이스만 미리 만들어두고, 실제 구현은 나중. 예:
```cpp
CudaBackend::CudaBackend() {
#ifndef MPMSTD_BACKEND_CUDA
  throw std::runtime_error("CudaBackend in CPU build");  // ← stub
#endif
  // 진짜 구현은 M5' 에서
}
```

→ **GPU-ready 골격은 처음부터, 구현은 단계적으로**. Option C 전략의 핵심 (13_final_plan).

### 패턴 5 — Namespace 로 가짜-GPU 레이어 만들기

`namespace mpmstd::parallel::cuda_helpers` 는 "GPU 가 있다면 어떻게 했을 일을 묶어둔 곳". CPU 빌드에서도 같은 이름으로 호출되지만 내부에서 `memcpy` / `malloc` 으로 흉내냄.

이 이름공간 덕에 **모든 메모리/스트림 호출이 한 곳을 통과**. 미래에 pinned memory 같은 게 도입되어도 그 한 곳만 바꾸면 끝.

---

## 3. `mpi/` 폴더 — 프로세스간 통신

이 폴더 4개 파일이 풀려는 문제: "여러 프로세스를 안전하게 시작하고, 3D 격자를 그들에게 나눠주고, 매 step 마다 이웃과 데이터 교환".

읽는 순서는 의존성을 따라가는 게 자연스러움: `MpiContext` → `MpiTopology` → `Subdomain` → `cuda_aware_mpi`.

### 3.1 `mpi_context.hpp + .cpp` — MPI 의 시작과 끝

#### 클래스: `MpiContext`

**역할**: `MPI_Init` 과 `MPI_Finalize` 의 RAII 래퍼. 프로세스당 정확히 1개 만들면 된다.

**왜 RAII?** 위 패턴 1 참조.

**저장하는 정보**:
- `world_rank_`, `world_size_` — 전체 MPI 세계에서 내 번호와 총 인원
- `node_rank_`, `node_size_`, `node_comm_` — **같은 물리 노드 안의** 내 번호와 인원
- `hostname_` — 로깅용

**노드-로컬 통신자가 왜 필요한가?** CUDA 빌드에서 "한 랭크 = 한 GPU" 매핑을 하려면 노드 안의 인덱스가 필요하다. 예를 들어 노드 2개, 노드당 GPU 4개 = 총 8 랭크인 환경에서:
- `world_rank = 5` 인 랭크는 노드 B 의 두 번째 GPU 를 써야 함
- `cudaSetDevice(world_rank)` = `cudaSetDevice(5)` → 노드에 GPU 4개뿐인데 5번을 찾음 → 에러
- `cudaSetDevice(node_rank)` = `cudaSetDevice(1)` → OK

→ `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` 가 같은 노드 랭크끼리 묶어준다.

**생성자가 하는 일** (cpp 파일을 보면서):
1. `MPI_Initialized` 로 이미 초기화됐는지 체크 (테스트 프레임워크가 이미 했을 수 있음)
2. 아니면 `MPI_Init_thread(MPI_THREAD_SINGLE)` 호출 — 단일 스레드 모드, 가장 빠름
3. world rank/size 저장
4. 노드-로컬 comm 만들고 거기서 rank/size 저장
5. `MPI_Get_processor_name` 로 호스트네임 저장

**소멸자**:
- `node_comm_` 을 `MPI_Comm_free` (이건 사용자가 만든 거니까 직접 해제)
- `MPI_COMM_WORLD` 는 MPI 가 관리 → free 안 함
- `MPI_Finalized` 체크 후 `MPI_Finalize`

**복사 금지**: `MPI_Finalize` 가 두 번 불리면 죽음. 컴파일 시간에 막아둠.

#### 사용 예시 (미래의 main)

```cpp
int main(int argc, char** argv) {
  MpiContext mpi(&argc, &argv);
  // ... 다른 모든 MPI 사용 객체가 mpi 를 참조로 받음 ...
}  // 자동으로 MPI_Finalize
```

### 3.2 `mpi_topology.hpp + .cpp` — 3D 격자 토폴로지

#### 구조체: `CartComm1D`

한 축의 1D sub-communicator 정보 한 묶음:
- `comm` — 그 축의 sub-communicator
- `rank` — 그 축 내의 내 순서
- `nprocs` — 그 축의 총 랭크 수
- `west_rank`, `east_rank` — 음/양 방향 이웃 (없으면 `MPI_PROC_NULL`)

**왜 한 묶음으로?** 이웃과의 통신을 할 때 매번 sub-comm + 이웃 랭크를 찾는 게 번거로움. 한 번 계산해서 캐싱.

#### 클래스: `MpiTopology`

**역할**: 3D 가상 좌표계를 만들고, 축별 1D sub-comm 3개를 미리 계산.

**저장하는 정보**:
- `cart_comm_` — 3D Cartesian 통신자 (`MPI_Cart_create` 결과)
- `dims_[3]` — 축별 랭크 수, 예: 2×2×2
- `periodic_[3]` — 축별 주기성
- `coords_[3]` — 이 랭크의 3D 좌표
- `axis_[3]` — X/Y/Z 축의 `CartComm1D` 셋

#### 생성자가 하는 일

1. **검증**: `dims[0]*dims[1]*dims[2] == world_size` 인지. 아니면 throw.
2. **`MPI_Cart_create`**: 3D 가상 토폴로지 만들기. `reorder=0` 으로 랭크 재배치 금지 (재현성 보장).
3. **`MPI_Cart_coords`**: 내 (i, j, k) 좌표 알아냄.
4. **`build_axis_comm_(0..2)`** 호출: 3개 sub-comm 생성.

#### `build_axis_comm_(axis)` 의 핵심 트릭

**`MPI_Cart_sub`** 으로 한 축만 살리고 나머지를 collapse:
```cpp
int remain_dims[3] = {0, 0, 0};
remain_dims[axis]  = 1;          // 이 축만 살림
MPI_Cart_sub(cart_comm_, remain_dims, &a.comm);
```

예: 2×2×2 토폴로지에서 X 축 sub-comm 만들면 → 8 랭크가 4개의 1D sub-comm 으로 묶임 (한 줄에 2 랭크씩, 4줄).

그 다음 **`MPI_Cart_shift`** 으로 양쪽 이웃 미리 계산:
```cpp
MPI_Cart_shift(a.comm, /*direction=*/0, /*disp=*/1, &a.west_rank, &a.east_rank);
```
- `direction=0` — 1D sub-comm 의 유일한 축
- `disp=1` — 1칸 거리 이웃
- 비주기 + 경계 랭크는 `MPI_PROC_NULL` 반환 → `MPI_Sendrecv` 가 자동으로 no-op

#### 소멸자

4개 comm (cart_comm + 3 sub-comm) 모두 `MPI_Comm_free`.

#### 미래 예고 (헤더 주석)

> "The 2D sub-communicators ... for the pressure-FFT transpose are introduced later (M4)."

→ M4 (압력 FFT) 에서는 X-Y 평면 sub-comm, Y-Z 평면 sub-comm 도 필요해진다 (PaScaL_TCS 의 `comm_x1n2` 같은 것). 지금은 1D 3개만.

### 3.3 `subdomain.hpp + .cpp` — 도메인 분할 + Halo 교환

이 폴더에서 가장 큰 파일이자 가장 자주 호출되는 객체. 매 시간 step 마다 `exchange_halo()` 가 6번 통신.

#### 보조 함수: `compute_para_range(n_global, nprocs, rank)`

**역할**: 전역 인덱스 `[0, n_global-1]` 을 nprocs 랭크에게 연속 블록으로 나눔. 나머지가 있으면 **앞 랭크가 1개씩 더 가져감**.

```cpp
const int base      = n_global / nprocs;
const int remainder = n_global % nprocs;
r.start = rank * base + (rank < remainder ? rank : remainder);
const int count = base + (rank < remainder ? 1 : 0);
```

예: `n_global=10, nprocs=3` → 랭크 0=[0..3], 1=[4..6], 2=[7..9] (4, 3, 3 개)

**왜 이 규칙?** **PaScaL_TDMA_C 의 `para_range.cpp` 와 같은 규칙**. 다르면 분산 TDMA 가 격자와 안 맞아서 솔버가 망함. 외부 라이브러리 호환성 강제.

#### 클래스: `Subdomain`

**역할 두 가지**:
1. 도메인 분할 정보 보관 (내 범위, 크기, 오프셋)
2. Halo 교환 (`exchange_halo`)

#### 저장 값과 용어

| 용어 | 의미 |
|---|---|
| `n_global[d]` | 전역 내부 셀 개수 (축 d) |
| `n_interior[d]` | **이 랭크가** 가진 내부 셀 개수 |
| `n_total[d]` | `n_interior[d] + 2*kHaloWidth` = 실제 배열 stride |
| `offset[d]` | 이 랭크 첫 내부 셀의 전역 인덱스 |
| `kHaloWidth` | ghost 셀 폭 = 1 (현재) |
| **slab** | 한 면 (한 평면) — halo 교환 단위 |

`kHaloWidth` 는 [src/common/macros.hpp](../../src/common/macros.hpp) 에 `inline constexpr int kHaloWidth = 1;` 로 정의.

#### 메모리 레이아웃 (중요)

C row-major:
```
stride_z = 1                       (k 가 가장 빠른 축)
stride_y = n_total[2]
stride_x = n_total[1] * n_total[2]
linear   = (i * n_total[1] + j) * n_total[2] + k
```

**Fortran (column-major) 와 반대**. PaScaL_TCS 의 `A(i,j,k)` 를 포팅할 때 머릿속에서 인덱스 순서를 뒤집어야 한다.

#### `build_datatypes_()` — MPI derived datatype 미리 만들기

**왜 derived datatype 이 필요한가?**

C row-major 에서 한 면 (slab) 의 메모리 패턴:
- **z-축 면 (xy 평면 한 장)**: 연속 메모리. count=n_total[0]*n_total[1] 로 그냥 전송 가능.
- **y-축 면 (xz 평면 한 장)**: z 연속 + x 점프 = strided.
- **x-축 면 (yz 평면 한 장)**: 더 심한 strided.

→ MPI 에게 "이런 패턴" 을 datatype 으로 등록해두면 매 step 마다 패턴 재계산 없이 재사용. `MPI_Type_create_subarray`.

축 3개 × 면 2개 = 6개 datatype 미리 만들고 `commit`.

#### 4개 slab offset

축마다 4개 평면 위치 기억:

| slab role | 인덱스 | 의미 |
|---|---|---|
| 0: send→minus | `kHaloWidth` | 첫 내부 평면 — 왼쪽 이웃에게 보낼 데이터 |
| 1: recv←plus | `n_total - kHaloWidth` | 위쪽 halo — 오른쪽 이웃에게서 받을 자리 |
| 2: send→plus | `n_total - 2*kHaloWidth` | 마지막 내부 평면 — 오른쪽에 보낼 데이터 |
| 3: recv←minus | `0` | 아래쪽 halo — 왼쪽에서 받을 자리 |

한 축 단면 (kHaloWidth=1):
```
인덱스:        0    1    2   ...  N-2  N-1
              [H] [int][int] ... [int] [H]
               ↑    ↑              ↑    ↑
        recv←minus send→minus  send→plus recv←plus
            (3)      (0)         (2)     (1)
```

#### `exchange_halo(data)` — 매 step 호출되는 메인 API

6면 ghost 교환. 축마다 2번의 `MPI_Sendrecv`:

```cpp
for (int axis = 0; axis < 3; ++axis) {
  // 첫째: send→minus, recv←plus
  MPI_Sendrecv(send_to_minus_ptr,   datatype, west_rank, ...,
                recv_from_plus_ptr, datatype, east_rank, ...);

  // 둘째: send→plus, recv←minus
  MPI_Sendrecv(send_to_plus_ptr,    datatype, east_rank, ...,
                recv_from_minus_ptr, datatype, west_rank, ...);
}
```

**왜 데드락이 안 걸리나?** `MPI_Sendrecv` 가 send/recv 를 atomic 하게 처리. MPI 구현이 알아서 ordering 잡음.

**`MPI_PROC_NULL` 의 역할**: 비주기 경계의 끝 랭크는 이웃이 `MPI_PROC_NULL`. `Sendrecv` 가 자동으로 no-op → boundary 처리 코드 분기 안 해도 됨.

**현재 한계** (헤더 주석):
> "we exchange faces only (no edges/corners). If a stencil later needs corners, we extend this routine."

→ 5점/7점 stencil 은 면만 있으면 충분. M3 에서 cross-derivative (viscous) 가 corner 필요해지면 확장 예정. **현재 결정이 깨질 수 있는 지점을 코드 안에 적어둔 것**.

### 3.4 `cuda_aware_mpi.hpp + .cpp` — 거의 빈 stub

M0 시점에는 사실상 빈 파일이지만 의도적으로 존재한다.

**헤더** ([cuda_aware_mpi.hpp](../../src/parallel/mpi/cuda_aware_mpi.hpp)) 에는 단 하나의 함수:

```cpp
constexpr bool is_cuda_aware_mpi_enabled() {
#if defined(MPMSTD_BACKEND_CUDA) && !defined(MPMSTD_NO_CUDA_AWARE_MPI)
  return true;
#else
  return false;
#endif
}
```

`.cpp` 는 빈 namespace.

**왜 미리 존재하나?**
- M5' 에서 host-staging fallback (클러스터가 CUDA-aware MPI 를 지원 안 할 때) 들어갈 자리
- 빌드 시스템이 .cpp 파일을 기대 — 빈 namespace 라도 있어야 .o 생성
- 사용자 코드가 `if constexpr (is_cuda_aware_mpi_enabled())` 로 미리 분기 가능

→ **인터페이스 stub** 의 전형.

---

## 4. `backend/` 폴더 — 메모리/동기화 추상

이 폴더 4개 파일이 풀려는 문제: "위 레이어가 메모리 할당이나 디바이스 동기화를 할 때, CPU 빌드인지 GPU 빌드인지 모르게 하기".

### 4.1 `backend.hpp` — 추상 인터페이스

#### 클래스: `Backend` (추상)

가상 함수 4개:
- `alloc(bytes)` — 메모리 할당
- `free(ptr)` — 해제
- `synchronize()` — 디바이스/스트림 동기화
- `name()` — 로깅용 식별자

#### 왜 이렇게 작은가?

이 인터페이스에 **핫루프 커널 (predict, project, FFT) 은 의도적으로 들어있지 않다**. 헤더 주석:

> "Hot-loop kernels are NOT methods on this interface; instead, equation modules call into kernel functions whose CPU/CUDA implementations are chosen at build time."

이유: 핫루프를 가상 메서드로 만들면 매 step 마다 vtable 분기 → GPU 인라이닝 깨짐 → 성능 폭망. 그래서 핫루프는 **빌드 시간 분기** (`kernels_cpu.cpp` vs `kernels_cuda.cu`) 로 처리. Backend 는 자주 안 불리는 것만 가진다.

### 4.2 `cpu_backend.hpp + .cpp` — CPU 구현

```cpp
void* CpuBackend::alloc(std::size_t bytes) {
  return cuda_helpers::device_alloc(bytes);
}
```

`cuda_helpers::device_alloc` 은 CPU 빌드에서 `posix_memalign` (64-byte 정렬 호스트 메모리) 으로 fallback. 즉 `CpuBackend::alloc` 도 결국 정렬 malloc.

**왜 직접 `malloc` 안 쓰고 cuda_helpers 를 통하나?** 메모리 할당 진입점을 **한 곳** 으로 모으기. 미래에 pinned memory, NUMA-aware 등 도입할 때 그 한 곳만 바꾸면 됨.

`synchronize()` 는 CPU 빌드에서 no-op.

### 4.3 `cuda_backend.hpp + .cpp` — CUDA 구현 (stub + 실구현)

CPU 빌드에서는:
```cpp
CudaBackend::CudaBackend() {
#ifndef MPMSTD_BACKEND_CUDA
  throw std::runtime_error("CudaBackend in CPU build, use CpuBackend.");
#endif
}
```

→ **CPU 빌드에서도 컴파일은 된다** (헤더가 존재). 다만 실행 시점에 `new CudaBackend()` 하면 throw. 이것이 Option C 의 "GPU-ready 골격" 의 의미.

CUDA 빌드에서는 `cudaMalloc/Free/DeviceSynchronize` 로 동작.

### 4.4 `main.hpp` — Factory

```cpp
inline std::unique_ptr<Backend> make_default_backend() {
#ifdef MPMSTD_BACKEND_CUDA
  return std::make_unique<CudaBackend>();
#else
  return std::make_unique<CpuBackend>();
#endif
}
```

미래의 `apps/rbc/main.cpp` 첫머리:
```cpp
auto backend = parallel::make_default_backend();
```
→ 빌드에 맞는 객체가 알아서 나온다. 분기 코드를 사용자가 안 쓴다.

---

## 5. `cuda/` 폴더 — CUDA API 래퍼

이 폴더의 모든 함수/클래스가 `namespace mpmstd::parallel::cuda_helpers` 에 들어있다. "GPU 가 있다면 어떻게 했을 일들의 가짜-GPU 레이어".

### 5.1 `error_check.hpp` — 거의 빈 헤더

```cpp
#include "common/macros.hpp"
```

이게 전부. **왜 존재?** `CUDA_CHECK` 매크로만 가져오고 싶을 때 가벼운 진입점. `common/main.hpp` 다 가져오기 부담스러울 때 사용.

`CUDA_CHECK` 매크로는 `common/macros.hpp` 에서:
```cpp
#define CUDA_CHECK(call) do {                       \
  cudaError_t err = (call);                          \
  if (err != cudaSuccess) {                          \
    std::fprintf(stderr, "CUDA error: %s\n", ...);   \
    std::abort();                                    \
  }                                                  \
} while(0)
```
CPU 빌드에서는 `((void)0)` 로 정의 — 영(0) 비용.

### 5.2 `cuda_runtime.hpp + .cpp` — 디바이스 관리

3개 자유 함수:

#### `initialize_device(rank_in_node)`

CUDA 빌드:
```cpp
int n_dev = 0;
cudaGetDeviceCount(&n_dev);
const int dev = rank_in_node % n_dev;
cudaSetDevice(dev);
return dev;
```

`% n_dev` 는 **round-robin fallback**. 한 노드에 8 랭크 / 4 GPU 면 0→0, 1→1, 2→2, 3→3, 4→0, ... 으로 매핑. 보통은 랭크 수 = GPU 수 권장.

CPU 빌드: `-1` 반환 (no-op).

미래의 main:
```cpp
MpiContext mpi(&argc, &argv);
int dev = cuda_helpers::initialize_device(mpi.node_rank());  // ← node_rank 가 핵심
```

#### `synchronize_device()`

CUDA: `cudaDeviceSynchronize()`. CPU: no-op.

#### `device_count()`

CUDA: `cudaGetDeviceCount`. CPU: 0.

### 5.3 `cuda_memory.hpp + .cpp` — 메모리 (한 함수 두 얼굴의 전형)

6개 자유 함수:

| 함수 | CUDA 빌드 | CPU 빌드 |
|---|---|---|
| `device_alloc(bytes)` | `cudaMalloc` | `posix_memalign(64)` |
| `device_free(ptr)` | `cudaFree` | `std::free` |
| `copy_host_to_device` | `cudaMemcpy(H2D)` | `std::memcpy` |
| `copy_device_to_host` | `cudaMemcpy(D2H)` | `std::memcpy` |
| `copy_device_to_device` | `cudaMemcpy(D2D)` | `std::memcpy` |
| `copy_*_async(stream)` | `cudaMemcpyAsync` | `std::memcpy` (stream 무시) |

호출자는 분기 없이 같은 이름. **이 패턴이 backend/ 와 위쪽 코드가 빌드를 모르게 만드는 비결.**

`device_alloc` 의 CPU fallback 이 `memset(0)` 까지 하는 이유: `cudaMalloc` 의 동작과 맞추려는 게 아니라, 단순히 valgrind 등의 uninit-read 경고 회피용. 디버깅 편의.

### 5.4 `cuda_stream.hpp + .cpp` — Stream RAII

#### 클래스: `Stream`

저장 값: `void* native_` — opaque 핸들. CUDA 빌드면 `cudaStream_t`, CPU 빌드면 `nullptr`.

**왜 opaque (void\*)?** 헤더에 `<cuda_runtime.h>` 를 안 끌어오기 위함. 그 헤더는 NVCC 전용 매크로/구조체가 많아서 일반 컴파일러로 빌드할 때 노이즈가 큼. `void*` 로 감춰두면 헤더는 깔끔하고, `.cpp` 안에서만 캐스팅.

생성자/소멸자:
- CUDA 빌드: `cudaStreamCreate` / `cudaStreamDestroy`
- CPU 빌드: 둘 다 빈 함수

복사 금지: stream 소유권은 unique.

`synchronize()`: `cudaStreamSynchronize` 또는 no-op.

#### 함수: `default_stream()`

```cpp
Stream& default_stream() {
  static Stream s;
  return s;
}
```

**Meyers singleton 패턴**. 함수 안의 static 으로 한 번만 만들어지고, 프로세스 끝까지 살아있음. C++11 이후 thread-safe 보장. CUDA 의 "legacy default stream" (id=0) 흉내.

#### 미래 확장 예고

헤더 주석:
> "Future expansion (M5'+): wrap as a class with a pool of cudaStream_t objects and allow round-robin assignment to overlap compute / halo exchange / IO."

→ M5'+ 에서 stream pool 로 확장해서 컴퓨트와 halo exchange 를 서로 다른 stream 에 두면 overlap 가능. 지금은 single stream.

### 5.5 `cuda_launch.hpp` — block/grid 사이즈 계산 (header-only)

3개 구조체 + 1개 inline 함수.

```cpp
struct LaunchExtent3D { int n1, n2, n3; };
struct BlockShape3D   { int x = 8, y = 8, z = 4; };   // 256 threads/block
struct GridShape3D    { int x, y, z; };

inline GridShape3D compute_grid_shape(LaunchExtent3D ext, BlockShape3D block) {
  GridShape3D g;
  g.x = (ext.n1 + block.x - 1) / block.x;     // ceiling division
  g.y = (ext.n2 + block.y - 1) / block.y;
  g.z = (ext.n3 + block.z - 1) / block.z;
  return g;
}
```

**`(a + b - 1) / b` = ceiling division**. N 셀을 B 개씩 처리하려면 `(N + B - 1) / B` 개 블록.

**왜 block 기본값이 8×8×4?** 일반적 sweet spot. row-major 에서 z 가 inner 라 z 가 작아도 한 block 당 데이터 청크가 큼.

CPU 빌드에서도 컴파일/호출 가능 — 결과는 안 쓰일 뿐.

미래 (M5' kernels_cuda.cu):
```cpp
LaunchExtent3D ext{n1, n2, n3};
BlockShape3D block;
GridShape3D grid = compute_grid_shape(ext, block);
predict_kernel<<<dim3(grid.x, grid.y, grid.z),
                 dim3(block.x, block.y, block.z)>>>(...);
```

### 5.6 `nvtx_range.hpp` — NVTX 프로파일링 RAII (header-only)

NVIDIA Nsight 프로파일러용 범위 마커.

#### 클래스: `NvtxRange`

생성자:
- CUDA: `nvtxRangePushA(name)` — Nsight 타임라인에 "이 범위 시작"
- CPU: 아무것도 안 함

소멸자: `nvtxRangePop()`.

#### 매크로: `MPMSTD_NVTX_RANGE(name)`

```cpp
#define MPMSTD_NVTX_RANGE(name) \
  NvtxRange _mpmstd_nvtx_range_##__LINE__(name)
```

**트릭**: `##__LINE__` 으로 변수 이름에 라인 번호를 붙여 한 함수 안 여러 범위가 충돌 안 나게.

사용:
```cpp
void predict() {
  MPMSTD_NVTX_RANGE("predict");
  // ... 코드 ...
}  // 스코프 끝나면 자동 pop
```

CPU 빌드에서는 **영(0) 오버헤드** — 빈 클래스, 인라인됨.

---

## 6. 의존성 정리

이 폴더 안에서 누가 누구를 부르는지:

```
Backend (추상)
   ↑
   ├── CpuBackend ─────┐
   │                   ├──→ cuda_helpers::device_alloc/free/synchronize
   └── CudaBackend ────┘                   (cuda_memory + cuda_runtime)


MpiContext
   ↓
MpiTopology (참조)
   ↓
Subdomain (참조)
   └── 내부에서 MPI 직접 호출 (Sendrecv, Type_create_subarray)


cuda_aware_mpi  ←──  Subdomain (M5' 이후 사용)


cuda_helpers (자유 함수 집합)
   ├── cuda_runtime
   ├── cuda_memory
   ├── cuda_stream
   └── nvtx_range / cuda_launch (header-only)
```

**위쪽 레이어에서 보는 진입점**:
- 메모리 → `backend.alloc(bytes)`  (안에서 cuda_helpers 호출)
- halo 교환 → `subdomain.exchange_halo(ptr)`
- 디바이스 동기화 → `backend.synchronize()`
- 프로파일링 마크 → `MPMSTD_NVTX_RANGE("name")` 매크로

---

## 7. 파일 한눈 표

| 파일 | 종류 | 역할 | 빌드 분기 |
|---|---|---|---|
| `parallel/main.hpp` | umbrella | 하위 3 main include | – |
| `mpi/main.hpp` | umbrella | mpi 4 헤더 include | – |
| `mpi/mpi_context` | RAII 클래스 | MPI_Init/Finalize + node comm | – |
| `mpi/mpi_topology` | RAII 클래스 | 3D Cart + 3 sub-comm | – |
| `mpi/subdomain` | RAII 클래스 | 분할 + 6 datatype + halo 교환 | – |
| `mpi/cuda_aware_mpi` | stub | M5' 의 host-staging 자리 | constexpr |
| `backend/main.hpp` | factory | `make_default_backend()` | #ifdef |
| `backend/backend.hpp` | abstract | alloc/free/sync 인터페이스 | – |
| `backend/cpu_backend` | impl | cuda_helpers 위임 | – |
| `backend/cuda_backend` | impl | CPU 빌드는 throw | #ifdef |
| `cuda/main.hpp` | umbrella | cuda 6 헤더 include | – |
| `cuda/error_check.hpp` | re-export | CUDA_CHECK 가벼운 진입점 | – |
| `cuda/cuda_runtime` | 자유 함수 | initialize_device, sync, count | #ifdef |
| `cuda/cuda_memory` | 자유 함수 | alloc/free/copy 6개 | #ifdef |
| `cuda/cuda_stream` | RAII 클래스 | Stream + default_stream | #ifdef |
| `cuda/cuda_launch.hpp` | helpers | grid/block sizing | – |
| `cuda/nvtx_range.hpp` | RAII 클래스 | NVTX 범위 마커 + 매크로 | #ifdef |

---

## 8. 이 폴더를 한 줄로 요약

**"이 빌드에 몇 개의 랭크가 있고, GPU 가 있는지 없는지의 모든 디테일을 위쪽 코드에서 가리는 추상화 레이어. RAII 와 `#ifdef` 와 umbrella 패턴이 반복된다. M0 에서 골격 + CPU 실구현, M5' 에서 GPU 살아남."**

---

## 9. 처음 보는 사람에게 추천하는 읽기 순서

1. **이 문서 §2 (5가지 패턴) 먼저 읽기** — 패턴이 머리에 들어와야 코드가 빨리 읽힘.
2. **`common/macros.hpp` 열어보기** — `MPMSTD_HD`, `CUDA_CHECK`, `kHaloWidth` 등 빌드 매크로 분기를 어디서 정의하는지 확인.
3. **`mpi_context.cpp`** — 가장 작고 자기완결적. RAII 가 뭔지 체감.
4. **`cuda_memory.cpp`** — "한 함수 두 얼굴" 의 가장 깔끔한 예.
5. **`backend.hpp` + `cpu_backend.cpp` + `cuda_backend.cpp`** — 추상화의 작은 예.
6. **`mpi_topology.cpp`** — `MPI_Cart_create`, `MPI_Cart_sub`, `MPI_Cart_shift` 의 콤보.
7. **`subdomain.cpp`** — 가장 큰 파일. `MPI_Type_create_subarray` 와 `exchange_halo`. 시간을 좀 들여서 읽기.
8. **나머지 (`cuda_stream`, `cuda_launch`, `nvtx_range`)** — 작은 utility 들. 빠르게 훑기.

---

## 10. M2 로 가기 전 체크리스트

이 폴더 읽기를 마쳤다면 다음 질문에 답할 수 있어야 한다:

- [ ] `MpiContext` 의 소멸자는 무엇을 보장하는가? 왜 RAII 인가?
- [ ] `node_rank` 는 `world_rank` 와 무엇이 다른가? 왜 둘 다 필요한가?
- [ ] `MpiTopology` 의 `axis_[3]` 가 가진 정보 5개는 무엇인가?
- [ ] `Subdomain` 이 매 step 부르는 `exchange_halo` 의 비용은 몇 번의 MPI 호출인가?
- [ ] `kHaloWidth` 는 어디 정의되어 있고, 값은 몇인가? 왜 그 값인가?
- [ ] row-major 인덱싱에서 가장 빠른 축은? Fortran 과 어떻게 다른가?
- [ ] `Backend` 가 "작게" 유지되는 이유는?
- [ ] `cuda_helpers::device_alloc` 은 CPU 빌드에서 무엇을 호출하는가?
- [ ] `CudaBackend` 의 생성자가 CPU 빌드에서 어떻게 동작하는가? 왜?
- [ ] `make_default_backend()` 는 무엇을 반환하는가? 어디서 분기되는가?
- [ ] `Stream` 의 `native_` 가 왜 `void*` 인가?
- [ ] `MPMSTD_NVTX_RANGE` 매크로의 `##__LINE__` 트릭은 무엇을 해결하는가?

모두 답할 수 있으면 M2 (ScalarEquation) 으로 넘어가도 좋다.
