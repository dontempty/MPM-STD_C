# Optimization Backlog

> 이 파일은 **나중에 최적화할 항목** 을 기록하는 백로그입니다.
> 현재 M0–M2 단계의 코드는 *기능 완성 + 정확성 검증* 에 집중. 성능 최적화는
> M5' (GPU 빌드 활성화) 시점부터 본격 적용.

각 항목은 다음 형식:
```
[priority] item — impact, effort, status
  files: which files would change
  notes: context, references
```

---

## 🔴 High Priority (M5' 시작 시 적용)

### O1. Laplacian + Convection 커널 fuse
- **Impact**: T 를 한 번만 읽음 → ScalarEquation step 의 메모리 트래픽 ~30% 감소
- **Effort**: Small (1 통합 커널 작성)
- **Status**: TODO (M5')
- **Files**:
  - `src/equation/scalar/kernels/kernels.hpp` — 새 함수 `compute_explicit_rhs(...)`
  - `src/equation/scalar/kernels/kernels_cpu.cpp` + `kernels_cuda.cu`
  - `src/equation/scalar/scalar_equation.cpp` — step() 의 두 호출이 하나로
- **Notes**: M2 에서 발견. `laplacian_explicit_rhs` + `add_convection_rhs` 가 동일 stencil 패턴이라 자연스럽게 합쳐짐.

### O2. Shared memory tiling for stencil 커널
- **Impact**: Laplacian / convection stencil 의 메모리 트래픽 ~7× 감소 (셀당 7-point read 가 1-point + halo로)
- **Effort**: Medium (CUDA 표준 패턴이지만 디테일 주의)
- **Status**: TODO (M5')
- **Files**: 모든 `*/kernels/kernels_cuda.cu`
- **Notes**: MPM-STD Fortran 의 `real, shared, dimension(...)` 패턴 직역. 블록 크기 (8×8×4 or 16×16×4) 튜닝 필요.

### O3. CUDA-aware MPI 검증 + 폴백 코드
- **Impact**: 멀티노드 halo exchange 2× ↑ (host-staging 회피)
- **Effort**: Small (런타임 체크 + fallback path)
- **Status**: TODO (M5' 환경 검증 단계)
- **Files**:
  - `src/parallel/mpi/cuda_aware_mpi.hpp/.cpp`
  - `src/parallel/mpi/subdomain.cpp` (exchange_halo)
- **Notes**: NVHPC SDK 의 OpenMPI 는 기본 CUDA-aware. 검증 + 안 되는 경우 host-staging 폴백.

---

## 🟡 Medium Priority (M5' 후반)

### O4. scatter_from_tdma + add_increment fuse (마지막 stage)
- **Impact**: ScalarEquation step 의 1 메모리 패스 절감
- **Effort**: Small
- **Status**: TODO
- **Files**:
  - `src/equation/scalar/kernels/kernels*.cpp`
  - `src/equation/scalar/scalar_equation.cpp`
- **Notes**: M2 의 마지막 stage `scatter_from_tdma(d, delta, D)` + `add_increment(T, delta)` → `T[i,j,k] += D_solution[tdma_idx]` 직접.

### O5. X/Y 축 build_adi_bands 의 strided 메모리 접근
- **Impact**: X/Y 축 ADI stage 의 대역폭 ~2× ↑ (현재 non-coalesced)
- **Effort**: Medium (shared memory tiled transpose)
- **Status**: TODO
- **Files**: `equation/*/kernels/kernels_cuda.cu`
- **Notes**: 32×32 shared mem tile 사용. CUDA 표준 전치 패턴. Z 축은 자연 layout 으로 OK, X/Y 만 필요.

### O6. CUDA Graphs (kernel launch overhead 감소)
- **Impact**: Latency-bound 케이스 (작은 격자, ~128³) 의 step time 5–10% ↑
- **Effort**: Medium
- **Status**: TODO
- **Files**: `src/integrator/time_stepper.{hpp,cpp}` (M5)
- **Notes**: step() 의 모든 kernel call 시퀀스를 한 번 record → 매 step `cudaGraphLaunch` 한 번. 5 μs/launch × 10 launches → 1 μs total.

### O7. NVTX ranges 모든 hot path 에 추가
- **Impact**: 프로파일링 가능 (성능 자체는 0)
- **Effort**: Small
- **Status**: TODO (M5' 와 동시)
- **Files**: `equation/*/kernels/*.cu`, `scalar_equation.cpp`, etc.
- **Notes**: `parallel/cuda/nvtx_range.hpp` 의 `MPMSTD_NVTX_RANGE("name")` 매크로 사용. CPU 빌드 no-op.

---

## 🟢 Low Priority (M6+ 또는 필요 시)

### O8. A/B/C TDMA bands 의 broadcast 인터페이스
- **Impact**: 메모리 ~3 × n_sys 절감 (512³ 에서 3 GB 절감!)
- **Effort**: **Very Large** (PaScaL_TDMA fork 또는 새 backend 필요)
- **Status**: 보류 (영향은 크지만 PaScaL_TDMA 라이브러리 변경 필요)
- **Files**: 별도 TDMA 백엔드
- **Notes**: A, B, C 는 (i, j) 무관 → `n_row` 크기로 충분. PaScaL_TDMA 의 `[n_row × n_sys]` 인터페이스가 강제. fork 또는 wrapper-level optimization 필요.

### O9. GPU 메모리 풀
- **Impact**: 임시 버퍼 alloc/dealloc 비용 0 으로 (`cudaMalloc` ~1 ms / call)
- **Effort**: Medium
- **Status**: TODO (M5' 후반)
- **Files**: `src/parallel/cuda/cuda_memory.{hpp,cpp}` 확장
- **Notes**: 현재는 ScalarEquation 등이 ctor 에서 alloc → 매 step 에 재사용. 하지만 momentum predictor 의 dU/dV/dW 같은 *임시 버퍼* 는 step 안에서 일시 사용 → 메모리 풀로 재사용.

### O10. Single precision 옵션 검증
- **Impact**: GPU 메모리 / 대역폭 2× ↑ (정확도는 미세 손실)
- **Effort**: Small (이미 `MPMSTD_PRECISION=single` 토글 있음)
- **Status**: 인터페이스만, M5'+ 에서 실측
- **Files**: 빌드 시스템에 이미 반영됨
- **Notes**: 정확성 검증 후 production-grade 격자에서 검토.

### O11. Async compute–IO overlap
- **Impact**: 큰 격자 + 빈번한 IO 시 IO 시간 숨김
- **Effort**: Large
- **Status**: 미정
- **Files**: `src/post/`, `src/parallel/cuda/cuda_stream.cpp`
- **Notes**: instant_io / stats 가 GPU stream 과 분리되도록.

### O12. Field 의 device-primary 메모리 모델
- **Impact**: Host buffer alloc 절감 (큰 격자에서 메모리 절약)
- **Effort**: Medium
- **Status**: 미정
- **Files**: `src/field/scalar_field.cpp`
- **Notes**: 현재 host buffer 가 항상 alloc. IO/restart 시점에만 alloc 하도록 변경. 단 코드 인터페이스 변경 필요.

---

## 📋 일반 M5' 시작 시 체크리스트

GPU 빌드 활성화 (M5') 시 같이 점검할 것:

- [ ] `parallel/cuda/cuda_runtime.cpp` 구현 (`cudaSetDevice` per-rank)
- [ ] `parallel/cuda/cuda_memory.cpp` 구현 (`cudaMalloc` 등)
- [ ] `parallel/backend/cuda_backend.cpp` 의 throw stub 제거
- [ ] `parallel/mpi/cuda_aware_mpi.cpp` 의 CUDA-aware 감지
- [ ] 모든 `kernels_cuda.cu` 의 throw stub → 실제 `__global__` 커널
- [ ] `pascal_tdma_cuda_backend.cpp` 의 throw stub 제거
- [ ] NVHPC SDK 빌드 환경 확인 (gpu=cc80 등)
- [ ] PaScaL_TDMA_C 의 CUDA 빌드 (USE_CUDA=1) 확인
- [ ] heat_gpu 빌드/실행 검증 (의존성 확인)
- [ ] CPU 빌드 결과 ↔ GPU 빌드 결과 일치 (1e-8 tol) 회귀

---

## 발견 일자별 기록

- **2026-05-26** (M2 Phase 3 완료 후): O1–O7 식별 (Laplacian/convection fuse, shared mem tiling, CUDA-aware MPI, scatter+add fuse, strided I/O, CUDA Graphs, NVTX)
- (다음 발견 시 여기에 추가)
