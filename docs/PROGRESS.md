# MPM-STD_C 리팩토링 진행 기록 (rev.2 실행 로그)

마스터 설계: [REFACTOR_PLAN.md](REFACTOR_PLAN.md) · 브랜치: `refactor/rev2` (main = 검증된 Re_tau=180 채널, 회귀 기준선) · 방식: **병행 신축** (신트리 `mpmstd/`를 기존 `src/` 옆에 새로 짓고, 파리티 도달 시 교체).

> 이 문서는 마일스톤 완료마다 갱신한다. 상세 설계는 REFACTOR_PLAN.md, 동결 API는 `tests/spike/README.md`.

## 마일스톤 현황

| 단계 | 내용 | 상태 | 커밋 |
|---|---|---|---|
| **P-0.5** | 인터페이스 스파이크 (core/solve API 동결 + halo/banded PoC + 듀얼빌드) | ✅ 완료 | `9e6df3b` |
| **P0** | 6계층 트리 전체 스켈레톤 + 듀얼빌드 lib/apps/tests | ✅ 완료 | `4a178fc` |
| P1 | CPU 자유함수 본문 포팅 + Re_tau=180 회귀 | ✅ 완료 | 4321294…596169d |
| P2 | channel main 가독화 (§7 레시피) | ✅ 완료 | — |
| P3 / P3b | solve 일반화·BC-agnostic / cavity vs Ghia | ⏳ | — |
| P4 / P5 | GPU core+멀티GPU 통신 / GPU 커널 | ⏳ | — |
| P6 / P7 | DHVC(Fig 7) / NOB RBC(Fig 9) | ⏳ | — |
| **P8** | **multi-GPU 최종 검증 (Fig 7 + Fig 9)** | ⏳ | — |
| P9 / P10 | LES / IBM | ⏳ | — |

---

## P-0.5 — 인터페이스 스파이크 ✅ (`9e6df3b`)

**동결 API** (전문: `tests/spike/README.md`)
- `core`: `CpuField`/`GpuField` (분리 타입, virtual Backend 폐기), `Bands[n_row×n_sys]`, `ScalarSystem`/`MomentumSystem`/`PressureSystem`, `exchange_halo_{cpu,gpu}(Field&, const Subdomain&)`, `bind_gpu_to_local_rank_{cpu,gpu}` (= `cudaSetDevice(node_rank)`)
- `solve`: `solve_banded_cpu` (스파이크는 Thomas, P1에서 동일 시그니처로 PaScaL_TDMA 교체)

**결정 (스파이크가 푼 모호점)**
- halo 인자는 `MpiTopology`가 아니라 **`Subdomain`** (datatypes가 거기 있음) — 계획서 §5 불일치#1 해소
- rank↔GPU는 `MpiContext::node_rank()`(MPI_COMM_TYPE_SHARED)가 이미 존재 → `cudaSetDevice`만 추가 (C2, 1 rank=1 GPU)
- `CpuField`는 shape를 값복사로 보유한 "dumb" 데이터, 통신은 호출마다 Subdomain 전달
- **듀얼빌드 = 폴더(cpu/ vs gpu/) 기준**, 결합빌드는 단일 NVHPC 컴파일러 → OpenMPI/NVHPC MPI ABI 혼합 회피. `make cpu`는 CUDA-free. `kHaloWidth=1`.

**검증**: 헤더 7개 standalone 컴파일 · `halo_poc` np=2/4 PASS · `banded_poc` 1D Poisson max|err|=4.9e-5 · baseline 무손상 · **GPU(gpu01, 2×A100): 듀얼빌드 LINK + device-to-device CUDA-aware MPI halo 실행, 1 rank=1 GPU** (P4 선제 de-risk).

---

## P0 — 6계층 스켈레톤 + 듀얼빌드 ✅ (`4a178fc`)

**구조** (`mpmstd/`, 각 계층 `cpu/`+`gpu/`, §5 자유함수 + no-op `// TODO(Pn)` 스텁)
- `core` (+ `grid`/`boundary`/`config` = 기존 host-single 타입 재export)
- `solve` (banded + poisson 스텁)
- `equation` (scalar / momentum[const+var 점성, solve_momentum이 U,V,W+커플링 한 번에 M2] / pressure)
- `physics` (buoyancy/properties/forcing/ibm/les + 파라미터 타입)
- `post` (statistics/diagnostics/io) · `driver` (cfl/monitor/restart)
- **io/monitor/restart는 host 전용(cpu only)** → gpu `.cu` 14개 vs cpu `.cpp` 17개
- 앱: `mpmstd/apps/{channel,cavity,dhvc,rbc}` 스켈레톤 · 테스트: 레포 `tests/{unit/test_smoke_cpu, integration, regression}`

**빌드**: `mpmstd/Makefile` — 폴더 기반 듀얼빌드, `build/$(BACKEND)/mpmstd/`로 격리(기존 `src/` 빌드와 무충돌)

**⚠ GPU 빌드 핵심 교훈**: cuda 빌드는 **모든 TU(host `cpu/*.cpp` 포함)를 `nvc++ -cuda`(=ALL_CUFLAGS)로** 컴파일해야 함. ① nvc++가 Makefile.inc의 GCC전용 `-Wno-cast-function-type`(WARN)·`-MP`(DEPFLAGS) 거부, ② `macros.hpp`가 `-DMPMSTD_BACKEND_CUDA`에서 `<cuda_runtime.h>` include → 경로는 `-cuda`에서만 추가됨. 공유 `Makefile.inc`는 안 건드려 CPU baseline 보존. (기존 `src/`의 cuda 경로는 한 번도 검증된 적 없어 이 결함이 잠복)

**검증**
- `make cpu`: 17 lib TU + 4앱 + smoke 빌드/실행; cpu lib에 `_cpu`만, `_gpu` **0** (폴더 선택 검증)
- `make BACKEND=cuda` (job 29049, gpu01): 17 cpu `.cpp` + 14 gpu `.cu` → **하나의 libmpmstd에 `_cpu`/`_gpu` 양쪽 공존** (`exchange_halo_cpu`+`exchange_halo_gpu`+`solve_momentum_gpu`); 앱/테스트 링크; A100 실행

---

## P1 — CPU 자유함수 본문 포팅 (진행중)

검증된 기존 `src/` 수치를 신구조 자유함수로 충실히 이식. 전략: 순수 커널(raw-pointer 자유함수)은 mpmstd로 **복사**(자기완결), 분산 TDMA는 `TdmaRegistry` **재사용**, BC는 **CpuField용 자유함수로 이식**(ScalarField/virtual Backend 의존 제거).

**✅ momentum (이식 완료·컴파일)**
- `mpmstd/equation/momentum/kernels.*` — `compute_mpmstd_rhs`/`build_adi_bands`/`block_couple_dV,dU`/`scatter_from_tdma`/`add_increment` 복사(검증된 수치 보존)
- `assemble_momentum_const_visc_cpu` — 성분별 explicit BW-ADI RHS (점성+대류, conv_f=1.0, W z-stagger). momentum은 T·P 비의존
- `solve_momentum_cpu` (M2) — 성분별 3-sweep ADI(`build_adi_bands`+TDMA+`scatter`, sweep order·cyclic/solve는 BC도출) + 블록 하삼각 커플링(`block_couple_dV`→halo/ghost→`block_couple_dU`) 한 번에
- `update_velocity_cpu` — `add_increment`
- `mpmstd/core/boundary_ops.*` — `apply_ghost_cpu`(Dirichlet zero/antisym·Neumann)·`modify_tdma_row_cpu`(B+=A/A=0 fold)를 **CpuField/포인터 자유함수로 이식** (기존 BoundaryApplier=ScalarField 기반 → 의존 제거)
- `MomentumSystem` = 성분별 RHS/증분 + ADI 핑퐁 + tridiagonal 밴드 워크스페이스

**✅ pressure (이식 완료·컴파일)**
- `mpmstd/equation/pressure/pressure_base.*` + `pressure_engine.*` — 검증된 pencil-FFT `PressureSolver`(base + engine)를 복사 후 **필드 접촉점 3곳만 CpuField로 적응**(divergence 읽기·dP unpack·projection); 전치/FFT/분산 z-TDMA/파수/플랜은 그대로(수치 보존). FieldRegistry·ScalarField BoundaryApplier 의존 제거(BC는 `apply_ghost_cpu`)
- `solve_pressure_cpu` (번들) — div(U*) → C→I→FFTx → I→Y→FFTy → 분산 z-TDMA → 역변환 → I→C → dP unpack → projection 한 번에. 엔진은 `PressureSystem`이 `shared_ptr`로 **지연 생성·재사용**

**✅ forcing + cfl (이식 완료·컴파일)**
- `physics/forcing/` — `apply_body_force_cpu`(U+=-dt·dPdx)·`channel_total_volume_cpu`·`channel_bulk_velocity_cpu`·`apply_mass_flow_correction_cpu`(dPdx 진화). dP/dx 상태는 main이 보유, 자유함수가 명시적 read/update
- `driver/cfl/` — `compute_cfl_dt_cpu` (speed=global-max(|u|/dx+…), dt=min(MaxCFL/speed, cap))

**✅ statistics + restart (이식 완료·컴파일)**
- `post/statistics` — `Stats` + `init`/`accumulate_statistics_cpu`(전역 nx·ny 정규화, Welford)/`write_statistics_cpu`(Allreduce + rank0 Tecplot)
- `post/io` — `write_restart_cpu`/`read_restart_cpu` (MPI-IO 글로벌 C-order; 기존 채널과 동일 포맷 → 동결 난류 필드 직접 로드)

→ **모든 CPU 수치 자유함수 이식 완료** (momentum·pressure·forcing·cfl·statistics·restart, 전부 `make cpu` GREEN).

**✅ channel main 통합 (§7 레시피) + 스모크 검증**
- `mpmstd/apps/channel/main.cpp` — 등온 채널 §7 레시피: cfl → assemble_momentum_const_visc → solve_momentum(ADI+커플링) → update_velocity → forcing(body+mass-flow) → halo/ghost → solve_pressure(div+Poisson+project) → statistics. **T·mu 선언 안 함**(등온). restart 로드 IC.
- Makefile `SRC_INFRA`에 host-single infra(config·logger·grid·stretching·problem·problem_loader·domain_topology·face_bc·tdma_registry·pascal_cpu) 추가; 채널 앱이 lib+FFTW+PaScaL 링크 성공.
- **로그인 노드 스모크(16³, np=2, zero IC + mass-flow)**: pipeline 무crash, **div ~ 1e-16(기계 영점)**, Ub→1.0000 — momentum+pressure+projection 통합 정확성 검증.

**✅ P1e Re_tau=180 난류 회귀 PASS** (로그인 노드 login01, 48코어, 시스템-OpenMPI 바이너리, np=4×4×2=32, 동결 난류 restart 256³ 로드, 30스텝)
- **지속 확인**: div ~ 5e-14(난류 범위; 라미나~1e-15), dt~0.015(난류 변동에 CFL 제한, 라미나 cap 0.05 아님), Ub=1.0000, dPdx 난류 수준 → relaminarize 안 함
- **통계 = 난류 DNS 일치**: U⁺ 로그-법칙(중심선 U=1.24), **u_rms⁺ 피크 ≈ 2.7**(KMM 2.65), Reynolds 전단 -uw~u_tau², **Re_tau ≈ 167**(21샘플; →180 근접)
- → 검증된 기존 솔버와 동일 거동. **신구조 자유함수 채널이 수치적으로 정확.**

**✅ nvc++ ICE 해결 (FaceBc 리팩토링, 방안 b)**: `FaceBc`의 `std::function` 값(BcValueFn)을 **POD `real_t value`로 교체**(미사용 함수형 오버로드 제거). 시간 의존 BC는 콜백 저장이 아니라 **자유함수가 매 스텝 `value`를 갱신**(rev.2 "데이터=struct, 연산=자유함수"; GPU 커널에서 std::function 평가 불가 문제도 해소). 결과: `make BACKEND=cuda`가 **gpu01에서 lib+apps 전부 빌드 성공**(BUILD_EXIT=0, lower_dynamic_init ICE 0건) → **듀얼빌드 복구**. CPU 동작 **비트-동일 보존**(스모크 div~5e-16, Re_tau=180 회귀 동일).
- 참고(환경): 컴퓨트 노드(cpu/gpu)는 nvhpc(HPC-X)만, **시스템 OpenMPI 없음** → 시스템-OpenMPI 바이너리는 login01(48코어)에서만 실행. CPU 단독 회귀는 login01에서 256³ 32랭크 수용.

---

## P2 — channel main 가독화 (§7 레시피) ✅
- **`core::sync_field_cpu(field, fbc, sub)`** 추가 = `exchange_halo_cpu` + `apply_ghost_cpu` 한 번에("solve 후 이 필드를 이웃·경계와 일치"). 채널 루프·IC·`solve_momentum` 증분의 halo+ghost 쌍을 전부 한 줄로 접음 → ghost 누락 방지 + 잡음 제거.
- 채널 루프를 §7 단계 주석((1) CFL → (2) momentum → (3) forcing → (4) pressure → (5) post)로 정리. 과한 추상화(컨텍스트 번들 등)는 의도적으로 안 함 — 명시 인자가 데이터 흐름을 그대로 드러냄(교수님 "이름으로 기능이 드러나게").
- **동작 비트-동일 보존**(스모크 div 6.09e-16; Re_tau=180 회귀 div 4.63e-14, u_rms 0.1717, U_c 1.2448 — 변화 없음).

---

## 빌드·실행 빠른 참조

```bash
# CPU (로그인 노드에서 가능)
make -C mpmstd cpu
build/cpu/mpmstd/bin/tests/unit/test_smoke_cpu
mpirun --mca btl ^openib -np 2 build/cpu/mpmstd/bin/apps/channel

# GPU (A100 노드; 포그라운드로 제출해 결과를 바로 확인)
sbatch --wait mpmstd/build_gpu.sh && cat log/mpmstd_gpu_*.out

# 스파이크 (P-0.5)
make -C tests/spike cpu && mpirun -np 2 build/cpu/spike/halo_poc
```

---

## 다음 (P2)
channel main 가독화(§7 레시피 정리; assemble/solve/project 분리 등) + 코드리뷰. 이후 P3/P3b(BC-agnostic·cavity), P4 GPU. (nvc++ ICE는 FaceBc 리팩토링으로 이미 해결 → 듀얼빌드 그린.)
