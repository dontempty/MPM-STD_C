# 메모리 관리 계획 — memory::Pool: 장 크기 배열 전부를 하나의 관리자가 내준다 (2026-11-03)

전 단계: `docs/Memory/plan.md` (2026-10-23, 솔버 밴드·스펙트럼만 `equation::Workspace` 5 슬롯으로 공용화).
이 문서는 그 다음 단계다 — **앱이 만드는 장(field)과 물성, 전치 버퍼, 스크래치까지** 같은 관리자 아래로 모은다.

## 0. 한 줄 요약

랭크 하나가 들고 있는 장 크기(ghost 포함, 1.09 GB @512³) 배열은 약 28 개인데, 그중 공용 관리를 받는 것은 Workspace 5 개뿐이다.
나머지 — 상태 7, 통계 2, rhs 3, 물성 6, 압력 전치 버퍼 4, 순환 TDMA 스크래치 1 — 는 각자 `cudaMalloc` 한다.
한 스텝 안에서 이들의 수명은 겹치지 않는 구간이 많다 (rhs 는 momentum 안, 열 물성은 scalar 안, 전치 버퍼는 압력 안).
**시작할 때 블록 N 개를 한 번에 받고, 쓰는 쪽이 필요할 때 빌리고 끝나면 돌려주는 풀** 로 바꾸면
512³ np 8 기준 **32.3 GB → 약 24.6 GB** (블록 28 → 20), 수치는 비트 동일이 게이트다.

실측(2026-11-03, `MPMSTD_MEM_REPORT=1`, 512³ np 8 2×2×2, 랭크 max):

| 단계 | 누적 GB | 증가 | 정체 |
|---|---|---|---|
| CUDA + MPI(GTL) 컨텍스트 | 2.41 | 2.41 | 앱이 못 줄임 (np 1 은 0.64) |
| 앱 장 10 (U V W T P dPhat dP0 rhs_U V W) | 13.30 | 10.89 | 각자 `Field` |
| 물성 6 | 19.83 | 6.53 | `PropertyFields` 의 `Field` 6 |
| Workspace 5 슬롯 | 25.21 | 5.38 | 유일한 공용 풀 |
| momentum·scalar 솔버 | 25.56 | 0.35 | PaScaL plan 축약계 (작음) |
| 압력 솔버 | 30.11 | 4.56 | **전치 send/recv 4 개** (np > 1) + plan |
| 통계 | 32.29 | 2.18 | `mean_t_sum_`, `mean_w_sum_` (scalability 앱은 쓰지도 않음) |
| step 1 뒤 (np 1 만) | +1.07 | | PaScaL 순환 스크래치 (첫 풀이 때 할당) |

---

## 1. 배열 분류 — 수명이 설계의 전부다

N⁺ = ghost 포함 장 원소 수. "블록" = N⁺ 개 double 하나 (§3.1 크기 규칙).

### 1.1 상주 (resident) — 스텝을 넘어 살아 있다. 풀에서 **한 번 빌리고 돌려주지 않는다**

| 배열 | 개수 | 근거 |
|---|---|---|
| U V W T P | 5 | 체크포인트 항목 |
| dPhat, dP0 (압력 예측자 이력) | 2 | 체크포인트 항목, `advance_pressure_predictor` 가 매 스텝 갱신 |
| μ̂, 1/ρ̂ (운동량 물성) | 2 | `dt_viscous`(스텝 머리)가 **직전 스텝 끝**(properties_2)의 μ̂ 를 읽는다 → 스텝 경계를 넘는다 |
| 통계 `mean_t_sum_`, `mean_w_sum_` | 2 | 3D 시간평균 T·W (U_pl, Re_pl, 시간평균 T PDF). 누적 시작 전까지는 빌리지 않는다 |
| **합** | **11** (통계 전 9) | |

### 1.2 스텝 임시 (step-scoped) — 스텝 안의 한 구간에서만 산다. 앱이 빌리고 앱이 돌려준다

| 배열 | 개수 | 사는 구간 |
|---|---|---|
| 열 물성 4 (κ̂, dκ̂/dT, 1/ĉ_p, d(1/ĉ_p)/dT) | 4 | properties_1 → scalar 끝. scalar 조립(x·y·z 세 스윕 전부)만 읽는다. 통계의 ⟨κ̂⟩(z) 는 평면평균 1D 라 표본 때 T 에서 다시 만들면 된다 |
| rhs_U V W (→ 안에서 δU δV δW 로 덮어씀) | 3 | momentum_rhs → momentum_solve 끝 (plan.md §9: solve 가 rhs 를 덮어쓴다) |
| rhs_dP (→ 안에서 dP) | 1 | pressure_rhs → pressure_correct·predictor 끝 |

### 1.3 호출 임시 (call-scoped) — 솔버 함수 안에서 빌리고 **함수가 끝나면서 돌려준다**

| 소유자 | 배열 | 개수 | 비고 |
|---|---|---|---|
| scalar | A B C D0 D1 | 5 | 지금 Workspace 0~4 |
| momentum (성분마다) | A B C D0 D1 | 5 | 지금 Workspace 0~4 |
| pressure | real_lines↔band_c, spectrum_x, spectrum_xy, band_a↔fft_work, band_b | 4~5 | 지금 Workspace 0~4 (동시 최대 4: band a·b·c + spectrum_xy) |
| pressure 전치 | send_buffer, recv_buffer (전치 종류마다) | 0 | **전치 중에는 스펙트럼 슬롯이 비어 있다** → 그 블록을 쓰면 추가 0 (§4 단계 1) |
| PaScaL 순환 TDMA | z 스크래치 (nsys·nrow) | 1 | 지금은 `static` 공용 (`pascal_tdma_cuda.cu cyclic_scratch`) → 호출자가 블록을 넘기면 0 추가 |
| pressure (x 벽 앱만) | mirror_spectrum | 2 | DCT 거울 (cavity 등). rbc 에는 없다 |

### 1.4 풀 밖에 남기는 것 (장 크기가 아니다)

PaScaL plan 축약계·bigbuf (≈ 24/nrow 블록; momentum·scalar 합 0.36 GB, 압력 z 는 nzp 가 작아지면 커져 np 64 에서 ≈ 0.75 블록 — 압력 z 행렬 미리 분해와 함께 따로 다룬다),
halo send/recv (면 크기), λ 표·격자 표 (1D), 통계 프로파일·히스토그램의 호출마다 `cudaMalloc` (1D, 작음 — 성능 문제이지 메모리 문제가 아님), CUDA·MPI 컨텍스트.

---

## 2. 예산 — 블록이 몇 개 필요한가

풀 용량 = 상주 합 + **스텝 안에서 동시에 살아 있는 임시 블록의 최대값** (합이 아니다 — 돌려준 블록은 다음 구간이 다시 쓴다).

| 구간 | 스텝 임시 | 호출 임시 | 동시 |
|---|---|---|---|
| dt·properties_1 | 열 물성 4 (빌림) | 0 | 4 |
| **scalar** | 열 물성 4 | 밴드 5 | **9** |
| sync_T·momentum_rhs | rhs 3 (열 물성은 돌려줌) | 0 | 3 |
| momentum | rhs 3 | 밴드 5 | 8 |
| properties_2 (μ̂·1/ρ̂ 만 갱신) | 0 | 0 | 0 |
| pressure_rhs·solve·correct·predictor | rhs_dP 1 | 압력 4~5 (전치 포함 0 추가, np 1 순환 스크래치 0 추가) | 6 |
| 통계 표본 | 0 (κ̂ 평면평균은 임시 1 블록이면 충분) | 0 | 1 |

→ 임시 최대 **9** (scalar). 용량 = 11 + 9 = **20 블록** (통계 전 18, scalability 앱 18).

| 구성 | 블록 | 512³ np 8 예상 | GPU 당 상한 (95 GB 가용) |
|---|---|---|---|
| 지금 | ≈ 28 | 32.3 GB (실측) | ≈ 740³ |
| 이 계획 | 20 (통계 포함) | 2.4 + 20 × 1.09 + 0.4 ≈ **24.6 GB** | ≈ 830³ |
| 이 계획 + 열 물성을 scalar 커널 안에서 T 로 바로 계산 (후속) | 19 | ≈ 23.5 GB | ≈ 850³ |
| 참고: 1024³/GPU 가 되려면 | ≤ 11 | | 상주만 11 → 임시 0 이어야 한다. 밴드를 장보다 훨씬 작게 쪼개는 TDMA 구조 변경 없이는 불가능 |

주의: **Workspace 5 를 고정으로 두어도 블록 수는 같다** (고정 5 + 떠다니는 max(열 물성 4, rhs 3, rhs_dP 1) = 9).
절감은 rhs·물성·전치·통계·스크래치의 **수명을 드러내는 데서** 오고, 밴드를 호출마다 빌리게 바꾸는 것은 절감이 아니라
한 가지 기제로 통일하고 예산을 실행 중에 검증할 수 있게 하는 데 의미가 있다.

---

## 3. 설계 — `mpmstd::memory::Pool`

### 3.1 객체

```
// src/memory/pool.hpp  (양 빌드 공용; 저장은 GPU = cudaMalloc 한 번, CPU = std::vector 한 번)
namespace mpmstd::memory {

class Block;                                  // 빌린 블록 핸들 (아래)

class Pool {
 public:
  // block_elements 개 double 짜리 블록 capacity 개를 한 번에 받는다. 생성 뒤 재할당·성장 없음.
  Pool(std::size_t block_elements, int capacity);
  Pool(const Pool&) = delete;

  // 빈 블록 n 개(연속)를 빌린다. 모자라면 throw — 메시지에 지금 보유자 전부를 적는다 (예산 실수를 조용히 넘기지 않는다).
  // zero = true 면 0 으로 채운다 (상주 장: 기존 Field 생성자 계약과 같다). 임시 블록은 기본 false — 내용 보장 없음.
  Block acquire(const char* who, int n = 1, bool zero = false);

  std::size_t block_elements() const;
  int capacity() const;
  int in_use() const;
  int high_water() const;                     // 지금까지의 최대 동시 사용 — 예산 검증용
  void report(std::FILE*) const;              // 보유자 목록 + 최고 수위 (MPMSTD_MEM_REPORT 때 앱이 부른다)
};

// 소유권 핸들. 소멸 = 반환 (RAII — "함수가 끝남과 동시에 전달"). move 전용.
class Block {
 public:
  real_t* data();  const real_t* data() const;
  std::size_t elements() const;               // n × block_elements
  void release();                             // 소멸 전에 미리 돌려주고 싶을 때
  ~Block();
};
}
```

- 할당: `capacity × round_up(block_elements × 8 B, 256 B)` 한 덩어리. 블록 k 의 주소 = base + k × stride.
- 빈 블록 관리: 비트맵 + 선형 탐색 (capacity ≤ 수십이라 충분). n > 1 은 연속 구간을 찾는다.
- 반환된 블록은 즉시 재사용 가능 — **내용은 보존하지 않는다.** 빌리는 쪽은 "쓰기 전에 읽지 않는다" 가 계약이다 (지금 Workspace 와 같다).
- 디버그 모드 (`MPMSTD_POOL_POISON=1`): 반환할 때 NaN 으로 채운다 → 돌려준 블록을 몰래 읽는 코드가 바로 터진다. 게이트 전에 한 번 돌려 본다.
- 스트림·스레드: 기본 스트림 하나에서 차례로 쓰는 전제 (Workspace 와 같은 불변 조건). MPI 가 블록을 쓰는 동안(동기 Alltoallv)은 호출이 끝나야 반환되므로 안전.

### 3.2 크기 규칙

`block_elements = max(장 N⁺, 각 솔버의 required_slot_elements)`.
- rbc (x 주기): 압력 밴드 n_band = 2·(nx/2+1)·ny·nz = N·(1+2/nx) ≤ N⁺ = (nx+2)(ny+2)(nz+2). 평면 경로 라인 ny·nzp·nx ≤ N. **장 크기 하나로 전부 맞는다.**
- x 벽(DCT) 앱: real_lines = 2N, mirror_spectrum ≈ 2N → 블록 하나로 안 된다. 블록을 2 배로 키우면 20 블록 전부가 2 배가 되므로, 대신 **`acquire(who, n = 2)`** 로 연속 2 블록을 빌린다 (용량에 +2~4 를 더 잡는다). rbc 경로에는 영향 없다.
- 전치 send/recv: 원소 수 = 내부 N ≤ N⁺. 블록 하나.
- 순환 스크래치: nsys·nrow ≤ N⁺. 블록 하나.

### 3.3 `Field` — 소유에서 "블록을 감싼 이름표" 로

```
// field/gpu/field.hpp, field/cpu/field.hpp
Field(int nx_total, int ny_total, int nz_total, std::string name);          // (유지) 스스로 cudaMalloc — 아직 안 옮긴 앱용
Field(memory::Block&& block, int nx_total, int ny_total, int nz_total, std::string name);  // (추가) 블록을 입는다. 소멸 = 블록 반환
```
- 두 번째 생성자는 `block.elements() ≥ nx·ny·nz` 를 검사하고 (상주 장은 `zero = true` 로 빌려서) 기존 "0 으로 초기화" 계약을 지킨다.
- 커널·halo·체크포인트·통계는 `Field&` 와 `data()/view()` 만 보므로 **호출부 변경 없음.** `write_checkpoint` 의 `{"U", &U}` 도 그대로.
- 소유 생성자는 지우지 않는다 (다른 11 개 앱이 그대로 빌드된다). 옮기는 앱만 두 번째 생성자로 바꾼다.

### 3.4 솔버 — 선언은 정적 함수, 빌리기는 호출 안에서

```
// 각 솔버: 예산용 선언 (생성 전에 앱이 부른다)
static std::size_t scalar::Solver::required_slot_elements(sub);     // (유지) 블록 크기
static int         scalar::Solver::transient_blocks();              // (추가) 5
static int         momentum::Solver::transient_blocks();            // 5
static int         pressure::Solver::transient_blocks(sub);         // 4 (슬랩·평면 경로) / 5 (전치 4 회) / +2 (DCT)

// 생성자: Workspace& 대신 memory::Pool& 를 받아 보관
// solve_*: 들어오면서 빌리고 나가면서 돌려준다
void scalar::solve_variable_property(Solver& s, ...) {
  memory::Block A = s.pool->acquire("scalar.A"), B = ..., C = ..., D0 = ..., D1 = ...;
  ... (기존 코드: A.data() 등)
}   // 여기서 다섯 블록 반환
```
- momentum: 성분 dU → dV → dW 가 같은 다섯 블록을 차례로 쓴다 — 지금과 같다 (함수 머리에서 한 번 빌려 끝까지).
- pressure: 생성자가 슬롯 주소를 멤버에 묶어 두던 것(`real_lines = ws.slot(0)` …)을 `solve()` 머리에서 빌리는 것으로 바꾼다.
  - cuFFT: 작업 영역은 `cufftSetWorkArea` 를 **매 solve 머리에서** 다시 건다 (호출 비용 무시 가능). 플랜 자체는 주소에 안 묶인다.
  - CPU(FFTW): 플랜이 배열 주소에 묶인다 → `fftw_execute_dft_r2c(plan, in, out)` 계열(new-array execute) 로 바꾸면 같은 정렬의 다른 주소에 실행할 수 있다. CPU 는 이 서버에서 검증 불가 — 문서에 남기고 CPU 빌드는 후순위.
  - 전치: `run_exchange(e, src, dst, …, send_block, recv_block)` 로 블록을 인자로 받는다. 평면 경로 block_to_planes 때 spectrum_x·spectrum_xy·band_a·band_b 자리가 비어 있으므로 그중 둘을 넘긴다 (solve 안에서 빌린 블록을 역할만 바꿔 쓴다 — Workspace 의 ↔ 쌍과 같은 방식). `PencilExchange::send_buffer/recv_buffer` 멤버는 CPU·index 대조 경로용으로 남기되 GPU 기본 경로에서는 `assign` 하지 않는다.
- PaScaL 순환: `launch_tdma_cycl_many(..., double* scratch)` 로 호출자가 넘긴다. `cyclic_scratch()` static 은 넘기지 않았을 때의 대안으로 남긴다 (기존 경로 보존).
- `equation::Workspace` 는 당분간 **Pool 위의 얇은 겉면**으로 다시 구현한다 (생성 때 5 블록 빌려 `slot(k)` 로 돌려줌) — 안 옮긴 앱이 그대로 돈다. 세 솔버가 Pool 로 넘어가면 지운다.

### 3.5 물성 — 둘로 가른다

```
struct MomentumProperties { Field viscosity, inverse_density; };                       // 상주 2 (앱이 Pool 에서 한 번 빌림)
struct ThermalProperties  { Field conductivity, conductivity_derivative,
                            inverse_heat_capacity, inverse_heat_capacity_derivative; }; // 스텝 임시 4 (properties_1 ~ scalar 끝)
void update_momentum_properties(MomentumProperties&, const Field& T, model, sub);
void update_thermal_properties (ThermalProperties&,  const Field& T, model, sub);
// (유지) PropertyFields + update_properties — 안 옮긴 앱용. 커널은 같은 다항식을 쓰므로 값은 비트 동일.
```
- 스텝 순서는 그대로다: properties_1 = 운동량 2 + 열 4 갱신 (clip 이 T 를 바꾸므로 properties_2 와 중복이 아니다), properties_2 = 운동량 2 만.
- 통계 `accumulate` 의 `properties.viscosity` 는 상주 μ̂ 를 그대로 읽고, `⟨κ̂⟩(z)` 는 T 에서 평면평균 커널 안에서 κ̂(T) 를 계산하거나 임시 1 블록에 만들어 평균한다. 표본 간격마다 한 번이라 비용 무시.
- 후속 (이 계획 밖): 열 물성 4 를 scalar 조립 커널 안에서 T 로 바로 계산 → 임시 최대 9 → 8. 커널이 대역폭에 묶여 있어 느려지지 않을 가능성이 크지만 측정 전이다.

### 3.6 앱 (rbc_nob, rbc_nob_scalability) 의 스텝

```
Pool pool(block_elements, budget);                          // §2 — 상주 11 + 임시 9 (+ DCT 여분)
Field U(pool.acquire("U", 1, true), nx, ny, nz, "U"), V, W, T, P, dPhat, dP0;   // 상주 7
MomentumProperties mprops(pool);                            // 상주 2
Statistics statistics(cfg, sub, pool);                      // 3D 누적 2 는 첫 accumulate 때 빌린다

매 스텝:
  dt_cfl, dt_viscous(mprops.viscosity)
  { ThermalProperties tprops(pool);                         // 임시 4
    update_momentum_properties(mprops, T); update_thermal_properties(tprops, T);
    scalar::solve_variable_property(scalar_solver, ..., tprops, mprops, ...);   // 안에서 밴드 5 → 동시 9
  }                                                         // 열 물성 반환
  sync_T
  { Field rhs_U(pool.acquire("rhs_U"), ...), rhs_V, rhs_W;  // 임시 3
    momentum::build_variable_property_rhs(...); momentum::solve_variable_property(...);   // 안에서 밴드 5 → 8
  }                                                         // rhs 반환
  sync_UVW_1
  update_momentum_properties(mprops, T)                     // properties_2
  { Field rhs_dP(pool.acquire("rhs_dP"), ...); Field& dP = rhs_dP;   // 임시 1
    pressure::build_variable_property_rhs(...); pressure::solve(...);  // 안에서 4~5 → 6
    pressure::correct_velocity_and_accumulate_variable_property(...);
    sync_UVW_2; advance_pressure_predictor(dPhat, dP0, dP);
  }                                                         // rhs_dP 반환
  clip, monitor, statistics.accumulate(...)
```
`rhs_dP.fill(0)` 을 루프 밖에서 한 번만 하던 것(ghost 0 보장)은 빌릴 때 `zero = true` 로 매 스텝 0 으로 — 비용은 memset 한 번. ghost 를 매 스텝 0 으로 만들어도 지금과 값이 같은지(조립이 ghost 를 읽는지)는 게이트로 확인한다; 다르면 압력 rhs 조립이 ghost 를 직접 쓰도록 바꾼다.

---

## 4. 단계 — 매 단계 게이트: `check_gate.py` (rbc_nob 모니터 줄 비트 동일) + `test_pressure_pencil`/`test_pressure_wall` 체크섬 + `MPMSTD_MEM_REPORT` 로 블록 수 확인

| 단계 | 내용 | 블록 변화 (512³ np 8) | 위험 |
|---|---|---|---|
| 0 | `memory::Pool` + `Block`, `Field` 블록 생성자, `Workspace` 를 Pool 위 겉면으로. 모든 앱 빌드·게이트 불변 | 0 | 낮음 |
| 1 | 압력 전치 send/recv → solve 안의 빈 블록. PaScaL 순환 스크래치 → 호출자 블록 | **−4** (np > 1), −1 (np 1) | 낮음 — 포인터만 바뀐다 |
| 2 | rbc_nob·scalability: 물성 둘로 가르기, rhs 3·rhs_dP 를 스텝 임시로, 통계 3D 누적을 첫 표본 때 | −3 (rhs) −4 (열 물성) −2 (통계, scalability) | 중간 — 수명이 바뀐다. POISON 모드로 한 번 |
| 3 | 세 솔버가 Workspace 대신 Pool 에서 호출마다 빌리기. cuFFT 작업 영역 매 solve 재설정. Workspace 삭제 | 0 (기제 통일, high_water 로 예산 검증) | 중간 — 압력 생성자의 주소 고정 가정 해제 |
| 4 (후속) | 열 물성을 scalar 커널 안에서 T 로 계산 | −1 | 성능 측정 필요 |

단계 2 까지 끝나면 28 → 20 (scalability 18). 단계마다 커밋, 옮기지 않은 앱은 기존 생성자·`PropertyFields`·`Workspace` 로 그대로 돈다.

---

## 5. 열어 둔 점

- 1024³/GPU 는 이 계획으로 안 된다 (§2). 그 목표면 TDMA 밴드를 "장 하나" 가 아니라 "시스템 묶음" 단위로 쪼개 풀어 임시 블록을 장보다 작게 만드는 별도 설계가 필요하다 — 통신(PaScaL 축약계)과 조립을 묶음마다 다시 부르므로 호출 횟수가 늘고, 그 비용은 측정해 봐야 안다.
- PaScaL 압력 z plan (np 64 에서 ≈ 0.75 블록) 은 압력 z 행렬 미리 분해(a′·c′ 를 생성 때 한 번)와 같이 다룬다.
- 통계 표본의 1D `cudaMalloc`/`cudaFree` 반복은 메모리 문제가 아니라 동기화 비용 — Pool 과 별개로 1D 작은 영역 하나를 재사용하게 바꾸면 된다.
