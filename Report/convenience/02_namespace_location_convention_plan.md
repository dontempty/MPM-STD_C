# MPM-STD_C 수정계획서 — "namespace = 파일 위치" 규약 제정 및 이행

- 작성일: 2026-07-24
- 상위 계획: [01_deabstraction_user_friendliness_plan.md](01_deabstraction_user_friendliness_plan.md)의 **1단계(이름 정리)를 구체화**한 하위 계획. 0단계(죽은 코드 삭제)는 완료·검증됨.
- 전제: 계산 로직 불변. 본 계획의 변경은 전부 **컴파일 타임 상표(namespace) 이동·별칭 제거·파일/디렉터리 개명**뿐이며, 목적 코드가 1바이트도 달라질 이유가 없다 → CPU `validate` 필드덤프 byte-비교가 그대로 합격 판정이 된다(0단계에서 증명된 게이트).

---

## 1. 문제 정의

사용자 지적 예시 — `apps/rbc/main_gpu_nob.cu:406`:

```cpp
core::Subdomain sub(topo, n_global);
```

`Subdomain`의 실제 정의는 `src/parallel/mpi/subdomain.hpp`(`mpmstd::parallel::mpi::Subdomain`)인데,
`src/core/mpi_topology.hpp:25-27`이 이를 `core::`로 **재수출(using-alias)** 한다:

```cpp
namespace mpmstd::core {
using MpiContext  = parallel::mpi::MpiContext;
using MpiTopology = parallel::mpi::MpiTopology;
using Subdomain   = parallel::mpi::Subdomain;   // ← 위치를 속이는 별칭
}
```

결과: 독자가 `core::Subdomain`을 보고 `src/core/`를 뒤지면 정의가 없다. grep은 별칭 헤더에 먼저 걸리고, IDE 점프는 한 단계를 더 거친다. **namespace가 "위치 정보"가 아니라 "호출 편의 상표"로 쓰이고 있는 것**이 근본 원인이다.

같은 문제의 변형이 세 유형 있다:

### 유형 A — 타 모듈 재수출 별칭 (위치를 속임)

| 별칭 헤더 | 별칭 | 실제 정의 위치 | 호출부 수(src+apps) |
|---|---|---|---|
| `core/mpi_topology.hpp:25-27` | `core::MpiContext` / `MpiTopology` / `Subdomain` | `src/parallel/mpi/` | 11 / 10 / 32 |
| `core/grid.hpp:9-10` | `core::Grid` / `AxisConfig` | `src/grid/` | 16 / 0 |
| `core/boundary.hpp:10-12` | `core::Boundary` / `BoundaryCondition` / `BcKind` | `src/boundary/` (`boundary::Problem`) | 0 / 37 / 0 |
| `core/boundary_ops.hpp:12` | `core::FieldBoundary` | `src/boundary/field_boundary.hpp` | 2 |
| `core/config.hpp:8` | `core::Config` | `src/config/` | 0 |

합계 **약 108개 호출부**가 가짜 위치를 보고 있고, 별칭 4개(`Boundary`, `BcKind`, `AxisConfig`, `Config`)는 사용처 0인 죽은 별칭이다. 특히 `core::boundary.hpp`는 같은 클래스(`boundary::Problem`)에 **이름 셋(Problem/Boundary/BoundaryCondition)**을 부여해 grep을 3갈래로 쪼갠다.

### 유형 B — 폴더 구조를 무시한 평탄(flat) namespace (위치 정밀도 손실)

| 실제 경로 | 현재 namespace | 문제 |
|---|---|---|
| `src/equation/momentum/…` `scalar/…` `pressure/solve.hpp` | 전부 `mpmstd::equation` (평탄) | `equation::solve_scalar_nob_cpu`를 보고 momentum/scalar/pressure 중 어느 폴더인지 알 수 없음. 호출부 **73개** |
| `src/physics/buoyancy/…` `forcing/…` `properties/…` | 전부 `mpmstd::physics` (평탄) | `physics::add_buoyancy_force_cpu` — 폴더 3개 중 어디인지 이름이 말해주지 않음. 호출부 **48개** |

(반례로 `parallel::mpi`, `linear_solver::tdma`, `equation::pressure`(엔진 클래스), `equation::momentum::kernels` / `scalar::kernels`는 이미 경로를 따르고 있다 — 즉 코드베이스 절반은 이미 규약대로다. 규약이 없어서 나머지 절반이 표류했을 뿐이다.)

### 유형 C — 경로에 존재하지 않는 레벨·축약 별칭·경로/이름 불일치

| 항목 | 위치 | 문제 |
|---|---|---|
| `equation::scalar::kernels::stencil` | `scalar_var_stencil.hpp` | `stencil`은 디렉터리에 없음. 게다가 `scalar_gpu.cu:105,161,188`에서 `using namespace …::stencil;`로 무자격 사용 |
| `namespace sk = scalar::kernels;` / `namespace mk = momentum::kernels;` | `scalar_cpu.cpp:15`, `momentum_cpu.cpp:16` | 축약 별칭 — 팀 규칙("축약 금지")과 위치 추적성 모두 위반 |
| `mpmstd::parallel::cuda_helpers` | 디렉터리는 `src/parallel/cuda/` | namespace와 폴더명이 다름 (`cuda` ≠ `cuda_helpers`) |
| `mpmstd::core::gpu` | `core/gpu/gpu_grid.hpp/.cu` | 다른 모든 `cpu/`·`gpu/` 폴더는 namespace 레벨을 만들지 않고 `_cpu`/`_gpu` 접미사를 쓰는데 여기만 예외 |
| `apps/rbc/rbc_common.hpp:34` | `using namespace mpmstd;`가 **헤더**에 | 이 헤더를 포함하는 모든 TU의 전역을 오염 |

---

## 2. 제정할 규약 (MPM-STD_C Namespace Convention v1)

> **한 줄 요약: `mpmstd::a::b::Name`을 보면 정의는 반드시 `src/a/b/` 안에 있다. 역으로 `src/a/b/`의 모든 선언은 `mpmstd::a::b`에 있다.** (`::`를 `/`로 바꾸면 디렉터리가 된다.)

### 선언 규칙 (D)

- **D1. 경로 사상**: `src/<경로>/파일`의 모든 선언은 `namespace mpmstd::<경로>`에 둔다. 디렉터리 중첩 = namespace 중첩.
- **D2. cpu/·gpu/ 잎 디렉터리는 namespace 레벨을 만들지 않는다.** 백엔드 구분은 기존 규칙대로 함수 접미사 `_cpu`/`_gpu`가 담당한다. (예: `src/driver/cpu/cfl_cpu.cpp` → `mpmstd::driver`.)
- **D3. 허용되는 비(非)경로 잎은 두 개뿐**:
  - `kernels` — 방정식 모듈의 저수준 커널 계층(공개 solve/assemble API와 구분되는, 실재하는 역할 구분). 예: `equation::momentum::kernels`.
  - `detail` — 헤더 내부 구현 전용.
  그 외의 비경로 레벨(현재의 `stencil` 등)은 금지.
- **D4. 익명 namespace는 `.cpp`/`.cu` 파일-로컬 내부 구현에만.** 헤더에는 금지.
- **D5. 예외 — `src/common/`**: 프로젝트 공용 어휘(`index_t`, `Direction`, 매크로)만은 `mpmstd::` 직속에 둔다. "namespace 경로가 없는 이름 = `src/common/`"이 역추적 규칙이 된다. common에는 타입 별칭·enum·매크로 외에 함수를 추가하지 않는다.

### 호출 규칙 (C)

- **C1. 다른 모듈의 이름은 항상 모듈 경로부터 자격(qualify)해서 쓴다**: `parallel::mpi::Subdomain`, `grid::Grid`, `equation::scalar::solve_scalar_nob_cpu`. (프로젝트 내부에서 `mpmstd::` 접두는 생략 가능 — C++ 상대 조회가 처리하며, 나머지 경로만으로 위치가 확정된다.)
- **C2. 타 모듈 이름을 자기 namespace로 들여오는 재수출(using-alias/using-declaration)은 금지.** 이것이 이번 사고(`core::Subdomain`)의 직접 원인이다. 별칭은 **같은 모듈 내부의 템플릿 인스턴스 축약**(예: `core::CpuFields = core::FieldStore<core::CpuField>`)에만 허용.
- **C3. `using namespace`는 두 곳에만 허용**: ① 앱 `main` 소스 파일 최상단의 `using namespace mpmstd;` 1회(최상위 접두만 벗김 — 호출부에는 여전히 `equation::…`, `parallel::mpi::…`가 남으므로 추적성 유지), ② 함수 본문-로컬로 **같은 모듈** 하위 namespace를 여는 경우(GPU 커널에서 `kernels` 헬퍼 다량 사용 시). 헤더에서는 전면 금지.
- **C4. namespace 별칭은 축약 금지.** `namespace sk = …;` 같은 두 글자 별칭 대신 전체 경로를 쓴다(팀 규칙 "축약 없는 완전한 이름"의 namespace 판).

### 강제 장치

- `scripts/check_namespace_convention.sh` (신규, ~50줄 grep/awk):
  1. 각 소스의 `namespace mpmstd::…` 선언 ↔ 파일 경로 일치 검사(D1·D2·D3),
  2. 헤더의 `using namespace` 검출(C3),
  3. `using X = <다른 최상위 모듈>::…` 재수출 검출(C2),
  4. 2글자 이하 namespace 별칭 검출(C4).
  `make lint` 타깃으로 편입, 회귀 스크립트와 함께 상시 실행. (고아 `test/` 트리는 4단계 병합 전까지 검사 제외.)

---

## 3. 목표 상태 매핑 표

| 경로 | 현재 | 규약 적용 후 | 조치 |
|---|---|---|---|
| `src/common/` | `mpmstd` 직속 | 변경 없음 (D5 예외) | — |
| `src/config/`, `src/grid/`, `src/boundary/`, `src/driver/`, `src/post/`, `src/core/`(자기 타입), `src/parallel/mpi/`, `src/linear_solver/tdma/` | 이미 경로 일치 | 변경 없음 | — |
| `core/mpi_topology.hpp`의 별칭 3개 | `core::Subdomain` 등 | **삭제** → 호출부는 `parallel::mpi::Subdomain` | 53곳 치환 |
| `core/grid.hpp`, `core/config.hpp`, `core/boundary.hpp`, `boundary_ops.hpp:12` 별칭 | `core::Grid`/`core::BoundaryCondition` 등 | **삭제** → `grid::Grid`, `boundary::BoundaryCondition` | 55곳 치환 |
| `boundary::Problem` 클래스명 | `Problem` (별칭 경유로만 `BoundaryCondition`) | 클래스 자체를 `boundary::BoundaryCondition`으로 개명 (호출부 37곳이 이미 이 이름을 씀 — 실사용 다수결) | rename |
| `src/equation/{momentum,scalar}/solve.hpp·assemble.hpp` API | `mpmstd::equation` 평탄 | `equation::momentum`, `equation::scalar` | 호출부 ~50곳 |
| `src/equation/pressure/solve.hpp` API | `mpmstd::equation` 평탄 (엔진만 `::pressure`) | `equation::pressure`로 합류 | 호출부 ~23곳 |
| `…::kernels` | 경로에 없는 잎 | **유지** (D3 승인 잎) | — |
| `…::kernels::stencil` | 경로에 없는 2중 잎 | `kernels`로 흡수, `using namespace …stencil` 3곳은 C3-②로 정리 | 소규모 |
| `namespace sk/mk` | 축약 별칭 | 삭제, 전체 경로 표기 | 2파일 |
| `src/parallel/cuda/` ↔ `parallel::cuda_helpers` | 폴더·namespace 불일치 | **디렉터리를 `src/parallel/cuda_helpers/`로 개명** (namespace 유지 쪽 선택 — `mpmstd::parallel::cuda`는 libcu++의 전역 `::cuda`와 조회 충돌 위험이 있어 피함) | 파일 이동 + Makefile 글롭 |
| `core/gpu/gpu_grid.*`의 `core::gpu` | 유일한 backend-잎 namespace | `mpmstd::core`로 흡수 (D2 정합; 이름은 이미 `GpuGrid`류라 충돌 없음) | 소규모 |
| `apps/*` | `using namespace mpmstd;` (main 6곳 + **헤더 1곳**) | main 소스 1회는 C3-①로 합법; `rbc_common.hpp:34`는 제거하고 자격 표기 (3단계 main 통합 시 헤더 자체가 소멸 예정이지만 그 전까지 위반 상태로 두지 않음) | 1파일 |

적용 후 사용자 예시는 다음이 된다:

```cpp
// namespace가 곧 경로: src/parallel/mpi/subdomain.hpp
parallel::mpi::Subdomain sub(topo, n_global);
// 온도 해석: src/equation/scalar/
equation::scalar::solve_scalar_nob_gpu(...);
// 부력: src/physics/buoyancy/
physics::buoyancy::add_buoyancy_force_nob_gpu(...);
```

---

## 4. 이행 순서 (커밋 단위, 각각 독립 검증)

원칙: **한 커밋 = 한 종류의 치환.** 치환은 sed 일괄이 아니라 **컴파일러 주도**로 한다 — 별칭/구 namespace를 먼저 제거하면 컴파일 에러 목록이 곧 완전한 작업 목록이다(과거 `%d→%domain` sed 사고 재발 방지).

| 커밋 | 내용 | 규모 | 비고 |
|---|---|---|---|
| **N1** | 유형 A 박멸: `core/` 재수출 별칭 5개 헤더에서 제거 → 호출부 ~108곳을 실제 namespace로 치환. 죽은 별칭 4개는 그냥 삭제. `core/mpi_topology.hpp`는 별칭 제거 후 실기능(`bind_gpu_to_local_rank_*`)만 남으므로 `core/gpu_binding.hpp`로 개명 | ~110곳 | 사용자 지적 사항의 직접 해결. 최우선 |
| **N2** | `boundary::Problem` → `boundary::BoundaryCondition` 클래스 개명 (N1과 분리 — rename과 재배치를 섞지 않음) | ~40곳 | |
| **N3** | 유형 B-1: `equation` 평탄 API를 `equation::momentum / scalar / pressure`로 분할 (파일 이동 없음 — 선언을 감싼 namespace만 변경) | ~73곳 | 함수명 자체(`_ob`/`_nob` 통일 등)는 건드리지 않음 — 그것은 01 보고서 1단계의 별도 커밋 |
| **N4** | 유형 B-2: `physics` → `physics::buoyancy / forcing / properties` 분할 | ~48곳 | |
| **N5** | 유형 C 정리: `stencil` 잎 흡수, `sk`/`mk` 별칭 제거, `core::gpu` 흡수, `rbc_common.hpp`의 헤더 using 제거 | 4~5파일 | |
| **N6** | `src/parallel/cuda/` → `src/parallel/cuda_helpers/` 디렉터리 개명 + Makefile 글롭·include 경로 수정 | 파일 이동 | 유일하게 빌드 스크립트를 건드리는 커밋 — 단독 분리 |
| **N7** | `scripts/check_namespace_convention.sh` 작성, `make lint` 편입, 규약 본문을 `docs/namespace_convention.md`로 저장(본 보고서 §2 전재) | 신규 ~50줄 | 이후 모든 단계(3·4단계 포함)의 가드레일 |

의존성: N1→N2는 순서 필수(같은 헤더를 만짐), N3·N4·N5는 상호 독립, N6·N7은 마지막. 전체 예상 소요 1~2일(대부분 컴파일러 에러 따라가는 기계적 치환).

**01 계획서와의 접합**: 본 N-시리즈는 1단계의 전반부다. 1단계 후반부(타입명 `*System`→`*LinearSystem`, `engine`→`pressure_solver`, `_ob/_nob` 함수명 통일, 우산 `main.hpp` 정리)는 N-시리즈 완료 후 별도 커밋으로 진행한다 — namespace 이동과 이름 변경이 한 diff에 섞이면 리뷰가 불가능해지기 때문. 3단계(main 통합)는 N-시리즈 **후**에 하는 것이 유리하다: 새로 쓰는 통합 main이 처음부터 규약 준수 상태로 작성되고 lint의 보호를 받는다.

---

## 5. 검증 (0단계에서 확립한 프로토콜 재사용)

각 커밋마다:

1. **CPU**: `make cpu` (기본 `/usr/bin/mpicxx` — ⚠️ Intel oneAPI 소싱 금지, OpenMPI 충돌) → `validate` 결정론 200스텝 → **필드덤프 8개 byte-비교**. namespace 변경은 코드젠에 영향이 없으므로 1바이트라도 다르면 실수다. (⚠️ 바이너리 md5 비교는 무의미 — 경로 문자열 등으로 원래 달라짐. 필드덤프가 진짜 게이트.)
2. **GPU**: gpu 노드에서 clean `make BACKEND=cuda apps` 에러 0 + `rbc_gpu_nob` np=2 스모크(32³×64, 20스텝, div~1e-12). 입출력은 공유 NFS 경로 사용, `mpirun --oversubscribe --bind-to none`.
3. N7 이후로는 `make lint`가 회귀 스크립트에 합류.

---

## 6. 기대 효과

- **정방향 추적**: 어떤 호출부든 `::`→`/` 치환만으로 정의 파일 디렉터리가 나온다. IDE 없이 grep/ls만으로 탐색 가능.
- **역방향 추적**: 파일을 열면 namespace 선언이 곧 "이 코드를 누가 어떤 이름으로 부르는가"를 말해준다.
- **grep 단일화**: `BoundaryCondition` 하나로 통일되어 Problem/Boundary/BoundaryCondition 3갈래 검색이 사라진다.
- **부수 효과**: 유형 A 별칭 헤더가 사라지면서 `core/`가 "자기 타입만 가진 진짜 core"가 되고, include 그래프에서 core→(grid,boundary,config,parallel) 역방향 의존 4건이 제거된다.
- 규약이 lint로 강제되므로, 이후 3·4단계와 신규 기여 코드가 다시 표류하지 않는다.

## 부록 — 근거 수치 (2026-07-24 grep 기준)

- 재수출 별칭 호출부: `core::Subdomain` 32, `core::BoundaryCondition` 37, `core::Grid` 16, `core::MpiContext` 11, `core::MpiTopology` 10, `core::FieldBoundary` 2 (합 ~108); 죽은 별칭: `core::Boundary`/`BcKind`/`AxisConfig`/`Config` 각 0
- 평탄 namespace 호출부: `equation::` 73, `physics::` 48 (src+apps, namespace 선언 제외)
- 앱은 이미 `grid::` 18, `config::` 18, `boundary::` 9곳을 실제 namespace로 호출 중 — 부분적으로는 이미 규약대로 사용되고 있음
- `using namespace`: 앱 main 6곳(합법화 대상) + `rbc_common.hpp:34`(헤더, 위반) + `scalar_gpu.cu` 함수-로컬 3곳(C3-②로 정리)
- namespace 선언 분포: `mpmstd::core` 32, `mpmstd::equation`(평탄) 24, `mpmstd::physics`(평탄) 24, `mpmstd::boundary` 20, `mpmstd::parallel::mpi` 16, `mpmstd::parallel::cuda_helpers` 16, `mpmstd::linear_solver::tdma` 14, 익명 21, `core::gpu` 4, `sk`/`mk` 각 1
