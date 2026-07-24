# MPM-STD_C Namespace Convention v1

> **한 줄 요약: `mpmstd::a::b::Name`을 보면 정의는 반드시 `src/a/b/` 안에 있다. 역으로 `src/a/b/`의 모든 선언은 `mpmstd::a::b`에 있다.** (`::`를 `/`로 바꾸면 디렉터리가 된다.)

근거·배경: [`Report/convenience/02_namespace_location_convention_plan.md`](../Report/convenience/02_namespace_location_convention_plan.md).
강제: `scripts/check_namespace_convention.sh` (`make lint`).

## 선언 규칙 (D)

- **D1. 경로 사상**: `src/<경로>/파일`의 모든 선언은 `namespace mpmstd::<경로>`에 둔다. 디렉터리 중첩 = namespace 중첩.
- **D2. `cpu/`·`gpu/` 잎 디렉터리는 namespace 레벨을 만들지 않는다.** 백엔드 구분은 함수 접미사 `_cpu`/`_gpu`가 담당. (예: `src/driver/cpu/cfl_cpu.cpp` → `mpmstd::driver`.)
- **D3. 허용되는 비(非)경로 잎은 두 개뿐**: `kernels`(방정식 모듈의 저수준 커널 계층), `detail`(헤더 내부 구현 전용). 그 외 비경로 레벨(`stencil` 등) 금지.
- **D4. 익명 namespace는 `.cpp`/`.cu` 파일-로컬 내부 구현에만.** 헤더 금지.
- **D5. 예외 — `src/common/`**: 프로젝트 공용 어휘(`real_t`, `Direction`, 매크로)는 `mpmstd::` 직속. "namespace 경로가 없는 이름 = `src/common/`". common엔 타입 별칭·enum·매크로 외 함수를 추가하지 않는다.

## 호출 규칙 (C)

- **C1. 다른 모듈의 이름은 항상 모듈 경로부터 자격(qualify)한다**: `parallel::mpi::Subdomain`, `grid::Grid`, `equation::scalar::solve_scalar_nob_cpu`. (내부에서 `mpmstd::` 접두는 생략 가능 — 상대 조회가 처리.)
- **C2. 타 모듈 이름을 자기 namespace로 들여오는 재수출(using-alias/using-declaration)은 금지.** (이번 정리의 발단 `core::Subdomain`이 그 사례.) 별칭은 **같은 모듈 내부 템플릿 인스턴스 축약**(`core::CpuFields = core::FieldStore<core::CpuField>`)에만 허용.
- **C3. `using namespace`는 두 곳에만**: ① 앱 `main` 소스 최상단 `using namespace mpmstd;` 1회, ② 함수 본문-로컬로 **같은 모듈** 하위 namespace 열기. 헤더 전면 금지.
- **C4. namespace 별칭 축약 금지** (`namespace mk = …;` 같은 2글자 별칭 금지 — 전체 경로 표기).

## 예시

```cpp
// namespace가 곧 경로: src/parallel/mpi/subdomain.hpp
parallel::mpi::Subdomain sub(topo, n_global);
// 온도 해석: src/equation/scalar/
equation::scalar::solve_scalar_nob_gpu(...);
// 부력: src/physics/buoyancy/
physics::buoyancy::add_buoyancy_force_nob_gpu(...);
```

## 이행 이력 (2026-07-24, N-시리즈)

- **N1** core/ 재수출 별칭 5헤더 제거(`core::Subdomain`→`parallel::mpi::Subdomain` 등 108곳) + `boundary::Problem`→`boundary::BoundaryCondition` + `core/mpi_topology.hpp`→`core/gpu_binding.hpp`.
- **N3** `equation` 평탄 → `equation::{momentum,scalar,pressure}`.
- **N4** `physics` 평탄 → `physics::{buoyancy,forcing,properties}`.
- **N5** `stencil` 잎 흡수, `sk`/`mk` 별칭 제거, `core::gpu`→`core`, `rbc_common.hpp` 헤더 using 정리.
- **N6** `src/parallel/cuda/`→`src/parallel/cuda_helpers/`.
- **N7** 본 문서 + `check_namespace_convention.sh` + `make lint`.

각 단계는 CPU `validate` 필드덤프 byte-비교로 로직 불변을 검증(namespace 변경은 코드젠 무관 → 1비트라도 다르면 실수). GPU측 `.cu`는 clean `make BACKEND=cuda apps` + rbc_gpu_nob 스모크로 별도 검증.
