# 0단계 실행 기록 — 죽은 코드 삭제

- 작성일: 2026-07-24
- 근거 계획: [`01_deabstraction_user_friendliness_plan.md`](01_deabstraction_user_friendliness_plan.md) §4 "0단계 — 삭제"
- 원칙: **검증 완료된 계산 로직(커널·밴드 조립·스텐실·시간전진 순서)은 한 줄도 바꾸지 않는다.** 허용 변경 = 삭제 · 죽은 include 제거 · 미사용 멤버 제거뿐.
- 결과: **37개 파일 삭제 + 헤더 5개 편집**, CPU 빌드 green, 로직 불변 수치 증명 완료(→ [`validate.md`](validate.md)).

---

## 1. 삭제 전 검증 방법 (참조 0 재확인)

계획서는 "조사에서 이미 참조 0 확인됨"이라 했지만, 과거 sed 사고·monitor.csv 파괴 전례가 있어 **삭제 직전 직접 재검증**했다. 검증 범위 = 빌드에 실제로 들어가는 트리(`src/` + `apps/` + `tests/`), 고아 `test/`(단수)는 제외.

### ⚠️ grep 함정 하나를 실제로 밟았다가 잡았음

첫 검증에서 `grep -e 'A|B|C'`를 **`-E` 없이** 써서 `|`가 리터럴로 취급됨 → 교체(alternation) 패턴이 전부 **거짓 음성**(항상 "매칭 없음")이었다. 모든 대상이 "clean"으로 나온 게 그래서였고, 이건 이 프로젝트가 경계하는 바로 그 종류의 실수다. **`grep -rEn`(확장 정규식)으로 전면 재검증**하여 실제 참조를 확인했다. 이때 계획서 §2.4가 옳았음(예: `boundary/main.hpp`가 실제로 `boundary_applier.hpp`를 include)도 드러났다.

### 확정된 참조 관계 (재검증 결과)

| 대상 | 실제 참조 | 조치 |
|---|---|---|
| `src/solve/*`, `core::Bands` | `tests/unit/test_smoke_cpu.cpp`(빌드됨) + `tests/spike/`(POC) 만 사용. `system.hpp:3`은 **bands.hpp를 include하지만 실제로 Bands를 안 씀**(구조체는 raw `vector A,B,C,D` 사용) → 죽은 include | solve/·bands.hpp 삭제, 스모크·spike 삭제, system.hpp include 제거 |
| `parallel/backend/` | `make_default_backend` 호출부는 고아 `test/`(미빌드) 14곳뿐. `parallel/main.hpp:4`가 우산으로 include | backend/ 삭제, parallel/main.hpp include 제거 |
| `boundary/boundary_applier.*` | 코드 참조는 `boundary/main.hpp:9` include 1곳 + 주석 몇 개. (`boundary/main.hpp`는 아무도 include 안 하는 미사용 우산) | boundary_applier 삭제, boundary/main.hpp include 제거 |
| `driver/monitor·restart` | `print_monitor_cpu`/`restart_due_cpu` 호출부 0 (앱이 자체 인라인) | .hpp + cpu/*.cpp 삭제 |
| `post/diagnostics` | 대문자 `compute_Re_delta_star`/`compute_center_temp` 스텁, 호출부 0. 살아있는 건 `statistics`의 소문자 `compute_re_delta_star` | .hpp + cpu + gpu 삭제 |
| `physics/les`, `physics/ibm` | 참조 0 (TODO no-op) | 삭제 |
| `filtered_tdma_cpu_backend.hpp` | 참조 0, `tdma/main.hpp`도 include 안 함 | 삭제 |
| `TdmaSolver::set_rho/set_eps_constant` | 삭제되는 FilteredTDMA만 override, 기본 no-op, 호출부 0. Pascal 백엔드는 override 안 함 | 제거 |
| `PressureSystem::transform` + `enum Transform` | `.transform`/`->transform` 접근 0 (Fft/Dct 미구현 자리표시자) | 제거 |
| `bc_kind.hpp` `Wall/Inflow/Outflow` | 삭제되는 boundary_applier + 자기 `name()` 스위치에서만 사용. TOML `z="wall"`은 **별도 enum** `AxisKind::Wall`(problem_loader)을 거쳐 per-field Dirichlet/Neumann로 매핑 → `BcKind::Wall`은 미사용 예약 슬롯 | 제거 |
| `apps/rbc/main_gpu_{ob,nob}.cpp` | Makefile은 `.cu`를 빌드(줄 170-180). `.cpp`는 빌드 안 됨 | 삭제 |

---

## 2. 삭제한 파일 (37개, git 추적 → 복구 가능)

```
# 죽은 수치 계층
src/solve/banded_solver.hpp
src/solve/poisson_solver.hpp
src/solve/cpu/banded_solver_cpu.cpp
src/solve/cpu/poisson_solver_cpu.cpp
src/solve/gpu/poisson_solver_gpu.cu
src/core/bands.hpp
src/linear_solver/tdma/filtered_tdma_cpu_backend.hpp
src/linear_solver/fft/                     # 빈 디렉터리(추적 파일 없음)

# 미사용 런타임-다형성 계층 (프로덕션 커널은 통과하지 않음)
src/parallel/backend/backend.hpp
src/parallel/backend/cpu_backend.hpp
src/parallel/backend/cpu_backend.cpp
src/parallel/backend/cuda_backend.hpp
src/parallel/backend/cuda_backend.cpp
src/parallel/backend/main.hpp

# TODO no-op 스텁
src/physics/les/les.hpp
src/physics/les/cpu/les_cpu.cpp
src/physics/les/gpu/les_gpu.cu
src/physics/ibm/ibm.hpp
src/physics/ibm/cpu/ibm_cpu.cpp
src/physics/ibm/gpu/ibm_gpu.cu

# 호출부 0 (앱이 자체 인라인)
src/driver/monitor.hpp
src/driver/restart.hpp
src/driver/cpu/monitor_cpu.cpp
src/driver/cpu/restart_cpu.cpp

# 죽은 대문자 진단 스텁 (살아있는 건 statistics의 소문자판)
src/post/diagnostics.hpp
src/post/cpu/diagnostics_cpu.cpp
src/post/gpu/diagnostics_gpu.cu

# 고아 BC 시스템
src/boundary/boundary_applier.hpp
src/boundary/boundary_applier.cpp

# 빌드 안 되는 죽은 main (GPU는 .cu를 빌드)
apps/rbc/main_gpu_ob.cpp
apps/rbc/main_gpu_nob.cpp

# 삭제된 solve/Bands에 의존하던 스모크·POC 테스트
tests/unit/test_smoke_cpu.cpp
tests/spike/Makefile
tests/spike/README.md
tests/spike/banded_poc.cpp
tests/spike/halo_poc.cpp
tests/spike/link_gpu.cpp
tests/spike/spike_gpu.sh

# (추적 안 되던 빌드 산출물도 정리) test_smoke_cpu.o, tests/spike/*.o
```

---

## 3. 편집한 헤더 5개 (죽은 include · 미사용 멤버 제거)

삭제된 파일을 가리키던 매달린(dangling) include와, 그와 함께 죽은 멤버를 제거. **sed 일괄 금지, Edit(정확 문자열 치환)으로 수행.**

| 파일 | 변경 |
|---|---|
| `src/core/system.hpp` | ① 죽은 `#include "core/bands.hpp"` 제거(구조체가 Bands 미사용) ② 미사용 `enum class Transform { Fft, Dct, Tdma }` 제거 ③ `PressureSystem`의 미사용 멤버 `std::array<Transform,3> transform{...}` 제거 + 주석 정리 |
| `src/parallel/main.hpp` | 삭제된 `parallel/backend/main.hpp` include 제거 |
| `src/boundary/main.hpp` | 삭제된 `boundary/boundary_applier.hpp` include 제거 |
| `src/linear_solver/tdma/tdma_solver.hpp` | 미사용 가상 메서드 `set_rho`/`set_eps_constant`(기본 no-op) 제거 |
| `src/boundary/bc_kind.hpp` | 예약 슬롯 `Wall/Inflow/Outflow` enum 값 + `bc_kind_name()`의 해당 case 제거, 상단 주석 정리 |

> **참고 — 왜 이 편집들이 로직을 안 바꾸는가:** 제거한 것은 전부 "미참조"임을 grep으로 확인했고, 더 결정적으로 **빌드가 성공했다는 것 자체가 컴파일러/링커가 미참조를 확인해준 것**이다(참조됐다면 undefined reference / no member 에러가 났을 것). 미사용 구조체 멤버·미사용 가상 메서드를 지우면 **바이너리 레이아웃**(구조체 크기·vtable 크기)은 바뀌지만 **런타임 동작은 불변**이다. 이 불변성을 수치로 증명한 것이 [`validate.md`](validate.md).

---

## 4. 빌드 결과

```bash
# ⚠️ Intel oneAPI를 소싱하지 말 것 (아래 §6)
cd MPM-STD_C
make clean
make BACKEND=cpu all
# → LIB libmpmstd.a + 7개 앱(channel cavity dhvc rbc validate rbc_cpu_ob rbc_cpu_nob) 전부 링크, EXIT 0
```

라이브러리 전체가 깨끗이 재컴파일되고 7개 앱이 모두 링크됨. `make clean`을 먼저 한 이유: `ar rcs`는 아카이브에서 사라진 소스의 스테일 오브젝트를 자동 제거하지 않으므로, 삭제한 `.o`(diagnostics_cpu.o 등)가 lib에 남지 않게 하기 위함.

---

## 5. 이번 단계에서 **하지 않은 것** (의도적)

| 항목 | 이유 |
|---|---|
| `apps/rbc/main.cpp:50-122` 헬퍼 4개 dedup | 이건 "죽은 코드"가 아니라 **활성 코드 중복**(`rbc` 바이너리가 실제 사용). 게다가 `main.cpp`는 3단계에서 통째로 삭제 예정 → 곧 삭제될 파일을 위험하게 편집할 이유 없음. **3단계로 이월.** |
| `log/*.out` (SLURM 로그 9개) | 죽은 코드가 아니라 실험 로그. 삭제는 사용자 판단. |
| 고아 `test/`(단수, 알짜 테스트 21개) | 빌드에 없어 깨지지 않음. 삭제 심볼(make_default_backend/solve_banded/boundary_applier)을 참조하나, 계획서대로 **4단계에서 병합·수정**. 지금 지우면 살릴 테스트를 잃음. |
| GPU 빌드 검증 | 로그인 노드는 `.cu` 컴파일 불가. 남은 `.cu`/`.cuh`가 삭제 헤더를 참조하지 않음은 grep으로 확인했으나, `make BACKEND=cuda`는 **GPU 노드 sbatch 필요**. |

---

## 6. 환경 주의 — CPU 빌드에 Intel oneAPI 소싱 금지

`CXX := mpicxx` = `/usr/bin/mpicxx`(**시스템 OpenMPI**, `libmpi.so.40`). FFTW3·PaScaL_TDMA도 이 OpenMPI에 링크됨. 만약 `source /opt/intel/oneapi/setvars.sh`(FDS 프로젝트용)를 소싱하면 Intel MPI(`libmpi.so.12`)가 PATH·링크 경로에 끼어들어:

```
undefined reference to symbol 'ompi_mpi_double'   # OpenMPI 심볼인데 Intel libmpi엔 없음
libmpi.so.40: DSO missing from command line
```

로 **앱 링크가 실패**한다(라이브러리 컴파일은 통과하므로 놓치기 쉬움). 기본 `/usr/bin/mpicxx`로 빌드할 것.

---

## 7. 다음 단계

계획서 §6 우선순위대로 **0→1→4가 사실상 무위험**. 0단계가 끝났으니 다음은 **1단계(이름·우산 헤더 정리)** 또는 **4단계(설정·빌드·문서)**. 2·3단계는 §7 회귀 체계(→ [`validate.md`](validate.md)의 방법을 스크립트화)를 갖춘 뒤 진행 권장.
