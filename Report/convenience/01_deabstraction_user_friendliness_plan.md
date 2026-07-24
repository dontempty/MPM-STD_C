# MPM-STD_C 사용자 편의성 개선 보고서 — 추상화 완화 계획

- 작성일: 2026-07-24
- 전제: **channel / DHVC / RBC 3개 케이스에서 검증 완료된 계산 로직(커널·수치 기법)은 한 줄도 바꾸지 않는다.**
  본 보고서의 모든 제안은 "코드를 지우거나, 이름을 바꾸거나, 같은 코드를 한 곳으로 옮기는" 변경만 포함한다.
- 근거 조사: src 전체(~14,100줄, ~130파일) + apps(~3,600줄) + 빌드/설정/문서/스크립트 전수 탐색.

---

## 1. 요약 (TL;DR)

코드가 어려운 이유는 "추상화가 정교해서"가 아니라, **절반쯤 지은 추상화의 잔해**가 실제 사용 경로 위에 겹쳐져 있기 때문이다. 조사 결과:

1. **죽은 코드가 사용 경로를 가리고 있다.** `src/solve/` 전체, `physics/les`, `physics/ibm`, `driver/monitor·restart`, `post/diagnostics` 스텁, `boundary/boundary_applier`(고아 BC 시스템), `FilteredTDMACpuBackend`(구현부 없는 헤더), `parallel/backend/`(테스트에서만 호출), RBC 죽은 main 2개 등 — **약 1,500~2,000줄이 아무 계산에도 쓰이지 않으면서** 독자가 "이게 진짜 경로인가?"를 매번 판별하게 만든다.
2. **구현이 하나뿐인 가상 인터페이스**가 간접 참조를 늘린다. 압력 해석은 호출부에서 실제 루프까지 **약 7단계(가상 호출 2겹)**를 따라가야 한다.
3. **앱 main이 5+개 케이스 × (CPU/GPU) × (OB/NOB)로 복붙**되어 있다. 전형적인 RBC main 130줄 중 **90~100줄이 복붙 보일러플레이트**이고, `apps/rbc/`에는 빌드조차 되지 않는 main이 2개 있다.
4. **사용자 표면(빌드·입력·문서)이 코드보다 더 큰 진입 장벽**이다. 의존성 경로가 개인 홈 디렉터리로 하드코딩되어 있고, `.toml` 확장자를 쓰지만 실제 파서는 INI이며, readme가 존재하지 않는 파일·키를 안내한다.

권장 실행 순서는 **삭제(0단계) → 이름·헤더 정리(1단계) → 단일구현 인터페이스 평탄화(2단계) → 앱 main 통합(3단계) → 설정·문서 정비(4단계)**이며, 각 단계 후 3케이스 회귀 비교로 "로직 불변"을 기계적으로 보증한다(§7).

---

## 2. 현황 진단

### 2.1 죽은 코드 — 삭제해도 아무 계산이 안 바뀌는 것들

| 대상 | 규모 | 근거 |
|---|---|---|
| `src/solve/` 계층 전체 (`banded_solver`, `poisson_solver`, `core::Bands`) | ~90줄 + 헤더 | `poisson_solver_cpu.cpp:5`는 `/* TODO(P3) */` no-op. `solve_banded_cpu` 호출자 0. 실제 압력은 `equation/pressure/`가 담당 |
| `src/linear_solver/tdma/filtered_tdma_cpu_backend.hpp` | 헤더 1개 | `.cpp` 구현부가 아예 없고, 참조 0 |
| `TdmaSolver::set_rho / set_eps_constant` | 인터페이스 2메서드 | 위 죽은 백엔드만을 위해 존재하는 기본 no-op (`tdma_solver.hpp:38-40`) |
| `src/parallel/backend/` (`Backend`/`CpuBackend`/`CudaBackend`/팩토리) | ~154줄 | `make_default_backend()` 호출처가 **테스트뿐**. 커널은 이 계층을 통과하지 않음 (`backend.hpp:10-14` 주석 스스로 인정). `CudaBackend`는 "M5'에서 구현 예정" 셸 |
| `src/physics/les/`, `src/physics/ibm/` | ~28줄 | 전부 `/* TODO(P9|P10) */` no-op, 호출자 0 |
| `src/driver/monitor.hpp`, `src/driver/restart.hpp` | ~40줄 | 호출자 0 — 앱들이 자체 `div_max()`/printf를 인라인함 |
| `src/post/diagnostics.hpp`의 `compute_Re_delta_star_cpu`(대문자 R)/`compute_center_temp_cpu` | ~26줄 | `out=0; TODO` 스텁. 살아있는 함수는 소문자 `compute_re_delta_star_cpu`(`post/statistics.hpp:30`) — **살아있는 함수와 죽은 함수가 대소문자만 다른 이름으로 공존** |
| `src/boundary/boundary_applier.*` | ~340줄 | 존재하지 않는 헤더(`field/scalar_field.hpp`)에 의존하는 고아 BC 시스템. `test/`(고아 테스트 트리)만 사용. 런타임 BC는 `core::apply_ghost_cpu / modify_tdma_row_cpu / sync_field_cpu` |
| `apps/rbc/main_gpu_ob.cpp`, `main_gpu_nob.cpp` | ~320줄 | Makefile 어디에서도 빌드하지 않는 죽은 main (GPU 타깃은 `.cu` 쪽을 빌드) |
| `apps/rbc/main.cpp:50-122`의 헬퍼 4개 | ~75줄 | `rbc_common.hpp:38-106`과 **글자 단위 동일 복사본** |
| `src/linear_solver/fft/` 빈 디렉터리, `core::PressureSystem::transform` enum(`Dct` 미구현), `bc_kind.hpp:13-23`의 `Wall/Inflow/Outflow` 예약 슬롯 | — | 전부 만들다 만 자리표시자 |
| 루트 `test_smoke_cpu.o`, `tests/spike/*.o`, `build/`의 마이크로벤치 바이너리, `log/`의 SLURM 산출물 | — | 잡동사니 |

**합계: 계산 결과에 영향 없이 삭제 가능한 코드가 약 1,500~2,000줄.** 이것이 독해 난이도의 1차 원인이다 — 새 독자는 "두 개의 BC 시스템 중 어느 쪽이 진짜인가", "`solve/`와 `equation/pressure/` 중 어디가 압력인가", "대문자 `Re_delta_star`와 소문자 중 뭘 불러야 하나"를 스스로 판별해야 한다.

### 2.2 구현이 하나뿐인 인터페이스 (speculative generality)

| 인터페이스 | 실제 구현 수 | 문제 |
|---|---|---|
| `PressureSolverBase` (`pressure_base.hpp:22`) | 1 (`PressureSolver`) | 주석은 "FFT and DCT solvers"를 약속하지만 DCT는 없음. 게다가 GPU의 `PressureSolverGpu`(`pressure_engine_gpu.cuh:265`)는 이 베이스를 **상속하지 않는** 별개 클래스 — CPU/GPU 통일에도 실패한 단일 구현 가상 베이스 |
| `parallel::Backend` | 빌드당 1 (`#ifdef` 선택) | 런타임 다형성이 실제로 일어나지 않음. 게다가 프로덕션 호출자 0 (§2.1) |
| `TdmaSolver` 가상 + `TdmaRegistry` | 2 (CPU/CUDA, `#ifdef`로 컴파일 시 결정) | 유일하게 실제로 쓰이는 가상 계층. 그런데 GPU 경로는 `static_cast`로 구체 타입으로 되돌려 씀(`scalar_gpu.cu:250`, `momentum_gpu.cu:1319`) — **가상 인터페이스가 스스로 무력화됨**. 컴파일 시 이미 결정되므로 빌드별 구체 타입 별칭으로 충분 |

### 2.3 간접 참조 깊이 — "호출부에서 실제 루프까지"

압력 해석 체인 (7단계, 가상 호출 2겹):

```
apps/rbc/main.cpp  equation::solve_pressure_cpu(...)
 → pressure/cpu/pressure_cpu.cpp:83   지연 생성된 make_shared<PressureSolver> 를 PressureSystem::engine 에 저장
 → pressure_cpu.cpp:94                poi.engine->solve(...)          ← 가상 ①  (PressureSolverBase)
 → pressure_engine.hpp:39             PressureSolver::solve → transpose/fft/solve_tdma_z_
 → pressure_engine.hpp:110            tdma_z_->solve_many(...)        ← 가상 ②  (TdmaSolver)
 → pascal_tdma_cpu_backend.cpp        get_or_create_plan → plan.solver->solve
 → 외부 라이브러리 PaScaL_TDMA_C      실제 전진소거/후진대입 루프
```

스칼라(에너지) ADI도 유사하게 5~6단계다. 각 단계가 개별적으로는 정당해 보여도, 합쳐지면 "온도 확산이 어디서 계산되는지"를 찾는 데 파일 6개를 열어야 한다.

### 2.4 이름 문제

- **`System`** (`core/system.hpp`) — "한 방정식의 Ax=b 작업공간(RHS+밴드+핑퐁 버퍼)"이라는 뜻인데 이름에서 전혀 유추 불가. `PressureSystem`은 `shared_ptr<void>` 3개(`engine_gpu`, `dPhat_gpu_field`, `dP0_gpu_field`)를 타입 소거로 들고 있어 더 불투명.
- **`engine` / `engine_gpu`** — 압력 FFT/전치/TDMA 솔버 객체. "engine"은 아무 정보가 없다.
- **`Domain`** (`core/domain.hpp`) — 물리 도메인이 아니라 `grid+topo+sub+tdma+gpu_grid` 참조 묶음. `gpu_grid`는 `const void*`.
- **`main.hpp` 우산 헤더 8개+** — `parallel/main.hpp`, `mpi/main.hpp`, `cuda/main.hpp`, `backend/main.hpp`, `tdma/main.hpp`, `linear_solver/main.hpp`, `config/main.hpp`, `grid/main.hpp`, `boundary/main.hpp`(죽은 `boundary_applier.hpp`를 끌고 들어옴)… 엔트리 포인트와 이름이 충돌하고 무엇이 포함되는지 숨긴다. 일부는 존재하지 않는 하위 헤더("fft/main.hpp added in M4")를 주석으로 약속.
- **OB/NOB 명명 불일치** — 스칼라는 `_ob`/`_nob`, 모멘텀은 `_const_visc`/`_var_visc`, DHVC는 물성≡1을 채워 `_nob`/`_var` 경로로 OB를 돌린다(`apps/dhvc/main.cpp:129-143`). "`_var` 커널이 물성 1이면 OB와 동일"이라는 사실은 `momentum/kernels.hpp:38-70` 주석을 읽어야만 알 수 있다 — **이름이 실제 사용법을 배반하는 대표 사례**.
- **두 개의 무관한 "Backend" 계열** — `parallel::CpuBackend/CudaBackend` vs `PaScaLTDMACpuBackend/CudaBackend`.
- 주석 전반의 `P0…P10`, `M2…M5'`, `rev.2 §`, `Report/30` 같은 내부 로드맵 태그 — 신규 독자에게는 해독 불가 소음.

### 2.5 앱 main 복붙 (정량)

- 모든 main이 **동일한 ~40줄 셋업 서문**(MPI→logger→config→subdomain→grid→BC→TDMA→domain→필드 5개 할당)을 복붙 (rbc `main.cpp:126-171`, dhvc, channel, validate 전부 동일 확인).
- `apps/rbc/`에 **main 7개**: 모놀리스 `main.cpp`(427줄, `#ifdef`+`if(nob)`로 4조합 전부 내장), 분리형 4개, 죽은 것 2개. `main_cpu_ob.cpp` vs `main_cpu_nob.cpp`는 118/142줄 중 **차이가 ~25줄뿐**.
- 실제 프로덕션 바이너리는 `rbc_gpu_nob`(=`main_gpu_nob.cu`, 714줄)뿐인데, restart/checkpoint·`monitor.csv`·`stats_t*.dat`·T-clip 등 **핵심 기능이 이 파일에만 있다**. 즉 "대표"처럼 보이는 `main.cpp`와 나머지 4개는 기능이 뒤처진 사실상의 사본.
- `apps/dhvc/main.cpp`(259줄)와 `main_gpu.cu`(742줄)도 같은 케이스의 손 유지 이중본.
- 시간 루프의 호출 순서(CFL→스칼라→모멘텀 조립→부력→예측압력구배→모멘텀 해석→update/sync→압력→누적)는 모든 케이스에서 불변인데, **5+개 main에 손으로 재타이핑**되어 있고 OB/NOB 분기(`if(nob) … else …`)가 매 호출부마다 반복된다(rbc `main.cpp:328-349`).

### 2.6 설정 시스템

- 파일은 `*.toml`인데 파서는 손수 만든 **INI 리더** (`config.hpp:14-27` 스스로 "Minimal INI-style" 명시). TOML 배열·중첩 테이블·인라인 테이블은 **조용히 오파싱**된다. `[bc.z.minus]`가 동작하는 것도 평평한 섹션명으로 취급되기 때문.
- 기본값은 소스의 `cfg.get_or<T>(…, default)` 3번째 인자를 grep해야만 발견 가능하고, **바이너리마다 다르다** (예: 통계 시작이 `main.cpp:193`은 `nstat_start=20000` 스텝 기반, `main_gpu_nob.cu`는 `t_stat_start=7500.0` 시간 기반 — 같은 개념, 다른 키, 다른 기본값).
- 섹션명 오타 → 조용히 기본값 사용, 경고 없음 (`config.cpp:91-98`). 미지 키 검출·`np1*np2*np3 == mpirun -np` 교차 검증 없음(불일치 시 조용히 잘못 돎).
- 키 레퍼런스 문서가 없다. readme §7의 키 목록은 실제와 불일치 — RBC가 읽지 않는 `nstat`/`warmstart_io` 등을 싣고, 실제로 읽는 `t_stat_out`/`nrestart_write`/`stats_reset`/`gravity_axis`는 누락.

### 2.7 빌드·실행·문서·트리 위생

- **의존성 경로 하드코딩**: `Makefile.inc:16` `PASCAL_TDMA_DIR=/shared/home/wel1come1234/...`, `:20` FFTW3 — 새 사용자는 첫 `make`에서 즉시 막힘.
- 기본 입력 `apps/rbc/input.toml`이 **존재하지 않고**, 실험 코드명이 붙은 TOML 9개(`input_p8_deltat20_256_cold_cfl03.toml` 등)만 있으며 설명 색인이 없다.
- CUDA 빌드는 nvc++ ICE 회피를 위해 `cpu/*.cpp`를 `-DMPMSTD_BACKEND_CPU`로 컴파일하면서 `tdma_registry.cpp` 등 2개 파일만 수동으로 `-DMPMSTD_BACKEND_CUDA`를 강제하는 **파일별 예외 패치** 상태 (`Makefile:60-107`) — 한 TU의 매크로와 링크되는 레지스트리의 매크로가 서로 다른 취약 구조.
- **`test/`(단수) vs `tests/`(복수)**: 루트 Makefile은 `tests/`(사실상 스모크 1개)만 빌드하고, 풍부한 단위·통합 테스트 21개가 있는 `test/`는 자체 Makefile을 가진 채 **고아** 상태. readme §11은 `test/`의 내용물을 `tests/` 것처럼 설명.
- 낡은 문서: `docs/PROGRESS.md`는 이미 출하된 P6/P7/P8을 "⏳ 미완"으로 표시, `docs/REFACTOR_PLAN.md`는 끝난 이주를 미래형으로 서술, readme §7은 존재하지 않는 `plot_rbc_fig9.py`를 안내, `OPTIMIZATIONS.md`는 현재와 다른 경로 참조.
- **실행 산출물이 소스 트리 안으로** 쓰인다(`dir_stats = "apps/rbc/..."`). 그 결과 `apps/rbc/` 아래 수십 개의 결과 디렉터리·백업이 소스와 섞여 있고, `apps/channel/turb_out/`과 `turb_out_old/`가 동일 파일로 공존.
- 출력 파일(`monitor.csv`, `stats_t*.dat`)의 **컬럼 정의 문서가 없다** — `main_gpu_nob.cu:299-326`을 읽어야 열 순서를 알 수 있다.

---

## 3. 개선 원칙

1. **계산 로직 동결.** 커널 함수 본문, 밴드 조립식, 스텐실, 시간 전진 순서는 수정 금지. 허용되는 변경은 삭제·이름 변경·파일 이동·호출부 정리·문서화뿐.
2. **추상화는 "구현이 2개 이상 실재할 때만"** 남긴다. 구현 1개짜리 가상 베이스는 구체 클래스로 병합.
3. **"데이터=struct, 알고리즘=자유 함수 + `_cpu`/`_gpu` 접미사"라는 현재의 기본 설계는 유지한다.** 이 패턴 자체는 읽기 쉽다. 문제는 그 위에 얹힌 잔해다. (새로운 `TimeStepper` 클래스 같은 추상화를 **추가하지 않는다** — 명시적 시간 루프가 이 프로젝트의 독해 편의에 오히려 유리.)
4. **축약 없는 완전한 이름** (기존 팀 규칙과 동일: `domain`, `momentum`, `pressure` 등 완전 명칭).
5. 각 단계는 독립적으로 머지 가능해야 하며, 단계마다 3케이스 회귀(§7)를 통과해야 다음 단계로 간다.

---

## 4. 단계별 실행 계획

### 0단계 — 삭제 (위험도: 없음, 효과: 즉시)

§2.1 표의 항목 전부. 구체적으로:

```
삭제:  src/solve/                              (banded_solver, poisson_solver, core/bands.hpp 포함)
삭제:  src/linear_solver/fft/                  (빈 디렉터리)
삭제:  src/linear_solver/tdma/filtered_tdma_cpu_backend.hpp
삭제:  src/parallel/backend/                   (테스트의 make_default_backend 호출부 함께 정리)
삭제:  src/physics/les/  src/physics/ibm/      (TODO no-op 스텁)
삭제:  src/driver/monitor.hpp  src/driver/restart.hpp   (호출자 0)
삭제:  src/post/diagnostics.hpp + cpu/gpu 스텁 (죽은 대문자 Re_delta_star 쌍)
삭제:  src/boundary/boundary_applier.*         (고아 BC — test/ 정리와 함께)
삭제:  apps/rbc/main_gpu_ob.cpp  main_gpu_nob.cpp        (빌드되지 않는 main)
삭제:  apps/rbc/main.cpp:50-122 헬퍼 4개 → rbc_common.hpp 사용으로 대체
삭제:  TdmaSolver::set_rho / set_eps_constant, PressureSystem::transform enum,
       bc_kind.hpp 의 Wall/Inflow/Outflow 예약 슬롯
정리:  루트 test_smoke_cpu.o, tests/spike/*.o, log/*.out, build/ 마이크로벤치 바이너리
```

예상 감소: **소스 ~1,500줄 + 트리 잡동사니 다수.** 삭제 전 `git grep`으로 참조 0 재확인(조사에서 이미 확인됨) 후, 각 항목을 **개별 커밋**으로 지워 되돌리기 쉽게 한다.

### 1단계 — 이름·헤더 정리 (위험도: 낮음 — 기계적 rename)

| 현재 | 변경안 | 이유 |
|---|---|---|
| `ScalarSystemT` / `MomentumSystemT` / `PressureSystem` | `ScalarLinearSystem` / `MomentumLinearSystem` / `PressureLinearSystem` (또는 `*Workspace`) | "Ax=b 작업공간"임이 이름에 드러나게 |
| `PressureSystem::engine` | `pressure_solver` | 실체 그대로 |
| 우산 헤더 `*/main.hpp` (8개+) | 삭제하고 호출부에서 **명시적 include** (또는 최소한 `mpi/mpi.hpp`처럼 모듈명 헤더로 개명) | "main.hpp"는 엔트리 포인트와 충돌하고 의존성을 숨김 |
| 스칼라 `_ob`/`_nob` vs 모멘텀 `_const_visc`/`_var_visc` | 한 규칙으로 통일: **`_constant_property` / `_variable_property`** (물리 모델명 OB/NOB는 앱·문서 층위에만) | "`_var`+물성1 = OB"라는 함정 제거. 커널명은 수치적 사실만 말하게 |
| `parallel::*Backend` (0단계에서 삭제) | — | TDMA 쪽 `*Backend`만 남아 충돌 해소 |
| 주석의 `P0…P10 / M2…M5' / rev.2 §` 태그 | 제거 또는 `docs/history.md`로 이동 | 신규 독자 소음 제거 |

rename은 `sed` 일괄이 아니라 컴파일러를 이용해(옛 이름 삭제 → 빌드 에러 지점 수정) 진행한다. ※ 과거 `%d→%domain` sed 사고 전례 있음.

### 2단계 — 단일 구현 인터페이스 평탄화 (위험도: 중간, 로직 이동 없음)

1. **`PressureSolverBase` 제거** — 유일 파생 `PressureSolver`를 구체 클래스로. `pressure_cpu.cpp:94`의 가상 호출 1겹이 사라지고, 압력 체인이 7→6단계.
2. **`TdmaSolver` 가상 계층을 컴파일 시 별칭으로** — 선택이 이미 `#ifdef`(`tdma_registry.cpp:13-17`)로 컴파일 시 확정되고, GPU 경로는 이미 `static_cast`로 구체 타입을 쓴다. `using TdmaBackend = PaScaLTDMACpuBackend;`(빌드별) + `TdmaRegistry`는 그 구체 타입 3개(축별) 보관으로 단순화. 가상 호출 1겹 추가 제거. *(선택적 — 부담되면 보류 가능. 다만 `static_cast` 우회가 이미 있는 만큼 정직해지는 쪽을 권장.)*
3. `PressureSystem`의 `shared_ptr<void>` 3개 — GPU 빌드 전용 구조라면 최소한 필드명에 실체를 쓰고(`pressure_engine_gpu` 등), 주석으로 실제 타입과 캐스팅 지점을 명시. (타입 소거 해소는 헤더 의존성 재설계가 필요하므로 이름·주석만으로 1차 방어.)

### 3단계 — 앱 main 통합 (위험도: 중간, 효과: 사용자 체감 최대)

원칙: **시간 루프는 각 main에 명시적으로 남긴다** (그게 이 프로젝트에서 가장 읽기 좋은 형태). 통합하는 것은 루프가 아니라 **루프 밖 보일러플레이트와 케이스 변형 수**다.

1. **공통 셋업 헬퍼 1개**: 모든 main의 동일한 ~40줄 서문을 `apps/common/setup.hpp`의 `SimulationSetup setup_from_config(argc, argv)` 하나로. 반환 struct에 ctx/logger/grid/bc/domain/fields. 각 main은 `auto sim = setup_from_config(...);` 한 줄 + 케이스 고유 IC/물성만 남는다. (main당 ~40줄 → 케이스 고유 ~30줄.)
2. **RBC main 7개 → 1개**: 기능이 가장 완전한 `main_gpu_nob.cu`(restart·monitor.csv·stats 보유)를 기준으로 삼고, OB/NOB는 지금처럼 런타임 `nob` 플래그, CPU/GPU는 지금처럼 `#ifdef MPMSTD_BACKEND_CUDA`로 한 파일에서 분기. 모놀리스 `main.cpp`와 분리형 4개는 폐기. `diff`가 ~25줄뿐인 파일들을 별도로 유지할 이유가 없고, **기능 격차(다른 main엔 restart가 없음)도 자동 해소**된다.
3. **DHVC도 동일**: `main.cpp`/`main_gpu.cu` 이중본 → `#ifdef` 한 파일.
4. OB/NOB 분기 반복(`if(nob) solve_x_nob else solve_x_ob` × 6곳)은 그대로 두되 **한 파일에만** 존재하게 되므로 자연히 완화. (분기를 함수 포인터·전략 객체로 감싸는 것은 새 추상화이므로 하지 않는다.)

예상: apps ~3,600줄(+dhvc_gpu 742) → **~1,200줄**, 바이너리 수와 기능 격차 소멸.

### 4단계 — 설정·빌드·문서·트리 (위험도: 낮음, 신규 사용자 체감 최대)

1. **설정 정직화** — 둘 중 하나를 택일:
   - (a) 확장자를 `.ini`로 바꾸고 문서에서 "TOML" 표기 제거 (변경 최소), 또는
   - (b) toml11 같은 단일 헤더 TOML 파서로 교체 (`config.hpp:27` 주석이 이미 예정)
   - 어느 쪽이든: **미지 키 경고**(오타 방어), **`np1*np2*np3` 검증**(assert + 명확한 에러 메시지), 시작 시 **유효 설정값 전체를 로그로 에코**(기본값이 뭐였는지 grep 없이 확인 가능).
2. **설정 키 레퍼런스 1문서** — `docs/config_reference.md`: 섹션·키·타입·기본값·읽는 바이너리를 표로. 통계 시작 키(`nstat_start` vs `t_stat_start`)처럼 바이너리마다 갈라진 키는 이 기회에 하나로 통일.
3. **케이스별 주석 달린 템플릿 입력** — `apps/rbc/input.toml`(기본), `apps/channel/input.toml`, `apps/dhvc/input.toml`. 실험 코드명 TOML 9개는 `apps/rbc/inputs_archive/`로 이동 + 한 줄 색인 README.
4. **빌드 이식성** — `Makefile.inc`의 하드코딩 경로를 `PASCAL_TDMA_DIR ?=` 환경변수 우선 + 미설정 시 **명확한 에러 메시지**("export PASCAL_TDMA_DIR=... 하세요")로. readme 상단에 5줄짜리 Quick Start(모듈 로드→export→make→mpirun) 추가.
5. **출력 문서화** — `monitor.csv`/`stats_t*.dat` 컬럼 정의를 `docs/output_formats.md`에, 그리고 **파일 첫 줄에 헤더 행 출력**(기존 파일 파서 호환이 걱정되면 `#` 주석 행으로).
6. **출력을 소스 트리 밖으로** — 템플릿 입력의 `dir_stats`/`dir_out` 기본을 `runs/<case>/`로. 기존 `apps/rbc/` 아래 결과 디렉터리·백업은 `runs/archive/`로 이동.
7. **테스트 트리 단일화** — 고아 `test/`(알짜 21개)를 루트 Makefile 타깃으로 편입하고 `tests/`의 스모크·회귀와 병합, 한 디렉터리로. `boundary_applier` 의존 테스트는 0단계에서 함께 폐기.
8. **문서 현행화** — `docs/PROGRESS.md`·`REFACTOR_PLAN.md`·`OPTIMIZATIONS.md`는 `docs/history/`로 이동(폐기 대신 보존), readme의 죽은 참조(§7 plot 스크립트, §9 스크립트 경로, §11 테스트 설명, §7 키 목록) 수정.

---

## 5. 하지 않기로 제안하는 것

- **CPU/GPU 커널 본문 통합** (공유 로직을 `MPMSTD_HD` 헤더로 올리는 팀 규칙 적용): 방향은 옳지만 커널 코드 이동은 "계산 로직 불변" 원칙과 충돌 위험이 있다. 이번 정리 범위에서 제외하고, 이후 별도 과제로(케이스별 byte/허용오차 회귀를 갖춘 뒤) 진행 권장. 특히 CPU와 GPU의 OB 스칼라 스킴이 현재 **의도적으로 다르다**(CPU=완전 명시, GPU=var-스텐실 준음해)는 점 때문에, 섣부른 통합은 결과를 바꾼다.
- **TimeStepper/Simulation 클래스 도입**: 새 추상화 추가는 이번 목표(추상화 완화)와 정반대. 명시적 시간 루프 유지.
- **config 기반 케이스 레지스트리**("한 바이너리가 모든 케이스"): 케이스=main 파일이라는 현 구조가 오히려 이해하기 쉽다. main 수만 줄인다(3단계).
- **CUDA 빌드의 파일별 매크로 패치 해소**(`Makefile:60-107`): nvc++ ICE라는 외부 제약이 원인이므로, 지금은 Makefile 주석으로 이유를 남기는 선에서 유지. 컴파일러 업데이트 시 재검토.

---

## 6. 우선순위·효과 요약

| 단계 | 작업 | 위험 | 사용자 체감 | 예상 규모 |
|---|---|---|---|---|
| 0 | 죽은 코드 삭제 | 없음 | ★★★ (가짜 경로 소멸) | −1,500~2,000줄, 1~2일 |
| 1 | 이름·우산 헤더 정리 | 낮음 | ★★☆ | rename 다수, 1~2일 |
| 2 | 단일구현 가상층 평탄화 | 중간 | ★★☆ (압력 체인 7→5단계) | 2~3일 |
| 3 | 앱 main 통합 (rbc 7→1, dhvc 2→1, 공통 셋업) | 중간 | ★★★ | apps −2,400줄, 3~5일 |
| 4 | 설정·빌드·문서·트리 | 낮음 | ★★★ (신규 사용자) | 문서+소규모 코드, 2~3일 |

0→1→4는 사실상 무위험이므로 먼저 진행하고, 2·3은 §7 회귀 체계를 갖춘 뒤 진행을 권장한다.

---

## 7. "로직 불변" 검증 전략

각 단계(및 0단계는 각 삭제 커밋) 후:

1. **3케이스 고정 회귀**: channel / DHVC / RBC 각각 짧은 결정적 런(예: 500~2,000스텝, 고정 dt 또는 동일 CFL 설정, 동일 격자·np)을 기준 커밋과 비교.
   - **CPU 빌드: byte 단위 비교** 가능(`monitor.csv`, `stats_*.dat`, restart `.bin` diff). 삭제·rename만 한 단계에서 1비트라도 다르면 실수.
   - **GPU 빌드: 허용오차 비교** — atomic 연산으로 run-to-run 비결정이므로 byte 비교 금지. `div_max`·Θ_c·Nu·평균 프로파일을 상대오차(예: 1e-12~1e-10)로 비교.
2. **빌드 매트릭스**: `make cpu`와 `make BACKEND=cuda apps`(GPU 노드에서) 둘 다 각 커밋에서 통과. GPU `.cu` 컴파일은 로그인 노드 불가에 유의.
3. **회귀 스크립트를 저장소에 커밋** (`scripts/regression_three_cases.sh`) — 이후 어떤 정리·최적화에도 재사용되는 자산이 된다. 실행은 `sbatch --wait`로 포그라운드 확인.
4. 3단계(main 통합)는 통합 전후 **같은 입력으로 CPU 빌드 byte 비교**를 필수 관문으로: `rbc_cpu_nob`(구) vs 통합 main CPU 빌드(신)가 동일해야 통과.

---

## 부록 A — 정리 후 목표 디렉터리 스케치

```
MPM-STD_C/
├── apps/
│   ├── common/setup.hpp            # 공통 셋업 (신규, 유일한 신규 파일)
│   ├── channel/ main.cpp input.toml
│   ├── dhvc/    main.cpp(#ifdef GPU) input.toml
│   └── rbc/     main.cpp(#ifdef GPU, nob 런타임) input.toml inputs_archive/
├── src/
│   ├── common/  config/  grid/  boundary/     # boundary_applier 제거됨
│   ├── core/                                   # bands.hpp 제거, *LinearSystem 개명
│   ├── parallel/ (cuda/ mpi/)                  # backend/ 제거
│   ├── linear_solver/tdma/                     # filtered 제거, fft/ 제거
│   ├── equation/ (momentum/ pressure/ scalar/) # PressureSolverBase 병합
│   ├── physics/ (buoyancy/ forcing/ properties/)  # les/ ibm/ 제거
│   ├── driver/ (cfl)                           # monitor/restart 헤더 제거
│   └── post/ (io statistics interp)            # diagnostics 스텁 제거
├── test/                                       # 단일 테스트 트리 (구 test/ + tests/ 병합)
├── docs/  config_reference.md  output_formats.md  history/
├── runs/                                       # 모든 실행 산출물 (소스 트리 밖)
└── readme.md                                   # Quick Start 5줄 + 현행화
```

## 부록 B — 주요 근거 파일 (재확인용)

- 죽은 계층: `src/solve/cpu/poisson_solver_cpu.cpp:5`, `src/physics/les/cpu/les_cpu.cpp:5`, `src/boundary/boundary_applier.hpp`
- 단일 구현 가상: `src/equation/pressure/pressure_base.hpp:22` ↔ `pressure_engine.hpp:28`; GPU 비상속 `pressure_engine_gpu.cuh:265`
- 가상 우회 static_cast: `src/equation/scalar/gpu/scalar_gpu.cu:250`, `src/equation/momentum/gpu/momentum_gpu.cu:1319`
- OB=단위물성 NOB 트릭: `apps/dhvc/main.cpp:129-143`, `src/equation/momentum/kernels.hpp:38-70`
- 복붙 정량: `apps/rbc/main.cpp:50-122` ↔ `apps/rbc/rbc_common.hpp:38-106`; `main_cpu_ob.cpp` ↔ `main_cpu_nob.cpp` (diff ~25줄)
- INI-파서: `src/config/config.hpp:14-27`, 조용한 기본값 `config.cpp:91-98`
- 빌드 취약점: `Makefile:60-107`(파일별 매크로 예외), `Makefile.inc:16,20`(경로 하드코딩)
