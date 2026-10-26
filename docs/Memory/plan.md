# 솔버 작업 버퍼 공유 계획 — scalar·momentum·pressure 가 한 작업 영역을 돌려쓴다 (2026-10-23)

## 0. 한 줄 요약

세 솔버(scalar, momentum, pressure)가 각자 장(field) 크기의 작업 버퍼를 들고 있다 — 밴드 a·b·c·d·d2 와 압력의
스펙트럼 버퍼. 한 스텝 안에서 세 풀이는 **차례로** 돌기 때문에 이 버퍼들이 동시에 살아 있는 순간이 없다.
그런데도 각자 소유하므로 한 솔버가 쓰는 동안 나머지 솔버의 버퍼는 논다. 이것을 **앱이 한 번 선언한 작업 영역
(`equation::Workspace`, 장 크기 슬롯 5개)** 로 모으고 세 솔버가 빌려 쓰게 바꾼다. 셀당 장 8개(A 단계) + 4개(B 단계)가
줄어 GPU 당 메모리가 258 B/cell → 약 194 → 약 162 B/cell 이 된다. 수치·계산 순서는 바꾸지 않는다 (체크포인트 비트 동일이 게이트).

원칙 (사용자 지시 2026-10-23):
- 솔버 **버퍼만** 공유한다. 앱이 소유한 U V W T P·rhs·dP 와 통계 배열은 건드리지 않는다.
- momentum 의 증분 δU δV δW 는 dU → dV → dW 순서 커플링 때문에 세 성분이 동시에 필요하므로 momentum 객체가 계속 소유한다.
  공유하는 것은 한 스윕 안에서만 쓰이는 TDMA 밴드다 — 세 성분이 같은 밴드를 차례로 쓴다 (지금도 그렇다).
- scalar 는 x → y → z 스윕이 차례이고 중간 상태가 밴드 핑퐁(d ↔ d2)뿐이라 그대로 공용 밴드에 올린다.

전제: 1단계 메모리 정리(2026-10-23, 미커밋)가 끝난 상태 — stage 버퍼·legacy 경로 삭제, 순환 TDMA z 공용화, cuFFT 작업 영역 공용화.
이 문서의 "현재" 는 그 상태다.

---

## 1. 현재 상태 — 누가 무엇을 얼마나 들고 있나

N = 랭크의 내부 셀 수 (i1·i2·i3). "장 1개" = N 개 double. 수명은 DHVC 상수물성 한 스텝 기준
(scalar → momentum → pressure 순; 가변물성 앱도 순서는 같다).

| 소유자 | 버퍼 | 크기 | 살아 있는 구간 | 비고 |
|---|---|---|---|---|
| `scalar::Solver` | band_a b c d d2 | 5 × N | scalar 풀이 안 (x·y·z 스윕) | d ↔ d2 핑퐁 (band_layout.hpp) |
| `momentum::Solver` | band_a b c d d2 | 5 × N | 성분 하나의 스윕 안 — dU, dV, dW 가 차례로 재사용 | scalar 와 **같은 크기·같은 용도** |
| `momentum::Solver` | delta_u v w (Field, halo 포함) | 3 × N⁺ | 성분 풀이 시작 ~ add_increment | 커플링이 이웃 증분을 읽음 → **공유 안 함** |
| `pressure::Solver` | band_a b c | 3 × N·(1+2/nx) | build_z ~ tdma_z | nsys = 2·(nx/2+1)·ny → N 보다 2/nx 만큼 큼 |
| `pressure::Solver` | real_lines | N | pack ~ FFT x 정방향 / FFT x 역방향 ~ scatter | 가운데 구간은 죽어 있음 |
| `pressure::Solver` | spectrum_x | N·(1+2/nx) | FFT x 정방향 ~ reorder / reorder 역 ~ FFT x 역방향 | |
| `pressure::Solver` | spectrum_xy | N·(1+2/nx) | reorder ~ reorder 역 (FFT y, TDMA 의 D, 평균 합) | |
| `pressure::Solver` | fft_work (GPU) | ≈ N | FFT 실행 중에만 | 1단계에서 3 → 1 로 줄인 것 |
| `pressure::Solver` | mirror_spectrum | 2N 급 | x 벽(DCT) 앱만 (cavity) | 주기 x 앱에는 없음 |
| (공용) | 순환 TDMA z | N | 순환 커널 한 번 안 | 1단계에서 공용화 |
| `*::Solver` | TdmaPlan ×7 | 작음 (≈ 0.4 N 합계) | 상시 | scalar·momentum 의 plan_x/y/z 는 인자까지 동일 |

합계: scalar 5 + momentum 8 + pressure 7 + z 1 = **장 21개**가 솔버 몫. 앱 몫 10개(U V W T P rhs_U rhs_V rhs_W rhs_P dP)를 더해
31개 ≈ 258 B/cell (DHVC 1랭크 실측, halo·λ 표·plan 포함).

동시에 살아 있는 최대 묶음은 **momentum 한 성분 풀이 중**: 밴드 5 + δ 3 = 8 (δ 는 공유 대상이 아님). 압력 풀이 안의 동시 최대는
TDMA 시점의 band a·b·c + spectrum_xy = 4. 따라서 **슬롯 5개면 어느 솔버도 모자라지 않는다.**

---

## 2. 설계 — `equation::Workspace`

### 2.1 객체

```
// src/equation/workspace.hpp  (새 파일, 양 빌드 공용 — 메모리는 field::Buffer 가 백엔드별로)
namespace mpmstd::equation {
struct Workspace {
  static constexpr int kSlots = 5;
  explicit Workspace(std::size_t slot_elements);      // 슬롯 5개를 slot_elements 개씩 한 번 할당
  real_t* slot(int k);                                 // 0 ≤ k < kSlots
  std::size_t slot_elements() const;
  // 재할당 없음 — 생성 뒤 주소 고정 (FFTW 플랜이 주소에 묶이므로, §6)
};
}
```

각 솔버는 자기가 필요로 하는 슬롯 크기를 정적 함수로 알려 준다. 앱은 셋의 최대값으로 Workspace 를 만든다.

```
std::size_t scalar::Solver::required_slot_elements(sub);          // i1·i2·i3
std::size_t momentum::Solver::required_slot_elements(sub);        // i1·i2·i3
std::size_t pressure::Solver::required_slot_elements(grid, sub);  // max(nsys·nz_local, real_lines, spectrum_x, spectrum_xy[, mirror])
```

### 2.2 솔버가 빌려 쓰는 방식

생성자에 `Workspace&` 를 받아 포인터로 보관하고, solve 함수가 밴드 포인터를 `ws.slot(k)` 로 얻는다. 지금 코드가 밴드를
전부 `real_t*` 로 다루므로 (`solver.band_a.data()` → `ws.slot(0)`) 커널·조립 코드는 손대지 않는다.

```
momentum::Solver::Solver(sub, Workspace& ws)   // band_a..d2 멤버 삭제, plan·delta·wall_closure 는 그대로
scalar::Solver::Solver(sub, Workspace& ws)     // band_a..d2 멤버 삭제
pressure::Solver::Solver(grid, sub, Workspace& ws)

// solve 진입부 (세 솔버 공통 형태)
if (ws.slot_elements() < required) throw std::invalid_argument("...Workspace too small");
real_t* A = ws.slot(0); real_t* B = ws.slot(1); real_t* C = ws.slot(2);
real_t* D0 = ws.slot(3); real_t* D1 = ws.slot(4);
```

앱 쪽 변경은 세 줄이다 (7개 앱 + test_pressure_wall + channel_pencil_test).

```
equation::Workspace ws(std::max({momentum::Solver::required_slot_elements(sub),
                                 scalar::Solver::required_slot_elements(sub),
                                 pressure::Solver::required_slot_elements(grid, sub)}));
momentum::Solver momentum_solver(sub, ws);
scalar::Solver   scalar_solver(sub, ws);
pressure::Solver pressure_solver(grid, sub, ws);
```

scalar 가 없는 앱(channel, boundary_layer)은 max 에서 scalar 항만 뺀다.

### 2.3 슬롯 배정표

| 슬롯 | scalar 풀이 | momentum 성분 풀이 | pressure 풀이 (A 단계) | pressure 풀이 (B 단계) |
|---|---|---|---|---|
| 0 | band_a | band_a | band_a | real_lines ↔ band_c (수명 분리, §3 B) |
| 1 | band_b | band_b | band_b | spectrum_x |
| 2 | band_c | band_c | band_c | spectrum_xy (= TDMA 의 D) |
| 3 | band_d (D0) | band_d (D0) | — | band_a ↔ fft_work (수명 분리) |
| 4 | band_d2 (D1) | band_d2 (D1) | — | band_b |

A 단계에서 pressure 는 real_lines·spectrum_x·spectrum_xy·fft_work 를 계속 소유한다. B 단계에서 그것까지 슬롯으로 옮긴다.

### 2.4 불변 조건 (코드 주석과 이 문서에 둔다)

1. **한 스텝 안에서 세 솔버의 solve 는 겹치지 않고 차례로 불린다.** 모든 커널이 기본 CUDA 스트림 하나에 들어간다.
   나중에 scalar 와 momentum 을 다른 스트림에서 겹쳐 돌리는 최적화를 넣으면 Workspace 를 둘로 나눠야 한다.
2. **솔버는 slot 의 내용이 호출 사이에 보존된다고 가정하지 않는다.** 모든 밴드는 solve 안에서 조립해서 쓰고 버린다
   (지금도 그렇다 — build_x 가 매 호출 A·B·C·D 를 새로 쓴다).
3. **Workspace 주소는 생성 뒤 고정.** CPU 빌드의 FFTW 플랜이 real_lines·spectrum 주소에 묶이기 때문 (B 단계).
   그래서 "자라는 풀" 이 아니라 "앱이 최대값으로 한 번 만드는 풀" 로 한다.
4. **통계·진단은 slot 을 읽지 않는다.** 읽고 싶은 중간량이 생기면 그 배열은 공유 대상에서 뺀다.

---

## 3. 단계별 변경

### A 단계 — 밴드 공유 (장 −8)

- `src/equation/workspace.hpp` 신설 (Buffer 5개 소유, slot(k), required 계산은 각 솔버에).
- scalar: `band_a..d2` 멤버 삭제, 생성자 `(sub, ws)`, `solve_constant_property`·`solve_variable_property` 의 포인터 취득을 slot 으로.
- momentum: 같은 방식. `adi_sweeps_constant`·`adi_sweeps_variable_property` 는 이미 `real_t* A,B,C,D0,D1` 로 받는다 → 호출부만 변경.
- pressure: `band_a b c` 멤버 삭제 → slot 0·1·2. nsys·nz_local ≤ slot_elements 를 생성자가 검사.
- 앱 9개: Workspace 생성 + 생성자 인자. `tools/scale/instrument_main.py` 가 만드는 `main_scale.cpp` 는 main.cpp 에서 재생성.
- 효과: scalar 5 + pressure 3 = **장 8개** 감소.

### B 단계 — 압력 스펙트럼·라인·FFT 작업 영역도 슬롯으로 (장 −4)

압력 풀이 안의 수명표(슬랩 경로):

| 단계 | real_lines | spectrum_x | spectrum_xy | band a b c | fft_work |
|---|---|---|---|---|---|
| pack_lines | 씀 | | | | |
| FFT x 정방향 | 읽음 | 씀 | | | 사용 |
| reorder x→y | | 읽음 | 씀 | | |
| FFT y 정방향 | | | 제자리 | | 사용 |
| build_z, tdma_z | | | D (읽고 씀) | 씀·읽음 | |
| mean_remove, FFT y 역방향 | | | 제자리 | | 사용 |
| reorder y→x | | 씀 | 읽음 | | |
| FFT x 역방향 | 씀 | 읽음 | | | 사용 |
| scatter | 읽음 | | | | |

- real_lines 와 band_c 의 수명이 서로 떨어져 있다 (real_lines 는 앞·뒤, band_c 는 가운데) → **슬롯 0 공유**.
- fft_work 는 FFT 실행 중에만, band_a 는 build_z ~ tdma 에만 → **슬롯 3 공유**. cufftSetWorkArea 는 주소를 한 번 묶지만
  그 주소를 다른 용도로 쓰는 시점에 FFT 가 실행되지 않으므로 문제없다.
- spectrum_x → 슬롯 1, spectrum_xy → 슬롯 2, band_b → 슬롯 4.
- CPU(FFTW): 플랜을 slot 주소로 만든다 (주소 고정, §2.4-3).
- 펜슬 경로(npx·npy > 1): 같은 버퍼를 쓰므로 같은 배정. `pencil_exchange` 의 송수신 버퍼는 PencilLayout 소유 — 이번 범위 밖.
- x 벽(DCT, cavity): mirror_spectrum 은 pressure 가 계속 소유 (2N 급 하나). real_lines·spectrum 이 2N 이므로 cavity 의
  required_slot_elements 가 2N 급이 되어 scalar·momentum 밴드도 그 크기 슬롯을 쓴다 — 개수는 같고 크기만 커지므로 손해는 없다
  (지금은 cavity 가 밴드 10개 + 압력 버퍼 2N 급 5개를 따로 갖는다).
- 효과: real_lines·spectrum_x·spectrum_xy·fft_work = **장 4개** 감소. pressure 가 소유하는 장 크기 버퍼가 0 이 된다 (DCT 앱은 mirror 1개).

### C 단계 (선택, 작음) — TdmaPlan 공유

scalar 와 momentum 의 plan_x/y/z 는 `create(i2·i3, x.comm)` 등 인자까지 같다. Workspace 가 plan 3개를 소유하고 두 솔버가 참조하면
plan 6 → 3. 절감은 ≈ 0.2 N (축약계·통신 버퍼가 시스템당 수십 개 값) 이라 B 단계까지 끝난 뒤 여유가 있을 때만 한다.
pressure 의 plan_z 는 nsys 가 달라 공유 불가.

---

## 4. 효과 — DHVC Ra 1e10, 4×1×1 상자, H200 ×8 (GPU 당 N = 4.69억, 장 1개 = 3.49 GiB)

| 단계 | 솔버 장 수 | 전체 장 수 (앱 10 + z 1 포함) | B/cell (추정) | GPU 당 | H200 140 GiB 대비 |
|---|---|---|---|---|---|
| 현재 (1단계 뒤) | 21 | 31 | 258 (실측) | 113 GiB | 80 % |
| A 단계 | 13 | 23 | ≈ 194 | ≈ 85 GiB | 60 % |
| A + B | 9 | 19 | ≈ 162 | ≈ 71 GiB | 50 % |
| A + B + C | 9 | 19 | ≈ 160 | ≈ 70 GiB | 50 % |

A + B 뒤에 새로 가능해지는 것:

| 케이스 | GPU 당 셀 | 필요 (162 B/cell) | 판정 |
|---|---|---|---|
| Ra 1e9 축소 상자 4×2×1 (1600×332×1600), V100 ×8 (32 GB) | 1.06억 | 16 GiB | **들어감** (지금은 27 GiB 로 불가) |
| Ra 1e9 원래 상자 8×4×1 (3200×664×1600), A100 ×8 (80 GB) | 4.25억 | 64 GiB | **들어감** (86 %) |
| Ra 1e10 4×2×1 (3200×732×3200), H200 ×8 | 9.37억 | 141 GiB | 안 됨 (앱 몫 rhs_P·dP 까지 빌려 써야 91 %) |

---

## 5. 검증

게이트는 **변경 전 바이너리와 체크포인트 비트 동일**이다 (GPU 통계 축소의 atomicAdd 가 끼는 앱은 변경 전끼리의 반복 차이 ≤ 1e-12 와
같은 수준). 1단계 검증과 같은 절차(`scratchpad/memcheck/run.sh` 방식: 변경 전 2회 + 변경 후 1회, 300 스텝, 모니터 + 체크포인트).

| 앱 | 덱 | 보는 것 |
|---|---|---|
| dhvc | input_dhvc_smoke_gpu.ini | scalar·momentum 상수물성 + 압력 주기 x |
| rbc_nob | input_rbc_nob_smoke.ini | 가변물성 경로 (solve_variable_property 양쪽) — 비트 동일 기대 |
| channel | input_channel_smoke.ini | scalar 없는 앱, LES 가변점성 경로 |
| cavity | 스모크 덱 | **x 벽 DCT** 경로 (mirror_spectrum, 2N 슬롯) |
| thermal_boundary_layer | laminar_check | 스칼라 여러 개가 한 Solver 를 차례로 쓰는 경우 |
| channel_pencil_test | run_compare_v100x8.sh 의 2x2x2 | 펜슬 경로 + CPU 빌드(FFTW 주소 고정) 직렬 비교 |

추가로: 각 앱의 nvidia-smi 최대 메모리를 변경 전후로 재서 셀당 절감량이 §4 추정과 맞는지 확인 (1단계 때 예상 1,338 MiB vs 실측 1,366 MiB).

디버그 보조(선택): `MPMSTD_WORKSPACE_POISON=1` 이면 solve 진입·종료 때 slot 을 NaN 으로 채운다 → 죽은 데이터를 읽는 배정 실수가 즉시 터진다.
B 단계의 슬롯 0·3 공유를 검증할 때 쓴다.

---

## 6. 위험과 제약

- **FFTW 주소 바인딩(CPU)**: 플랜 생성 뒤 slot 이 재할당되면 플랜이 낡은 주소를 쓴다 → Workspace 는 재할당 API 를 두지 않는다.
  pressure 생성자는 `ws.slot_elements() ≥ required` 를 검사하고 아니면 throw (조용한 오답 금지).
- **B 단계의 슬롯 공유(0: real_lines↔band_c, 3: fft_work↔band_a)**: §3 수명표가 근거. 압력 풀이 순서를 바꾸는 수정이 들어오면 표를 다시 본다.
- **스트림 겹치기 최적화와 충돌**: §2.4-1. 솔버 헤더에 "Workspace 는 단일 스트림 순차 실행 전제" 를 적는다.
- **PhaseTimer 계측(main_scale.cpp)**: 구간 이름·동기화 지점은 그대로 — 영향 없음. 단 main.cpp 를 고치므로 instrument_main.py 로 재생성.
- **진행 중인 런**: Ra 1e10 잡(944112)은 `build_gpu_mem` 바이너리(1단계)로 돈다. 이 작업 중 `build_gpu_mem` 은 건드리지 않고
  `build_gpu` 만 다시 빌드한다. 재시작 체인부터 새 바이너리를 쓴다.
- **범위 밖**: 앱 소유 배열(rhs_U V W, rhs_P, dP)의 공유, 통계 배열, PencilLayout 의 교환 버퍼, momentum 의 δ 세 개.

---

## 7. 순서

1. A 단계 구현 → §5 검증 → 커밋 (1단계 메모리 정리·legacy 삭제와 함께 또는 바로 뒤에).
2. `build_gpu` 재빌드, 앱별 메모리 실측 표를 이 문서 §4 에 채움.
3. B 단계 구현 → 검증(특히 cavity DCT·CPU FFTW·펜슬) → 커밋.
4. C 단계는 보류 — A+B 뒤 메모리가 더 필요해질 때.
5. 끝나면 Ra 1e9 축소 상자를 V100 ×8 로, 원래 8×4 상자를 A100 ×8 로 돌릴 수 있는지 적재 시험으로 확인.

---

## 8. 진행 결과 (2026-10-23)

### 8.1 구현 — A + B 단계 완료 (C 는 보류)
- `src/equation/workspace.hpp` 신설 (슬롯 5개, `slot(k)`, `require(need, who)`).
- scalar·momentum: `band_a..d2` 멤버 삭제 → `Workspace*` + `required_slot_elements(sub)`; solve 진입부에서 슬롯 0~4 를 A·B·C·D0·D1 로.
- pressure: `band_a b c`·`real_lines`·`spectrum_x`·`spectrum_xy`·`fft_work` 가 모두 슬롯 포인터 (0 = real_lines ↔ band_c, 1 = spectrum_x,
  2 = spectrum_xy, 3 = band_a ↔ fft_work, 4 = band_b). cuFFT 작업 영역이 슬롯보다 크면 `fft_work_owned` 로 후퇴. `mirror_spectrum`(DCT)만 소유.
- 앱 9개: `Workspace workspace(std::max({...required_slot_elements...}))` 한 줄 + 생성자 인자. main_scale.cpp 4개 재생성.
- GPU(cc70/80/90)·CPU(fftw3 모듈) 빌드 통과.

### 8.2 결과 동일성 (로그인 노드 A100, 300 스텝, 변경 전 = 1단계 정리 뒤 바이너리 `build_gpu/bin/before_ws/`)

| 앱 | 덱 | 체크포인트 | 모니터 (시간 열 제외) |
|---|---|---|---|
| rbc_nob (가변물성) | smoke | **비트 동일** | 동일 |
| rbc_ob | smoke | **비트 동일** | 동일 |
| cavity (x 벽 DCT, 2N 슬롯) | 1e3 | **비트 동일** | (300 스텝 안에 모니터 없음) |
| thermal_boundary_layer (스칼라 2개) | laminar_check | **비트 동일** | 동일 |
| boundary_layer | blasius_coarse | **비트 동일** | 동일 |
| dhvc | smoke_gpu | 1e-13 급 (변경 전끼리의 반복 차와 같은 수준 — 통계 축소 atomicAdd) | max\|div\| 열만 1e-13 급 |
| channel | smoke | 1e-13 급 (같은 이유) | max\|div\| 열만 |
| test_pressure_wall | — | stdout 동일 (EOC 1.998 PASS) | |
| channel_pencil_test | CPU 1x1x1 before vs new | **비트 동일** (FFTW 주소 고정 확인) | |
| channel_pencil_test | GPU 2x1x1·1x1x2 vs CPU 직렬 | max\|ΔU\| 1.2e-14 | |

### 8.3 메모리 (DHVC 1 랭크, 416×244×192 = 1,949만 셀, nvidia-smi 최대)

| | 최대 메모리 | 셀당 | 비고 |
|---|---|---|---|
| 1단계 정리 뒤 | 5,286 MiB | 258 B | |
| A + B | **3,486 MiB** | **161 B** | −1,800 MiB (예상 −12 장 = 1,784 MiB) |

Ra 1e10 4×1×1 (GPU 당 4.69억 셀, H200 ×8) 환산: 161 B/cell → **약 70 GiB/GPU (50 %)**. §4 추정(71 GiB)과 일치.

### 8.4 성능 동일성 (잡 944408, gpu04 V100 1장, 변경 전·후 번갈아 3회, `apps/rbc_nob/scalability/sbatch_ws_perf.sh`)

| 측정 | 변경 전 | A + B | 차이 |
|---|---|---|---|
| rbc_nob 256³ np1 step_total (ms/step, 3회 중앙값; 반복 폭 0.07 ms) | 83.38 | 83.80 | +0.5 % |
| 〃 momentum_solve / scalar_solve / pressure_solve | 45.74 / 14.74 / 5.91 | 46.14 / 14.74 / 5.92 | +0.9 % / 0 / +0.1 % |
| 〃 TDMA x·y 스윕 (성분당) | 2.55 | 2.57 | +0.03 ms (+1 %) |
| dhvc 256³ np1 wall s/step (모니터, 3회) | 0.0650 | 0.0650 | 0 % |

- rbc_nob 의 +0.4 ms 는 반복 폭(0.07 ms)보다 커서 체계적이지만 0.5 % — 구간별로는 ±1 % 로 양방향(scalar.tdma_y −1.2 %, momentum.tdma_x +1.1 %).
  커널·산술은 그대로이고 바뀐 것은 버퍼의 디바이스 메모리 위치뿐이라 배치 효과로 본다. 통신이 끼는 np > 1 은 영향 없음(공유는 랭크 안 버퍼).
- 판정: **결과 동일(§8.2) + 성능 동일(0 ~ +0.5 %) + 메모리 −38 %(§8.3).** C 단계는 보류.
