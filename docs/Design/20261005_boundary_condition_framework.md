# 축별 경계조건(BC) 프레임워크 + de Vahl Davis 2D 열 cavity 검증 (2026-10-05~)

> 설계·구현·검증을 한 문서에 단계별로 누적한다. 각 단계는 "무엇을 바꿨나 / 무엇을
> 어떻게 검증했나 / 결과" 세 줄 구조. 계획 정본은 `docs/Plan/plan_20261003.md` §8-19.

## 0. 목표와 범위 (사용자 지시 요약)

- 입력에서 **축마다** BC 를 고르고(periodic | wall | …), Dirichlet·Neumann 을 둘 다
  덮는다. 지금까지 코드는 x·y 주기 + z 벽이 하드코딩이었다.
- 최종 검증 = **de Vahl Davis 열 cavity 를 2D 로** (ny=1 축 붕괴, 3축 전부 푸는 비용
  회피). 참조: de Vahl Davis (1983) Ra 1e3~1e6, Pr 0.71, 벤치 Nū 1.118 / 2.243 /
  4.519 / 8.800.
- CPU·GPU **둘 다 구현**(.cu 쌍둥이 규칙), 검증은 **CPU 만** (문제가 작아 충분).
- 벽 속도는 기본 no-slip(Dirichlet 0), **slip 은 override 로** 지정 (사용자 제안).

## 1. 설계

### 1.1 입력 문법 (`[boundary]`, rbc 설정 스키마 공유: rbc / rbc_nob / dhvc / cavity)

```ini
[boundary]
x = wall            ; periodic | wall | dirichlet | neumann   (축 kind)
y = periodic
z = wall
t_x_minus = 0.5     ; <field>_<axis>_<side> = 숫자 → Dirichlet(값)
t_x_plus  = -0.5
t_z_minus = adiabatic   ; adiabatic | slip | neumann → Neumann 0 (동의어)
t_z_plus  = adiabatic
v_z_plus  = slip        ; 접선 속도 free-slip 예시
```

- field ∈ u v w p t, axis ∈ x y z, side ∈ minus plus → 30개 옵션 키 + 축 3개.
- 섹션이 **없으면 RBC 관례(x·y periodic, z wall)** — 기존 덱 전부 무수정.
- 축 kind → 필드 기본값 (MPM-STD_C problem_loader 관례):
  periodic → 전 필드 Periodic / wall → U V W T Dirichlet 0, **P Neumann 0(불가변)** /
  dirichlet → 전부 Dirichlet 0 / neumann → 전부 Neumann 0.
- ★ MPM-STD_C 와 다른 점: override 가 **종류도** 바꾼다(숫자=Dirichlet,
  adiabatic·slip=Neumann). de Vahl Davis 가 같은 wall 축에서 측벽 T Dirichlet·상하벽 T
  Neumann 을 요구하고, slip 벽은 접선 속도만 Neumann 이기 때문.
- fail-loud 규칙: 주기 축에 override / 벽 P override / **벽 법선 속도를 Neumann**
  (불투과 위반) → 전부 "파일: [boundary] …" 메시지로 throw.

### 1.2 내부 표현 — `src/boundary/boundary_condition.{hpp,cpp}`

```
enum BcKind {Periodic, Dirichlet, Neumann};  enum AxisKind {Periodic, Wall, Dirichlet, Neumann}
struct FaceBc { BcKind kind; real_t value; }
struct BoundaryCondition { AxisKind axis[3]; FaceBc face[field 5][axis 3][side 2];
                           is_periodic(axis); at(Field, axis, side); validate(); }
make_boundary_condition(axis_kind_text[3], override_text[5][3][2])  // 기본값 전개 → override → validate
```

이 테이블 하나를 각 층이 본다 (단계 2~3 에서 소비):
| 층 | 보는 것 | 처리 |
|---|---|---|
| Topology(MPI cart) | `is_periodic(a)` | 주기 플래그 |
| ghost 채움 | `at(field,a,side)` | Dirichlet: 셀중심 → ghost=2v−interior(mirror), 면변수 법선 → 면값 고정 / Neumann: ghost=interior |
| ADI 스윕 | `is_periodic(a)` | cyclic TDMA vs 벽 fold(`wall_rows` 의 z 처리를 축 매개변수화) |
| 압력 변환 | `at(P,a,·).kind` | periodic→FFT, Neumann→DCT(REDFT10/01), Dirichlet→DST(불필요, 미구현) |

### 1.3 단계 (각 단계마다 기존 앱 비트 일치 게이트)

| 단계 | 내용 | 게이트 |
|---|---|---|
| 1 | BC 구조체 + `[boundary]` 로더 (소비자 없음) | 채널 md5 `2336c4a5` · rbc Nu 30.6631 · 파싱/거부 |
| 2 | x(·y) 벽 ghost + momentum/scalar 스윕 fold 축 매개변수화 | 동일 게이트 (x periodic 경로 불변) |
| 3 | 압력 x-DCT (CPU FFTW REDFT, GPU cuFFT 미러확장) | 동일 게이트 + Poisson 해석해 EOC 2차 |
| 4 | `apps/cavity` + de Vahl Davis Ra 1e3~1e6 (ny=1) | Nū 벤치 대비 ±1% |

## 2. 단계별 구현·검증 기록

### 단계 1 — BC 구조체 + 로더 ✅ (2026-10-05)

**바꾼 것**
- 신설 `src/boundary/boundary_condition.{hpp,cpp}` (위 §1.2; 순수 로직, MPI 무관).
- `apps/rbc/settings.{hpp,cpp}`: `Settings::boundary` (`axis_kind[3]`,
  `override_text[5][3][2]`, 전개 결과 `condition`), 바인딩 33키를 루프 생성(정적
  키 저장소), `make_settings` 에서 전개(실패 시 파일명 접두), `format_settings` 에
  `[boundary]` 에코(왕복 가능 — 비어 있지 않은 override 만 출력).
- rbc_nob / dhvc 는 `#include "../rbc/settings.cpp"` 로 자동 공유 (⚠ depfile 이
  이 include 를 추적 못 함 → 해당 obj 수동 삭제 후 재빌드).

**검증 (cpu04, np=2, `tmp_const_gate/gate_bc.sh`)**
| 항목 | 결과 |
|---|---|
| 채널 스모크 `channel_stats_00000200.dat` md5 | `2336c4a569f3…` 기준과 **비트 일치** |
| rbc 스모크 FINAL | `Nu = 30.6631, theta_c = -0.00361` **동일** |
| cavity형 `[boundary]` 파싱 | 에코 9줄 정상(x wall / y periodic / z wall + t·v override) |
| `u_x_minus = slip` (벽 법선 Neumann) | 거부: `u_x_minus: wall-normal velocity must be Dirichlet (impermeable)` |
| `t_y_minus = 0.5` (주기 축 override) | 거부: `'t_y_minus' given but axis y is periodic` |

**교훈** — sbatch `-o`·입력 ini 를 `/tmp` 스크래치에 두면 계산 노드에서 안 보여 잡이
로그 없이 즉사(기존 메모리 교훈 재확인). 전부 NFS(`tmp_const_gate/`)로.

### 단계 2 — x(·y) 벽 ghost + ADI 스윕 벽 닫기 축 매개변수화 ✅ (2026-10-05)

**바꾼 것** (기존 앱 코드 무수정 — 전부 라이브러리 안)
- `parallel/topology`: `AxisComm::periodic` (축 전체 주기 여부 — 내부 랭크는 이웃만
  보고 알 수 없어 필요) + `Topology::axis(a)` / `Subdomain::axis(a)` 인덱스 접근자.
- `boundary/wall.{hpp,cpp,cu}`: 본체를 **(axis, side) 한 벽면 단위**로 재작성
  `fill_wall_mirror/value/zero_gradient(f, sub, axis, side, …)`, `zero_wall_faces` —
  z 전용 이름은 axis=2 를 minus→plus 순으로 두 번 부르는 래퍼 (원래 순서와 동일).
  GPU 커널은 평면 (p,q) 디코드 + 축 슬롯으로 일반화.
- `boundary/apply.{hpp,cpp}` 신설: BoundaryCondition 테이블 → 함수 선택
  (속도 법선 성분 = 벽면 0 / 접선 Dirichlet = value, Neumann = zero_gradient;
  T Dirichlet = mirror, Neumann = zero_gradient; P = zero_gradient).
- `momentum/wall_rows`: `fold_wall_value(…, axis, …)`, `pin_minus_wall_face_row(…, axis, …)`
  본체 + z 이름 래퍼(legacy_ob 용). `momentum/solve.cpp`: 스윕마다
  `solve_sweep(plan, axis, own_axis)` — 주기면 순환 TDMA, 벽이면 fold(+own 축이면
  pin)+비주기 TDMA; δ 증분 ghost 닫기도 3축 루프(`close_delta_wall_ghosts`).
- `scalar/scalar_stencil.hpp`: `build_cell` 벽 플래그 2→6개(x/y/z lo/hi), 첫 스윕(x)
  축이 벽이면 그 행 δ 비대각 제거(나중 스윕의 full_offdiag 와 같은 닫기);
  `system.{cpp,cu}` 는 `WallMask`(3축 소유)로 플래그 공급. `scalar/solve.cpp` 도
  축별 cyclic/비주기 선택 (벽 행 닫기는 마스크가 하므로 fold 없음).

**검증 (cpu04 np=2 `gate_bc2.sh`; GPU 는 gpu02 에서 4앱 컴파일만)**
| 항목 | 결과 |
|---|---|
| 채널 스모크 md5 | `2336c4a5…` **비트 일치** |
| rbc 스모크 | `Nu = 30.6631, theta_c = -0.00361` 동일 |
| rbc_nob 스모크 (가변물성 스칼라 — 마스크 확장 영향권) | `theta_c = 0.01503, Nu = 46.0362` 동일 |
| GPU 빌드 (`build_gpu_bc/`, rbc·rbc_nob·dhvc·channel) | 성공 (경고 3건은 기존) |

x 벽 자체의 물리 검증은 압력 DCT(단계 3) 없이는 불가 → 단계 4 cavity 에서 함께.

### 단계 3 — 압력 Poisson x 벽(Neumann) = DCT 경로 ✅ (2026-10-05)

**바꾼 것** (`src/equation/pressure/`)
- `Solver`: `x_neumann = !topology.x().periodic` 로 경로 선택. x 모드 수
  `n_mode_x` = Nx/2+1 (주기 r2c) / Nx (벽 DCT) — 이름을 `n_half_x`→`n_mode_x` 로
  바꿨고 하류(reorder·bands·TDMA·unpack)는 전부 "x 모드 수"로만 보므로 무수정.
- **DCT 는 거울 확장 2N 실수 FFT + 트위들**로 구현 — cuFFT 에 DCT 가 없어서
  CPU(FFTW)/GPU(cuFFT) 가 **같은 알고리즘**을 쓴다(식은 `solve_kernels.hpp` 머리말):
  forward `y=[x, rev x]` → r2c₂ₙ → `X[k]=Re(Y[k]e^{−iπk/2N})` (= REDFT10);
  backward `Y[k]=X[k]e^{+iπk/2N}, Y[N]=0` → c2r₂ₙ → 앞 N 개 (= REDFT01); 왕복 배율
  2N → scale 1/(2Nx·Ny). 커널 4종(`pack_rhs_to_mirrored_lines`, `dct_twiddle_forward/
  backward`, `scatter_mirrored_lines_to_field`) CPU+GPU 쌍둥이. DCT 실수 모드는 복소
  버퍼 실수부에 두고 허수부 시스템(0)은 그냥 푼다 — 단순함 우선.
- 고유값 `make_lambda_x_neumann`: λ_m = (1−cos(πm/Nx))·2/dx² (CaNS 'NN' 표와 동일),
  (0,0) 모드 핀·평균 제거는 그대로. 끝의 ghost 는 비주기 축 전부 zero_gradient.
- y 벽은 미구현(생성자 throw) — cavity 는 ny=1 주기로 2D.
- 검증 앱 `apps/test_pressure_wall` 신설 (x벽·z벽·ny=1, φ=cos(πx/Lx)cos(πz/Lz)).

**검증 (cpu04 `gate_bc3.sh`)**
| 항목 | 결과 |
|---|---|
| 채널 md5 / rbc / rbc_nob 스모크 | 전부 **불변** (주기 경로 비트 일치) |
| ① 이산 일관성 (rhs = ∇²_h φ, n=32) | max\|φ_h−φ\| = **6.3e-15** (기계 정밀도) |
| ② 해석 rhs EOC (n=16→32→64) | 3.19e-3 → 8.02e-4 → 2.01e-4, **EOC 1.992 / 1.998** |
| np=1 vs np=2 (z 분할) | 동일 (6.1e-15 / 같은 EOC) |
| GPU | 5앱 컴파일 성공 (`build_gpu_bc/`); **실행 검증은 보류** — gpu01/02 GPU 전부 DHVC 1e7/1e8 캠페인이 gres 로 점유 |

**교훈** — 수정 직후 다른 노드(gpu02)에서 make 하면 NFS 속성 캐시 때문에 옛 mtime
을 봐서 `.cu.o` 를 재컴파일하지 않을 수 있다(undefined reference 로 드러남). 원격
빌드 전엔 바뀐 TU 의 .o 를 지우거나 1분쯤 기다릴 것.

### 단계 4 — `apps/cavity` + de Vahl Davis 2D 검증 (2026-10-05, 진행 중)

**만든 것**
- `apps/cavity/main.cpp` (+ `cavity_settings.cpp` 1줄 TU): 하드코딩된 벽이 **하나도
  없는** 첫 앱 — Topology 주기 플래그·ghost 채움(`boundary/apply`)·에너지 벽 닫기
  전부 `[boundary]` 테이블에서. z 부력(`add_buoyancy_z_constant_property`), 상수물성
  경로, ny=1. 진단: Nu_hot/Nu_cold(2차 편측차분, +솔버 일관 1차), 중앙선 u_max/w_max
  (확산 스케일 ×√(Ra·Pr)), 전역 max|U|V|W|, T_center; 통계 파일에 중앙선 3단면.
- 덱 `input_cavity_{1e3,1e4,1e5,1e6}.ini` (128², 1e6 은 256²; 균일 격자 — x 는 DCT
  전제), `sbatch_cavity_ladder.sh`(4 케이스 병렬), `README.md`, `plot/plot_cavity.ipynb`.

**중간 검증 (운동량·압력·스칼라를 분리해서)**
| 시험 | 결과 |
|---|---|
| lid-driven Re=100 (T 비활성; `u_z_plus = 1`) | u_min −0.203@z0.44 (Ghia −0.211@0.453), w_min −0.244@x0.816 (Ghia −0.245@0.805), x벽 근처 w 매끈 → **운동량+x벽+DCT 정상** |
| 전도 섭동 감쇠 (dt=1e-5, x벽 Dirichlet) | 1스텝 ΔT_center −2.34e-6 = 해석값, 이후 감쇠율 일치 → **스칼라 x벽 정상(작은 dt)** |
| lid-driven 유동 속 수동 스칼라 (Pe≈71) | T 유계, Nu_hot/cold 수렴 → 이류+벽 정상 |
| 부력 결합 Ra=1e3 (최초) | **폭주**: 벽 셀 T → −20, 이웃 > 0.5 (최대원리 위반), Nu → 1e5 |

**★ 원인 — 벽 행 음해 닫기와 ghost 정책의 불일치 (중요 발견)**
에너지 ADI 의 벽 행 닫기는 PaScaL 충실 "drop"(δθ_ghost = 0: 비대각만 제거). 거울
ghost(θ_g = 2θ_w − θ₁)의 명시 연산자는 벽 셀 대각이 3κ/dx² 인데 음해 쪽은 2κ/dx² —
δ-형 CN 의 벽 셀 모드 증폭률 g = 1 − dt·λ_e/(1 + dt·λ_i/2) → dt·λ ≫ 1 에서 1 − 2·(3/2)
= **−2 (불안정)**. 실측: Ra=1e3·128²·dt 0.005 → dt·3α/dx² = 9.2 → g ≈ −1.24 폭주;
수동 lid 케이스(2.8 → g −0.45)·감쇠 시험(≪1)은 안정 범위라 안 보였다. RBC/DHVC 는
dt·κ/dz² ≲ 0.03 이라 무해(그래서 지금까지 드러나지 않음).
수정: `scalar::Solver::wall_closure[axis][side]` (0 = drop 기본 → 기존 앱 비트 불변,
−1 = Dirichlet 거울 일치 B −= A, +1 = Neumann 영구배 일치 B += A) +
`set_wall_closure(solver, bc, Field::T)`; `WallMask::closure_*` 로 조립(CPU/GPU 공용
stencil)에 전달. 이는 기존 메모리 교훈(modify_tdma_row 닫기 = ghost 정책과 일치)의
스칼라판. ⚠ RBC/DHVC 의 z 벽도 원리상 같은 조건부 불안정(dt·κ/dz² ≳ 3)을 가진다 —
현재 운전점에선 무관하나 거친 격자·큰 dt 로 가면 −1 로 바꿀 것.

**결과 (128², 수정 후; 1e6 256² 및 1e5 256² 격자수렴 런 진행 중)**
| Ra | Nu (ours) | bench | err | u_max (z) | bench | w_max (x) | bench | 정상 도달 |
|---|---|---|---|---|---|---|---|---|
| 1e3 | 1.1260 | 1.118 | +0.7% | 3.723 (0.184↔0.816) | 3.649 (0.813) | 3.775 (0.176) | 3.697 (0.178) | t≈9 |
| 1e4 | 2.3016 | 2.243 | +2.6% | 16.385 (0.176↔0.824) | 16.178 (0.823) | 19.992 (0.113) | 19.617 (0.119) | t≈60 |
| 1e5 | 4.7273 | 4.519 | +4.6% | 35.44 (0.145↔0.855) | 34.73 (0.855) | 70.60 (0.059) | 68.59 (0.066) | t≈150 |
| 1e6 | (진행 중) | 8.800 | | | 64.63 (0.850) | | 219.36 (0.0379) | |
max|div| 3.6e-15 (DCT 투영 정확), Nu_hot = Nu_cold 전 자리 일치(중심 대칭).

**편향 해석 → 확인(2026-10-05 밤)**: Nu 양의 편향이 Ra 와 함께 커진 원인은 접선 속도의
**value ghost(1차)** — 벽 전단을 V₁/dx 로 봐(참값 V₁/(dx/2)) **벽 마찰이 절반**이 되어
경계층이 가속되는 것. 증거: ① 1e5 를 256² 로 가면 Nu 오차 4.6% → 2.3% (정확히 반감 =
1차), ② 논문 자체의 2차 노드 스킴은 h=0.0125 에서 ±0.5% 이내(Table III).

### 단계 4b — 논문(de Vahl Davis 1983, `apps/Ref [4].pdf`) Table V 10개 항목 전수 검증 ✅

사용자 지시 "논문 읽고 검증 수치 선정 → 검증". 선정: Table V 전부 — |ψ|_mid, |ψ|_max(x,z),
u_max(z), w_max(x), Nu̅(공동 평균), Nu_½(연직 중앙면), Nu_0(hot 벽), Nu_max(z), Nu_min(z).
- 앱에 최종 2D 장 덤프(`cavity_field_<step>.dat`: U x면·W z면·T 셀중심) 추가, 후처리
  `plot/benchmark.py`(numpy 만): ψ = ∫u dz (노드), Q = uT − ∂T/∂x 면적분·선적분, 벽 국소 Nu
  2차 편측차분, 극값·위치는 논문처럼 격자 극값 주변 5점 4차 다항식 보간.
- 내부 일관성: ψ 상단벽 잔차 ~1e-13(연속성), Nu̅ = Nu_½ = Nu_0 (에너지 수지, 4자리).
- 2차 벽을 위해 `[boundary] velocity_ghost = value | mirror` 신설 (기본 value = 기존 앱
  불변). mirror 는 운동량 벽 행 닫기 B −= A (`momentum::set_wall_closure`, `fold_wall`
  CPU/GPU) 및 δ ghost 거울과 **짝** — 스칼라와 같은 조건부 불안정을 피하려면 필수.

| Ra=1e3 (128²) | value | mirror | bench | Ra=1e4 (128²) | value | mirror | bench | Ra=1e5 (128²) | value | mirror | bench |
|---|---|---|---|---|---|---|---|---|---|---|---|
| \|ψ\|_mid | 1.2075 (+2.9%) | **1.1752 (+0.10%)** | 1.174 | | 5.168 (+1.9%) | **5.075 (+0.08%)** | 5.071 | | 9.309 (+2.2%) | **9.125 (+0.16%)** | 9.111 |
| \|ψ\|_max (x,z) | — | — | =mid | | — | — | =mid | | 9.849 | **9.626 (+0.14%) @(0.284,0.601)** | 9.612 @(0.285,0.601) |
| u_max (z) | 3.723 (+2.0%) | **3.650 (+0.03%) @0.813** | 3.649 @0.813 | | 16.386 (+1.3%) | **16.183 (+0.03%) @0.823** | 16.178 @0.823 | | 35.45 (+2.1%) | **34.76 (+0.08%) @0.855** | 34.73 @0.855 |
| w_max (x) | 3.775 (+2.1%) | **3.698 (+0.03%) @0.178** | 3.697 @0.178 | | 19.994 (+1.9%) | **19.631 (+0.07%) @0.119** | 19.617 @0.119 | | 70.70 (+3.1%) | **68.67 (+0.12%) @0.066** | 68.59 @0.066 |
| Nu̅ | 1.1260 (+0.7%) | **1.1179 (−0.01%)** | 1.118 | | 2.301 (+2.6%) | **2.246 (+0.15%)** | 2.243 | | 4.725 (+4.6%) | **4.532 (+0.28%)** | 4.519 |
| Nu_½ | 1.1260 | **1.1179 (−0.01%)** | 1.118 | | 2.301 | **2.246 (+0.15%)** | 2.243 | | 4.725 | **4.532 (+0.28%)** | 4.519 |
| Nu_0 | 1.1260 (+0.8%) | **1.1179 (+0.08%)** | 1.117 | | 2.302 (+2.8%) | **2.246 (+0.37%)** | 2.238 | | 4.727 (+4.8%) | **4.533 (+0.52%)** | 4.509 |
| Nu_max (z) | 1.533 (+1.9%) | **1.507 (+0.12%) @0.088** | 1.505 @0.092 | | 3.662 (+3.8%) | **3.536 (+0.23%) @0.144** | 3.528 @0.143 | | 8.322 (+7.8%) | **7.771 (+0.70%) @0.080** | 7.717 @0.081 |
| Nu_min (z=1) | 0.681 (−1.6%) | **0.691 (−0.13%)** | 0.692 | | 0.576 (−1.7%) | **0.585 (−0.17%)** | 0.586 | | 0.703 (−3.6%) | **0.728 (−0.15%)** | 0.729 |

**판정**: mirror(2차 벽) 로 **1e3~1e5 전 항목 ≤ 0.7%**(대부분 ≤ 0.3%), 위치 3자리 일치 —
논문이 벤치마크 정확도로 밝힌 "1% 이내"를 만족. value(1차 벽)는 +1~8% 로 Ra 와 함께 악화
(1e5 256² 에서 반감 → 1차). Ra=1e6 은 256² mirror 런 진행 중(잡 35638; value 256² 는
Nu 9.177 = +4.3%).
