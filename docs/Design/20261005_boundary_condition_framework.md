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
