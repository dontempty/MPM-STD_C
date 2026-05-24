# BC 처리 설계 보고서

> 대상: C++ MPM-STD 솔버의 경계조건 (Boundary Condition) 시스템
> 목적: **RBC** 와 **Channel** 두 문제를 동일 소스로 해결하기 위한 BC 추상화 설계
> 기준 컨벤션: **z = wall-normal** (Fortran MPM-STD 와 일치, RBC 기본값)
> 1차 지원 BC: **Periodic / Dirichlet / Neumann** 3 종만
> 미래 확장: Wall / Inflow / Outflow (인터페이스 hook 만 마련, 미구현)

---

## 1. 설계 원칙

[04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md), [03_MPM-STD_Fortran_analysis.md](03_MPM-STD_Fortran_analysis.md), [01_CaNS_analysis.md](01_CaNS_analysis.md), [02_PyFR_analysis.md](02_PyFR_analysis.md) 분석을 종합한 5 가지 원칙:

| # | 원칙 | 근거 |
|---|---|---|
| 1 | 축의 periodicity 는 **도메인 전역 공유** | MPI Cart `periods[]` 와 1:1. U 만 periodic 같은 비물리적 조합 불허 |
| 2 | non-periodic 면의 BC 종류·값은 **변수별 독립** | wall에서 U=0, ∂P/∂n=0, T=T_wall — 변수마다 다름 |
| 3 | 1차는 **Periodic/Dirichlet/Neumann 3 종**만 구현 | 단순성. RBC·Channel 둘 다 이걸로 충분 |
| 4 | 확장 BC (Wall, Inflow, Outflow) 는 **enum + dispatch** 로 미리 자리만 | 후일 자식 클래스 / case 추가로 확장 |
| 5 | 명시되지 않은 면은 **RBC 기본값** (z=wall-normal) 으로 자동 채움 | 사용자 보일러플레이트 제거 |

---

## 2. 데이터 모델

### 2.1 BC 종류 enum

```cpp
// include/mpmstd/bc_kind.hpp
namespace mpmstd {

enum class BcKind {
  // === 1차 구현 ===
  Periodic,    // 축 토폴로지에서 자동 도출 (직접 지정도 가능)
  Dirichlet,   // φ = value
  Neumann,     // ∂φ/∂n = value

  // === 미래 확장 (인터페이스만, 1차에서는 호출 시 throw) ===
  Wall,        // U,V,W → Dirichlet 0; P → Neumann 0; T → 사용자 지정
  Inflow,      // 시공간 함수 (속도/스칼라 분포)
  Outflow,     // convective 또는 zero-gradient
};

}  // namespace mpmstd
```

→ enum 값을 미리 둠으로써 switch 문에 case 추가만으로 확장 가능. 1차에서는 미지원 case 에 `throw std::runtime_error("not implemented")`.

### 2.2 FaceBc — 한 면의 BC

```cpp
// include/mpmstd/face_bc.hpp
#include <functional>

namespace mpmstd {

using BcValueFn = std::function<double(double x, double y, double z, double t)>;

struct FaceBc {
  BcKind   kind  = BcKind::Periodic;
  BcValueFn value = [](double,double,double,double){ return 0.0; };

  // 편의 생성자: 상수 값
  static FaceBc periodic();
  static FaceBc dirichlet(double v);
  static FaceBc neumann  (double v);
  static FaceBc dirichlet(BcValueFn f);
  static FaceBc neumann  (BcValueFn f);
};

}  // namespace mpmstd
```

→ 값은 항상 `BcValueFn`. 상수는 캡쳐 람다로 래핑 (인라인 함수 호출 비용 무시 가능).

### 2.3 FieldBoundary — 한 필드의 6 면

```cpp
// include/mpmstd/field_boundary.hpp
#include <array>

namespace mpmstd {

enum class Direction { X = 0, Y = 1, Z = 2 };
enum class Side      { Minus = 0, Plus = 1 };

class FieldBoundary {
public:
  // 6 면 인덱싱: [axis*2 + side]
  FaceBc& face(Direction d, Side s) { return faces_[2*int(d) + int(s)]; }
  const FaceBc& face(Direction d, Side s) const { return faces_[2*int(d) + int(s)]; }

private:
  std::array<FaceBc, 6> faces_;
};

}  // namespace mpmstd
```

### 2.4 DomainTopology — 축별 periodicity

```cpp
// include/mpmstd/domain_topology.hpp
namespace mpmstd {

enum class AxisTopology { Periodic, NonPeriodic };

struct DomainTopology {
  std::array<AxisTopology, 3> axis{
    AxisTopology::Periodic,     // x
    AxisTopology::Periodic,     // y
    AxisTopology::NonPeriodic   // z = wall-normal (RBC 기본)
  };

  bool is_periodic(Direction d) const {
    return axis[int(d)] == AxisTopology::Periodic;
  }

  // periodic 축 먼저, non-periodic 마지막 — ADI sweep 순서
  std::array<Direction, 3> sweep_order() const;

  // wall 축 = non-periodic 축 (1 차에선 1 개 가정)
  std::optional<Direction> wall_axis() const;
};

}  // namespace mpmstd
```

### 2.5 Problem — 모든 변수 BC 의 집합체

```cpp
// include/mpmstd/problem.hpp
namespace mpmstd {

class Problem {
public:
  DomainTopology topology;

  // 변수별 6 면 BC
  FieldBoundary U, V, W, P, T;
  // std::vector<FieldBoundary> scalars;   // 미래

  // 생성자: RBC 기본값으로 초기화 (z=wall-normal)
  Problem();

  // 미지정 면에 RBC 기본값을 채워 넣음 (사용자가 일부만 설정한 경우)
  void apply_rbc_defaults();

  // 일관성 검증: topology 와 면 BC 가 모순되는지 체크
  void validate() const;
};

}  // namespace mpmstd
```

---

## 3. RBC 기본값 — `Problem()` 생성자가 자동 설정

```cpp
// src/problem.cpp
Problem::Problem() {
  // 1) topology: z = wall-normal
  topology.axis[int(Direction::X)] = AxisTopology::Periodic;
  topology.axis[int(Direction::Y)] = AxisTopology::Periodic;
  topology.axis[int(Direction::Z)] = AxisTopology::NonPeriodic;

  // 2) 모든 변수 의 x, y 면 → Periodic
  for (auto* f : {&U, &V, &W, &P, &T}) {
    for (auto d : {Direction::X, Direction::Y})
      for (auto s : {Side::Minus, Side::Plus})
        f->face(d, s) = FaceBc::periodic();
  }

  // 3) z 면 (wall) 에서의 변수별 기본값
  //    U, V, W: no-slip → Dirichlet 0
  for (auto* f : {&U, &V, &W})
    for (auto s : {Side::Minus, Side::Plus})
      f->face(Direction::Z, s) = FaceBc::dirichlet(0.0);

  // 4) P: ∂P/∂n = 0 → Neumann 0
  for (auto s : {Side::Minus, Side::Plus})
    P.face(Direction::Z, s) = FaceBc::neumann(0.0);

  // 5) T: hot bottom, cold top (RBC 표준)
  T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet(+0.5);
  T.face(Direction::Z, Side::Plus ) = FaceBc::dirichlet(-0.5);
}
```

→ 사용자가 `Problem p;` 한 줄만 적으면 **완전한 RBC 셋업** 이 됨.

---

## 4. 사용 예시

### 4.1 기본 RBC — 1 줄

```cpp
mpmstd::Problem p;
// 끝. 자동으로 z=wall-normal RBC 설정 완료.
solver.solve(p);
```

### 4.2 RBC 의 벽 온도만 변경

```cpp
mpmstd::Problem p;     // 기본 RBC
p.T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet(+1.0);   // hot
p.T.face(Direction::Z, Side::Plus ) = FaceBc::dirichlet(-1.0);   // cold
// 나머지 (벽 속도, 압력 Neumann, x/y periodic) 는 기본값 유지
```

### 4.3 Channel flow (RBC 와 같은 z=wall 토폴로지, T 비활성)

```cpp
mpmstd::Problem p;     // 기본 RBC 셋업 (T 까지 자동)
// T 면 BC 를 그대로 두되, ThermalSolver 자체를 비활성화:
config.enable_thermal = false;
// → T.faces 는 무시됨 (ThermalSolver 가 안 돌면 BC 안 읽음)
```

→ Channel 은 T 만 끄면 됨. BC 구조 자체는 RBC 와 동일.

### 4.4 Inflow/Outflow Channel 의 미래 형태 (현재 throw)

```cpp
mpmstd::Problem p;
p.topology.axis[int(Direction::X)] = AxisTopology::NonPeriodic;   // x 도 non-periodic

p.U.face(Direction::X, Side::Minus) = {BcKind::Inflow,
  [](double, double, double y, double t){ return 1.5*(1 - y*y); }};
p.U.face(Direction::X, Side::Plus ) = {BcKind::Outflow, {}};
// 1차 구현에서는 BoundaryApplier 가 Inflow/Outflow case 에서 throw
// → 미래에 케이스 추가만으로 동작
```

### 4.5 사용자가 일부만 명시한 경우

```cpp
mpmstd::Problem p;
p.T.face(Direction::Z, Side::Minus) = FaceBc::dirichlet(2.5);
// T(+z), U/V/W/P 면들은 명시 안 함
// → 생성자에서 이미 RBC 기본값으로 채워둠 → OK
```

→ `apply_rbc_defaults()` 를 굳이 다시 호출할 필요 없음. **생성자가 한 번에 처리**.

---

## 5. 솔버 내부 — BoundaryApplier 가 디스패치

```cpp
// include/mpmstd/boundary_applier.hpp
class BoundaryApplier {
public:
  void apply_ghost(Field& f, const FieldBoundary& fbc, double t) const {
    for (int a = 0; a < 3; ++a) for (int s = 0; s < 2; ++s) {
      auto d = Direction(a); auto side = Side(s);
      const FaceBc& fb = fbc.face(d, side);
      switch (fb.kind) {
        case BcKind::Periodic:
          /* MPI halo 가 이미 처리 — skip */
          break;
        case BcKind::Dirichlet:
          fill_ghost_dirichlet(f, d, side, fb.value, t);
          break;
        case BcKind::Neumann:
          fill_ghost_neumann  (f, d, side, fb.value, t);
          break;

        // === 1차 미구현 ===
        case BcKind::Wall:
        case BcKind::Inflow:
        case BcKind::Outflow:
          throw std::runtime_error("BcKind not implemented yet");
      }
    }
  }

  // TDMA matrix row 보정 — non-periodic 축의 wall 면에서
  void modify_tdma_row(Direction d, const FieldBoundary& fbc,
                       double* A, double* B, double* C, double* D,
                       int n_sys, int n_row) const;
};
```

→ **변수별 분기 없음**. 같은 함수가 U, V, W, P, T 에 모두 동작. 차이는 `FieldBoundary` 의 내용뿐.

---

## 6. ADI sweep 순서 자동 도출

```cpp
// src/domain_topology.cpp
std::array<Direction, 3> DomainTopology::sweep_order() const {
  std::array<Direction, 3> order;
  int k = 0;
  // periodic 축 먼저
  for (int i = 0; i < 3; ++i)
    if (is_periodic(Direction(i))) order[k++] = Direction(i);
  // non-periodic (wall) 축 마지막
  for (int i = 0; i < 3; ++i)
    if (!is_periodic(Direction(i))) order[k++] = Direction(i);
  return order;
}
```

| 케이스 | topology | sweep_order() |
|---|---|---|
| RBC (z=wall, 기본) | P,P,N | X → Y → Z |
| PaScaL_TCS (y=wall) | P,N,P | X → Z → Y |
| 미래 Channel-IO (x,z=wall) | N,P,N | Y → X → Z |

→ PaScaL_TCS 의 z→x→y 와 Fortran MPM-STD 의 x→y→z 가 **같은 함수로** 도출됨.

---

## 7. validate() — 일관성 체크

```cpp
void Problem::validate() const {
  for (auto* f : {&U, &V, &W, &P, &T}) {
    for (int a = 0; a < 3; ++a) {
      auto d = Direction(a);
      bool axis_periodic = topology.is_periodic(d);
      for (auto s : {Side::Minus, Side::Plus}) {
        bool face_periodic = (f->face(d, s).kind == BcKind::Periodic);
        if (axis_periodic != face_periodic)
          throw std::runtime_error(
            "BC mismatch: axis periodicity != face BcKind::Periodic");
      }
    }
  }
}
```

→ 축이 periodic 이면 그 축의 모든 면은 `Periodic` 이어야 한다는 제약 강제. 생성자가 자동으로 맞춰주지만, 사용자가 수동으로 깨뜨릴 가능성 차단.

---

## 8. 확장 시나리오 — 미래에 Wall/Inflow/Outflow 를 더할 때

### 8.1 Wall 추가 (편의용)

`BoundaryApplier::apply_ghost` 의 case 만 추가:

```cpp
case BcKind::Wall:
  // 사용자가 어떤 변수인지 모르므로, 호출 측 (MomentumSolver/PressureSolver/...) 가
  // 자기 변수에 맞는 fb 를 미리 변환해서 전달하는 방식이 더 깔끔.
  // 또는 Field 에 VariableKind 태그를 두고 dispatch:
  switch (field_kind_of(f)) {
    case U: case V: case W: fill_ghost_dirichlet(f, d, side, /*0*/, t); break;
    case P:                 fill_ghost_neumann  (f, d, side, /*0*/, t); break;
    case T: throw std::runtime_error("Wall on T requires explicit value");
  }
  break;
```

→ enum 자리만 미리 비워뒀으니 case 추가만으로 동작. 다른 코드 손댈 필요 없음.

### 8.2 Inflow 추가

```cpp
case BcKind::Inflow:
  fill_ghost_dirichlet(f, d, side, fb.value, t);   // 일단 Dirichlet 과 동일 처리
  // 추후 convective 보정이나 다른 처리는 InflowPolicy 로 분리
  break;
```

### 8.3 Outflow 추가

```cpp
case BcKind::Outflow:
  apply_outflow_convective(f, d, side, fb.value, t);
  break;
```

→ 어떤 경우든 **이미 마련된 enum case + dispatch 한 곳만 채우면 됨**.

---

## 9. 헤더 / 소스 배치

```
include/mpmstd/
  direction.hpp              ← Direction, Side enum
  bc_kind.hpp                ← BcKind enum
  face_bc.hpp                ← FaceBc + 편의 생성자
  field_boundary.hpp         ← FieldBoundary (6 면)
  domain_topology.hpp        ← DomainTopology + sweep_order, wall_axis
  problem.hpp                ← Problem (전체 묶음)
  boundary_applier.hpp       ← apply_ghost, modify_tdma_row

src/
  face_bc.cpp                ← FaceBc::periodic/dirichlet/neumann
  field_boundary.cpp         ← (대부분 inline)
  domain_topology.cpp        ← sweep_order, wall_axis
  problem.cpp                ← 생성자 RBC 기본값, validate
  boundary_applier.cpp       ← case 별 fill_ghost_*, modify_tdma_row
```

---

## 10. 1차 구현 체크리스트

- [ ] `Direction`, `Side`, `AxisTopology`, `BcKind` enum 정의
- [ ] `FaceBc` struct + `BcValueFn` typedef + 편의 정적 생성자
- [ ] `FieldBoundary` 6 면 컨테이너
- [ ] `DomainTopology::sweep_order()`, `wall_axis()`
- [ ] `Problem` 생성자에서 RBC 기본값 (z = wall-normal) 자동 채움
- [ ] `Problem::validate()` 일관성 체크
- [ ] `BoundaryApplier::apply_ghost()` Periodic/Dirichlet/Neumann 3 case 구현
- [ ] `BoundaryApplier::apply_ghost()` Wall/Inflow/Outflow 3 case 는 throw stub
- [ ] `BoundaryApplier::modify_tdma_row()` Dirichlet/Neumann 행 변경 ([feedback_wall_bc_zero_ghost.md](../../.claude/projects/-shared-home-wel1come1234-workspace/memory/feedback_wall_bc_zero_ghost.md) 의 zero-ghost + matrix flag-drop 정책 반영)
- [ ] Unit test: 각 BC 종류, sweep_order 자동 도출, validate 의 mismatch 검출

---

## 11. 다른 분석 보고서와의 정합성

- [01_CaNS_analysis.md](01_CaNS_analysis.md): CaNS 의 `cbc(0:1,3)` 문자 코드 패턴 → 본 보고서의 `enum BcKind` + `FieldBoundary::face(d,s)`
- [02_PyFR_analysis.md](02_PyFR_analysis.md): PyFR 의 BC class 계층 → 1차에서는 enum + dispatch 로 단순화, 확장 시 자식 클래스 도입 가능
- [03_MPM-STD_Fortran_analysis.md](03_MPM-STD_Fortran_analysis.md): Fortran MPM-STD 의 변수별 BC 배열 (XMBC/YMBC/ZMBC) + z 하드코딩 → 본 보고서의 변수별 `FieldBoundary` + 축 자동 결정
- [04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md): PaScaL_TCS 의 `UBCup/bt/...` 변수별 배열 + y 하드코딩 → 동일하게 변수별 + 축 자동 결정

→ 본 BC 설계는 네 솔버의 BC 처리 방식을 **일관된 한 모델로 통합**한 것. RBC·Channel·미래 Inflow-Outflow 가 같은 데이터 구조로 표현됨.

---

## 12. 결론

**1차 구현 (Periodic/Dirichlet/Neumann 3 종)** 으로 RBC 와 Channel 두 문제를 같은 솔버에서 풀 수 있다:

- **RBC**: `Problem p;` 한 줄. z=wall-normal 자동, 벽 속도 0, 압력 Neumann 0, T hot/cold 자동.
- **Channel (forced)**: 동일한 `Problem p;` + `config.enable_thermal = false`. ThermalSolver 비활성.
- **Channel (Inflow/Outflow)**: 1 차 미지원이지만 enum case 와 dispatch slot 이 이미 마련되어 있어, 미래에 case 채우기만 하면 동작.

**핵심 디자인 가치**:
1. **변수별·면별 독립** BC (CaNS 식) + **축 토폴로지는 공유** (물리적 일관성)
2. **RBC 기본값 자동 채움** — 사용자 보일러플레이트 제거
3. **enum 슬롯 미리 확보** — 확장 시 다른 코드 손 대지 않음
4. **sweep order / TDMA cyclic / matrix row 보정** 이 모두 `Problem` 에서 자동 도출 — PaScaL_TCS·Fortran MPM-STD 의 하드코딩 원천 차단
