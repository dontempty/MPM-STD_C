#pragma once
// 경계조건(BC) 자료구조 — 입력 [boundary] 섹션을 "필드별·축별·면별 BC" 테이블로
// 전개한 것. 솔버의 각 층(Topology 주기 플래그·ghost 채움·ADI 스윕 cyclic/벽·
// 압력 변환 FFT/DCT)이 이 테이블 하나를 보고 자기 처리를 고른다.
// (2026-10-05 설계; MPM-STD_C problem_loader 의 topology+override 모델 1:1)
//
// ── 입력 ([boundary]) ──────────────────────────────────────────────────────
//   x = periodic | wall | dirichlet | neumann        축 kind (y, z 동일)
//   <field>_<axis>_<side> = <숫자> | adiabatic | slip  면별 오버라이드
//       field ∈ u v w p t,  axis ∈ x y z,  side ∈ minus plus
//       숫자      → 그 면 Dirichlet(그 값)   예: t_x_minus = 0.5, u_z_plus = 1.0 (lid)
//       adiabatic → 그 면 Neumann 0 (온도: 단열)
//       slip      → 그 면 Neumann 0 (접선 속도: free-slip)   adiabatic/slip/neumann 은 동의어
//
// ── 축 kind → 필드별 기본값 (MPM-STD_C 관례) ─────────────────────────────────
//   periodic : 전 필드 Periodic (override 무시)
//   wall     : U V W T Dirichlet 0 (no-slip·벽온도 0),  P Neumann 0 (고정, 불가변)
//   dirichlet: 전 필드 Dirichlet 0
//   neumann  : 전 필드 Neumann 0
//
// ★ MPM-STD_C 와 다른 한 가지: override 가 값뿐 아니라 **종류**도 바꾼다
//   (숫자=Dirichlet / adiabatic·slip=Neumann). de Vahl Davis 가 같은 wall 에서 측벽
//   T=Dirichlet, 상하벽 T=Neumann 을 요구하고, slip 벽은 접선 속도만 Neumann 이기
//   때문. 벽 축의 **법선 속도**(x벽의 U 등)는 불투과라 Neumann 으로 못 바꾼다(validate).
//
// 좌표 관례: side 0 = minus(−), 1 = plus(+). field 인덱스는 Field enum.

#include <array>
#include <string>
#include <vector>

#include "common/types.hpp"

namespace mpmstd::boundary {

enum class BcKind { Periodic, Dirichlet, Neumann };
enum class AxisKind { Periodic, Wall, Dirichlet, Neumann };
enum class Field : int { U = 0, V = 1, W = 2, P = 3, T = 4 };
constexpr int kFieldCount = 5;

struct FaceBc {
  BcKind kind = BcKind::Periodic;
  real_t value = 0;
};

struct BoundaryCondition {
  std::array<AxisKind, 3> axis{AxisKind::Periodic, AxisKind::Periodic, AxisKind::Periodic};
  // face[field][axis][side]
  FaceBc face[kFieldCount][3][2];

  bool is_periodic(int axis_index) const { return axis[axis_index] == AxisKind::Periodic; }
  const FaceBc& at(Field f, int axis_index, int side) const {
    return face[static_cast<int>(f)][axis_index][side];
  }
  // 일관성 검사: 주기 축의 모든 필드가 Periodic, 비주기 축에 Periodic 없음,
  // wall 축의 P 는 Neumann. 위반 시 throw.
  void validate() const;
};

AxisKind parse_axis_kind(const std::string& text);     // "periodic"… → enum (틀리면 throw)
const char* axis_kind_name(AxisKind kind);

// 면 override 문자열 하나를 FaceBc 로: 숫자 → Dirichlet(값), "adiabatic" → Neumann 0.
// 빈 문자열은 "지정 안 함"(nullopt 의미) — 호출자가 걸러서 부른다.
FaceBc parse_face_override(const std::string& text);

// 축 kind 3개 + override 문자열 표[field][axis][side] → BC 테이블 (기본값 전개 후
// override 적용). override 가 빈 문자열이면 그 면은 기본값 유지.
BoundaryCondition make_boundary_condition(
    const std::array<std::string, 3>& axis_kind_text,
    const std::string (&override_text)[kFieldCount][3][2]);

std::string format_boundary_condition(const BoundaryCondition& bc);  // 로그 에코용

}  // namespace mpmstd::boundary
