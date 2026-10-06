#include "boundary/boundary_condition.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace mpmstd::boundary {

namespace {

constexpr const char* kFieldName[kFieldCount] = {"u", "v", "w", "p", "t"};
constexpr const char* kAxisName[3] = {"x", "y", "z"};
constexpr const char* kSideName[2] = {"minus", "plus"};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

void fail(const std::string& what) { throw std::runtime_error("boundary: " + what); }

const char* bc_kind_name(BcKind k) {
  switch (k) {
    case BcKind::Periodic: return "periodic";
    case BcKind::Dirichlet: return "dirichlet";
    case BcKind::Neumann: return "neumann";
    case BcKind::Inflow: return "inflow";
    case BcKind::Outflow: return "outflow";
    case BcKind::Continuity: return "continuity";
  }
  return "?";
}

}  // namespace

AxisKind parse_axis_kind(const std::string& text) {
  const std::string s = lower(text);
  if (s == "periodic") return AxisKind::Periodic;
  if (s == "wall") return AxisKind::Wall;
  if (s == "dirichlet") return AxisKind::Dirichlet;
  if (s == "neumann") return AxisKind::Neumann;
  if (s == "inflow_outflow") return AxisKind::InflowOutflow;
  fail("unknown axis kind '" + text + "' (periodic | wall | dirichlet | neumann | inflow_outflow)");
  return AxisKind::Periodic;
}

const char* axis_kind_name(AxisKind kind) {
  switch (kind) {
    case AxisKind::Periodic: return "periodic";
    case AxisKind::Wall: return "wall";
    case AxisKind::Dirichlet: return "dirichlet";
    case AxisKind::Neumann: return "neumann";
    case AxisKind::InflowOutflow: return "inflow_outflow";
  }
  return "?";
}

FaceBc parse_face_override(const std::string& text) {
  const std::string s = lower(text);
  if (s == "adiabatic" || s == "slip" || s == "neumann") return {BcKind::Neumann, 0};
  if (s == "continuity") return {BcKind::Continuity, 0};
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0')
    fail("bad face override '" + text + "' (number | adiabatic | slip | continuity)");
  return {BcKind::Dirichlet, static_cast<real_t>(v)};
}

BoundaryCondition make_boundary_condition(
    const std::array<std::string, 3>& axis_kind_text,
    const std::string (&override_text)[kFieldCount][3][2],
    const std::string& velocity_ghost_text) {
  BoundaryCondition bc;
  {
    const std::string v = lower(velocity_ghost_text);
    if (v == "value") bc.velocity_ghost = VelocityGhost::Value;
    else if (v == "mirror") bc.velocity_ghost = VelocityGhost::Mirror;
    else fail("unknown velocity_ghost '" + velocity_ghost_text + "' (value | mirror)");
  }
  // ① 축 kind → 필드별 기본값
  for (int a = 0; a < 3; ++a) {
    bc.axis[a] = parse_axis_kind(axis_kind_text[a]);
    for (int f = 0; f < kFieldCount; ++f)
      for (int side = 0; side < 2; ++side) {
        FaceBc& fb = bc.face[f][a][side];
        switch (bc.axis[a]) {
          case AxisKind::Periodic: fb = {BcKind::Periodic, 0}; break;
          case AxisKind::Wall:
            fb = (f == static_cast<int>(Field::P)) ? FaceBc{BcKind::Neumann, 0}
                                                   : FaceBc{BcKind::Dirichlet, 0};
            break;
          case AxisKind::Dirichlet: fb = {BcKind::Dirichlet, 0}; break;
          case AxisKind::Neumann: fb = {BcKind::Neumann, 0}; break;
          case AxisKind::InflowOutflow:
            if (f == static_cast<int>(Field::P)) fb = {BcKind::Neumann, 0};
            else if (side == 0) fb = (f == a) ? FaceBc{BcKind::Inflow, 0} : FaceBc{BcKind::Dirichlet, 0};
            else fb = (f == static_cast<int>(Field::T)) ? FaceBc{BcKind::Neumann, 0}
                                                        : FaceBc{BcKind::Outflow, 0};
            break;
        }
      }
  }
  // ② override — 주기 축은 지정 자체가 오류(의도 불명을 fail-loud)
  for (int f = 0; f < kFieldCount; ++f)
    for (int a = 0; a < 3; ++a)
      for (int side = 0; side < 2; ++side) {
        const std::string& text = override_text[f][a][side];
        if (text.empty()) continue;
        const std::string key = std::string(kFieldName[f]) + "_" + kAxisName[a] + "_" +
                                kSideName[side];
        if (bc.axis[a] == AxisKind::Periodic)
          fail("'" + key + "' given but axis " + kAxisName[a] + " is periodic");
        if (f == static_cast<int>(Field::P)
            && (bc.axis[a] == AxisKind::Wall || bc.axis[a] == AxisKind::InflowOutflow))
          fail("'" + key + "': pressure on a wall/open boundary is always Neumann 0 (not overridable)");
        const FaceBc parsed = parse_face_override(text);
        if (parsed.kind == BcKind::Continuity && f != a)
          fail("'" + key + "': continuity is only for the wall-normal velocity component");
        bc.face[f][a][side] = parsed;
      }
  bc.validate();
  return bc;
}

void BoundaryCondition::validate() const {
  for (int a = 0; a < 3; ++a)
    for (int f = 0; f < kFieldCount; ++f)
      for (int side = 0; side < 2; ++side) {
        const FaceBc& fb = face[f][a][side];
        const std::string where = std::string(kFieldName[f]) + "_" + kAxisName[a] + "_" +
                                  kSideName[side];
        if (axis[a] == AxisKind::Periodic) {
          if (fb.kind != BcKind::Periodic) fail(where + ": periodic axis must be periodic");
          continue;
        }
        if (fb.kind == BcKind::Periodic) fail(where + ": non-periodic axis has periodic face");
        if (axis[a] == AxisKind::Wall) {
          if (f == static_cast<int>(Field::P) && fb.kind != BcKind::Neumann)
            fail(where + ": wall pressure must be Neumann");
          // 벽의 법선 속도(축 a 의 성분 a)는 불투과 → Dirichlet 만
          if (f == a && fb.kind != BcKind::Dirichlet && fb.kind != BcKind::Continuity)
            fail(where + ": wall-normal velocity must be Dirichlet (impermeable) or continuity (free-stream)");
        }
      }
}

std::string format_boundary_condition(const BoundaryCondition& bc) {
  std::ostringstream os;
  os << "  velocity_ghost: " << (bc.velocity_ghost == VelocityGhost::Mirror ? "mirror" : "value")
     << "\n";
  for (int a = 0; a < 3; ++a) {
    os << "  " << kAxisName[a] << ": " << axis_kind_name(bc.axis[a]);
    if (bc.axis[a] != AxisKind::Periodic) {
      for (int f = 0; f < kFieldCount; ++f)
        for (int side = 0; side < 2; ++side) {
          const FaceBc& fb = bc.face[f][a][side];
          os << "  " << kFieldName[f] << (side ? "+" : "-") << "=" << bc_kind_name(fb.kind);
          if (fb.kind == BcKind::Dirichlet) os << "(" << fb.value << ")";
          if (fb.kind == BcKind::Inflow) os << "(profile)";
        }
    }
    os << "\n";
  }
  return os.str();
}

}  // namespace mpmstd::boundary
