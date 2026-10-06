#include "boundary/boundary_settings.hpp"

#include <stdexcept>

namespace mpmstd::boundary {

namespace {
const char* const kField[] = {"u", "v", "w", "p", "t"};
const char* const kAxis[] = {"x", "y", "z"};
const char* const kSide[] = {"minus", "plus"};
}  // namespace

void append_bindings(std::vector<config::Binding>& b, BoundarySettings& s) {
  // 키 이름은 바인딩이 const char* 로 들고 있으므로 정적 저장소에 둔다 (프로세스 수명).
  static std::string key_storage[3 + kFieldCount * 3 * 2];
  int n = 0;
  b.push_back({"boundary", "velocity_ghost", &s.velocity_ghost, false});
  b.push_back({"boundary", "inflow_profile", &s.inflow_profile, false});
  for (int a = 0; a < 3; ++a) {
    key_storage[n] = kAxis[a];
    b.push_back({"boundary", key_storage[n++].c_str(), &s.axis_kind[a], false});
  }
  for (int f = 0; f < kFieldCount; ++f)
    for (int a = 0; a < 3; ++a)
      for (int side = 0; side < 2; ++side) {
        key_storage[n] = std::string(kField[f]) + "_" + kAxis[a] + "_" + kSide[side];
        b.push_back({"boundary", key_storage[n++].c_str(), &s.override_text[f][a][side],
                     false});
      }
}

void build(BoundarySettings& s, const config::IniFile& ini) {
  try {
    s.condition = make_boundary_condition({s.axis_kind[0], s.axis_kind[1], s.axis_kind[2]},
                                          s.override_text, s.velocity_ghost);
    if (s.condition.axis[0] == AxisKind::InflowOutflow && s.inflow_profile.empty())
      throw std::runtime_error("x = inflow_outflow needs inflow_profile (blasius | file:<path>)");
  } catch (const std::exception& e) {
    throw std::runtime_error(ini.name() + ": [boundary] " + e.what());
  }
}

void format(const BoundarySettings& s, std::string& out) {
  const auto add = [&out](const std::string& line) {
    out += line;
    out += '\n';
  };
  add("[boundary]");
  for (int a = 0; a < 3; ++a) add(std::string(kAxis[a]) + " = " + s.axis_kind[a]);
  add("velocity_ghost = " + s.velocity_ghost);
  if (!s.inflow_profile.empty()) add("inflow_profile = " + s.inflow_profile);
  for (int f = 0; f < kFieldCount; ++f)
    for (int a = 0; a < 3; ++a)
      for (int side = 0; side < 2; ++side) {
        const std::string& t = s.override_text[f][a][side];
        if (!t.empty()) add(std::string(kField[f]) + "_" + kAxis[a] + "_" + kSide[side] + " = " + t);
      }
  add("");
}

}  // namespace mpmstd::boundary
