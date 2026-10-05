#include "settings.hpp"

#include <filesystem>
#include <vector>

#include "config/schema.hpp"

namespace rbc {

using mpmstd::config::Binding;
using mpmstd::config::IniFile;
using mpmstd::config::format_number;
using mpmstd::config::require;

namespace {

// ── 바인딩 표 — 이 앱이 아는 키의 전체 목록 ─────────────────────────────────
// 새 설정은 settings.hpp 에 멤버 한 줄, 여기에 한 줄을 추가하면 끝이다.
// 여기 없는 키는 전부 unknown-key 에러가 된다 (schema.cpp).

std::vector<Binding> make_bindings(Settings& s) {
  return {
      {"mesh", "nx", &s.mesh.nx, true},
      {"mesh", "ny", &s.mesh.ny, true},
      {"mesh", "nz", &s.mesh.nz, true},

      {"domain", "lx", &s.domain.lx, true},
      {"domain", "ly", &s.domain.ly, true},
      {"domain", "lz", &s.domain.lz, true},

      {"stretch", "uniform_z", &s.stretch.uniform_z, false},
      {"stretch", "gamma_z", &s.stretch.gamma_z, false},

      {"process", "npx", &s.process.npx, false},
      {"process", "npy", &s.process.npy, false},
      {"process", "npz", &s.process.npz, false},

      {"physics", "rayleigh", &s.physics.rayleigh, true},
      {"physics", "prandtl", &s.physics.prandtl, true},
      {"physics", "delta_temperature", &s.physics.delta_temperature, false},
      {"physics", "gravity_axis", &s.physics.gravity_axis, false},

      {"initial", "perturb", &s.initial.perturb, false},
      {"initial", "restart_from", &s.initial.restart_from, false},
      {"initial", "warmstart_from", &s.initial.warmstart_from, false},

      {"time", "end_time", &s.time.end_time, true},
      {"time", "dt_max", &s.time.dt_max, false},
      {"time", "max_cfl", &s.time.max_cfl, false},
      {"time", "max_step", &s.time.max_step, false},

      {"output", "stats_dir", &s.output.stats_dir, false},
      {"output", "monitor_interval", &s.output.monitor_interval, false},
      {"output", "stats_start_time", &s.output.stats_start_time, false},
      {"output", "stats_interval", &s.output.stats_interval, false},
      {"output", "checkpoint_interval", &s.output.checkpoint_interval, false},
      {"output", "checkpoint_dir", &s.output.checkpoint_dir, false},
  };
}

// ── 의미 검증 — 값은 맞게 파싱됐지만 물리적으로 말이 안 되는 것 ─────────────

void validate(const IniFile& ini, const Settings& s) {
  require(ini, s.mesh.nx > 0, "'mesh.nx' must be > 0");
  require(ini, s.mesh.ny > 0, "'mesh.ny' must be > 0");
  require(ini, s.mesh.nz > 0, "'mesh.nz' must be > 0");

  require(ini, s.domain.lx > 0, "'domain.lx' must be > 0");
  require(ini, s.domain.ly > 0, "'domain.ly' must be > 0");
  require(ini, s.domain.lz > 0, "'domain.lz' must be > 0");

  if (s.stretch.uniform_z) {
    require(ini, s.stretch.gamma_z == 0,
            "'stretch.gamma_z' is set but 'stretch.uniform_z' is true — "
            "remove gamma_z or set uniform_z = false");
  } else {
    require(ini, s.stretch.gamma_z > 0,
            "'stretch.gamma_z' must be > 0 when 'stretch.uniform_z' is false");
  }

  require(ini, s.process.npx > 0, "'process.npx' must be > 0");
  require(ini, s.process.npy > 0, "'process.npy' must be > 0");
  require(ini, s.process.npz > 0, "'process.npz' must be > 0");

  require(ini, s.physics.rayleigh > 0, "'physics.rayleigh' must be > 0");
  require(ini, s.physics.prandtl > 0, "'physics.prandtl' must be > 0");
  require(ini, s.physics.delta_temperature >= 0,
          "'physics.delta_temperature' must be >= 0");
  require(ini, s.physics.gravity_axis >= 0 && s.physics.gravity_axis <= 2,
          "'physics.gravity_axis' must be 0, 1, or 2");

  require(ini, s.initial.perturb >= 0, "'initial.perturb' must be >= 0");
  require(ini, s.time.end_time > 0, "'time.end_time' must be > 0");
  require(ini, s.time.dt_max > 0, "'time.dt_max' must be > 0");
  require(ini, s.time.max_cfl > 0, "'time.max_cfl' must be > 0");
  require(ini, s.time.max_step >= 0, "'time.max_step' must be >= 0");

  require(ini, s.output.monitor_interval > 0, "'output.monitor_interval' must be > 0");
  require(ini, s.output.stats_start_time >= 0, "'output.stats_start_time' must be >= 0");
  require(ini, s.output.stats_interval >= 0, "'output.stats_interval' must be >= 0");
  require(ini, s.output.checkpoint_interval >= 0,
          "'output.checkpoint_interval' must be >= 0");
}

}  // namespace

Settings make_settings(const IniFile& ini) {
  Settings s;
  mpmstd::config::apply_bindings(ini, make_bindings(s));
  validate(ini, s);
  return s;
}

Settings load_settings(const std::string& path) {
  Settings s = make_settings(IniFile::load(path));
  // 출력은 앱 폴더 소유: 상대 stats_dir 는 입력 .ini 가 있는 폴더 기준으로 푼다
  // (어느 디렉터리에서 실행해도 결과가 리포 루트에 흩어지지 않는다).
  // 끝의 '/' 는 떼어 둔다 — 파일명 조립에서 "a//b" 가 되는 것 방지.
  std::filesystem::path stats(s.output.stats_dir);
  if (stats.is_relative())
    stats = std::filesystem::path(path).parent_path() / stats;
  s.output.stats_dir = stats.lexically_normal().string();
  while (s.output.stats_dir.size() > 1 && s.output.stats_dir.back() == '/')
    s.output.stats_dir.pop_back();
  std::filesystem::path ck(s.output.checkpoint_dir);
  if (ck.is_relative()) ck = std::filesystem::path(path).parent_path() / ck;
  s.output.checkpoint_dir = ck.lexically_normal().string();
  if (!s.initial.restart_from.empty()) {
    std::filesystem::path rs(s.initial.restart_from);
    if (rs.is_relative()) rs = std::filesystem::path(path).parent_path() / rs;
    s.initial.restart_from = rs.lexically_normal().string();
  }
  if (!s.initial.warmstart_from.empty()) {
    std::filesystem::path rs(s.initial.warmstart_from);
    if (rs.is_relative()) rs = std::filesystem::path(path).parent_path() / rs;
    s.initial.warmstart_from = rs.lexically_normal().string();
  }
  return s;
}

std::string format_settings(const Settings& s) {
  std::string out;
  const auto add = [&out](const std::string& line) {
    out += line;
    out += '\n';
  };

  add("[mesh]");
  add("nx = " + std::to_string(s.mesh.nx));
  add("ny = " + std::to_string(s.mesh.ny));
  add("nz = " + std::to_string(s.mesh.nz));
  add("");
  add("[domain]");
  add("lx = " + format_number(s.domain.lx));
  add("ly = " + format_number(s.domain.ly));
  add("lz = " + format_number(s.domain.lz));
  add("");
  add("[stretch]");
  add(std::string("uniform_z = ") + (s.stretch.uniform_z ? "true" : "false"));
  if (!s.stretch.uniform_z) add("gamma_z = " + format_number(s.stretch.gamma_z));
  add("");
  add("[process]");
  add("npx = " + std::to_string(s.process.npx));
  add("npy = " + std::to_string(s.process.npy));
  add("npz = " + std::to_string(s.process.npz));
  add("");
  add("[physics]");
  add("rayleigh = " + format_number(s.physics.rayleigh));
  add("prandtl = " + format_number(s.physics.prandtl));
  add("delta_temperature = " + format_number(s.physics.delta_temperature));
  add("gravity_axis = " + std::to_string(s.physics.gravity_axis));
  add("");
  add("[initial]");
  add("perturb = " + format_number(s.initial.perturb));
  add("restart_from = \"" + s.initial.restart_from + "\"");
  add("warmstart_from = \"" + s.initial.warmstart_from + "\"");
  add("");
  add("[time]");
  add("end_time = " + format_number(s.time.end_time));
  add("dt_max = " + format_number(s.time.dt_max));
  add("max_cfl = " + format_number(s.time.max_cfl));
  add("max_step = " + std::to_string(s.time.max_step));
  add("");
  add("[output]");
  // 경로는 항상 따옴표로 감싼다 — '#' 이나 ';' 이 들어 있어도 재파싱이 안전하다.
  add("stats_dir = \"" + s.output.stats_dir + "\"");
  add("monitor_interval = " + std::to_string(s.output.monitor_interval));
  add("stats_start_time = " + format_number(s.output.stats_start_time));
  add("stats_interval = " + format_number(s.output.stats_interval));
  add("checkpoint_interval = " + std::to_string(s.output.checkpoint_interval));
  add("checkpoint_dir = \"" + s.output.checkpoint_dir + "\"");
  return out;
}

}  // namespace rbc
