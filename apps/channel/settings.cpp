#include "settings.hpp"

#include <filesystem>
#include <vector>

#include "config/schema.hpp"

namespace channel {

using mpmstd::config::Binding;
using mpmstd::config::IniFile;
using mpmstd::config::format_number;
using mpmstd::config::require;

namespace {

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

      {"physics", "reynolds_bulk", &s.physics.reynolds_bulk, true},
      {"physics", "target_bulk_velocity", &s.physics.target_bulk_velocity, false},

      {"boundary", "velocity_ghost", &s.boundary.velocity_ghost, false},
      {"les", "model", &s.les.model, false},
      {"les", "c_s", &s.les.c_s, false},

      {"time", "dt_max", &s.time.dt_max, false},
      {"time", "max_cfl", &s.time.max_cfl, false},
      {"time", "max_step", &s.time.max_step, true},

      {"initial", "pert_amp", &s.initial.pert_amp, false},
      {"initial", "restart_from", &s.initial.restart_from, false},

      {"output", "stats_dir", &s.output.stats_dir, false},
      {"output", "monitor_interval", &s.output.monitor_interval, false},
      {"output", "stats_start_step", &s.output.stats_start_step, false},
      {"output", "stats_write_interval", &s.output.stats_write_interval, false},
      {"output", "budget_interval", &s.output.budget_interval, false},
      {"output", "checkpoint_interval", &s.output.checkpoint_interval, false},
      {"output", "checkpoint_dir", &s.output.checkpoint_dir, false},
  };
}

void validate(const IniFile& ini, const Settings& s) {
  require(ini, s.mesh.nx > 0, "'mesh.nx' must be > 0");
  require(ini, s.mesh.ny > 0, "'mesh.ny' must be > 0");
  require(ini, s.mesh.nz > 0, "'mesh.nz' must be > 0");
  require(ini, s.domain.lx > 0, "'domain.lx' must be > 0");
  require(ini, s.domain.ly > 0, "'domain.ly' must be > 0");
  require(ini, s.domain.lz > 0, "'domain.lz' must be > 0");
  require(ini, s.boundary.velocity_ghost == "value" || s.boundary.velocity_ghost == "mirror",
          "'boundary.velocity_ghost' must be value | mirror");
  require(ini, s.les.model == "none" || s.les.model == "constant_vreman",
          "'les.model' must be none | constant_vreman");
  require(ini, s.les.c_s > 0, "'les.c_s' must be > 0");
  if (s.stretch.uniform_z) {
    require(ini, s.stretch.gamma_z == 0,
            "'stretch.gamma_z' is set but 'stretch.uniform_z' is true");
  } else {
    require(ini, s.stretch.gamma_z > 0,
            "'stretch.gamma_z' must be > 0 when 'stretch.uniform_z' is false");
  }
  require(ini, s.process.npx > 0, "'process.npx' must be > 0");
  require(ini, s.process.npy > 0, "'process.npy' must be > 0");
  require(ini, s.process.npz > 0, "'process.npz' must be > 0");
  require(ini, s.physics.reynolds_bulk > 0, "'physics.reynolds_bulk' must be > 0");
  require(ini, s.physics.target_bulk_velocity > 0,
          "'physics.target_bulk_velocity' must be > 0");
  require(ini, s.time.dt_max > 0, "'time.dt_max' must be > 0");
  require(ini, s.time.max_cfl > 0, "'time.max_cfl' must be > 0");
  require(ini, s.time.max_step > 0, "'time.max_step' must be > 0");
  require(ini, s.initial.pert_amp >= 0, "'initial.pert_amp' must be >= 0");
  require(ini, s.output.monitor_interval > 0, "'output.monitor_interval' must be > 0");
  require(ini, s.output.stats_start_step >= 0,
          "'output.stats_start_step' must be >= 0");
  require(ini, s.output.stats_write_interval > 0,
          "'output.stats_write_interval' must be > 0");
  require(ini, s.output.budget_interval >= 0, "'output.budget_interval' must be >= 0");
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
  add("reynolds_bulk = " + format_number(s.physics.reynolds_bulk));
  add("target_bulk_velocity = " + format_number(s.physics.target_bulk_velocity));
  add("");
  add("[boundary]");
  add("velocity_ghost = " + s.boundary.velocity_ghost);
  add("");
  add("[les]");
  add("model = " + s.les.model);
  add("c_s = " + format_number(s.les.c_s));
  add("");
  add("[time]");
  add("dt_max = " + format_number(s.time.dt_max));
  add("max_cfl = " + format_number(s.time.max_cfl));
  add("max_step = " + std::to_string(s.time.max_step));
  add("");
  add("[initial]");
  add("pert_amp = " + format_number(s.initial.pert_amp));
  add("");
  add("[output]");
  add("stats_dir = \"" + s.output.stats_dir + "\"");
  add("monitor_interval = " + std::to_string(s.output.monitor_interval));
  add("stats_start_step = " + std::to_string(s.output.stats_start_step));
  add("stats_write_interval = " + std::to_string(s.output.stats_write_interval));
  add("budget_interval = " + std::to_string(s.output.budget_interval));
  add("checkpoint_interval = " + std::to_string(s.output.checkpoint_interval));
  add("checkpoint_dir = \"" + s.output.checkpoint_dir + "\"");
  add("restart_from = \"" + s.initial.restart_from + "\"");
  return out;
}

}  // namespace channel
