#include "settings.hpp"

#include <vector>

#include "config/schema.hpp"

namespace rbc_nob_scalability {

using mpmstd::config::Binding;
using mpmstd::config::IniFile;
using mpmstd::config::format_number;
using mpmstd::config::require;

namespace {

// ── 바인딩 표 — 이 앱이 아는 키의 전체 목록 (여기 없는 키는 unknown-key 에러) ──
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

      {"initial", "perturb", &s.initial.perturb, false},

      {"time", "max_step", &s.time.max_step, true},
      {"time", "dt_max", &s.time.dt_max, false},
      {"time", "max_cfl", &s.time.max_cfl, false},
      {"time", "viscous_coupling_number", &s.time.viscous_coupling_number, false},

      {"output", "monitor_interval", &s.output.monitor_interval, false},

      {"timing", "warmup_steps", &s.timing.warmup_steps, false},
      {"timing", "solver_phases", &s.timing.solver_phases, false},
  };
}

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

  require(ini, s.initial.perturb >= 0, "'initial.perturb' must be >= 0");
  require(ini, s.time.dt_max > 0, "'time.dt_max' must be > 0");
  require(ini, s.time.max_cfl > 0, "'time.max_cfl' must be > 0");
  require(ini, s.time.viscous_coupling_number >= 0, "'time.viscous_coupling_number' must be >= 0");

  require(ini, s.output.monitor_interval > 0, "'output.monitor_interval' must be > 0");
  require(ini, s.timing.warmup_steps >= 0, "'timing.warmup_steps' must be >= 0");
  require(ini, s.time.max_step > s.timing.warmup_steps,
          "'time.max_step' must be > 'timing.warmup_steps' (nothing would be timed)");
}

}  // namespace

Settings load_settings(const std::string& path) {
  const IniFile ini = IniFile::load(path);
  Settings s;
  mpmstd::config::apply_bindings(ini, make_bindings(s));
  validate(ini, s);
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
  add("");
  add("[initial]");
  add("perturb = " + format_number(s.initial.perturb));
  add("");
  add("[time]");
  add("max_step = " + std::to_string(s.time.max_step));
  add("dt_max = " + format_number(s.time.dt_max));
  add("max_cfl = " + format_number(s.time.max_cfl));
  add("viscous_coupling_number = " + format_number(s.time.viscous_coupling_number));
  add("");
  add("[output]");
  add("monitor_interval = " + std::to_string(s.output.monitor_interval));
  add("");
  add("[timing]");
  add("warmup_steps = " + std::to_string(s.timing.warmup_steps));
  add(std::string("solver_phases = ") + (s.timing.solver_phases ? "true" : "false"));
  return out;
}

}  // namespace rbc_nob_scalability
