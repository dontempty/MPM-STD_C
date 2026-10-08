#include "settings.hpp"

#include <filesystem>
#include <vector>

#include "config/schema.hpp"

namespace boundary_layer {

using mpmstd::config::Binding;
using mpmstd::config::IniFile;
using mpmstd::config::format_number;
using mpmstd::config::require;

namespace {

std::vector<Binding> make_bindings(Settings& s) {
  std::vector<Binding> b = {
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
      {"physics", "u_infinity", &s.physics.u_infinity, false},
      {"physics", "reynolds", &s.physics.reynolds, true},
      {"inflow", "method", &s.inflow.method, false},
      {"inflow", "recycle_x", &s.inflow.recycle_x, false},
      {"inflow", "delta99", &s.inflow.delta99, false},
      {"inflow", "relaxation_time", &s.inflow.relaxation_time, false},
      {"inflow", "noise", &s.inflow.noise, false},
      {"trip", "amplitude", &s.trip.amplitude, false},
      {"trip", "x_center", &s.trip.x_center, false},
      {"trip", "length_x", &s.trip.length_x, false},
      {"trip", "length_z", &s.trip.length_z, false},
      {"trip", "time_scale", &s.trip.time_scale, false},
      {"trip", "modes", &s.trip.modes, false},
      {"trip", "end_time", &s.trip.end_time, false},
      {"physics", "convective_velocity", &s.physics.convective_velocity, false},
      {"initial", "profile", &s.initial.profile, false},
      {"initial", "restart_from", &s.initial.restart_from, false},
      {"time", "end_time", &s.time.end_time, true},
      {"time", "dt_max", &s.time.dt_max, false},
      {"time", "max_cfl", &s.time.max_cfl, false},
      {"time", "max_step", &s.time.max_step, false},
      {"output", "stats_dir", &s.output.stats_dir, false},
      {"output", "monitor_interval", &s.output.monitor_interval, false},
      {"output", "stats_interval", &s.output.stats_interval, false},
      {"output", "stats_start_time", &s.output.stats_start_time, false},
      {"output", "stations", &s.output.stations, false},
      {"output", "checkpoint_interval", &s.output.checkpoint_interval, false},
      {"output", "checkpoint_dir", &s.output.checkpoint_dir, false},
  };
  mpmstd::boundary::append_bindings(b, s.boundary);
  return b;
}

void validate(const IniFile& ini, const Settings& s) {
  require(ini, s.mesh.nx > 0 && s.mesh.ny > 0 && s.mesh.nz > 0, "'mesh.*' must be > 0");
  require(ini, s.domain.lx > 0 && s.domain.ly > 0 && s.domain.lz > 0, "'domain.*' must be > 0");
  require(ini, s.stretch.uniform_z || s.stretch.gamma_z > 0, "'stretch.gamma_z' must be > 0");
  require(ini, s.physics.u_infinity > 0, "'physics.u_infinity' must be > 0");
  require(ini, s.physics.reynolds > 0, "'physics.reynolds' must be > 0");
  require(ini, s.inflow.method == "blasius" || s.inflow.method == "recycling",
          "'inflow.method' must be blasius | recycling");
  require(ini, s.inflow.delta99 > 0 && s.inflow.relaxation_time > 0 && s.inflow.noise >= 0,
          "'inflow.delta99'/'relaxation_time' must be > 0, 'noise' >= 0");
  require(ini, s.physics.convective_velocity >= 0, "'physics.convective_velocity' must be >= 0");
  require(ini, s.trip.amplitude >= 0 && s.trip.length_x > 0 && s.trip.length_z > 0
                   && s.trip.time_scale > 0 && s.trip.modes > 0 && s.trip.end_time >= 0,
          "'trip.*': amplitude/end_time >= 0, length_x/length_z/time_scale/modes > 0");
  require(ini, s.initial.profile == "blasius" || s.initial.profile == "uniform",
          "'initial.profile' must be blasius | uniform");
  require(ini, s.time.end_time > 0, "'time.end_time' must be > 0");
  require(ini, s.time.dt_max > 0 && s.time.max_cfl > 0, "'time.dt_max'/'max_cfl' must be > 0");
  require(ini, s.output.monitor_interval > 0, "'output.monitor_interval' must be > 0");
  using mpmstd::boundary::AxisKind;
  require(ini, s.boundary.condition.axis[0] == AxisKind::InflowOutflow,
          "[boundary] x must be inflow_outflow for the boundary layer");
  require(ini, s.boundary.condition.axis[1] == AxisKind::Periodic, "[boundary] y must be periodic");
  require(ini, s.boundary.condition.axis[2] == AxisKind::Wall, "[boundary] z must be wall");
  require(ini, s.boundary.inflow_profile == "blasius" || s.boundary.inflow_profile == "recycling",
          "[boundary] inflow_profile must be blasius | recycling (= [inflow] method)");
  require(ini, s.boundary.inflow_profile == s.inflow.method,
          "[boundary] inflow_profile and [inflow] method must agree");
}

}  // namespace

Settings make_settings(const IniFile& ini) {
  Settings s;
  mpmstd::config::apply_bindings(ini, make_bindings(s));
  mpmstd::boundary::build(s.boundary, ini);
  validate(ini, s);
  return s;
}

Settings load_settings(const std::string& path) {
  Settings s = make_settings(IniFile::load(path));
  const auto resolve = [&](std::string& p) {
    if (p.empty()) return;
    std::filesystem::path q(p);
    if (q.is_relative()) q = std::filesystem::path(path).parent_path() / q;
    p = q.lexically_normal().string();
    while (p.size() > 1 && p.back() == '/') p.pop_back();
  };
  resolve(s.output.stats_dir);
  resolve(s.output.checkpoint_dir);
  resolve(s.initial.restart_from);
  return s;
}

std::string format_settings(const Settings& s) {
  std::string out;
  const auto add = [&out](const std::string& line) { out += line; out += '\n'; };
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
  add("u_infinity = " + format_number(s.physics.u_infinity));
  add("reynolds = " + format_number(s.physics.reynolds));
  add("convective_velocity = " + format_number(s.physics.convective_velocity));
  add("");
  add("[inflow]");
  add("method = " + s.inflow.method);
  add("recycle_x = " + format_number(s.inflow.recycle_x));
  add("delta99 = " + format_number(s.inflow.delta99));
  add("relaxation_time = " + format_number(s.inflow.relaxation_time));
  add("noise = " + format_number(s.inflow.noise));
  add("");
  add("[trip]");
  add("amplitude = " + format_number(s.trip.amplitude));
  add("x_center = " + format_number(s.trip.x_center));
  add("length_x = " + format_number(s.trip.length_x));
  add("length_z = " + format_number(s.trip.length_z));
  add("time_scale = " + format_number(s.trip.time_scale));
  add("modes = " + std::to_string(s.trip.modes));
  add("end_time = " + format_number(s.trip.end_time));
  add("");
  add("[initial]");
  add("profile = " + s.initial.profile);
  add("restart_from = \"" + s.initial.restart_from + "\"");
  add("");
  mpmstd::boundary::format(s.boundary, out);
  add("[time]");
  add("end_time = " + format_number(s.time.end_time));
  add("dt_max = " + format_number(s.time.dt_max));
  add("max_cfl = " + format_number(s.time.max_cfl));
  add("max_step = " + std::to_string(s.time.max_step));
  add("");
  add("[output]");
  add("stats_dir = \"" + s.output.stats_dir + "\"");
  add("monitor_interval = " + std::to_string(s.output.monitor_interval));
  add("stats_interval = " + format_number(s.output.stats_interval));
  add("stats_start_time = " + format_number(s.output.stats_start_time));
  add("stations = \"" + s.output.stations + "\"");
  add("checkpoint_interval = " + std::to_string(s.output.checkpoint_interval));
  add("checkpoint_dir = \"" + s.output.checkpoint_dir + "\"");
  return out;
}

}  // namespace boundary_layer
