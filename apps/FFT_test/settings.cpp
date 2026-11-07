#include "settings.hpp"

#include <vector>

#include "config/ini_parser.hpp"
#include "config/schema.hpp"

namespace fft_test {

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

      {"check", "fft", &s.check.fft, false},
      {"check", "accuracy", &s.check.accuracy, false},
      {"check", "contract", &s.check.contract, false},
      {"check", "tolerance", &s.check.tolerance, false},

      {"bench", "warmup", &s.bench.warmup, false},
      {"bench", "solves", &s.bench.solves, false},
      {"bench", "phases", &s.bench.phases, false},
      {"bench", "only", &s.bench.only, false},
  };
}

void validate(const IniFile& ini, const Settings& s) {
  require(ini, s.mesh.nx > 0 && s.mesh.ny > 0 && s.mesh.nz > 0, "'mesh.nx/ny/nz' must be > 0");
  require(ini, s.domain.lx > 0 && s.domain.ly > 0 && s.domain.lz > 0, "'domain.lx/ly/lz' must be > 0");
  if (!s.stretch.uniform_z) require(ini, s.stretch.gamma_z > 0, "'stretch.gamma_z' must be > 0");
  require(ini, s.process.npx > 0 && s.process.npy > 0 && s.process.npz > 0, "'process.np*' must be > 0");
  require(ini, s.check.tolerance > 0, "'check.tolerance' must be > 0");
  require(ini, s.bench.warmup >= 0, "'bench.warmup' must be >= 0");
  require(ini, s.bench.solves >= 0, "'bench.solves' must be >= 0");
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
  const auto flag = [](const bool v) { return std::string(v ? "true" : "false"); };
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
  add("uniform_z = " + flag(s.stretch.uniform_z));
  add("gamma_z = " + format_number(s.stretch.gamma_z));
  add("");
  add("[process]");
  add("npx = " + std::to_string(s.process.npx));
  add("npy = " + std::to_string(s.process.npy));
  add("npz = " + std::to_string(s.process.npz));
  add("");
  add("[check]");
  add("fft = " + flag(s.check.fft));
  add("accuracy = " + flag(s.check.accuracy));
  add("contract = " + flag(s.check.contract));
  add("tolerance = " + format_number(s.check.tolerance));
  add("");
  add("[bench]");
  add("warmup = " + std::to_string(s.bench.warmup));
  add("solves = " + std::to_string(s.bench.solves));
  add("phases = " + flag(s.bench.phases));
  if (!s.bench.only.empty()) add("only = " + s.bench.only);
  return out;
}

}  // namespace fft_test
