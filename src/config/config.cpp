#include "config/config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <cstdlib>
#include <cerrno>

namespace mpmstd::config {

std::string Config::trim_(std::string_view s) {
  std::size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  std::size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
  return std::string(s.substr(a, b - a));
}

Config Config::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Config::load: cannot open '" + path + "'");
  }

  Config cfg;
  cfg.source_path_ = path;

  std::string current_section = "";   // empty = no section
  std::string line;
  int line_no = 0;

  while (std::getline(in, line)) {
    ++line_no;

    // Strip comments (# or ;) — but only when not inside quotes (we don't
    // support quotes yet, so this is fine).
    for (char c : { '#', ';' }) {
      auto pos = line.find(c);
      if (pos != std::string::npos) {
        line.erase(pos);
      }
    }

    auto trimmed = trim_(line);
    if (trimmed.empty()) continue;

    if (trimmed.front() == '[' && trimmed.back() == ']') {
      current_section = trim_(std::string_view(trimmed).substr(1, trimmed.size() - 2));
      continue;
    }

    auto eq = trimmed.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error(
        "Config::load: '" + path + "' line " + std::to_string(line_no) +
        ": expected key = value, got '" + trimmed + "'");
    }
    auto key   = trim_(std::string_view(trimmed).substr(0, eq));
    auto value = trim_(std::string_view(trimmed).substr(eq + 1));
    cfg.set(current_section, key, value);
  }

  return cfg;
}

void Config::set(std::string_view section, std::string_view key, std::string_view value) {
  data_[std::string(section)][std::string(key)] = std::string(value);
}

bool Config::has(std::string_view section, std::string_view key) const {
  auto it = data_.find(std::string(section));
  if (it == data_.end()) return false;
  return it->second.find(std::string(key)) != it->second.end();
}

std::string Config::get_raw(std::string_view section, std::string_view key) const {
  auto it = data_.find(std::string(section));
  if (it == data_.end()) {
    throw std::runtime_error("Config: missing section '" + std::string(section) + "'");
  }
  auto kit = it->second.find(std::string(key));
  if (kit == it->second.end()) {
    throw std::runtime_error("Config: missing key '" + std::string(key) +
                              "' in section '" + std::string(section) + "'");
  }
  return kit->second;
}

std::optional<std::string>
Config::try_get_raw(std::string_view section, std::string_view key) const {
  auto it = data_.find(std::string(section));
  if (it == data_.end()) return std::nullopt;
  auto kit = it->second.find(std::string(key));
  if (kit == it->second.end()) return std::nullopt;
  return kit->second;
}

std::vector<std::string> Config::sections() const {
  std::vector<std::string> out;
  out.reserve(data_.size());
  for (auto& kv : data_) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

// ===== detail::parse_* =====
namespace detail {

double parse_double(std::string_view s) {
  std::string tmp(s);
  errno = 0;
  char* endp = nullptr;
  double v = std::strtod(tmp.c_str(), &endp);
  if (errno != 0 || endp == tmp.c_str()) {
    throw std::runtime_error("Config: cannot parse double from '" + tmp + "'");
  }
  return v;
}

int parse_int(std::string_view s) {
  std::string tmp(s);
  errno = 0;
  char* endp = nullptr;
  long v = std::strtol(tmp.c_str(), &endp, 10);
  if (errno != 0 || endp == tmp.c_str()) {
    throw std::runtime_error("Config: cannot parse int from '" + tmp + "'");
  }
  return static_cast<int>(v);
}

bool parse_bool(std::string_view s) {
  std::string tmp(s);
  std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                  [](unsigned char c) { return std::tolower(c); });
  if (tmp == "true"  || tmp == "1" || tmp == "yes" || tmp == "on")  return true;
  if (tmp == "false" || tmp == "0" || tmp == "no"  || tmp == "off") return false;
  throw std::runtime_error("Config: cannot parse bool from '" + std::string(s) + "'");
}

std::string parse_string(std::string_view s) {
  // Strip surrounding quotes if present.
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                         (s.front() == '\'' && s.back() == '\''))) {
    return std::string(s.substr(1, s.size() - 2));
  }
  return std::string(s);
}

} // namespace detail

} // namespace mpmstd::config
