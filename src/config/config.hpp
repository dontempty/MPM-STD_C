#pragma once

#include "common/types.hpp"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <optional>
#include <stdexcept>

namespace mpmstd::config {

// Minimal INI-style configuration.
//
// Format
// ------
//   ; or # comments
//   [section]
//   key = value
//
// Values are stored as strings; typed accessors parse on demand and throw
// std::runtime_error on missing keys / parse failures (caller can use the
// `_or` variants for defaults).
//
// This is intentionally simple — we can drop in a TOML / YAML parser later
// without changing the call sites.

class Config {
public:
  // Construct empty config (for tests).
  Config() = default;

  // Load from file. Path is also stored for diagnostic prints.
  static Config load(const std::string& path);

  const std::string& source_path() const { return source_path_; }

  // ----- raw access -----
  bool        has(std::string_view section, std::string_view key) const;
  std::string get_raw(std::string_view section, std::string_view key) const;
  std::optional<std::string>
              try_get_raw(std::string_view section, std::string_view key) const;

  // ----- typed access -----
  template <class T>
  T get(std::string_view section, std::string_view key) const;

  template <class T>
  T get_or(std::string_view section, std::string_view key, T fallback) const;

  // Convenience: list every section name (for debug dumps).
  std::vector<std::string> sections() const;

  // For tests / programmatic config.
  void set(std::string_view section, std::string_view key, std::string_view value);

private:
  using KV = std::unordered_map<std::string, std::string>;
  std::unordered_map<std::string, KV> data_;   // section -> key -> value
  std::string                          source_path_;

  static std::string trim_(std::string_view s);
};

// ===== Typed conversion =====
namespace detail {
double      parse_double(std::string_view s);
int         parse_int   (std::string_view s);
bool        parse_bool  (std::string_view s);
std::string parse_string(std::string_view s);
} // namespace detail

template <> inline std::string Config::get<std::string>(std::string_view sec, std::string_view k) const {
  return detail::parse_string(get_raw(sec, k));
}
template <> inline int Config::get<int>(std::string_view sec, std::string_view k) const {
  return detail::parse_int(get_raw(sec, k));
}
template <> inline double Config::get<double>(std::string_view sec, std::string_view k) const {
  return detail::parse_double(get_raw(sec, k));
}
template <> inline float Config::get<float>(std::string_view sec, std::string_view k) const {
  return static_cast<float>(detail::parse_double(get_raw(sec, k)));
}
template <> inline bool Config::get<bool>(std::string_view sec, std::string_view k) const {
  return detail::parse_bool(get_raw(sec, k));
}

template <class T>
T Config::get_or(std::string_view section, std::string_view key, T fallback) const {
  auto raw = try_get_raw(section, key);
  if (!raw.has_value()) return fallback;
  // Re-use the typed conversion path:
  Config tmp;
  tmp.set(section, key, *raw);
  return tmp.get<T>(section, key);
}

} // namespace mpmstd::config
