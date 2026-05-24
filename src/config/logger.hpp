#pragma once

#include <string>
#include <string_view>
#include <iosfwd>

namespace mpmstd::config {

// Rank-aware logger. Only the root rank writes to the configured stream
// (default: stderr). All other ranks see no-op calls.
//
// Usage:
//   Logger::init(world_rank);
//   Logger::info("starting solver with N = {}", N);   ← we don't have fmt yet;
//                                                       use printf style for now.
//   Logger::error("divergence > {:.3e}", div_max);

enum class LogLevel : int {
  Debug = 0,
  Info  = 1,
  Warn  = 2,
  Error = 3,
};

class Logger {
public:
  // Must be called once per process before any log call. world_rank == 0 means
  // this is the writer rank; others silently ignore log calls.
  static void init(int world_rank, LogLevel min_level = LogLevel::Info);

  static bool is_writer();
  static int  rank();

  // printf-style helpers (we add a fmt-based variant later if needed).
  static void debug(const char* fmt, ...);
  static void info (const char* fmt, ...);
  static void warn (const char* fmt, ...);
  static void error(const char* fmt, ...);

  // Plain string variants (no formatting).
  static void info_str (std::string_view msg);
  static void warn_str (std::string_view msg);
  static void error_str(std::string_view msg);

private:
  static int       rank_;
  static LogLevel  min_level_;
  static bool      initialized_;

  static void emit_(LogLevel level, const std::string& message);
  static const char* level_tag_(LogLevel l);
};

} // namespace mpmstd::config
