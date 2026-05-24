#include "config/logger.hpp"

#include <cstdarg>
#include <cstdio>
#include <vector>

namespace mpmstd::config {

int      Logger::rank_        = 0;
LogLevel Logger::min_level_   = LogLevel::Info;
bool     Logger::initialized_ = false;

void Logger::init(int world_rank, LogLevel min_level) {
  rank_        = world_rank;
  min_level_   = min_level;
  initialized_ = true;
}

bool Logger::is_writer() { return rank_ == 0; }
int  Logger::rank()      { return rank_; }

const char* Logger::level_tag_(LogLevel l) {
  switch (l) {
    case LogLevel::Debug: return "DEBUG";
    case LogLevel::Info:  return "INFO ";
    case LogLevel::Warn:  return "WARN ";
    case LogLevel::Error: return "ERROR";
  }
  return "?    ";
}

void Logger::emit_(LogLevel level, const std::string& message) {
  if (!is_writer()) return;
  if (static_cast<int>(level) < static_cast<int>(min_level_)) return;
  std::fprintf(stderr, "[mpmstd][%s] %s\n", level_tag_(level), message.c_str());
}

static std::string vformat(const char* fmt, std::va_list ap) {
  std::va_list copy;
  va_copy(copy, ap);
  int len = std::vsnprintf(nullptr, 0, fmt, copy);
  va_end(copy);
  if (len < 0) return "<format error>";
  std::vector<char> buf(static_cast<std::size_t>(len) + 1);
  std::vsnprintf(buf.data(), buf.size(), fmt, ap);
  return std::string(buf.data(), static_cast<std::size_t>(len));
}

void Logger::debug(const char* fmt, ...) {
  if (!is_writer()) return;
  va_list ap; va_start(ap, fmt);
  emit_(LogLevel::Debug, vformat(fmt, ap));
  va_end(ap);
}
void Logger::info(const char* fmt, ...) {
  if (!is_writer()) return;
  va_list ap; va_start(ap, fmt);
  emit_(LogLevel::Info, vformat(fmt, ap));
  va_end(ap);
}
void Logger::warn(const char* fmt, ...) {
  if (!is_writer()) return;
  va_list ap; va_start(ap, fmt);
  emit_(LogLevel::Warn, vformat(fmt, ap));
  va_end(ap);
}
void Logger::error(const char* fmt, ...) {
  if (!is_writer()) return;
  va_list ap; va_start(ap, fmt);
  emit_(LogLevel::Error, vformat(fmt, ap));
  va_end(ap);
}

void Logger::info_str (std::string_view msg) { emit_(LogLevel::Info,  std::string(msg)); }
void Logger::warn_str (std::string_view msg) { emit_(LogLevel::Warn,  std::string(msg)); }
void Logger::error_str(std::string_view msg) { emit_(LogLevel::Error, std::string(msg)); }

} // namespace mpmstd::config
