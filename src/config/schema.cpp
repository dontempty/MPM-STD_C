#include "config/schema.hpp"

#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace mpmstd::config {

namespace {

[[noreturn]] void fail_at(const IniFile& ini, const IniRecord& r, const std::string& what) {
  throw std::runtime_error(ini.name() + ":" + std::to_string(r.line) + ": " + what);
}

[[noreturn]] void fail_file(const IniFile& ini, const std::string& what) {
  throw std::runtime_error(ini.name() + ": " + what);
}

std::string qualified(const IniRecord& r) { return r.section + "." + r.key; }

// ── 값 파서 — 토큰 전체를 소비해야 성공 ─────────────────────────────────────

long parse_long(const IniFile& ini, const IniRecord& r) {
  std::size_t pos = 0;
  long v = 0;
  bool ok = !r.value.empty();
  if (ok) {
    try {
      v = std::stol(r.value, &pos);
    } catch (const std::exception&) {
      ok = false;
    }
  }
  if (ok && pos != r.value.size()) ok = false;  // "8e6", "3.5", "1 2" 거부
  if (!ok)
    fail_at(ini, r, "'" + qualified(r) + "' expects an integer, got '" + r.value + "'");
  return v;
}

int parse_int(const IniFile& ini, const IniRecord& r) {
  const long v = parse_long(ini, r);
  if (v < INT_MIN || v > INT_MAX)
    fail_at(ini, r, "'" + qualified(r) + "' is out of int range: '" + r.value + "'");
  return static_cast<int>(v);
}

double parse_double(const IniFile& ini, const IniRecord& r) {
  std::size_t pos = 0;
  double v = 0;
  bool ok = !r.value.empty();
  if (ok) {
    try {
      v = std::stod(r.value, &pos);
    } catch (const std::exception&) {
      ok = false;
    }
  }
  if (ok && (pos != r.value.size() || !std::isfinite(v))) ok = false;
  if (!ok)
    fail_at(ini, r, "'" + qualified(r) + "' expects a finite number, got '" + r.value + "'");
  return v;
}

bool parse_bool(const IniFile& ini, const IniRecord& r) {
  if (r.value == "true" || r.value == "1") return true;
  if (r.value == "false" || r.value == "0") return false;
  fail_at(ini, r, "'" + qualified(r) + "' expects true/false/1/0, got '" + r.value + "'");
}

void assign(const IniFile& ini, const IniRecord& r, int* p) { *p = parse_int(ini, r); }
void assign(const IniFile& ini, const IniRecord& r, long* p) { *p = parse_long(ini, r); }
void assign(const IniFile& ini, const IniRecord& r, double* p) { *p = parse_double(ini, r); }
void assign(const IniFile& ini, const IniRecord& r, bool* p) { *p = parse_bool(ini, r); }
void assign(const IniFile&, const IniRecord& r, std::string* p) { *p = r.value; }

}  // namespace

void apply_bindings(const IniFile& ini, const std::vector<Binding>& bindings) {
  // ① 모르는 키 → 즉시 에러
  for (const IniRecord& r : ini.records()) {
    bool known = false;
    for (const Binding& b : bindings) {
      if (r.section == b.section && r.key == b.key) {
        known = true;
        break;
      }
    }
    if (!known) fail_at(ini, r, "unknown key '" + qualified(r) + "'");
  }

  // ② 채우기 — (필수 누락 → throw) / (옵션 누락 → 기존 값 유지)
  for (const Binding& b : bindings) {
    const IniRecord* r = ini.find(b.section, b.key);
    if (r == nullptr) {
      if (b.required)
        fail_file(ini, std::string("missing required key '") + b.section + "." + b.key + "'");
      continue;
    }
    std::visit([&](auto* p) { assign(ini, *r, p); }, b.target);
  }
}

void require(const IniFile& ini, bool ok, const std::string& what) {
  if (!ok) fail_file(ini, what);
}

std::string format_number(real_t v) {
  char buffer[64];
  std::snprintf(buffer, sizeof buffer, "%.15g", static_cast<double>(v));
  return buffer;
}

}  // namespace mpmstd::config
