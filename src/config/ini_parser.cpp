#include "config/ini_parser.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mpmstd::config {

namespace {

std::string trim(std::string_view s) {
  std::size_t a = 0;
  std::size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return std::string(s.substr(a, b - a));
}

[[noreturn]] void fail(const std::string& name, int line, const std::string& what) {
  throw std::runtime_error(name + ":" + std::to_string(line) + ": " + what);
}

}  // namespace

const IniRecord* IniFile::find(std::string_view section, std::string_view key) const {
  for (const IniRecord& r : records_) {
    if (r.section == section && r.key == key) return &r;
  }
  return nullptr;
}

IniFile IniFile::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open settings file '" + path + "'");
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return parse(buffer.str(), path);
}

IniFile IniFile::parse(const std::string& text, std::string name) {
  IniFile ini;
  ini.name_ = std::move(name);

  std::istringstream in(text);
  std::string raw;
  std::string section;
  bool have_section = false;
  int line_no = 0;

  while (std::getline(in, raw)) {
    ++line_no;

    // 1) 주석 제거 — 큰따옴표 안의 # ; 는 보존한다.
    bool in_quote = false;
    std::size_t cut = raw.size();
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const char c = raw[i];
      if (c == '"') {
        in_quote = !in_quote;
      } else if (!in_quote && (c == '#' || c == ';')) {
        cut = i;
        break;
      }
    }
    if (in_quote) fail(ini.name_, line_no, "closing '\"' is missing");

    const std::string s = trim(std::string_view(raw).substr(0, cut));
    if (s.empty()) continue;

    // 2) 섹션 헤더
    if (s.front() == '[') {
      const std::size_t close = s.find(']');
      if (close == std::string::npos)
        fail(ini.name_, line_no, "']' is missing in section header '" + s + "'");
      if (!trim(std::string_view(s).substr(close + 1)).empty())
        fail(ini.name_, line_no, "unexpected text after ']' in '" + s + "'");
      section = trim(std::string_view(s).substr(1, close - 1));
      if (section.empty()) fail(ini.name_, line_no, "empty section name");
      have_section = true;
      continue;
    }

    // 3) key = value
    const std::size_t eq = s.find('=');
    if (eq == std::string::npos)
      fail(ini.name_, line_no, "expected 'key = value', got '" + s + "'");

    const std::string key = trim(std::string_view(s).substr(0, eq));
    if (key.empty()) fail(ini.name_, line_no, "empty key before '='");
    if (!have_section)
      fail(ini.name_, line_no, "key '" + key + "' appears before any [section]");

    std::string value = trim(std::string_view(s).substr(eq + 1));

    // 따옴표 벗기기 — " 로 시작하면 " 로 끝나야 한다.
    if (!value.empty() && value.front() == '"') {
      if (value.size() < 2 || value.back() != '"')
        fail(ini.name_, line_no, "value starts with '\"' but does not end with it");
      value = value.substr(1, value.size() - 2);
    }

    if (const IniRecord* prev = ini.find(section, key))
      fail(ini.name_, line_no,
           "duplicate key '" + section + "." + key + "' (first seen at line "
               + std::to_string(prev->line) + ")");

    ini.records_.push_back(IniRecord{section, key, value, line_no});
  }

  return ini;
}

}  // namespace mpmstd::config
