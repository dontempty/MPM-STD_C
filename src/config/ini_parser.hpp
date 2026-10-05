#pragma once
// INI 파서 — 파일을 (섹션, 키) → (값, 줄번호) 목록으로 바꾼다. 값의 의미는 모른다.
// 의미 해석(타입, 필수 여부, 범위)은 settings.cpp 가 한다. 이 분리 덕에 모든
// 에러 메시지에 "파일:줄" 을 넣을 수 있다. (docs/Plan/plan_20261003.md §7.3)
//
// 문법:
//   ; 와 # 는 주석 — 단, 큰따옴표 안에서는 주석 문자가 아니다
//   [section]
//   key = value            양쪽 공백은 제거된다
//   path = "has#hash/"     값이 " 로 시작하면 " 로 끝나야 하고, 따옴표는 벗겨진다
//
// 다음은 전부 예외(std::runtime_error, 메시지에 "파일:줄" 포함):
//   '=' 없는 줄 / 중복 키 / 섹션 선언 전의 키 / 닫히지 않은 따옴표 /
//   섹션 헤더 뒤의 잡문자 / 빈 섹션 이름 / 빈 키

#include <string>
#include <string_view>
#include <vector>

namespace mpmstd::config {

struct IniRecord {
  std::string section;
  std::string key;
  std::string value;  // 따옴표가 벗겨진 값
  int line = 0;       // 1-based, 에러 메시지용
};

class IniFile {
 public:
  // 파일에서 읽는다. 열기 실패 → throw.
  static IniFile load(const std::string& path);

  // 문자열에서 읽는다 (load 의 본체이자 테스트 진입점).
  // name 은 에러 메시지의 "파일" 자리에 들어간다.
  static IniFile parse(const std::string& text, std::string name);

  const std::string& name() const { return name_; }

  bool has(std::string_view section, std::string_view key) const {
    return find(section, key) != nullptr;
  }

  // 없으면 nullptr. (설정 파일은 수십 줄이라 선형 탐색으로 충분하다)
  const IniRecord* find(std::string_view section, std::string_view key) const;

  // 파일에 나온 순서 그대로 — settings 의 "모르는 키" 검사가 순회한다.
  const std::vector<IniRecord>& records() const { return records_; }

 private:
  std::string name_;
  std::vector<IniRecord> records_;
};

}  // namespace mpmstd::config
