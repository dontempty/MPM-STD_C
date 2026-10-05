#pragma once
// 설정 스키마 공용 기계장치 — "어떤 키가 있고 어디에 담기는가"(바인딩 표)를
// 받아서 실행한다. 표 자체는 **각 앱이 소유한다** (apps/<앱>/settings.cpp —
// 앱마다 읽는 변수가 다르기 때문. 2026-10-03 사용자 결정).
//
// 여기 있는 것: 모르는 키 검출, 타입 파싱(토큰 전체 소비), 필수 키 검사,
//               의미 검증용 require, 에코용 숫자 포맷.
// 여기 없는 것: 구체적인 키 목록·기본값·범위 규칙 — 전부 앱의 settings 소유.

#include <string>
#include <variant>
#include <vector>

#include "common/types.hpp"
#include "config/ini_parser.hpp"

namespace mpmstd::config {

using BindingTarget = std::variant<int*, long*, double*, bool*, std::string*>;

// 입력 키 하나 ↔ 설정 멤버 하나. target 의 타입이 곧 파싱 방식이다.
struct Binding {
  const char* section;
  const char* key;
  BindingTarget target;
  bool required;
};

// ① ini 의 모든 키가 bindings 에 있는지 확인 — 모르는 키(오타)는 "파일:줄" 포함
//    throw. 오타가 조용히 기본값으로 떨어지는 사고를 원천 차단한다.
// ② 각 바인딩을 채운다 — 타입 오류는 "파일:줄" 포함 throw, 필수 키 누락은
//    "파일" 수준 throw, 없는 옵션 키는 target 의 기존 값(= 멤버 초기화 기본값) 유지.
// 정수는 토큰 전체 소비를 요구한다 ("8e6" 은 8 이 아니라 에러 — MPM-STD_C 버그).
// 의미 검증(범위 등)은 호출자가 require 로 이어서 한다.
void apply_bindings(const IniFile& ini, const std::vector<Binding>& bindings);

// 의미 검증용: ok 가 아니면 "<파일>: <what>" 으로 throw.
void require(const IniFile& ini, bool ok, const std::string& what);

// 에코용 숫자 포맷 — %.15g: 입력 파일에 적는 값(15 유효숫자 이하)은 손실 없이
// 왕복한다 (에코 출력을 다시 입력으로 쓸 수 있다).
std::string format_number(real_t v);

}  // namespace mpmstd::config
