#pragma once
// [boundary] 섹션의 설정 — 앱 설정(rbc 계열·boundary_layer 등)이 공유하는 조각.
// 문법·기본값은 boundary_condition.hpp 머리말. 바인딩 표에 끼워 넣고(append_bindings),
// 파싱 뒤 전개(build)하고, 에코(format)한다. 섹션이 없으면 RBC 관례(x·y 주기, z 벽).
// (2026-10-06: rbc/settings.cpp 에서 분리 — 경계층 앱이 같은 코드를 쓰기 위해)

#include <string>
#include <vector>

#include "boundary/boundary_condition.hpp"
#include "config/ini_parser.hpp"
#include "config/schema.hpp"

namespace mpmstd::boundary {

struct BoundarySettings {
  std::string axis_kind[3] = {"periodic", "periodic", "wall"};
  std::string velocity_ghost = "value";   // value | mirror
  std::string inflow_profile = "";        // inflow_outflow 축이 있을 때: blasius | file:<경로>
  // 면 override 원문 [field u v w p t][axis][side minus plus]; "" = 지정 안 함
  std::string override_text[kFieldCount][3][2];
  BoundaryCondition condition;            // 전개 결과 (build 가 채움)
};

// 바인딩 표에 [boundary] 키 전부(옵션)를 덧붙인다 — 키 문자열은 정적 저장소에 산다.
void append_bindings(std::vector<config::Binding>& bindings, BoundarySettings& s);
// 파싱된 원문 → condition. 실패는 "파일명: [boundary] …" 로 throw.
void build(BoundarySettings& s, const config::IniFile& ini);
// 입력 파일 형식으로 에코 (왕복 가능: 비어 있지 않은 override 만).
void format(const BoundarySettings& s, std::string& out);

}  // namespace mpmstd::boundary
