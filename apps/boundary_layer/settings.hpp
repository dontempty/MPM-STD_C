#pragma once
// boundary_layer 앱 설정 — ZPG 평판 경계층 (유입 x−, 유출 x+, 벽 z=0, 자유류 z+).
// 무차원: 길이 δ*₀(유입 배제두께 = 1), 속도 U∞ → ν = U∞/Re_δ*₀. 자세한 설계는
// docs/Design/20261006_boundary_layer_validation.md.
#include <string>

#include "boundary/boundary_settings.hpp"
#include "config/ini_parser.hpp"

namespace boundary_layer {

struct Mesh {           // [mesh] — 전역 내부 셀 수
  int nx = 0, ny = 0, nz = 0;
};
struct Domain {         // [domain] — δ*₀ 단위
  double lx = 0, ly = 0, lz = 0;
};
struct Stretch {        // [stretch] — z 는 한쪽(벽) tanh (grid::make_tanh_wall_faces)
  bool uniform_z = false;
  double gamma_z = 2.5;
};
struct Process {
  int npx = 1, npy = 1, npz = 1;
};
struct Physics {                      // [physics]
  double u_infinity = 1.0;
  double reynolds = 450;              // Re = U∞·L/ν, L = 단위길이 1 (blasius: δ*₀ / recycling: δ99,in)
  double convective_velocity = 0;     // 유출 대류 속도 U_c (0 → U∞)
};
struct Inflow {                       // [inflow] — 유입 생성 방식 (boundary/recycling.hpp)
  std::string method = "blasius";     // blasius | recycling
  double recycle_x = 0;               // 재순환면 x (0 → 0.75 Lx)
  double delta99 = 1.0;               // 재순환: 유입 목표 δ99 (단위길이)
  double relaxation_time = 20;        // 재순환: ⟨Ū_r⟩ 완화 시간 (δ/U∞)
  double noise = 0.1;                 // 재순환 IC 섭동 진폭 (U∞ 배)
};
struct Initial {                      // [initial]
  std::string profile = "blasius";    // blasius | uniform
  std::string restart_from = "";
};
struct Time {                         // [time]
  double end_time = 0;                // 필수
  double dt_max = 0.5;
  double max_cfl = 0.8;
  long max_step = 0;                  // 0 = 제한 없음
};
struct Output {                       // [output]
  std::string stats_dir = "stats/";
  int monitor_interval = 100;
  double stats_interval = 0;          // 장 덤프 간격 (물리시간, 0 = 끝에만)
  long checkpoint_interval = 0;
  std::string checkpoint_dir = "checkpoints";
};

struct Settings {
  Mesh mesh;
  Domain domain;
  Stretch stretch;
  Process process;
  Physics physics;
  Inflow inflow;
  Initial initial;
  Time time;
  Output output;
  mpmstd::boundary::BoundarySettings boundary;  // [boundary] — 공용 조각
};

Settings load_settings(const std::string& path);
Settings make_settings(const mpmstd::config::IniFile& ini);
std::string format_settings(const Settings& s);

}  // namespace boundary_layer
