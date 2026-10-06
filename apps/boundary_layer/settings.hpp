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
  double reynolds_delta_star = 450;   // Re_δ*₀ = U∞ δ*₀/ν (Schlatter & Örlü 유입과 동일)
  double convective_velocity = 0;     // 유출 대류 속도 U_c (0 → U∞)
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
  Initial initial;
  Time time;
  Output output;
  mpmstd::boundary::BoundarySettings boundary;  // [boundary] — 공용 조각
};

Settings load_settings(const std::string& path);
Settings make_settings(const mpmstd::config::IniFile& ini);
std::string format_settings(const Settings& s);

}  // namespace boundary_layer
