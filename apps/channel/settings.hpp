#pragma once
// 채널 앱의 설정 — 이 앱이 읽는 모든 키와 기본값이 이 파일 하나에 보인다.
// (설정은 앱 소유 — rbc/settings.hpp 와 같은 규칙. 채널은 온도가 없다.)
//
// 문제 조건의 출처: Filtered_TDMA/apps/channel_gpu 의 Re_τ=180 입력
// (PARA_INPUT_re180_pascal.dat) — 256³, 2π×π×2, Re_b=2857, CFR(벌크 1 유지),
// tanh z (γ_channel_gpu=1.5; 우리 공식은 γ/2 관례라 gamma_z=3.0 이 동일 격자).

#include <string>

#include "config/ini_parser.hpp"

namespace channel {

struct Mesh {          // [mesh] — 전역 내부 셀 수
  int nx = 0;          // 필수, > 0.  x = 주류(주기)
  int ny = 0;          // 필수, > 0.  y = 스팬(주기)
  int nz = 0;          // 필수, > 0.  z = 벽수직
};

struct Domain {        // [domain] — 반높이 δ = lz/2 가 길이 척도
  double lx = 0;       // 필수, > 0
  double ly = 0;       // 필수, > 0
  double lz = 0;       // 필수, > 0
};

struct Stretch {       // [stretch]
  bool uniform_z = false;
  double gamma_z = 0;  // uniform_z = false 면 필수, > 0
};

struct Process {       // [process] — 곱 = MPI 랭크 수 (topology 가 검사)
  int npx = 1;
  int npy = 1;
  int npz = 1;
};

struct Physics {                    // [physics]
  double reynolds_bulk = 0;         // 필수, > 0.  Re_b = U_b·δ/ν → ν = U_b·δ/Re_b
  double target_bulk_velocity = 1;  // CFR 목표 벌크 속도 U_b
};

struct Boundary {                      // [boundary] — 채널은 z 벽 고정; 접선 속도 ghost 정책만 고른다
  std::string velocity_ghost = "value";  // value(1차: no-slip 면이 ghost 중심, u_τ 자 = dmz — 참조 Fortran 관례, 기본)
                                         // | mirror(2차: 벽면 no-slip, 닫기 B−=A, u_τ 자 = 벽면↔첫 중심)
};

struct Les {                           // [les] — LES 와점성 (physics/LES; docs/Design/20261006_les_design.md)
  std::string model = "none";          // none | constant_vreman  (WALE 는 다음)
  double c_s = 0.12;                   // Vreman: C_v = 2.5·C_s² (참조 Fortran 0.12 → 0.036; 원논문 0.17)
};

struct Time {              // [time]
  double dt_max = 0.05;    // dt 상한 (channel_gpu 의 dtStart 에 해당)
  double max_cfl = 1.0;
  long max_step = 0;       // 필수, > 0
};

struct Initial {           // [initial] — channel_gpu 의 vortex IC (1:1)
  double pert_amp = 0.5;   // 섭동 크기 / U_max
  std::string restart_from = "";  // 체크포인트 경로 — 비어 있으면 냉시작
};

struct Output {                      // [output]
  std::string stats_dir = "stats/";  // 통계 파일 디렉터리 (랭크 0 이 쓴다)
  int monitor_interval = 10;         // 모니터 한 줄 간격 (스텝)
  long stats_start_step = 0;         // 통계 누적 시작 스텝 (매 스텝 표본)
  long stats_write_interval = 10000; // 프로파일 파일 출력 간격 (스텝)
  long budget_interval = 0;          // 평균 운동량 수지 진단 간격 (0 = 끔).
                                     // 한 스텝 안 4지점의 벌크를 재므로 Allreduce
                                     // 4회가 더 든다 — 진단용으로만 켠다.
  long checkpoint_interval = 0;      // 재시작 체크포인트 간격 (스텝, 0 = 끔)
  std::string checkpoint_dir = "checkpoints";  // 체크포인트 디렉터리
};

struct Settings {
  Mesh mesh;
  Domain domain;
  Stretch stretch;
  Process process;
  Physics physics;
  Boundary boundary;
  Les les;
  Time time;
  Initial initial;
  Output output;
};

Settings load_settings(const std::string& path);
Settings make_settings(const mpmstd::config::IniFile& ini);
std::string format_settings(const Settings& s);

}  // namespace channel
