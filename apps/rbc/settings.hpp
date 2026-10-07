#pragma once
// RBC 앱의 설정 — "설정은 타입이다". 이 앱이 읽는 모든 키와 기본값이 이 파일
// 하나에 보인다. 멤버 이름 오타 = 컴파일 에러, 입력 키 오타 = throw.
//
// 설정은 앱마다 다를 수 있으므로 **각 앱이 자기 settings 를 소유한다**
// (2026-10-03 사용자 결정). 공용 기계장치(바인딩 실행·타입 파싱·모르는 키
// 검출)는 src/config/schema.hpp 에 있다.
//
// 규칙:
//   * 멤버 초기화 값 = 입력 파일에서 생략했을 때의 기본값
//   * "필수" 로 표시된 키는 누락 시 load_settings 가 throw
//   * 모르는 키 / 타입 불일치 / 범위 위반도 전부 throw (설계 규칙 D10)
//
// 입력 파일 문법은 src/config/ini_parser.hpp, 예시는 input_rbc_ra1e6.ini.

#include <string>

#include "boundary/boundary_settings.hpp"
#include "config/ini_parser.hpp"

namespace rbc {

struct Mesh {          // [mesh] — 전역 내부 셀 수 (ghost 제외)
  int nx = 0;          // 필수, > 0
  int ny = 0;          // 필수, > 0
  int nz = 0;          // 필수, > 0.  z = 벽수직 축
};

struct Domain {        // [domain] — 도메인 크기 (free-fall 무차원)
  double lx = 0;       // 필수, > 0
  double ly = 0;       // 필수, > 0
  double lz = 0;       // 필수, > 0
};

struct Stretch {       // [stretch] — z 방향 격자 스트레칭 (x, y 는 항상 균일)
  bool uniform_z = true;  // false 면 tanh 스트레칭 (양쪽 벽 집중)
  double gamma_z = 0;     // uniform_z = false 일 때 필수, > 0
};

struct Process {       // [process] — 프로세스 격자. 곱이 MPI 랭크 수와 같아야
  int npx = 1;         //   한다 (검사는 M3 의 topology 가 한다)
  int npy = 1;
  int npz = 1;
};

struct Physics {           // [physics]
  double rayleigh = 0;     // 필수, > 0
  double prandtl = 0;      // 필수, > 0
  double delta_temperature = 0;  // 0 = 상수물성(OB), > 0 = 글리세롤 변수물성(NOB)
  int gravity_axis = 2;    // 0 = x, 1 = y, 2 = z
};

struct Time {              // [time]
  double end_time = 0;     // 필수, > 0 (free-fall time 단위)
  double dt_max = 0.02;    // dt 상한 — dt = min(max_cfl/speed, dt_max)
  double max_cfl = 0.3;    // 명시적 대류라 1.0 금지가 운용 경험칙
  long max_step = 0;       // 0 = 스텝 수 제한 없음
};

struct Initial {          // [initial] — 대류 세포를 깨우는 섭동 (참조 main 1:1)
  double perturb = 0.02;   // W 다중모드 섭동 진폭
  std::string restart_from = "";  // 체크포인트 경로(같은 격자) — 비어 있으면 냉시작
  std::string warmstart_from = "";  // 체크포인트 경로(다른 격자 → trilinear 보간)
};

struct Output {                      // [output]
  std::string stats_dir = "stats/";  // 통계·모니터 출력 디렉터리
  int monitor_interval = 100;        // 모니터 한 줄 출력 간격 (스텝)
  double stats_start_time = 0;       // 통계 누적 시작 (물리시간)
  double stats_interval = 0;         // 프로파일 파일 출력 간격 (물리시간, 0 = 안 씀)
  long checkpoint_interval = 0;      // 재시작 체크포인트 간격 (스텝, 0 = 끔)
  std::string checkpoint_dir = "checkpoints";  // 체크포인트 디렉터리
  // ── dhvc 확장 통계 (apps/dhvc/statistics.hpp; rbc 는 읽기만 하고 안 쓴다) ──
  int stats_sample_interval = 1;     // 평면 프로파일 표본 간격 (스텝; 간격 안 dt 합이 가중)
  int conditional_interval = 0;      // 조건평균(|Ri_f| 분류) 표본 간격 (스텝, 0 = stats_sample_interval)
  int spectra_interval = 0;          // u 스펙트럼 표본 간격 (스텝, 0 = 끔)
  double richardson_threshold = 0.1; // low-|Ri_f| 상한 (Ng 2017: 0.1)
  double richardson_z_plus = 15;     // |Ri_f| 정의의 z⁺ (Ng 2017: 15)
};

// [boundary] 는 공용 조각 boundary/boundary_settings.hpp (경계층 앱과 공유).

struct Settings {
  Mesh mesh;
  Domain domain;
  Stretch stretch;
  Process process;
  Physics physics;
  Time time;
  Initial initial;
  Output output;
  mpmstd::boundary::BoundarySettings boundary;
};

// 파일 → Settings. 파싱·검증 실패는 전부 std::runtime_error (위치 포함).
Settings load_settings(const std::string& path);

// 이미 파싱된 IniFile → Settings (load_settings 의 본체; 테스트가 직접 쓴다).
Settings make_settings(const mpmstd::config::IniFile& ini);

// 설정 전체를 입력 파일과 같은 형식으로, 결정적 순서로 문자열화한다.
// 런 로그 맨 앞에 박는 용도이며, 출력을 다시 load_settings 에 넣으면 같은
// 설정이 나온다 (test_rbc_settings 의 왕복 테스트가 보증).
std::string format_settings(const Settings& s);

}  // namespace rbc
