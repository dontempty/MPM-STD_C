#pragma once
// rbc_nob_scalability 설정 — rbc_nob 확장성 측정 전용 앱이 읽는 키 전부 (docs/Scalability/plan.md §4.1).
//
// rbc 공용 설정(apps/rbc/settings.hpp)에서 측정 런이 쓰는 키만 남기고 — 재시작·통계·체크포인트·
// 경계 조각·gravity_axis(항상 z) 는 없다 — 측정 전용 [timing] 을 더했다. 규칙은 공용 설정과 같다:
//   * 멤버 초기화 값 = 입력 파일에서 생략했을 때의 기본값
//   * "필수" 키 누락, 모르는 키, 타입 불일치, 범위 위반은 전부 load_settings 가 throw
// 측정 런은 냉시작 + 스텝 수 고정(max_step)이다. 덱은 gen_decks.py 가 만든다.

#include <string>

#include "config/ini_parser.hpp"

namespace rbc_nob_scalability {

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

struct Process {       // [process] — 프로세스 격자. 곱이 MPI 랭크 수와 같아야 한다 (Topology 가 검사)
  int npx = 1;
  int npy = 1;
  int npz = 1;
};

struct Physics {           // [physics]
  double rayleigh = 0;     // 필수, > 0
  double prandtl = 0;      // 필수, > 0
  double delta_temperature = 0;  // 0 = 상수물성 극한, > 0 = 글리세롤 변수물성(NOB, 차원 K)
};

struct Time {              // [time]
  long max_step = 0;       // 필수, > timing.warmup_steps — 측정 런의 길이
  double dt_max = 0.02;    // dt 상한 — dt = min(max_cfl/speed, dt_max)
  double max_cfl = 0.3;
  double viscous_coupling_number = 0;  // > 0 이면 rbc_nob 과 같은 가변 점성 dt 한계 (0 = 끔)
};

struct Initial {           // [initial]
  double perturb = 0.02;   // W 다중모드 섭동 진폭 (rbc_nob 냉시작과 같은 IC)
};

struct Output {            // [output]
  int monitor_interval = 50;  // 모니터 한 줄 간격 (스텝) — monitor 구간으로 따로 잰다
};

struct Timing {            // [timing] — 구간 타이머 (common/phase_timer.hpp)
  long warmup_steps = 50;  // 앞의 이 스텝 수는 누적하지 않는다 (cuFFT 플랜·첫 할당·클럭 상승)
  bool solver_phases = false;  // true 면 momentum/scalar/pressure 풀이 안의 호출별 구간도 잰다
                               //   (구간 경계마다 동기화가 늘어 바깥 구간 시간이 조금 커진다)
};

struct Settings {
  Mesh mesh;
  Domain domain;
  Stretch stretch;
  Process process;
  Physics physics;
  Time time;
  Initial initial;
  Output output;
  Timing timing;
};

// 파일 → Settings. 파싱·검증 실패는 전부 std::runtime_error (위치 포함).
Settings load_settings(const std::string& path);

// 설정 전체를 입력 파일과 같은 형식으로, 결정적 순서로 문자열화한다 (런 로그 맨 앞에 박는다).
std::string format_settings(const Settings& s);

}  // namespace rbc_nob_scalability
