#pragma once
// FFT_test 설정 — 앱 하나가 소유하는 INI 스키마 (readme §4 규칙: 모르는 키·타입·범위 위반은 throw,
// 멤버 초기값 = 생략했을 때의 기본값, 로그 맨 앞 에코는 재파싱 가능).
//
//   [mesh] nx ny nz        [domain] lx ly lz        [stretch] uniform_z gamma_z
//   [process] npx npy npz  (순환 격자 q = npx·npy, 조건 q² | ny)
//   [check] fft accuracy contract tolerance         [bench] warmup solves phases only

#include <string>

namespace fft_test {

struct Mesh {
  int nx = 0;  // 필수
  int ny = 0;  // 필수
  int nz = 0;  // 필수 (z = 벽수직, 스트레칭 축)
};

struct Domain {
  double lx = 0;  // 필수
  double ly = 0;  // 필수
  double lz = 0;  // 필수
};

struct Stretch {
  bool uniform_z = false;  // 이 과제는 z 스트레칭이 기본 (사용자 결정 2026-11-06)
  double gamma_z = 2.6;    // tanh 양벽 집중 (RBC 관례값)
};

struct Process {
  int npx = 1;
  int npy = 1;
  int npz = 1;
};

struct Check {
  bool fft = true;         // 순환 FFT 단위 시험 (실수 평면파 → 스펙트럼 두 점, 정·역 왕복)
  bool accuracy = true;    // Poisson 정확도: 제조해 이산 일관성 + 난수 rhs 잔차 + 두 솔버 대조
  bool contract = false;   // CaNS·PaScaL 비교 모드 (plan.md §10): 해석해 RMS + 방법별 시간. lx·ly·lz 는 양의 짝수, z 균일
  double tolerance = 1e-9; // 상대 오차 허용치 (실패해도 bench 는 돌지만 결과에 FAIL 이 남는다)
};

struct Bench {
  int warmup = 5;      // 측정 전 각 솔버 풀이 수 (cuFFT·MPI 첫 호출, GTL 첫 집합통신)
  int solves = 20;     // 측정 풀이 수 (솔버마다; 기존 → 새 교대)
  bool phases = true;  // true 면 총시간 측정 뒤에 단계별 측정을 한 번 더 (경계마다 동기화)
  std::string only;    // 비교 모드에서 이 방법들만 잰다, 쉼표로 구분 (예: mpm_plane,mpm_pencil). 비우면 전부
};

struct Settings {
  Mesh mesh;
  Domain domain;
  Stretch stretch;
  Process process;
  Check check;
  Bench bench;
};

Settings load_settings(const std::string& path);
std::string format_settings(const Settings& s);

}  // namespace fft_test
