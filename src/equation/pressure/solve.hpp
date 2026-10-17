#pragma once
// 압력(증분 van Kan 투영) solve — ∇²(δp) = ∇·u*/dt 를 풀고 속도를 투영한다.
//
// 풀이 구조 (MPM-STD_C pressure_engine 의 OB 경로 1:1; 분할 범위만 다름):
//   x r2c FFT → y c2c FFT → (kx,ky) 모드별 z 분산 TDMA → 역 y → 역 x
//   → 1/(Nx·Ny) 정규화 → 전역 평균 제거(게이지) → halo + z Neumann ghost
//
// ★ 분할 범위: **z-슬랩 전용 (npx = npy = 1)** — x, y 가 랭크 안에서 완결이라
//   펜슬 전치가 아예 없다. 현 프로덕션 패턴(GPU 2장 = z 분할)과 일치한다.
//   x/y 분할이 필요해지면 C→I→Y 펜슬 전치(alltoall)를 이 앞단에 추가한다
//   (MPM-STD_C pressure_engine_cpu.cpp 의 transpose_* 가 참조 구현).
//   생성자가 검사해서 어기면 즉시 throw — 조용한 오답은 없다.
// ★ x, y 는 균일 격자 전제 (스펙트럴 + modified wavenumber) — 생성자가 검사.
//   x 는 주기(FFT) **또는 벽(Neumann → DCT-II/III, 2026-10-05 BC 프레임워크 단계 3)**:
//   Topology 의 x periodic 플래그로 고른다. DCT 는 cuFFT 에 없어 **거울 확장 2N
//   실수 FFT + 트위들**로 양 백엔드가 같은 알고리즘을 쓴다 (solve_kernels.hpp 식).
//   x 모드 수 n_mode_x = Nx/2+1 (FFT) / Nx (DCT, 실수 모드를 복소 버퍼 실수부에 둠 —
//   허수부 시스템은 0 을 푸는 낭비지만 하류 코드가 전부 그대로다). y 벽은 미구현
//   (생성자 throw; cavity 는 y 주기 ny=1 로 2D).
// ★ 백엔드: FFT = FFTW(CPU) / cuFFT(GPU), 팩·산포·평균 = solve_kernels.{cpp,cu}
//   쌍둥이, 버퍼 = field::Buffer (CPU=호스트, GPU=디바이스 상주). 이 파일과
//   solve.cpp(오케스트레이션)는 양쪽 빌드 공용이다.
//
// 사용 (매 스텝, main 조립):
//   rhs_p.fill(0);
//   equation::pressure::add_divergence(rhs_p, U, V, W, grid, sub, dt);
//   equation::pressure::solve(pressure_solver, grid, sub, rhs_p, dP);
//   equation::pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, dt);
//   equation::pressure::accumulate(P, dP);
//   (U, V, W 는 이후 교환 + 벽 처리로 ghost 를 새로 맞춘다)

#ifdef MPMSTD_GPU_FFT
#include <cufft.h>
#else
#include <fftw3.h>
#endif

#include "common/types.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "common/phase_timer.hpp"

namespace mpmstd::equation::pressure {

// FFT 플랜·스펙트럼 버퍼·λ 표·z TDMA 플랜 — 한 번 만들어 매 스텝 재사용한다.
struct Solver {
  Solver(const grid::Grid& grid, const parallel::Subdomain& sub);
  ~Solver();
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  int n_x = 0;        // = 전역 nx (npx = 1 이므로 로컬과 같다)
  int n_y = 0;
  int n_mode_x = 0;   // x 모드 수: n_x/2 + 1 (주기 r2c) / n_x (벽 DCT)
  int nz_local = 0;
  bool x_neumann = false;  // x 벽(DCT 경로) 인가

  field::Buffer lambda_x, lambda_y;  // 차분 고유값 표 (백엔드 메모리)
  linear_solver::TdmaPlan plan_z;
  field::Buffer band_a, band_b, band_c, band_d;

  // 스펙트럼 작업 버퍼 — FFT 플랜이 이 주소에 묶이므로 재할당 금지.
  // 복소 버퍼는 (실, 허) double 쌍의 평탄 배열로 둔다 (커널에서 std::complex 회피).
  field::Buffer real_lines;   // [(jy·nz + k)·Nx + i]        (DCT: 라인 길이 2Nx 거울)
  field::Buffer spectrum_x;   // 2 × [(jy·nz + k)·Nxm + mx]   Nxm = n_mode_x
  field::Buffer spectrum_xy;  // 2 × [(k·Nxm + mx)·Ny + jy]
  field::Buffer mirror_spectrum;  // DCT 전용: 2 × [(jy·nz + k)·(Nx+1) + m]

#ifdef MPMSTD_GPU_FFT
  cufftHandle fft_x_forward = 0, fft_x_backward = 0, fft_y = 0;  // y 는 Z2Z 하나
#else
  fftw_plan forward_x = nullptr, backward_x = nullptr;
  fftw_plan forward_y = nullptr, backward_y = nullptr;
#endif

  // 확장성 측정용 (docs/Scalability/plan.md §1.1): nullptr 이 아니면 solve 가 자기 안의
  // 단계(팩·FFT·재배열·밴드 팩·z TDMA·언팩·산포·평균 제거·halo)를 "pressure.<단계>" 구간으로
  // 잰다. 수치는 불변 — 경계마다 동기화만 추가된다.
  mpmstd::common::PhaseTimer* timer = nullptr;
};

// δp 를 구한다. 반환 시 phi 는 내부 + ghost(이음새·주기 랩·z Neumann)까지 완성 —
// 바로 correct_velocity_constant_property 에 쓸 수 있다. rhs 는 건드리지 않는다 (계획 §5.5).
void solve(Solver& solver, const grid::Grid& grid, const parallel::Subdomain& sub,
           const field::Field& rhs, field::Field& phi);

// 투영 속도 보정·압력 누적·예측자 전진은 correction.hpp 로 분리 (2026-10-05)
// — Poisson solve(이 파일)와 역할이 달라서다. 앱은 둘 다 include 한다.

}  // namespace mpmstd::equation::pressure
