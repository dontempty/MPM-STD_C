#pragma once
// 압력(증분 van Kan 투영) solve — ∇²(δp) = ∇·u*/dt 를 풀고 속도를 투영한다.
//
// 풀이 구조 (MPM-STD_C pressure_engine 의 OB 경로 1:1; 분할 범위만 다름):
//   x r2c FFT → y c2c FFT → (kx,ky) 모드별 z 분산 TDMA → 역 y → 역 x
//   → 1/(Nx·Ny) 정규화 → 전역 평균 제거(게이지) → halo + z Neumann ghost
//
// ★ 분할 범위: **3차원 분할 지원 (2026-10-20)**. npx = npy = 1 (z-슬랩)이면 기존 경로 그대로
//   (x, y 가 랭크 안에서 완결, 전치 없음 — 비트 동일). npx·npy > 1 이면 같은 z 블록의
//   xy-평면 그룹 안에서 펜슬 전치(pencil_transpose.hpp):
//     블록 →① x-펜슬 → x FFT →② y-펜슬 → y FFT → z 분산 TDMA → 역 y FFT →②⁻¹ → 역 x FFT →①⁻¹ 블록
//   z 는 원래 z 분할 그대로 두고 TDMA 가 z 통신자로 푼다. 전치는 백엔드 버퍼끼리(GPU = 디바이스↔디바이스,
//   CUDA-aware MPI; PaScaL_TDMA 와 같이 PASCAL_TDMA_MPI_MODE=host 면 호스트 경유).
//   조건: ny ≥ npx·npy, x 모드 수 ≥ npx·npy (생성자가 검사).
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
#include <memory>
#include <vector>

#include "equation/pressure/pencil_transpose.hpp"
#include "equation/workspace.hpp"
#include "linear_solver/tdma/tdma.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "common/phase_timer.hpp"

namespace mpmstd::equation::pressure {

// FFT 플랜·스펙트럼 버퍼·λ 표·z TDMA 플랜 — 한 번 만들어 매 스텝 재사용한다.
struct Solver {
  Solver(const grid::Grid& grid, const parallel::Subdomain& sub, Workspace& workspace);
  ~Solver();
  // 이 솔버가 요구하는 Workspace 슬롯 크기 — 슬랩 상한(real_lines = ny·nz_l·line_x, 스펙트럼·밴드 = 2·n_mode_x·ny·nz_l).
  static std::size_t required_slot_elements(const grid::Grid& grid, const parallel::Subdomain& sub);
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  int n_x = 0;        // 전역 nx (x 라인은 항상 전역 길이)
  int n_y = 0;        // 전역 ny (y 라인은 항상 전역 길이)
  int n_mode_x = 0;   // x 모드 수: n_x/2 + 1 (주기 r2c) / n_x (벽 DCT)
  int nz_local = 0;
  bool x_neumann = false;  // x 벽(DCT 경로) 인가
  // 펜슬 (npx·npy > 1) — 슬랩이면 pencil = nullptr, ny_lines = n_y, n_mode_local = n_mode_x
  std::unique_ptr<PencilLayout> pencil;
  int ny_lines = 0;       // x-펜슬의 로컬 y 라인 수
  int n_mode_local = 0;   // y-펜슬의 로컬 x 모드 수
  int mode_offset = 0;    // 그 첫 모드 번호 (λ_x 오프셋·(0,0) 핀 판정)

  field::Buffer lambda_x, lambda_y;  // 차분 고유값 표 (백엔드 메모리)
  linear_solver::TdmaPlan plan_z;
  // [최적화 2026-10-23] 밴드 a·b·c(D 는 spectrum_xy)·라인·스펙트럼·cuFFT 작업 영역은 공용 Workspace 슬롯을 가리키는
  //   포인터다 (docs/Memory/plan.md §2.3: 0 = real_lines ↔ band_c, 1 = spectrum_x, 2 = spectrum_xy, 3 = band_a ↔ fft_work,
  //   4 = band_b; ↔ = 수명이 겹치지 않는 쌍). 주소는 Workspace 생성 뒤 고정 — FFTW 플랜이 이 주소에 묶인다.
  //   거울 스펙트럼(DCT)만 솔버 소유.
  Workspace* workspace = nullptr;
  real_t* band_a = nullptr;
  real_t* band_b = nullptr;
  real_t* band_c = nullptr;
  // 복소 버퍼는 (실, 허) double 쌍의 평탄 배열로 둔다 (커널에서 std::complex 회피).
  real_t* real_lines = nullptr;   // [(jy·nz + k)·Nx + i]        (DCT: 라인 길이 2Nx 거울)
  real_t* spectrum_x = nullptr;   // 2 × [(jy·nz + k)·Nxm + mx]   Nxm = n_mode_x
  real_t* spectrum_xy = nullptr;  // 2 × [(k·Nxm + mx)·Ny + jy]
  field::Buffer mirror_spectrum;  // DCT 전용: 2 × [(jy·nz + k)·(Nx+1) + m]

#ifdef MPMSTD_GPU_FFT
  cufftHandle fft_x_forward = 0, fft_x_backward = 0, fft_y = 0;  // y 는 Z2Z 하나
  real_t* fft_work = nullptr;     // 세 계획 공용 cuFFT 작업 영역 = 슬롯 3 (cuFFT 요구가 슬롯보다 크면 fft_work_owned 로 후퇴)
  field::Buffer fft_work_owned;
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
// 바로 correct_velocity_constant_property 에 쓸 수 있다. rhs 는 첫 단계(라인 팩)에서만 읽고 phi 는 마지막
// 산포에서만 쓰므로 **rhs 와 phi 가 같은 Field 여도 된다** — 앱들은 rhs_P 자리에 dP 를 받는다 (2026-10-26, plan.md §9).
void solve(Solver& solver, const grid::Grid& grid, const parallel::Subdomain& sub,
           const field::Field& rhs, field::Field& phi);

// 투영 속도 보정·압력 누적·예측자 전진은 correction.hpp 로 분리 (2026-10-05)
// — Poisson solve(이 파일)와 역할이 달라서다. 앱은 둘 다 include 한다.

}  // namespace mpmstd::equation::pressure
