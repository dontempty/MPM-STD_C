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
// ★ x, y 는 주기 + 균일 격자 전제 (FFT + modified wavenumber) — 생성자가 검사.
//   x/y 벽(cavity)은 그 방향 FFT→DCT 교체가 본체다 (M5 주석과 같은 확장 지점).
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

namespace mpmstd::equation::pressure {

// FFT 플랜·스펙트럼 버퍼·λ 표·z TDMA 플랜 — 한 번 만들어 매 스텝 재사용한다.
struct Solver {
  Solver(const grid::Grid& grid, const parallel::Subdomain& sub);
  ~Solver();
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;

  int n_x = 0;        // = 전역 nx (npx = 1 이므로 로컬과 같다)
  int n_y = 0;
  int n_half_x = 0;   // = n_x/2 + 1 (r2c 절반 스펙트럼)
  int nz_local = 0;

  field::Buffer lambda_x, lambda_y;  // 차분 고유값 표 (백엔드 메모리)
  linear_solver::TdmaPlan plan_z;
  field::Buffer band_a, band_b, band_c, band_d;

  // 스펙트럼 작업 버퍼 — FFT 플랜이 이 주소에 묶이므로 재할당 금지.
  // 복소 버퍼는 (실, 허) double 쌍의 평탄 배열로 둔다 (커널에서 std::complex 회피).
  field::Buffer real_lines;   // [(jy·nz + k)·Nx + i]
  field::Buffer spectrum_x;   // 2 × [(jy·nz + k)·Nxh + mx]
  field::Buffer spectrum_xy;  // 2 × [(k·Nxh + mx)·Ny + jy]

#ifdef MPMSTD_GPU_FFT
  cufftHandle fft_x_forward = 0, fft_x_backward = 0, fft_y = 0;  // y 는 Z2Z 하나
#else
  fftw_plan forward_x = nullptr, backward_x = nullptr;
  fftw_plan forward_y = nullptr, backward_y = nullptr;
#endif
};

// δp 를 구한다. 반환 시 phi 는 내부 + ghost(이음새·주기 랩·z Neumann)까지 완성 —
// 바로 correct_velocity_constant_property 에 쓸 수 있다. rhs 는 건드리지 않는다 (계획 §5.5).
void solve(Solver& solver, const grid::Grid& grid, const parallel::Subdomain& sub,
           const field::Field& rhs, field::Field& phi);

// u ← u* − dt·∇φ (이 랭크 소유 면만; staggered 기울기 = Δφ/중심간거리).
// 하단 벽면 W 는 Neumann ghost 덕에 기울기가 정확히 0 — 자동으로 보존된다.
// 호출 뒤 U, V, W 의 ghost 는 낡는다 — 다음 사용 전 교환 + 벽 처리는 호출자 몫.
// (구현은 solve_kernels.{cpp,cu} — 백엔드별)
void correct_velocity_constant_property(field::Field& U, field::Field& V, field::Field& W,
                      const field::Field& phi, const grid::Grid& grid,
                      const parallel::Subdomain& sub, real_t dt);

// P += φ — ghost 포함 전체 칸. 둘 다 ghost 일관 상태라 덧셈만으로 P 의 ghost 도
// 유지된다 (재교환 불필요 — MPM-STD_C 의 pre.f90:818 관례 그대로).
void accumulate(field::Field& P, const field::Field& phi);

// ── 가변 밀도(NOB) 투영 (PaScaL_TCS §804-821 1:1) ───────────────────────────
// u −= dt·[ ∇(δp) + ((1/ρ̂)_면 − 1)·∇p̂ ].  rhs 의 ExtraTerm 과 짝 —
// 합치면 가변 밀도 투영이 반복 없이 닫힌다. p̂(dPhat)·1/ρ̂ ghost 최신 전제.
void correct_velocity_variable_property(field::Field& U, field::Field& V, field::Field& W,
                               const field::Field& phi,
                               const field::Field& pressure_predictor,
                               const field::Field& inverse_density,
                               const grid::Grid& grid, const parallel::Subdomain& sub,
                               real_t dt);

// 다음 스텝용 예측자 2차 외삽 (ghost 포함 전체; 호출 뒤 p̂ 를 교환+벽 채움):
//   previous ← (p̂ + previous)/2,   p̂ ← 2·δp − previous
void advance_pressure_predictor(field::Field& pressure_predictor,
                                field::Field& previous_predictor,
                                const field::Field& phi);

}  // namespace mpmstd::equation::pressure
