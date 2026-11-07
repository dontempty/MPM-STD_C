#pragma once
// FFTU 기반 압력 Poisson 솔버 v2 — 기존 equation::pressure::Solver 와 같은 이산 문제를 푼다:
//   ∇²_h φ = rhs,  x·y 주기·균일 (차분 고유값 λ), z 벽 Neumann·비균일 가능 (모드별 z 삼중대각, PaScaL 정확해),
//   (0,0) 모드 전역 첫 행 핀 + 평면평균의 z 산술평균 제거 (기존 solve.cpp 와 같은 게이지) → 두 해가 반올림 수준으로 같아야 한다.
//
//   rhs 블록 ─to_cyclic─▶ y 순환 x 라인 ─FFTU 정─▶ 스펙트럼 (x 모드 절반, y 모드를 xy 그룹에 순환 분배)
//     ─ z TDMA (모드마다, 원래 z 분할의 z 통신자) ─▶ ─FFTU 역─▶ 라인 ─to_block (×1/(nx·ny) − 평균)─▶ phi 내부
//     → halo 교환 + z 벽 zero_gradient ghost (기존 solve 와 같은 후처리)
//
// 전제 (생성자가 검사): x·y 주기, z 벽, x·y 균일, q² | ny (q = npx·npy), npz > 1 이면 nz_local ≥ 2.
// 버퍼: CyclicFft 의 슬롯 4 개를 재분배 scratch·TDMA 밴드와 나눠 쓴다.

#include "block_cyclic_exchange.hpp"
#include "common/phase_timer.hpp"
#include "cyclic_fft.hpp"
#include "cyclic_layout.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "linear_solver/tdma/tdma.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace fft_test {

class CyclicPoissonSolver {
 public:
  CyclicPoissonSolver(const mpmstd::grid::Grid& grid, const mpmstd::parallel::Subdomain& sub);
  CyclicPoissonSolver(const CyclicPoissonSolver&) = delete;
  CyclicPoissonSolver& operator=(const CyclicPoissonSolver&) = delete;

  void solve(const mpmstd::field::Field& rhs, mpmstd::field::Field& phi);

  const CyclicLayout& layout() const { return layout_; }
  CyclicFft& fft() { return fft_; }
  // z TDMA 가 MPI_Alltoallv 안에서 보낸 누적 초 (GPU 백엔드만 잰다) — 계산·통신 분리용
  double tdma_comm_seconds() const { return plan_z_.comm_seconds; }

  mpmstd::common::PhaseTimer* timer = nullptr;  // nullptr 이 아니면 fftu.* 구간을 잰다

 private:
  const mpmstd::grid::Grid& grid_;
  const mpmstd::parallel::Subdomain& sub_;
  CyclicLayout layout_;
  CyclicFft fft_;
  BlockCyclicExchange exchange_;
  mpmstd::field::Buffer lambda_mode_;  // 로컬 모드별 λx + λy
  mpmstd::field::Buffer lambda_zero_;  // 길이 1, 0 (기존 밴드 조립 함수의 λ_y 자리)
  mpmstd::linear_solver::ExactTdmaPlan plan_z_;
  bool owns_bottom_ = false, owns_top_ = false;
  bool holds_mean_mode_ = false;  // 그룹 랭크 0 — 로컬 모드 0 = 전역 (0,0)
};

}  // namespace fft_test
