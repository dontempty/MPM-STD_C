#pragma once
// 정확도 시험 (plan.md §6) — 순환 FFT 단위 시험과 Poisson 정확도·두 솔버 대조.

#include <mpi.h>

#include "cyclic_fft.hpp"
#include "cyclic_poisson.hpp"
#include "equation/pressure/solve.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace fft_test {

struct FftCheck {
  double delta_error = 0;      // 실수 평면파 → 스펙트럼 두 점, max |got − expected| / (nx·ny·(z+1))
  double roundtrip_error = 0;  // 정 → 역 = 원래 × nx·ny, 같은 정규화
  int modes_tested = 0;
};

// FFT 슬롯을 덮어쓴다 — Poisson 풀이 전에 부를 것.
FftCheck check_cyclic_fft(CyclicFft& fft, MPI_Comm comm);

struct PoissonCheck {
  // 제조해 φ = cos(2πx/Lx)·cos(2πy/Ly)·cos(πz/Lz), rhs = 이산 라플라시안 (stretched z 그대로)
  double manufactured_new = 0;  // max |φ_new − φ| / max |φ|
  double manufactured_old = 0;  // max |φ_old − φ| / max |φ|
  // 난수 rhs (전역 해시, 부피가중 평균 0 → Neumann 양립)
  double residual_new = 0;      // max |∇²_h φ_new − rhs| / max |rhs|
  double residual_old = 0;
  double difference = 0;        // max |φ_new − φ_old| / max |φ_old|
};

PoissonCheck check_poisson(CyclicPoissonSolver& new_solver, mpmstd::equation::pressure::Solver& old_solver,
                           const mpmstd::grid::Grid& grid, const mpmstd::parallel::Subdomain& sub,
                           double lx, double ly, double lz, mpmstd::field::Field& rhs,
                           mpmstd::field::Field& phi_new, mpmstd::field::Field& phi_old);

// CaNS·PaScaL 비교 문제 (COMPARISON_CONTRACT_HANGANG.md): φ = cos(πx)·cos(πy)·cos(πz), rhs = −3π²·φ (연속 식).
struct ContractError {
  double rms = 0;
  double linf = 0;
};
void fill_contract_rhs(mpmstd::field::Field& rhs, const mpmstd::grid::Grid& grid, const mpmstd::parallel::Subdomain& sub);
ContractError contract_error(const mpmstd::field::Field& phi, const mpmstd::grid::Grid& grid,
                             const mpmstd::parallel::Subdomain& sub);
// 2차 중심차분의 정확한 이산해로 계산한 RMS: |c − 1| / (2√2), c = 3π² / Σ (2/h²)(1 − cos πh). 균일 격자 전용.
double contract_expected_rms(const mpmstd::grid::Grid& grid);

// 난수 rhs 를 rhs 에 채운다 (bench 도 같은 rhs 를 쓴다).
void fill_random_rhs(mpmstd::field::Field& rhs, const mpmstd::grid::Grid& grid,
                     const mpmstd::parallel::Subdomain& sub);

}  // namespace fft_test
