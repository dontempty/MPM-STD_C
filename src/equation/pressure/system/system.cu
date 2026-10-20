#include "equation/pressure/system/system.hpp"

#include <cstddef>

#include <mpi.h>

#include "common/cuda_check.hpp"
#include "equation/pressure/pressure_stencil.hpp"

// GPU 백엔드 — system.cpp 의 쌍둥이 (make_lambda_* 는 호스트 공용이라 CPU TU 에만
// 있고, 여기는 밴드 조립 커널만 구현한다… 가 아니라, 빌드가 TU 를 통째로
// 바꾸므로 λ 생성도 여기 한 벌 더 있다 — 수식은 한 줄짜리 eigenvalue 뿐이고
// 그것도 stencil 이 아니라 호스트 전용이라 중복을 허용한다 (D3 의 "수식" =
// 셀 스텐실; 이 둘은 λ 표 생성자다).

namespace mpmstd::equation::pressure {

namespace {

const real_t two_pi = 2 * std::acos(real_t(-1));

real_t eigenvalue(const int mode, const int n, const real_t spacing) {
  return (1 - std::cos(two_pi * mode / n)) * 2 / (spacing * spacing);
}

enum : int { kThreads = 256 };

// 모드 계수 조립(1.2 ms/스텝, V100 256³). 계수는 스텝마다 같지만 PaScaL 이 a·c 를 제자리에서 바꿔 매 스텝 다시 쓴다
//   (2026-10-18: 나눗셈 → 역수는 효과 없음(저장 패턴이 병목)이라 되돌림; 스크래치 출력 방식은 라이브러리 수정이라 보류).
__global__ void build_kernel(real_t* A, real_t* B, real_t* C, const real_t* dz,
                             const real_t* dmz, const real_t* lambda_x,
                             const real_t* lambda_y, const int h, const int nz_local,
                             const int n_mode_x, const int n_mode_y,
                             const bool owns_bottom, const bool owns_top,
                             const bool pin_mean) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(nz_local) * n_mode_x * n_mode_y) return;
  const int jy = static_cast<int>(idx % n_mode_y);
  const int mx = static_cast<int>((idx / n_mode_y) % n_mode_x);
  const int row = static_cast<int>(idx / (static_cast<long long>(n_mode_y) * n_mode_x));
  const int k = row + h;
  const int nsys = 2 * n_mode_x * n_mode_y;

  stencil::Band band = stencil::z_mode_band(
      dz[k], dmz[k], dmz[k + 1], lambda_x[mx] + lambda_y[jy],
      owns_bottom && row == 0, owns_top && row == nz_local - 1);
  if (pin_mean && owns_bottom && row == 0 && mx == 0 && jy == 0) band = {0, 1, 0};  // (0,0) 핀
  const std::size_t p = static_cast<std::size_t>(row) * nsys
                        + 2 * (static_cast<std::size_t>(mx) * n_mode_y + jy);
  A[p] = band.a;
  B[p] = band.b;
  C[p] = band.c;
  A[p + 1] = band.a;
  B[p + 1] = band.b;
  C[p + 1] = band.c;
}

}  // namespace

std::vector<real_t> make_lambda_x(const grid::Grid& grid,
                                  const parallel::Subdomain& sub) {
  const int n = sub.x().n_global;
  const real_t dx = grid.x.cell_width[halo_width];
  std::vector<real_t> lambda(n / 2 + 1);
  for (int m = 0; m < static_cast<int>(lambda.size()); ++m)
    lambda[m] = eigenvalue(m, n, dx);
  return lambda;
}

std::vector<real_t> make_lambda_x_neumann(const grid::Grid& grid,
                                          const parallel::Subdomain& sub) {
  // = system.cpp 의 같은 함수 (x 벽 DCT-II 고유값)
  const int n = sub.x().n_global;
  const real_t dx = grid.x.cell_width[halo_width];
  const real_t pi = std::acos(real_t(-1));
  std::vector<real_t> lambda(n);
  for (int m = 0; m < n; ++m)
    lambda[m] = (1 - std::cos(pi * m / n)) * 2 / (dx * dx);
  return lambda;
}

std::vector<real_t> make_lambda_y(const grid::Grid& grid,
                                  const parallel::Subdomain& sub) {
  const int n = sub.y().n_global;
  const real_t dy = grid.y.cell_width[halo_width];
  std::vector<real_t> lambda(n);
  for (int m = 0; m < n; ++m) lambda[m] = eigenvalue(m, n, dy);
  return lambda;
}

void build_z_modes_system(real_t* A, real_t* B, real_t* C, const grid::Grid& grid,
                          const parallel::Subdomain& sub, const real_t* lambda_x,
                          const int n_mode_x, const real_t* lambda_y,
                          const int n_mode_y, const bool pin_mean) {
  const int nz_local = sub.z().n_local;
  const long long n = static_cast<long long>(nz_local) * n_mode_x * n_mode_y;
  build_kernel<<<static_cast<int>((n + kThreads - 1) / kThreads), kThreads>>>(
      A, B, C, grid.z.device_cell_width.data(),
      grid.z.device_center_distance.data(), lambda_x, lambda_y, halo_width, nz_local,
      n_mode_x, n_mode_y, sub.topology().z().minus_neighbor == MPI_PROC_NULL,
      sub.topology().z().plus_neighbor == MPI_PROC_NULL, pin_mean);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation::pressure
