#include "equation/pressure/solve.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MPMSTD_GPU_FFT
#include <cuda_runtime.h>
#endif

#include "boundary/wall.hpp"
#include "common/field_checks.hpp"
#include "equation/pressure/solve_kernels.hpp"
#include "equation/pressure/system/system.hpp"

// 오케스트레이션 전용 TU — 양쪽 빌드 공용. 팩·산포·축소는 solve_kernels.{cpp,cu},
// 밴드 조립은 system.{cpp,cu} 가 백엔드별로 구현한다. FFT 실행 래퍼만 ifdef.

#define PRESSURE_TIMER_BEGIN(name) \
  do { if (solver.timer) solver.timer->begin(name); } while (0)
#define PRESSURE_TIMER_END(name) \
  do { if (solver.timer) solver.timer->end(name); } while (0)

namespace mpmstd::equation::pressure {

namespace {

// FFT 고유값 표가 전제하는 균일 간격인지 — 어기면 조용히 틀리므로 여기서 막는다.
void require_uniform(const char* axis_name, const std::vector<real_t>& cell_width,
                     const int n_total) {
  const int h = halo_width;
  const real_t first = cell_width[h];
  for (int i = h; i < n_total - h; ++i)
    if (std::abs(cell_width[i] - first) > real_t(1e-12) * first)
      throw std::invalid_argument(std::string("pressure::Solver: ") + axis_name
                                  + " must be uniform (FFT eigenvalue premise)");
}

// ── FFT 실행 래퍼 — 백엔드 차이를 이 네 함수에 가둔다 ──────────────────────
// cuFFT 도 FFTW 도 비정규화 변환이라 공용 1/(Nx·Ny) 보정이 그대로 성립한다.
#ifdef MPMSTD_GPU_FFT

void check_cufft(const cufftResult status, const char* what) {
  if (status != CUFFT_SUCCESS)
    throw std::runtime_error(std::string("cuFFT failed: ") + what);
}

// cuFFT 실행은 비동기 — 다음 단계(커널·MPI)가 결과를 읽기 전에 동기화한다.
// x 변환의 스펙트럼 버퍼: 주기 = spectrum_x, 벽 DCT = mirror_spectrum (N+1 복소)
real_t* x_spectrum(Solver& s) {
  return s.x_neumann ? s.mirror_spectrum.data() : s.spectrum_x.data();
}
void run_forward_x(Solver& s) {
  check_cufft(cufftExecD2Z(s.fft_x_forward, s.real_lines.data(),
                           reinterpret_cast<cufftDoubleComplex*>(x_spectrum(s))),
              "ExecD2Z");
  cudaDeviceSynchronize();
}
void run_backward_x(Solver& s) {
  check_cufft(cufftExecZ2D(s.fft_x_backward,
                           reinterpret_cast<cufftDoubleComplex*>(x_spectrum(s)),
                           s.real_lines.data()),
              "ExecZ2D");
  cudaDeviceSynchronize();
}
void run_forward_y(Solver& s) {
  cufftDoubleComplex* xy = reinterpret_cast<cufftDoubleComplex*>(s.spectrum_xy.data());
  check_cufft(cufftExecZ2Z(s.fft_y, xy, xy, CUFFT_FORWARD), "ExecZ2Z fwd");
  cudaDeviceSynchronize();
}
void run_backward_y(Solver& s) {
  cufftDoubleComplex* xy = reinterpret_cast<cufftDoubleComplex*>(s.spectrum_xy.data());
  check_cufft(cufftExecZ2Z(s.fft_y, xy, xy, CUFFT_INVERSE), "ExecZ2Z inv");
  cudaDeviceSynchronize();
}

#else

void run_forward_x(Solver& s) { fftw_execute(s.forward_x); }
void run_backward_x(Solver& s) { fftw_execute(s.backward_x); }
void run_forward_y(Solver& s) { fftw_execute(s.forward_y); }
void run_backward_y(Solver& s) { fftw_execute(s.backward_y); }

#endif

}  // namespace

Solver::Solver(const grid::Grid& grid, const parallel::Subdomain& sub) {
  // ── 전제 검사 (헤더 ★ 참조) — 어기면 조용한 오답 대신 즉시 중단 ──
  if (sub.topology().x().nprocs != 1 || sub.topology().y().nprocs != 1)
    throw std::invalid_argument(
        "pressure::Solver: only z-slab decomposition is supported for now "
        "(npx = npy = 1); pencil transposes are the documented extension");
  if (!sub.topology().y().periodic)
    throw std::invalid_argument(
        "pressure::Solver: y must be periodic (wall y needs a y-DCT path — not implemented; "
        "x wall is supported via DCT)");
  require_uniform("x", grid.x.cell_width, sub.x().n_total);
  require_uniform("y", grid.y.cell_width, sub.y().n_total);

  x_neumann = !sub.topology().x().periodic;
  n_x = sub.x().n_global;
  n_y = sub.y().n_global;
  n_mode_x = x_neumann ? n_x : n_x / 2 + 1;
  nz_local = sub.z().n_local;

  // λ 표 — 호스트에서 만들어 그 빌드의 백엔드 메모리로.
  const std::vector<real_t> lambda_x_host =
      x_neumann ? make_lambda_x_neumann(grid, sub) : make_lambda_x(grid, sub);
  const std::vector<real_t> lambda_y_host = make_lambda_y(grid, sub);
  lambda_x.assign(lambda_x_host.size());
  lambda_x.upload(lambda_x_host);
  lambda_y.assign(lambda_y_host.size());
  lambda_y.upload(lambda_y_host);

  const int nsys = 2 * n_mode_x * n_y;  // 실·허수부를 나란한 시스템으로
  plan_z.create(nsys, sub.topology().z().comm);
  const std::size_t n_band = static_cast<std::size_t>(nsys) * nz_local;
  band_a.assign(n_band);
  band_b.assign(n_band);
  band_c.assign(n_band);

  // x 변환의 실제 길이/출력: 주기 = (Nx → Nx/2+1), 벽 DCT = 거울 (2Nx → Nx+1)
  const int line_x = x_neumann ? 2 * n_x : n_x;
  const int spec_x = x_neumann ? n_x + 1 : n_mode_x;
  real_lines.assign(static_cast<std::size_t>(n_y) * nz_local * line_x);
  spectrum_x.assign(2 * static_cast<std::size_t>(n_y) * nz_local * n_mode_x);
  spectrum_xy.assign(2 * static_cast<std::size_t>(nz_local) * n_mode_x * n_y);
  if (x_neumann) mirror_spectrum.assign(2 * static_cast<std::size_t>(n_y) * nz_local * spec_x);
  real_t* x_spectrum_target = x_neumann ? mirror_spectrum.data() : spectrum_x.data();

  // FFT 플랜 — 버퍼 주소에 묶인다 (Buffer 는 재할당하지 않는다). many 배치의
  // 의미는 양쪽 백엔드가 같다: x = 실수 r2c/c2r (howmany = Ny·nz), y = 복소
  // c2c (in-place).
  int length_x = line_x;
  const int howmany_x = n_y * nz_local;
  int length_y = n_y;
  const int howmany_y = n_mode_x * nz_local;
#ifdef MPMSTD_GPU_FFT
  check_cufft(cufftPlanMany(&fft_x_forward, 1, &length_x, nullptr, 1, line_x,
                            nullptr, 1, spec_x, CUFFT_D2Z, howmany_x),
              "PlanMany D2Z");
  check_cufft(cufftPlanMany(&fft_x_backward, 1, &length_x, nullptr, 1, spec_x,
                            nullptr, 1, line_x, CUFFT_Z2D, howmany_x),
              "PlanMany Z2D");
  check_cufft(cufftPlanMany(&fft_y, 1, &length_y, nullptr, 1, n_y, nullptr, 1, n_y,
                            CUFFT_Z2Z, howmany_y),
              "PlanMany Z2Z");
#else
  forward_x = fftw_plan_many_dft_r2c(
      1, &length_x, howmany_x, real_lines.data(), nullptr, 1, line_x,
      reinterpret_cast<fftw_complex*>(x_spectrum_target), nullptr, 1, spec_x,
      FFTW_ESTIMATE);
  backward_x = fftw_plan_many_dft_c2r(
      1, &length_x, howmany_x,
      reinterpret_cast<fftw_complex*>(x_spectrum_target), nullptr, 1, spec_x,
      real_lines.data(), nullptr, 1, line_x, FFTW_ESTIMATE);
  fftw_complex* xy = reinterpret_cast<fftw_complex*>(spectrum_xy.data());
  forward_y = fftw_plan_many_dft(1, &length_y, howmany_y, xy, nullptr, 1, n_y,
                                 xy, nullptr, 1, n_y, FFTW_FORWARD, FFTW_ESTIMATE);
  backward_y = fftw_plan_many_dft(1, &length_y, howmany_y, xy, nullptr, 1, n_y,
                                  xy, nullptr, 1, n_y, FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
}

Solver::~Solver() {
#ifdef MPMSTD_GPU_FFT
  if (fft_x_forward) cufftDestroy(fft_x_forward);
  if (fft_x_backward) cufftDestroy(fft_x_backward);
  if (fft_y) cufftDestroy(fft_y);
#else
  if (forward_x) fftw_destroy_plan(forward_x);
  if (backward_x) fftw_destroy_plan(backward_x);
  if (forward_y) fftw_destroy_plan(forward_y);
  if (backward_y) fftw_destroy_plan(backward_y);
#endif
}

void solve(Solver& solver, const grid::Grid& grid, const parallel::Subdomain& sub,
           const field::Field& rhs, field::Field& phi) {
  check_field_matches("pressure::solve", rhs, sub);
  check_field_matches("pressure::solve", phi, sub);
  if (solver.n_x != sub.x().n_global || solver.n_y != sub.y().n_global
      || solver.nz_local != sub.z().n_local)
    throw std::invalid_argument(
        "pressure::solve: Solver was created for a different subdomain");

  const int n_x = solver.n_x;
  const int n_y = solver.n_y;
  const int n_mode_x = solver.n_mode_x;
  const int nz_local = solver.nz_local;
  const int nsys = 2 * n_mode_x * n_y;
  const bool owns_bottom = sub.topology().z().minus_neighbor == MPI_PROC_NULL;

  // ① rhs 내부 → x-라인 팩 → x 변환(주기 r2c / 벽 거울 r2c+트위들 = DCT-II)
  //    → y-라인 재배열 → y c2c
  PRESSURE_TIMER_BEGIN("pressure.pack_lines");
  if (solver.x_neumann) {
    pack_rhs_to_mirrored_lines(solver.real_lines.data(), rhs, sub, n_x, n_y, nz_local);
  } else {
    pack_rhs_to_lines(solver.real_lines.data(), rhs, sub, n_x, n_y, nz_local);
  }
  PRESSURE_TIMER_END("pressure.pack_lines");
  PRESSURE_TIMER_BEGIN("pressure.fft_x_fwd");
  run_forward_x(solver);
  if (solver.x_neumann)
    dct_twiddle_forward(solver.spectrum_x.data(), solver.mirror_spectrum.data(),
                        n_y * nz_local, n_x);
  PRESSURE_TIMER_END("pressure.fft_x_fwd");
  PRESSURE_TIMER_BEGIN("pressure.reorder_x_to_y");
  reorder_x_to_y(solver.spectrum_xy.data(), solver.spectrum_x.data(), n_y, nz_local,
                 n_mode_x);
  PRESSURE_TIMER_END("pressure.reorder_x_to_y");
  PRESSURE_TIMER_BEGIN("pressure.fft_y_fwd");
  run_forward_y(solver);
  PRESSURE_TIMER_END("pressure.fft_y_fwd");

  // ② 모드별 z 삼중대각 — 실·허를 나란한 시스템으로 한 번에 (분산 TDMA, 비주기)
  // [최적화 2026-10-18] 스펙트럼 [(k·Nxm+mx)·Ny+jy]×2 와 밴드 [sys + k·nsys](nsys = 2·Nxm·Ny) 는 같은 인덱스라
  //   팩·언팩 복사 없이 spectrum_xy 를 D 로 바로 푼다. 특이 모드 행 핀(첫 두 원소 = 0)만 남는다.
  PRESSURE_TIMER_BEGIN("pressure.pack_bands");
  if (owns_bottom) pin_mean_row(solver.spectrum_xy.data());
  PRESSURE_TIMER_END("pressure.pack_bands");
  PRESSURE_TIMER_BEGIN("pressure.build_z");
  build_z_modes_system(solver.band_a.data(), solver.band_b.data(),
                       solver.band_c.data(), grid, sub, solver.lambda_x.data(),
                       n_mode_x, solver.lambda_y.data(), n_y);
  PRESSURE_TIMER_END("pressure.build_z");
  PRESSURE_TIMER_BEGIN("pressure.tdma_z");
  linear_solver::solve_many(solver.plan_z, solver.band_a.data(),
                            solver.band_b.data(), solver.band_c.data(),
                            solver.spectrum_xy.data(), nsys, nz_local);
  PRESSURE_TIMER_END("pressure.tdma_z");

  // ③ 전역 평균(게이지)을 DC 모드에서 — 평면 (x,y) 합 = 스펙트럼 (k,0,0) 실수부·scale·Nx·Ny 이므로
  //   평균 = Σ_k DC(k)·scale / Nz_global. 실공간 리덕션(2패스) 대신 nz_local 개 합 + Allreduce.
  //   [최적화 2026-10-18] 빼기는 산포 커널에 융합 (phi = line·scale − mean).
  PRESSURE_TIMER_BEGIN("pressure.mean_remove");
  const real_t scale = solver.x_neumann ? 1 / (static_cast<real_t>(2 * n_x) * n_y)
                                        : 1 / (static_cast<real_t>(n_x) * n_y);
  real_t local_dc = dc_mode_sum(solver.spectrum_xy.data(), nz_local, n_mode_x, n_y);
  real_t global_dc = 0;
  MPI_Allreduce(&local_dc, &global_dc, 1, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
  const real_t mean = global_dc * scale / static_cast<real_t>(sub.z().n_global);
  PRESSURE_TIMER_END("pressure.mean_remove");

  // ④ 역 y → 역 x → 정규화 산포 (평균 빼기 포함)
  PRESSURE_TIMER_BEGIN("pressure.fft_y_bwd");
  run_backward_y(solver);
  PRESSURE_TIMER_END("pressure.fft_y_bwd");
  PRESSURE_TIMER_BEGIN("pressure.reorder_y_to_x");
  reorder_y_to_x(solver.spectrum_x.data(), solver.spectrum_xy.data(), n_y, nz_local,
                 n_mode_x);
  PRESSURE_TIMER_END("pressure.reorder_y_to_x");
  PRESSURE_TIMER_BEGIN("pressure.fft_x_bwd");
  if (solver.x_neumann)  // DCT-III: 트위들 → 거울 c2r → 앞 N 개, 왕복 배율 2N
    dct_twiddle_backward(solver.mirror_spectrum.data(), solver.spectrum_x.data(),
                         n_y * nz_local, n_x);
  run_backward_x(solver);
  PRESSURE_TIMER_END("pressure.fft_x_bwd");
  PRESSURE_TIMER_BEGIN("pressure.scatter_lines");
  if (solver.x_neumann) {
    scatter_mirrored_lines_to_field(phi, solver.real_lines.data(), scale, sub, n_x, n_y,
                                    nz_local, mean);
  } else {
    scatter_lines_to_field(phi, solver.real_lines.data(), scale, sub, n_x, n_y, nz_local,
                           mean);
  }
  PRESSURE_TIMER_END("pressure.scatter_lines");

  // ⑤ ghost 완성: 이음새·주기 랩 → 벽 축 Neumann (조립 시 접은 벽 / DCT 와 같은
  //    정책의 ghost 짝 — conventions.md §6). 이 상태로 correct_velocity_* 에 바로 쓴다.
  PRESSURE_TIMER_BEGIN("pressure.halo");
  sub.exchange_halo(phi.data());
  for (int axis = 0; axis < 3; ++axis) {
    if (sub.topology().axis(axis).periodic) continue;
    boundary::fill_wall_zero_gradient(phi, sub, axis, 0);
    boundary::fill_wall_zero_gradient(phi, sub, axis, 1);
  }
  PRESSURE_TIMER_END("pressure.halo");
}

}  // namespace mpmstd::equation::pressure
