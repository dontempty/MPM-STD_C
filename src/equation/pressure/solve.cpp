#include "equation/pressure/solve.hpp"

#include <algorithm>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
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
  return s.x_neumann ? s.mirror_spectrum.data() : s.spectrum_x;
}
void run_forward_x(Solver& s) {
  check_cufft(cufftExecD2Z(s.fft_x_forward, s.real_lines,
                           reinterpret_cast<cufftDoubleComplex*>(x_spectrum(s))),
              "ExecD2Z");
  cudaDeviceSynchronize();
}
void run_backward_x(Solver& s) {
  check_cufft(cufftExecZ2D(s.fft_x_backward,
                           reinterpret_cast<cufftDoubleComplex*>(x_spectrum(s)),
                           s.real_lines),
              "ExecZ2D");
  cudaDeviceSynchronize();
}
void run_forward_y(Solver& s) {
  cufftDoubleComplex* xy = reinterpret_cast<cufftDoubleComplex*>(s.spectrum_xy);
  check_cufft(cufftExecZ2Z(s.fft_y, xy, xy, CUFFT_FORWARD), "ExecZ2Z fwd");
  cudaDeviceSynchronize();
}
void run_backward_y(Solver& s) {
  cufftDoubleComplex* xy = reinterpret_cast<cufftDoubleComplex*>(s.spectrum_xy);
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

// 펜슬 분할(npx·npy > 1)에서 평면 묶음 경로를 쓸지 — 기본 예, MPMSTD_PRESSURE_PENCIL=transpose4 면 전치 4 회 경로.
// 평면 경로가 못 쓰이는 분할(그룹 랭크당 z 행 < 2)도 전치 4 회로. 집합 호출 (PlaneLayout::usable 이 Allreduce).
bool use_plane_path(const parallel::Subdomain& sub) {
  if (sub.topology().x().nprocs * sub.topology().y().nprocs <= 1) return false;
  const char* mode = std::getenv("MPMSTD_PRESSURE_PENCIL");
  if (mode != nullptr && std::strcmp(mode, "transpose4") == 0) return false;
  return PlaneLayout::usable(sub);
}

std::size_t Solver::required_slot_elements(const grid::Grid&, const parallel::Subdomain& sub) {
  // 생성자의 ws.require 와 같은 식 — 펜슬이면 이 랭크의 y 조각·모드 조각 (PencilLayout 과 같은
  // split_evenly, 그룹 랭크 = y.rank·npx + x.rank), 슬랩(P = 1)이면 전역 ny·n_mode_x 그대로.
  // [2026-11-02] 전에는 펜슬도 슬랩 상한(전역 nx·ny·nz_l)을 써서 슬롯이 P = npx·npy 배로 컸다
  //   (4x4x4, 랭크당 512³ → 5 슬롯 86 GB, GH200 OOM).
  const auto& topo = sub.topology();
  const bool neumann = !topo.x().periodic;
  const int nx = sub.x().n_global, ny = sub.y().n_global;
  const std::size_t nzl = sub.z().n_local;
  const int n_mode_x = neumann ? nx : nx / 2 + 1;
  const std::size_t line_x = neumann ? 2 * nx : nx;
  const int groups = topo.x().nprocs * topo.y().nprocs;
  const int group_rank = topo.y().rank * topo.x().nprocs + topo.x().rank;
  std::size_t ny_lines = ny, n_mode_local = n_mode_x;
  if (use_plane_path(sub)) {   // 평면 묶음: 전역 ny × 모든 모드 × 내 평면 (PlaneLayout 과 같은 split_evenly)
    const std::size_t nzp = parallel::split_evenly(sub.z().n_local, groups, group_rank).count;
    return std::max({ny * nzp * line_x, 2 * ny * nzp * n_mode_x});
  }
  if (groups > 1 && ny >= groups && n_mode_x >= groups) {  // 조건 위반은 PencilLayout 이 알린다
    ny_lines = parallel::split_evenly(ny, groups, group_rank).count;
    n_mode_local = parallel::split_evenly(n_mode_x, groups, group_rank).count;
  }
  const std::size_t n_lines = ny_lines * nzl * line_x;
  const std::size_t n_sx = 2 * ny_lines * nzl * n_mode_x;
  const std::size_t n_sxy = 2 * nzl * n_mode_local * ny;  // == n_band
  return std::max({n_lines, n_sx, n_sxy});
}

Solver::Solver(const grid::Grid& grid, const parallel::Subdomain& sub, Workspace& ws) {
  // ── 전제 검사 (헤더 ★ 참조) — 어기면 조용한 오답 대신 즉시 중단 ──
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
  // x·y 가 쪼개져 있으면: 평면 묶음(기본, 전치 2 회) 또는 펜슬(전치 4 회). 슬랩이면 전치 없음 — 기존 경로
  if (use_plane_path(sub))
    planes = std::make_unique<PlaneLayout>(sub, x_neumann);
  else if (sub.topology().x().nprocs * sub.topology().y().nprocs > 1)
    pencil = std::make_unique<PencilLayout>(sub, n_mode_x, x_neumann);
  ny_lines = pencil ? pencil->my_y_count() : n_y;
  n_mode_local = pencil ? pencil->my_mode_count() : n_mode_x;
  mode_offset = pencil ? pencil->my_mode_offset() : 0;
  nz_lines = planes ? planes->my_plane_count() : nz_local;

  // λ 표 — 호스트에서 만들어 그 빌드의 백엔드 메모리로.
  const std::vector<real_t> lambda_x_host =
      x_neumann ? make_lambda_x_neumann(grid, sub) : make_lambda_x(grid, sub);
  const std::vector<real_t> lambda_y_host = make_lambda_y(grid, sub);
  lambda_x.assign(lambda_x_host.size());
  lambda_x.upload(lambda_x_host);
  lambda_y.assign(lambda_y_host.size());
  lambda_y.upload(lambda_y_host);

  const int nsys = 2 * n_mode_local * n_y;  // 실·허수부를 나란한 시스템으로 (로컬 x 모드)
  plan_z.create(nsys, planes ? planes->z_comm : sub.topology().z().comm);   // 평면 묶음: 전체 랭크 z 사슬
  const std::size_t n_band = static_cast<std::size_t>(nsys) * nz_lines;

  // x 변환의 실제 길이/출력: 주기 = (Nx → Nx/2+1), 벽 DCT = 거울 (2Nx → Nx+1)
  const int line_x = x_neumann ? 2 * n_x : n_x;
  const int spec_x = x_neumann ? n_x + 1 : n_mode_x;
  const std::size_t n_lines = static_cast<std::size_t>(ny_lines) * nz_lines * line_x;
  const std::size_t n_sx = 2 * static_cast<std::size_t>(ny_lines) * nz_lines * n_mode_x;
  const std::size_t n_sxy = 2 * static_cast<std::size_t>(nz_lines) * n_mode_local * n_y;  // == n_band
  // [최적화 2026-10-23] 밴드·라인·스펙트럼·cuFFT 작업 영역은 공용 Workspace 슬롯 (docs/Memory/plan.md §2.3·§3 B).
  //   수명이 겹치지 않는 쌍이 슬롯을 나눠 쓴다: 0 = real_lines(팩~FFT x 정방향, FFT x 역방향~산포) ↔ band_c(build_z~tdma),
  //   3 = band_a(build_z~tdma) ↔ fft_work(FFT 실행 중). solve() 의 단계 순서가 근거 — 순서를 바꾸면 배정을 다시 본다.
  workspace = &ws;
  ws.require(std::max({n_band, n_lines, n_sx, n_sxy}), "pressure::Solver");
  real_lines = ws.slot(0);
  band_c = ws.slot(0);
  spectrum_x = ws.slot(1);
  spectrum_xy = ws.slot(2);
  band_a = ws.slot(3);
  band_b = ws.slot(4);
  if (x_neumann) mirror_spectrum.assign(2 * static_cast<std::size_t>(ny_lines) * nz_lines * spec_x);
  real_t* x_spectrum_target = x_neumann ? mirror_spectrum.data() : spectrum_x;

  // FFT 플랜 — 버퍼 주소에 묶인다 (Buffer 는 재할당하지 않는다). many 배치의
  // 의미는 양쪽 백엔드가 같다: x = 실수 r2c/c2r (howmany = Ny·nz), y = 복소
  // c2c (in-place).
  int length_x = line_x;
  const int howmany_x = ny_lines * nz_lines;
  int length_y = n_y;
  const int howmany_y = n_mode_local * nz_lines;
#ifdef MPMSTD_GPU_FFT
  // [최적화 2026-10-23] 계획마다 cuFFT 가 작업 영역을 따로 잡던 것을 공용 하나로 (FFT 는 차례로 실행 → 겹치지 않음).
  //   자동 할당을 끄고 MakePlanMany 가 알려 준 크기의 최대값만큼 fft_work 를 잡아 세 계획에 붙인다. 수치 불변.
  std::size_t work_x_forward = 0, work_x_backward = 0, work_y = 0;
  for (cufftHandle* h : {&fft_x_forward, &fft_x_backward, &fft_y}) {
    check_cufft(cufftCreate(h), "Create");
    check_cufft(cufftSetAutoAllocation(*h, 0), "SetAutoAllocation");
  }
  check_cufft(cufftMakePlanMany(fft_x_forward, 1, &length_x, nullptr, 1, line_x,
                                nullptr, 1, spec_x, CUFFT_D2Z, howmany_x, &work_x_forward),
              "MakePlanMany D2Z");
  check_cufft(cufftMakePlanMany(fft_x_backward, 1, &length_x, nullptr, 1, spec_x,
                                nullptr, 1, line_x, CUFFT_Z2D, howmany_x, &work_x_backward),
              "MakePlanMany Z2D");
  check_cufft(cufftMakePlanMany(fft_y, 1, &length_y, nullptr, 1, n_y, nullptr, 1, n_y,
                                CUFFT_Z2Z, howmany_y, &work_y),
              "MakePlanMany Z2Z");
  const std::size_t work_bytes = std::max({work_x_forward, work_x_backward, work_y, sizeof(real_t)});
  const std::size_t work_elements = (work_bytes + sizeof(real_t) - 1) / sizeof(real_t);
  if (work_elements <= ws.slot_elements()) {
    fft_work = ws.slot(3);   // band_a 와 수명 분리 — FFT 실행 중에는 밴드가 살아 있지 않다
  } else {                   // cuFFT 가 슬롯보다 큰 작업 영역을 요구하는 드문 크기 → 솔버 소유 버퍼로 후퇴
    fft_work_owned.assign(work_elements);
    fft_work = fft_work_owned.data();
  }
  for (cufftHandle h : {fft_x_forward, fft_x_backward, fft_y})
    check_cufft(cufftSetWorkArea(h, fft_work), "SetWorkArea");
#else
  forward_x = fftw_plan_many_dft_r2c(
      1, &length_x, howmany_x, real_lines, nullptr, 1, line_x,
      reinterpret_cast<fftw_complex*>(x_spectrum_target), nullptr, 1, spec_x,
      FFTW_ESTIMATE);
  backward_x = fftw_plan_many_dft_c2r(
      1, &length_x, howmany_x,
      reinterpret_cast<fftw_complex*>(x_spectrum_target), nullptr, 1, spec_x,
      real_lines, nullptr, 1, line_x, FFTW_ESTIMATE);
  fftw_complex* xy = reinterpret_cast<fftw_complex*>(spectrum_xy);
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
  const bool owns_bottom = sub.topology().z().minus_neighbor == MPI_PROC_NULL;

  if (solver.planes) {
    // ── 평면 묶음 경로 (npx·npy > 1 기본, 2026-11-02) — 블록 →① 전체 xy 평면 × 내 평면 → 슬랩 경로와 같은
    //    x FFT · 재배열 · y FFT (통신 없음) → z 사슬(전체 랭크) PaScaL → 역순 →①⁻¹ 블록. 전치 2 회 (펜슬은 4 회).
    PlaneLayout& L = *solver.planes;
    const int nzp = solver.nz_lines;
    const int nsys = 2 * n_mode_x * n_y;
    PRESSURE_TIMER_BEGIN("pressure.plane.block_to_planes");
    run_exchange(L.block_to_planes, rhs.data(), solver.real_lines, L.comm, L.host_staging);
    PRESSURE_TIMER_END("pressure.plane.block_to_planes");
    PRESSURE_TIMER_BEGIN("pressure.fft_x_fwd");
    run_forward_x(solver);
    if (solver.x_neumann)
      dct_twiddle_forward(solver.spectrum_x, solver.mirror_spectrum.data(), n_y * nzp, n_x);
    PRESSURE_TIMER_END("pressure.fft_x_fwd");
    PRESSURE_TIMER_BEGIN("pressure.reorder_x_to_y");
    reorder_x_to_y(solver.spectrum_xy, solver.spectrum_x, n_y, nzp, n_mode_x);
    PRESSURE_TIMER_END("pressure.reorder_x_to_y");
    PRESSURE_TIMER_BEGIN("pressure.fft_y_fwd");
    run_forward_y(solver);
    PRESSURE_TIMER_END("pressure.fft_y_fwd");
    PRESSURE_TIMER_BEGIN("pressure.build_z");
    if (L.owns_bottom) pin_mean_row(solver.spectrum_xy);
    build_z_modes_system_rows(solver.band_a, solver.band_b, solver.band_c, grid, L.my_plane_offset(), nzp,
                              L.owns_bottom, L.owns_top, solver.lambda_x.data(), n_mode_x,
                              solver.lambda_y.data(), n_y, /*pin_mean=*/true);
    PRESSURE_TIMER_END("pressure.build_z");
    PRESSURE_TIMER_BEGIN("pressure.tdma_z");
    linear_solver::solve_many(solver.plan_z, solver.band_a, solver.band_b, solver.band_c,
                              solver.spectrum_xy, nsys, nzp);
    PRESSURE_TIMER_END("pressure.tdma_z");
    PRESSURE_TIMER_BEGIN("pressure.mean_remove");
    const real_t scale = solver.x_neumann ? 1 / (static_cast<real_t>(2 * n_x) * n_y)
                                          : 1 / (static_cast<real_t>(n_x) * n_y);
    real_t local_dc = dc_mode_sum(solver.spectrum_xy, nzp, n_mode_x, n_y);
    real_t global_dc = 0;
    MPI_Allreduce(&local_dc, &global_dc, 1, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
    const real_t mean = global_dc * scale / static_cast<real_t>(sub.z().n_global);
    PRESSURE_TIMER_END("pressure.mean_remove");
    PRESSURE_TIMER_BEGIN("pressure.fft_y_bwd");
    run_backward_y(solver);
    PRESSURE_TIMER_END("pressure.fft_y_bwd");
    PRESSURE_TIMER_BEGIN("pressure.reorder_y_to_x");
    reorder_y_to_x(solver.spectrum_x, solver.spectrum_xy, n_y, nzp, n_mode_x);
    PRESSURE_TIMER_END("pressure.reorder_y_to_x");
    PRESSURE_TIMER_BEGIN("pressure.fft_x_bwd");
    if (solver.x_neumann)
      dct_twiddle_backward(solver.mirror_spectrum.data(), solver.spectrum_x, n_y * nzp, n_x);
    run_backward_x(solver);
    PRESSURE_TIMER_END("pressure.fft_x_bwd");
    // ①⁻¹ 평면 → 블록 (정규화·평균 빼기를 팩에서; 장의 ghost 는 건드리지 않는다)
    PRESSURE_TIMER_BEGIN("pressure.plane.planes_to_block");
    run_exchange(L.planes_to_block, solver.real_lines, phi.data(), L.comm, L.host_staging, scale, mean);
    PRESSURE_TIMER_END("pressure.plane.planes_to_block");
  } else if (!solver.pencil) {
    // ── 슬랩 경로 (npx = npy = 1) — 10-18 최적화 경로 그대로 (pin_mean = true 만 추가) ──
    const int nsys = 2 * n_mode_x * n_y;
    // ① rhs 내부 → x-라인 팩 → x 변환(주기 r2c / 벽 거울 r2c+트위들 = DCT-II)
    //    → y-라인 재배열 → y c2c
    PRESSURE_TIMER_BEGIN("pressure.pack_lines");
    if (solver.x_neumann) {
      pack_rhs_to_mirrored_lines(solver.real_lines, rhs, sub, n_x, n_y, nz_local);
    } else {
      pack_rhs_to_lines(solver.real_lines, rhs, sub, n_x, n_y, nz_local);
    }
    PRESSURE_TIMER_END("pressure.pack_lines");
    PRESSURE_TIMER_BEGIN("pressure.fft_x_fwd");
    run_forward_x(solver);
    if (solver.x_neumann)
      dct_twiddle_forward(solver.spectrum_x, solver.mirror_spectrum.data(),
                          n_y * nz_local, n_x);
    PRESSURE_TIMER_END("pressure.fft_x_fwd");
    PRESSURE_TIMER_BEGIN("pressure.reorder_x_to_y");
    reorder_x_to_y(solver.spectrum_xy, solver.spectrum_x, n_y, nz_local,
                   n_mode_x);
    PRESSURE_TIMER_END("pressure.reorder_x_to_y");
    PRESSURE_TIMER_BEGIN("pressure.fft_y_fwd");
    run_forward_y(solver);
    PRESSURE_TIMER_END("pressure.fft_y_fwd");

    // ② 모드별 z 삼중대각 — 실·허를 나란한 시스템으로 한 번에 (분산 TDMA, 비주기)
    // [최적화 2026-10-18] 스펙트럼 [(k·Nxm+mx)·Ny+jy]×2 와 밴드 [sys + k·nsys](nsys = 2·Nxm·Ny) 는 같은 인덱스라
    //   팩·언팩 복사 없이 spectrum_xy 를 D 로 바로 푼다. 특이 모드 행 핀(첫 두 원소 = 0)만 남는다.
    PRESSURE_TIMER_BEGIN("pressure.pack_bands");
    if (owns_bottom) pin_mean_row(solver.spectrum_xy);
    PRESSURE_TIMER_END("pressure.pack_bands");
    PRESSURE_TIMER_BEGIN("pressure.build_z");
    build_z_modes_system(solver.band_a, solver.band_b,
                         solver.band_c, grid, sub, solver.lambda_x.data(),
                         n_mode_x, solver.lambda_y.data(), n_y, /*pin_mean=*/true);
    PRESSURE_TIMER_END("pressure.build_z");
    PRESSURE_TIMER_BEGIN("pressure.tdma_z");
    linear_solver::solve_many(solver.plan_z, solver.band_a,
                              solver.band_b, solver.band_c,
                              solver.spectrum_xy, nsys, nz_local);
    PRESSURE_TIMER_END("pressure.tdma_z");

    // ③ 전역 평균(게이지)을 DC 모드에서 — 평면 (x,y) 합 = 스펙트럼 (k,0,0) 실수부·scale·Nx·Ny 이므로
    //   평균 = Σ_k DC(k)·scale / Nz_global. 실공간 리덕션(2패스) 대신 nz_local 개 합 + Allreduce.
    //   [최적화 2026-10-18] 빼기는 산포 커널에 융합 (phi = line·scale − mean).
    PRESSURE_TIMER_BEGIN("pressure.mean_remove");
    const real_t scale = solver.x_neumann ? 1 / (static_cast<real_t>(2 * n_x) * n_y)
                                          : 1 / (static_cast<real_t>(n_x) * n_y);
    real_t local_dc = dc_mode_sum(solver.spectrum_xy, nz_local, n_mode_x, n_y);
    real_t global_dc = 0;
    MPI_Allreduce(&local_dc, &global_dc, 1, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
    const real_t mean = global_dc * scale / static_cast<real_t>(sub.z().n_global);
    PRESSURE_TIMER_END("pressure.mean_remove");

    // ④ 역 y → 역 x → 정규화 산포 (평균 빼기 포함)
    PRESSURE_TIMER_BEGIN("pressure.fft_y_bwd");
    run_backward_y(solver);
    PRESSURE_TIMER_END("pressure.fft_y_bwd");
    PRESSURE_TIMER_BEGIN("pressure.reorder_y_to_x");
    reorder_y_to_x(solver.spectrum_x, solver.spectrum_xy, n_y, nz_local,
                   n_mode_x);
    PRESSURE_TIMER_END("pressure.reorder_y_to_x");
    PRESSURE_TIMER_BEGIN("pressure.fft_x_bwd");
    if (solver.x_neumann)  // DCT-III: 트위들 → 거울 c2r → 앞 N 개, 왕복 배율 2N
      dct_twiddle_backward(solver.mirror_spectrum.data(), solver.spectrum_x,
                           n_y * nz_local, n_x);
    run_backward_x(solver);
    PRESSURE_TIMER_END("pressure.fft_x_bwd");
    PRESSURE_TIMER_BEGIN("pressure.scatter_lines");
    if (solver.x_neumann) {
      scatter_mirrored_lines_to_field(phi, solver.real_lines, scale, sub, n_x, n_y,
                                      nz_local, mean);
    } else {
      scatter_lines_to_field(phi, solver.real_lines, scale, sub, n_x, n_y, nz_local,
                             mean);
    }
    PRESSURE_TIMER_END("pressure.scatter_lines");
  } else {
    // ── 펜슬 경로 (npx·npy > 1) — 블록 →① x-펜슬 → x FFT →② y-펜슬 → y FFT → z TDMA → 역순 ──
    //    (2026-10-20, 10-18 최적화 레이아웃 기준: spectrum_xy 가 곧 TDMA 의 D, 평균은 DC 모드 합)
    PencilLayout& L = *solver.pencil;
    const int ny_lines = solver.ny_lines, nml = solver.n_mode_local;
    const int nsys = 2 * nml * n_y;
    const bool has_mean_mode = solver.mode_offset == 0;  // 로컬 모드 0 = 전역 (0,0) 모드인 랭크
    PRESSURE_TIMER_BEGIN("pressure.pencil_forward");
    // ① 블록 → x 라인 (주기: 길이 Nx / 벽: 거울 2Nx) → x 변환. 전치는 백엔드 버퍼끼리 (GPU↔GPU)
    //    (세부 타이머 pressure.pencil.* 는 측정 전용 — solver_phases 가 꺼져 있으면 no-op)
    PRESSURE_TIMER_BEGIN("pressure.pencil.block_to_lines");
    run_exchange(L.block_to_lines, rhs.data(), solver.real_lines, L.comm, L.host_staging);
    PRESSURE_TIMER_END("pressure.pencil.block_to_lines");
    PRESSURE_TIMER_BEGIN("pressure.pencil.fft_x_fwd");
    run_forward_x(solver);
    if (solver.x_neumann)
      dct_twiddle_forward(solver.spectrum_x, solver.mirror_spectrum.data(),
                          ny_lines * nz_local, n_x);
    PRESSURE_TIMER_END("pressure.pencil.fft_x_fwd");
    // ② x-펜슬 스펙트럼 → y-펜슬 → y 변환
    PRESSURE_TIMER_BEGIN("pressure.pencil.x_to_y");
    run_exchange(L.x_to_y_spectrum, solver.spectrum_x, solver.spectrum_xy, L.comm,
                 L.host_staging);
    PRESSURE_TIMER_END("pressure.pencil.x_to_y");
    PRESSURE_TIMER_BEGIN("pressure.pencil.fft_y_fwd");
    run_forward_y(solver);
    PRESSURE_TIMER_END("pressure.pencil.fft_y_fwd");
    PRESSURE_TIMER_END("pressure.pencil_forward");
    // z 삼중대각 — spectrum_xy 를 D 로 바로 (로컬 x 모드 조각, λ_x 오프셋). (0,0) 핀은 그 모드를 가진 랭크만
    PRESSURE_TIMER_BEGIN("pressure.tdma_z");
    PRESSURE_TIMER_BEGIN("pressure.pencil.build_z");
    if (owns_bottom && has_mean_mode) pin_mean_row(solver.spectrum_xy);
    build_z_modes_system(solver.band_a, solver.band_b,
                         solver.band_c, grid, sub,
                         solver.lambda_x.data() + solver.mode_offset, nml,
                         solver.lambda_y.data(), n_y, has_mean_mode);
    PRESSURE_TIMER_END("pressure.pencil.build_z");
    PRESSURE_TIMER_BEGIN("pressure.pencil.solve_z");
    linear_solver::solve_many(solver.plan_z, solver.band_a,
                              solver.band_b, solver.band_c,
                              solver.spectrum_xy, nsys, nz_local);
    PRESSURE_TIMER_END("pressure.pencil.solve_z");
    PRESSURE_TIMER_END("pressure.tdma_z");
    // 전역 평균(게이지) — DC 모드는 모드 오프셋 0 인 랭크(z 블록마다 하나)에만 있다
    PRESSURE_TIMER_BEGIN("pressure.mean_remove");
    const real_t scale = solver.x_neumann ? 1 / (static_cast<real_t>(2 * n_x) * n_y)
                                          : 1 / (static_cast<real_t>(n_x) * n_y);
    real_t local_dc = has_mean_mode ? dc_mode_sum(solver.spectrum_xy, nz_local, nml, n_y) : 0;
    real_t global_dc = 0;
    MPI_Allreduce(&local_dc, &global_dc, 1, MPI_DOUBLE, MPI_SUM, sub.topology().cart_comm());
    const real_t mean = global_dc * scale / static_cast<real_t>(sub.z().n_global);
    PRESSURE_TIMER_END("pressure.mean_remove");
    PRESSURE_TIMER_BEGIN("pressure.pencil_backward");
    PRESSURE_TIMER_BEGIN("pressure.pencil.fft_y_bwd");
    run_backward_y(solver);
    PRESSURE_TIMER_END("pressure.pencil.fft_y_bwd");
    // ②⁻¹ y-펜슬 → x-펜슬 → 역 x 변환
    PRESSURE_TIMER_BEGIN("pressure.pencil.y_to_x");
    run_exchange(L.y_to_x_spectrum, solver.spectrum_xy, solver.spectrum_x, L.comm,
                 L.host_staging);
    PRESSURE_TIMER_END("pressure.pencil.y_to_x");
    PRESSURE_TIMER_BEGIN("pressure.pencil.fft_x_bwd");
    if (solver.x_neumann)
      dct_twiddle_backward(solver.mirror_spectrum.data(), solver.spectrum_x,
                           ny_lines * nz_local, n_x);
    run_backward_x(solver);
    PRESSURE_TIMER_END("pressure.pencil.fft_x_bwd");
    // ①⁻¹ x 라인 → 블록 (정규화·평균 빼기를 팩 커널에서; 장의 ghost 는 건드리지 않는다)
    PRESSURE_TIMER_BEGIN("pressure.pencil.lines_to_block");
    run_exchange(L.lines_to_block, solver.real_lines, phi.data(), L.comm, L.host_staging,
                 scale, mean);
    PRESSURE_TIMER_END("pressure.pencil.lines_to_block");
    PRESSURE_TIMER_END("pressure.pencil_backward");
  }


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
