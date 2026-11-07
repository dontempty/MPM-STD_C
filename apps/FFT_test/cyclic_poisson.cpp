#include "cyclic_poisson.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "boundary/wall.hpp"
#include "device_loop.hpp"
#include "equation/pressure/solve_kernels.hpp"
#include "equation/pressure/system/system.hpp"

namespace fft_test {

namespace {

#define POISSON_BEGIN(name) \
  do { if (timer) timer->begin(name); } while (0)
#define POISSON_END(name) \
  do { if (timer) timer->end(name); } while (0)

void require_uniform(const char* axis_name, const std::vector<real_t>& cell_width, const int n_total) {
  const int h = mpmstd::halo_width;
  const real_t first = cell_width[h];
  for (int i = h; i < n_total - h; ++i)
    if (std::abs(cell_width[i] - first) > real_t(1e-12) * first)
      throw std::invalid_argument(std::string("CyclicPoissonSolver: ") + axis_name + " must be uniform");
}

// 기존 system.cu 의 eigenvalue 와 같은 식 (같은 연산 순서)
real_t eigenvalue(const int mode, const int n, const real_t spacing) {
  const real_t two_pi = 2 * std::acos(real_t(-1));
  return (1 - std::cos(two_pi * mode / n)) * 2 / (spacing * spacing);
}

}  // namespace

CyclicPoissonSolver::CyclicPoissonSolver(const mpmstd::grid::Grid& grid, const mpmstd::parallel::Subdomain& sub)
    : grid_(grid),
      sub_(sub),
      layout_(sub),
      fft_(layout_, static_cast<std::size_t>(sub.x().n_local) * sub.y().n_local * sub.z().n_local),
      exchange_(sub, layout_, fft_.host_staging()) {
  const auto& topo = sub.topology();
  if (!topo.x().periodic || !topo.y().periodic || topo.z().periodic)
    throw std::invalid_argument("CyclicPoissonSolver: needs x, y periodic and z wall");
  require_uniform("x", grid.x.cell_width, sub.x().n_total);
  require_uniform("y", grid.y.cell_width, sub.y().n_total);
  if (topo.z().nprocs > 1 && sub.z().n_local < 2)
    throw std::invalid_argument("CyclicPoissonSolver: distributed z TDMA needs nz_local >= 2");

  owns_bottom_ = topo.z().minus_neighbor == MPI_PROC_NULL;
  owns_top_ = topo.z().plus_neighbor == MPI_PROC_NULL;
  const CyclicDims& d = layout_.dims();
  holds_mean_mode_ = d.s == 0;

  const real_t dx = grid.x.cell_width[mpmstd::halo_width];
  const real_t dy = grid.y.cell_width[mpmstd::halo_width];
  std::vector<real_t> lambda(d.modes_local());
  for (int mode = 0; mode < d.modes_local(); ++mode) {
    int mode_x = 0, mode_y = 0;
    d.global_mode(mode, mode_x, mode_y);
    lambda[mode] = eigenvalue(mode_x, d.n_x, dx) + eigenvalue(mode_y, d.n_y, dy);
  }
  if (holds_mean_mode_ && lambda[0] != 0) throw std::logic_error("CyclicPoissonSolver: mode 0 is not (0,0)");
  lambda_mode_.assign(lambda.size());
  lambda_mode_.upload(lambda);
  lambda_zero_.assign(1);
  plan_z_.create(2 * d.modes_local(), topo.z().comm);
}

void CyclicPoissonSolver::solve(const mpmstd::field::Field& rhs, mpmstd::field::Field& phi) {
  const CyclicDims& d = layout_.dims();
  const int nz = d.nz;
  const int modes = d.modes_local();

  exchange_.to_cyclic(rhs, fft_.lines(), fft_.block_scratch(), fft_.cyclic_scratch(), timer);
  fft_.forward(timer);

  // z 삼중대각 — D = 스펙트럼 그대로 ([sys + z·nsys], nsys = 2·modes). 밴드는 지금 비어 있는 슬롯을 빌린다.
  real_t* spectrum = fft_.spectrum();
  POISSON_BEGIN("fftu.build_z");
  if (owns_bottom_ && holds_mean_mode_) mpmstd::equation::pressure::pin_mean_row(spectrum);
  mpmstd::equation::pressure::build_z_modes_system_rows(fft_.band_a(), fft_.band_b(), fft_.band_c(), grid_, 0, nz,
                                                        owns_bottom_, owns_top_, lambda_mode_.data(), modes,
                                                        lambda_zero_.data(), 1, /*pin_mean=*/holds_mean_mode_);
  POISSON_END("fftu.build_z");
  POISSON_BEGIN("fftu.tdma_z");
  mpmstd::linear_solver::solve_many(plan_z_, fft_.band_a(), fft_.band_b(), fft_.band_c(), spectrum, 2 * modes, nz);
  POISSON_END("fftu.tdma_z");

  POISSON_BEGIN("fftu.mean_remove");
  const real_t scale = 1 / (static_cast<real_t>(d.n_x) * d.n_y);
  const real_t local_dc = holds_mean_mode_ ? mpmstd::equation::pressure::dc_mode_sum(spectrum, nz, modes, 1) : 0;
  real_t global_dc = 0;
  MPI_Allreduce(&local_dc, &global_dc, 1, MPI_DOUBLE, MPI_SUM, sub_.topology().cart_comm());
  const real_t mean = global_dc * scale / static_cast<real_t>(sub_.z().n_global);
  POISSON_END("fftu.mean_remove");

  fft_.backward(timer);
  exchange_.to_block(fft_.lines(), phi, scale, mean, fft_.block_scratch(), fft_.cyclic_scratch(), timer);

  POISSON_BEGIN("fftu.halo");
  sub_.exchange_halo(phi.data());
  for (int axis = 0; axis < 3; ++axis) {
    if (sub_.topology().axis(axis).periodic) continue;
    mpmstd::boundary::fill_wall_zero_gradient(phi, sub_, axis, 0);
    mpmstd::boundary::fill_wall_zero_gradient(phi, sub_, axis, 1);
  }
  synchronize_device();
  POISSON_END("fftu.halo");
}

}  // namespace fft_test
