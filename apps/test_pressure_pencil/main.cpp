// 압력 Poisson 펜슬 경로(npx·npy > 1) 검증 — x 벽(DCT, 거울 라인)과 x 주기 둘 다, 불균등 분할.
//
// 격자: x 벽 또는 주기 · y 주기 · z 벽, 전부 균일, 일부러 P = npx·npy 로 안 나눠지는 크기.
// 제조해 φ = X(x) · cos(2π y / Ly) · cos(π z / Lz)   (X = cos(π x/Lx) 벽 / cos(2π x/Lx) 주기; 평균 0)
// 검사: 이산 일관성 — rhs = 2차 중심차분 ∇²_h φ 를 넣으면 solve 가 φ 를 기계 정밀도로 돌려줘야 한다.
// 출력의 checksum(Σφ, Σφ²; %.17e)은 팩·언팩 경로 대조용 (MPMSTD_PENCIL_PACK=index 와 기본 경로가 같아야 한다).
// 실행: srun -n 8 test_pressure_pencil [npx npy npz]   (기본 2 2 n/4; n 이 4 의 배수가 아니면 1 1 n). 출력 끝 PASS/FAIL.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../framework.hpp"
#include "boundary/wall.hpp"
#include "equation/pressure/solve.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"

using namespace mpmstd;
using mpmstd::field::Field;
using mpmstd::parallel::Subdomain;
using mpmstd::parallel::Topology;

namespace {

struct Result {
  real_t max_error;
  real_t sum, sum_sq;
};

Result run(const parallel::MpiContext& mpi, const int npx, const int npy, const int npz,
           const bool wall_x) {
  const int nx = 38, ny = 30, nz = 26;   // P = 4 로 안 나눠진다 (y 8,8,7,7 / 모드 불균등)
  const real_t lx = 1.3, ly = 0.9, lz = 1.0;
  const real_t pi = std::acos(real_t(-1));

  const Topology topo(mpi, npx, npy, npz, /*periodic_x=*/!wall_x, /*periodic_y=*/true,
                      /*periodic_z=*/false);
  const Subdomain sub(topo, nx, ny, nz);
  const grid::Grid g{
      grid::make_axis(grid::make_uniform_faces(nx, lx), {sub.x().n_global, sub.x().offset, sub.x().n_local}, "x"),
      grid::make_axis(grid::make_uniform_faces(ny, ly), {sub.y().n_global, sub.y().offset, sub.y().n_local}, "y"),
      grid::make_axis(grid::make_uniform_faces(nz, lz), {sub.z().n_global, sub.z().offset, sub.z().n_local}, "z"),
  };

  const int tx = sub.x().n_total, ty = sub.y().n_total, tz = sub.z().n_total;
  const int h = halo_width;
  Field phi_exact(tx, ty, tz, "phi_exact"), rhs(tx, ty, tz, "rhs"), phi(tx, ty, tz, "phi");
  const auto idx = [&](int i, int j, int k) { return (static_cast<std::size_t>(i) * ty + j) * tz + k; };

  std::vector<real_t> host(phi_exact.size(), 0);
  const real_t kx = wall_x ? pi / lx : 2 * pi / lx;
  for (int i = h; i < tx - h; ++i)
    for (int j = h; j < ty - h; ++j)
      for (int k = h; k < tz - h; ++k)
        host[idx(i, j, k)] = std::cos(kx * g.x.center[i]) * std::cos(2 * pi * g.y.center[j] / ly) *
                             std::cos(pi * g.z.center[k] / lz);
  phi_exact.upload(host);
  sub.exchange_halo(phi_exact.data());
  for (int axis = 0; axis < 3; ++axis) {
    if (topo.axis(axis).periodic) continue;
    for (int side : {0, 1}) boundary::fill_wall_zero_gradient(phi_exact, sub, axis, side);
  }
  phi_exact.download(host);

  std::vector<real_t> r(rhs.size(), 0);
  const real_t dx = g.x.cell_width[h], dy = g.y.cell_width[h], dz = g.z.cell_width[h];
  for (int i = h; i < tx - h; ++i)
    for (int j = h; j < ty - h; ++j)
      for (int k = h; k < tz - h; ++k)
        r[idx(i, j, k)] =
            (host[idx(i + 1, j, k)] - 2 * host[idx(i, j, k)] + host[idx(i - 1, j, k)]) / (dx * dx) +
            (host[idx(i, j + 1, k)] - 2 * host[idx(i, j, k)] + host[idx(i, j - 1, k)]) / (dy * dy) +
            (host[idx(i, j, k + 1)] - 2 * host[idx(i, j, k)] + host[idx(i, j, k - 1)]) / (dz * dz);
  rhs.upload(r);

  equation::Workspace workspace(equation::pressure::Solver::required_slot_elements(g, sub));
  equation::pressure::Solver solver(g, sub, workspace);
  equation::pressure::solve(solver, g, sub, rhs, phi);

  std::vector<real_t> p(phi.size());
  phi.download(p);
  real_t local[3] = {0, 0, 0};   // max error, Σφ, Σφ²
  for (int i = h; i < tx - h; ++i)
    for (int j = h; j < ty - h; ++j)
      for (int k = h; k < tz - h; ++k) {
        const real_t v = p[idx(i, j, k)];
        local[0] = std::max(local[0], std::abs(v - host[idx(i, j, k)]));
        local[1] += v;
        local[2] += v * v;
      }
  Result out{};
  MPI_Allreduce(&local[0], &out.max_error, 1, MPI_DOUBLE, MPI_MAX, topo.cart_comm());
  MPI_Allreduce(&local[1], &out.sum, 1, MPI_DOUBLE, MPI_SUM, topo.cart_comm());
  MPI_Allreduce(&local[2], &out.sum_sq, 1, MPI_DOUBLE, MPI_SUM, topo.cart_comm());
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  parallel::MpiContext mpi(&argc, &argv);
  const bool root = mpi.is_root();
  const int n = mpi.world_size();
  int npx = n % 4 == 0 ? 2 : 1, npy = n % 4 == 0 ? 2 : 1, npz = n % 4 == 0 ? n / 4 : n;
  if (argc == 4) {
    npx = std::atoi(argv[1]);
    npy = std::atoi(argv[2]);
    npz = std::atoi(argv[3]);
  }
  if (root) std::printf("process grid %dx%dx%d\n", npx, npy, npz);

  for (const bool wall_x : {true, false}) {
    const Result res = run(mpi, npx, npy, npz, wall_x);
    if (root)
      std::printf("%s x: max|phi_h - phi| = %.3e   checksum sum=%.17e sumsq=%.17e\n",
                  wall_x ? "wall    " : "periodic", res.max_error, res.sum, res.sum_sq);
    CHECK(res.max_error < 1e-12);
  }
  return root ? test::summary("test_pressure_pencil") : 0;
}
