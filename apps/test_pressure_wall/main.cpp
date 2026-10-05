// 압력 Poisson 의 x 벽(Neumann, DCT) 경로 검증 — BC 프레임워크 단계 3
// (docs/Design/20261005_boundary_condition_framework.md).
//
// 격자: x 벽 · y 주기(ny = 1, 2D) · z 벽, 전부 균일. 제조해
//   φ(x, z) = cos(π x / Lx) · cos(π z / Lz)     (양 벽 ∂φ/∂n = 0, 평균 0)
// 두 검사:
//   ① 이산 일관성: rhs = 2차 중심차분 ∇²_h φ (ghost 는 zero_gradient) 를 넣으면
//      solve 가 φ 를 **기계 정밀도**로 돌려줘야 한다 (스펙트럴 고유값이 차분
//      연산자와 정확히 일치한다는 뜻 — 투영 후 div 가 기계 0 이 되는 전제).
//   ② 수렴: rhs = 해석 −(π²/Lx² + π²/Lz²) φ 로 n = 16, 32, 64 를 풀어 EOC ≈ 2.
// 실행: mpirun -n 1 또는 -n 2 (z 분할) 로 돈다. 출력 끝 PASS/FAIL.

#include <mpi.h>

#include <cmath>
#include <cstdio>
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
  real_t max_error;  // max |φ_h − φ| (평균 제거 후)
};

Result run(const parallel::MpiContext& mpi, const int n, const bool discrete_rhs) {
  const real_t lx = 1.0, ly = 0.0625, lz = 1.0;
  const int nx = n, ny = 1, nz = n;
  const real_t pi = std::acos(real_t(-1));

  const Topology topo(mpi, 1, 1, mpi.world_size(), /*periodic_x=*/false,
                      /*periodic_y=*/true, /*periodic_z=*/false);
  const Subdomain sub(topo, nx, ny, nz);
  const auto faces_x = grid::make_uniform_faces(nx, lx);
  const auto faces_y = grid::make_uniform_faces(ny, ly);
  const auto faces_z = grid::make_uniform_faces(nz, lz);
  const grid::Grid g{
      grid::make_axis(faces_x, {sub.x().n_global, sub.x().offset, sub.x().n_local}, "x"),
      grid::make_axis(faces_y, {sub.y().n_global, sub.y().offset, sub.y().n_local}, "y"),
      grid::make_axis(faces_z, {sub.z().n_global, sub.z().offset, sub.z().n_local}, "z"),
  };

  const int tx = sub.x().n_total, ty = sub.y().n_total, tz = sub.z().n_total;
  const int h = halo_width;
  Field phi_exact(tx, ty, tz, "phi_exact"), rhs(tx, ty, tz, "rhs"), phi(tx, ty, tz, "phi");

  // 정확해 (내부) → ghost 는 halo 교환 + 벽 zero_gradient 로 채운다
  std::vector<real_t> host(phi_exact.size(), 0);
  const auto idx = [&](int i, int j, int k) {
    return (static_cast<std::size_t>(i) * ty + j) * tz + k;
  };
  for (int i = h; i < tx - h; ++i)
    for (int j = h; j < ty - h; ++j)
      for (int k = h; k < tz - h; ++k)
        host[idx(i, j, k)] =
            std::cos(pi * g.x.center[i] / lx) * std::cos(pi * g.z.center[k] / lz);
  phi_exact.upload(host);
  sub.exchange_halo(phi_exact.data());
  for (int axis : {0, 2})
    for (int side : {0, 1}) boundary::fill_wall_zero_gradient(phi_exact, sub, axis, side);
  phi_exact.download(host);

  // rhs
  std::vector<real_t> r(rhs.size(), 0);
  const real_t dx = g.x.cell_width[h], dz = g.z.cell_width[h];
  for (int i = h; i < tx - h; ++i)
    for (int j = h; j < ty - h; ++j)
      for (int k = h; k < tz - h; ++k) {
        if (discrete_rhs) {
          // 2차 중심차분 (x: 균일 dx, y: ny=1 주기라 이웃 = 자기 자신 → 0, z: 균일 dz)
          const real_t lap_x =
              (host[idx(i + 1, j, k)] - 2 * host[idx(i, j, k)] + host[idx(i - 1, j, k)])
              / (dx * dx);
          const real_t lap_z =
              (host[idx(i, j, k + 1)] - 2 * host[idx(i, j, k)] + host[idx(i, j, k - 1)])
              / (dz * dz);
          r[idx(i, j, k)] = lap_x + lap_z;
        } else {
          r[idx(i, j, k)] = -(pi * pi / (lx * lx) + pi * pi / (lz * lz)) * host[idx(i, j, k)];
        }
      }
  rhs.upload(r);

  equation::pressure::Solver solver(g, sub);
  equation::pressure::solve(solver, g, sub, rhs, phi);

  // 비교 (solve 는 전역 평균 0; 정확해도 평균 0 — 셀중심 cos 의 대칭 합)
  std::vector<real_t> p(phi.size());
  phi.download(p);
  real_t local_max = 0;
  for (int i = h; i < tx - h; ++i)
    for (int j = h; j < ty - h; ++j)
      for (int k = h; k < tz - h; ++k)
        local_max = std::max(local_max, std::abs(p[idx(i, j, k)] - host[idx(i, j, k)]));
  real_t global_max = 0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, topo.cart_comm());
  return {global_max};
}

}  // namespace

int main(int argc, char** argv) {
  parallel::MpiContext mpi(&argc, &argv);
  const bool root = mpi.is_root();

  // ① 이산 일관성 — 기계 정밀도
  const Result d = run(mpi, 32, /*discrete_rhs=*/true);
  if (root) std::printf("discrete consistency n=32: max|phi_h - phi| = %.3e\n", d.max_error);
  CHECK(d.max_error < 1e-12);

  // ② 수렴 — EOC ≈ 2
  const int ns[3] = {16, 32, 64};
  real_t err[3];
  for (int q = 0; q < 3; ++q) {
    err[q] = run(mpi, ns[q], /*discrete_rhs=*/false).max_error;
    if (root) std::printf("analytic rhs n=%d: max error = %.3e\n", ns[q], err[q]);
  }
  for (int q = 1; q < 3; ++q) {
    const real_t eoc = std::log(err[q - 1] / err[q]) / std::log(2.0);
    if (root) std::printf("EOC %d->%d = %.3f\n", ns[q - 1], ns[q], eoc);
    CHECK(eoc > 1.9 && eoc < 2.1);
  }
  return root ? test::summary("test_pressure_wall") : 0;
}
