// MPM-STD cavity 드라이버 — 입력 [boundary] 로 축별 BC 를 받는 첫 앱.
// 검증 문제: de Vahl Davis (1983) 차등가열 정방 공동 (2D, Pr = 0.71, Ra 1e3~1e6).
//   docs/Design/20261005_boundary_condition_framework.md 단계 4.
//
// 축 관례: x = 수평(hot 벽 x=0: T=+0.5, cold 벽 x=Lx: T=−0.5),
//          y = 스팬(주기; **ny = 1 이면 진짜 2D** — 축 붕괴, 별도 2D 코드 없음),
//          z = 연직(중력; 부력이 W 에 실림), 상하벽 단열.
// 어느 축이 벽이고 어느 면이 어떤 조건인지는 전부 입력 [boundary] 테이블이 정한다
// (이 파일엔 하드코딩된 벽이 없다 — Topology 주기 플래그도 테이블에서).
// 무차원: 자유낙하 스케일 — ν = √(Pr/Ra), α = 1/√(Ra·Pr), 부력계수 1.
//   벤치마크의 확산 스케일 속도(u·L/α)로 바꾸려면 √(Ra·Pr) 를 곱한다 (출력에 포함).
//
// 한 스텝 (dhvc 와 동일 recipe; 상수물성 전용 경로):
//   [CFL dt] → [에너지 Tⁿ⁺¹] → [운동량 rhs + 부력_z(Tⁿ⁺¹) − ∇Pⁿ → solve]
//   → [증분 투영] → [P 누적] → [진단]
//
// 진단 (모니터·FINAL): Nu_hot / Nu_cold = −(1/Lz)∫ ∂T/∂x|벽 dz (2차 편측차분),
//   u_max(x = Lx/2 연직 중앙선, 위치 z), w_max(z = Lz/2 수평 중앙선, 위치 x).
#include <mpi.h>
#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "boundary/apply.hpp"
#include "boundary/boundary_condition.hpp"
#include "equation/momentum/rhs/rhs.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/pressure/correction.hpp"
#include "equation/pressure/rhs/rhs.hpp"
#include "equation/pressure/solve.hpp"
#include "equation/scalar/solve.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "parallel/topology/topology.hpp"
#include "physics/buoyancy.hpp"
#include "post/checkpoint.hpp"
#include "post/statistics.hpp"
#include "rbc/settings.hpp"
#include "time/time_step.hpp"

using mpmstd::halo_width;
using mpmstd::real_t;
using mpmstd::boundary::BoundaryCondition;
using mpmstd::field::Field;
using mpmstd::parallel::Subdomain;
using mpmstd::parallel::Topology;
namespace boundary = mpmstd::boundary;
namespace momentum = mpmstd::equation::momentum;
namespace pressure = mpmstd::equation::pressure;
namespace scalar = mpmstd::equation::scalar;
namespace physics = mpmstd::physics;

namespace {

void sync_velocity(Field& U, Field& V, Field& W, const BoundaryCondition& bc,
                   const Subdomain& sub) {
  sub.exchange_halo(U.data());
  sub.exchange_halo(V.data());
  sub.exchange_halo(W.data());
  boundary::apply_velocity_wall_ghosts(U, V, W, bc, sub);
}

void sync_temperature(Field& T, const BoundaryCondition& bc, const Subdomain& sub) {
  sub.exchange_halo(T.data());
  boundary::apply_scalar_wall_ghosts(T, boundary::Field::T, bc, sub);
}

void sync_pressure(Field& P, const BoundaryCondition& bc, const Subdomain& sub) {
  sub.exchange_halo(P.data());
  boundary::apply_pressure_wall_ghosts(P, bc, sub);
}

// 냉시작 IC: 정지 유체 + x 전도 선형 온도 (hot → cold 벽값을 테이블에서 읽는다)
// + perturb·sin(πx/Lx)·sin(πz/Lz) (양 벽에서 0 — 전도만 있으면 e^{−α(π²/Lx²+π²/Lz²)t}
// 로 감쇠: 에너지 방정식 x 벽 처리의 독립 검증에 쓴다)
void cold_start_initial_condition(Field& T, const BoundaryCondition& bc,
                                  const mpmstd::grid::Grid& grid, const Subdomain& sub,
                                  const real_t lx, const real_t lz, const real_t perturb) {
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const real_t t_hot = bc.at(boundary::Field::T, 0, 0).value;
  const real_t t_cold = bc.at(boundary::Field::T, 0, 1).value;
  const real_t pi = std::acos(real_t(-1));
  std::vector<real_t> st(T.size(), 0.0);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
        st[(static_cast<std::size_t>(i) * ny + j) * nz + k] =
            t_hot + (t_cold - t_hot) * (grid.x.center[i] / lx)
            + perturb * std::sin(pi * grid.x.center[i] / lx)
                  * std::sin(pi * grid.z.center[k] / lz);
  T.upload(st);
}

struct Diagnostics {
  real_t t_center = 0;                   // T 중앙 셀 (감쇠 검증용)
  real_t abs_u_max = 0, abs_v_max = 0, abs_w_max = 0;  // 전역 max|·| (free-fall 단위; 진단)
  real_t nu_hot = 0, nu_cold = 0;        // 벽 평균 Nusselt (2차 편측차분)
  real_t nu_hot_1st = 0;                 // 솔버가 실제로 쓰는 1차 벽유속 (T₁−T_w)/(dx/2)
  real_t u_max = 0, z_of_u_max = 0;      // x = Lx/2 연직 중앙선 (확산 스케일)
  real_t w_max = 0, x_of_w_max = 0;      // z = Lz/2 수평 중앙선 (확산 스케일)
};

// 호스트로 내려 계산 (2D 소형 문제 — 모니터 주기마다 한 번이라 비용 무시).
Diagnostics compute_diagnostics(const Field& U, const Field& V, const Field& W, const Field& T,
                                const mpmstd::grid::Grid& grid, const Subdomain& sub,
                                const BoundaryCondition& bc, const real_t lx,
                                const real_t lz, const real_t velocity_scale) {
  const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
  const int h = halo_width;
  const auto at = [&](int i, int j, int k) {
    return (static_cast<std::size_t>(i) * ny + j) * nz + k;
  };
  std::vector<real_t> u, v, w, t;
  U.download(u);
  V.download(v);
  W.download(w);
  T.download(t);
  const int j = h;  // 2D: 첫(유일) y 셀. ny>1 이면 y 평균 대신 첫 평면 (검증은 ny=1)
  real_t amax[3] = {0, 0, 0};
  for (int i = h; i < nx - h; ++i)
    for (int jj = h; jj < ny - h; ++jj)
      for (int k = h; k < nz - h; ++k) {
        amax[0] = std::max(amax[0], std::abs(u[at(i, jj, k)]));
        amax[1] = std::max(amax[1], std::abs(v[at(i, jj, k)]));
        amax[2] = std::max(amax[2], std::abs(w[at(i, jj, k)]));
      }
  const real_t dx = grid.x.cell_width[h];
  const real_t t_hot = bc.at(boundary::Field::T, 0, 0).value;
  const real_t t_cold = bc.at(boundary::Field::T, 0, 1).value;

  // 벽 열유속: 2차 편측차분 (벽, dx/2, 3dx/2): f' = (−8 f_w + 9 f₁ − f₂) / (3dx)
  real_t flux_hot = 0, flux_cold = 0, flux_hot_1st = 0;
  for (int k = h; k < nz - h; ++k) {
    const real_t dz = grid.z.cell_width[k];
    flux_hot_1st += -(t[at(h, j, k)] - t_hot) / (dx / 2) * dz;
    const real_t g_hot = (-8 * t_hot + 9 * t[at(h, j, k)] - t[at(h + 1, j, k)]) / (3 * dx);
    const real_t g_cold =
        -(-8 * t_cold + 9 * t[at(nx - h - 1, j, k)] - t[at(nx - h - 2, j, k)]) / (3 * dx);
    flux_hot += -g_hot * dz;
    flux_cold += -g_cold * dz;
  }
  // 중앙선 최대: U 는 x 면 (면 i = h + nx_global/2 가 x = Lx/2), W 는 z 면
  real_t u_max_local = 0, z_u = 0;
  const int i_mid = h + sub.x().n_global / 2;
  for (int k = h; k < nz - h; ++k)
    if (std::abs(u[at(i_mid, j, k)]) > std::abs(u_max_local)) {
      u_max_local = u[at(i_mid, j, k)];
      z_u = grid.z.center[k];
    }
  real_t w_max_local = 0, x_w = 0;
  const int k_mid_global = sub.z().n_global / 2;
  if (sub.z().owns(k_mid_global)) {
    const int k_mid = sub.z().local(k_mid_global);
    for (int i = h; i < nx - h; ++i)
      if (std::abs(w[at(i, j, k_mid)]) > std::abs(w_max_local)) {
        w_max_local = w[at(i, j, k_mid)];
        x_w = grid.x.center[i];
      }
  }

  MPI_Comm comm = sub.topology().cart_comm();
  real_t flux[3] = {flux_hot, flux_cold, flux_hot_1st}, flux_sum[3];
  MPI_Allreduce(flux, flux_sum, 3, MPI_DOUBLE, MPI_SUM, comm);
  struct {
    double value;
    int rank;
  } in_u{std::abs(u_max_local), 0}, out_u, in_w{std::abs(w_max_local), 0}, out_w;
  MPI_Comm_rank(comm, &in_u.rank);
  in_w.rank = in_u.rank;
  MPI_Allreduce(&in_u, &out_u, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  MPI_Allreduce(&in_w, &out_w, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
  real_t loc_u[2] = {u_max_local, z_u}, loc_w[2] = {w_max_local, x_w};
  MPI_Bcast(loc_u, 2, MPI_DOUBLE, out_u.rank, comm);
  MPI_Bcast(loc_w, 2, MPI_DOUBLE, out_w.rank, comm);

  Diagnostics d;
  {
    real_t gmax[3];
    MPI_Allreduce(amax, gmax, 3, MPI_DOUBLE, MPI_MAX, comm);
    d.abs_u_max = gmax[0];
    d.abs_v_max = gmax[1];
    d.abs_w_max = gmax[2];
    real_t tc = 0;
    if (sub.z().owns(k_mid_global)) tc = t[at(h + sub.x().n_global / 2, j, sub.z().local(k_mid_global))];
    MPI_Allreduce(&tc, &d.t_center, 1, MPI_DOUBLE, MPI_SUM, comm);
  }
  const real_t delta_t = (t_hot - t_cold) != 0 ? (t_hot - t_cold) : real_t(1);
  d.nu_hot = flux_sum[0] / lz * lx / delta_t;
  d.nu_cold = flux_sum[1] / lz * lx / delta_t;
  d.nu_hot_1st = flux_sum[2] / lz * lx / delta_t;
  d.u_max = loc_u[0] * velocity_scale;
  d.z_of_u_max = loc_u[1];
  d.w_max = loc_w[0] * velocity_scale;
  d.x_of_w_max = loc_w[1];
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  mpmstd::parallel::MpiContext mpi(&argc, &argv);
#ifdef MPMSTD_GPU
  {
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (n_devices > 0) cudaSetDevice(mpi.node_rank() % n_devices);
  }
#endif
  if (argc < 2) {
    if (mpi.is_root()) std::fprintf(stderr, "usage: %s <input.ini>\n", argv[0]);
    return 1;
  }
  try {
    const rbc::Settings s = rbc::load_settings(argv[1]);
    const BoundaryCondition& bc = s.boundary.condition;
    if (s.physics.gravity_axis != 2 && s.physics.gravity_axis != 0)
      throw std::runtime_error("cavity: gravity_axis must be 2 (z 연직) or 0 (x 연직 — 축 교대 검증용)");
    if (s.physics.delta_temperature != 0)
      throw std::runtime_error("cavity: delta_temperature must be 0 (상수물성 OB)");
    if (!bc.is_periodic(1))
      throw std::runtime_error("cavity: [boundary] y must be periodic (ny = 1 for 2D)");
    // x 는 보통 wall 이지만 주기도 허용한다 (에너지 방정식 벽 처리 A/B 시험용 —
    // 그때 벽 Nu 진단은 의미 없다)

    if (mpi.is_root()) {
      std::printf("# settings loaded from %s  (%d ranks)\n", argv[1], mpi.world_size());
      std::fputs(rbc::format_settings(s).c_str(), stdout);
      std::printf("# boundary condition table:\n%s",
                  boundary::format_boundary_condition(bc).c_str());
      std::filesystem::create_directories(s.output.stats_dir);
    }

    const Topology topo(mpi, s.process.npx, s.process.npy, s.process.npz,
                        bc.is_periodic(0), bc.is_periodic(1), bc.is_periodic(2));
    const Subdomain sub(topo, s.mesh.nx, s.mesh.ny, s.mesh.nz);
    const auto faces_x = mpmstd::grid::make_uniform_faces(s.mesh.nx, s.domain.lx);
    const auto faces_y = mpmstd::grid::make_uniform_faces(s.mesh.ny, s.domain.ly);
    const auto faces_z =
        s.stretch.uniform_z
            ? mpmstd::grid::make_uniform_faces(s.mesh.nz, s.domain.lz)
            : mpmstd::grid::make_tanh_faces(s.mesh.nz, s.domain.lz, s.stretch.gamma_z);
    const mpmstd::grid::Grid grid{
        mpmstd::grid::make_axis(faces_x, {sub.x().n_global, sub.x().offset, sub.x().n_local}, "x"),
        mpmstd::grid::make_axis(faces_y, {sub.y().n_global, sub.y().offset, sub.y().n_local}, "y"),
        mpmstd::grid::make_axis(faces_z, {sub.z().n_global, sub.z().offset, sub.z().n_local}, "z"),
    };

    const real_t nu = std::sqrt(s.physics.prandtl / s.physics.rayleigh);
    const real_t Ct = real_t(1) / std::sqrt(s.physics.rayleigh * s.physics.prandtl);
    const real_t velocity_scale = std::sqrt(s.physics.rayleigh * s.physics.prandtl);  // → u·L/α

    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"),
        T(nx, ny, nz, "T"), P(nx, ny, nz, "P");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"), rhs_W(nx, ny, nz, "rhs_W");
    // [메모리 2026-10-26] 압력 우변·보정은 운동량 우변 메모리를 재사용한다 (docs/Memory/plan.md §9).
    //   한 스텝에서 rhs_U 의 역할: 운동량 우변 → (momentum::solve 안에서) δU → 압력 우변 → (pressure::solve 는 제자리) dP.
    //   운동량 → 압력 순서가 고정이라 두 역할이 동시에 살아 있는 순간이 없다. 되돌리려면 아래 두 참조를 Field 생성으로.
    Field& rhs_P = rhs_U;
    Field& dP = rhs_P;      // pressure::solve 가 rhs 자리에 해를 쓴다
  // [최적화 2026-10-23] 솔버 공용 작업 영역 — 세 솔버가 밴드·스펙트럼 버퍼를 빌려 쓴다 (docs/Memory/plan.md)
    mpmstd::equation::Workspace workspace(std::max({momentum::Solver::required_slot_elements(sub),
                                                    scalar::Solver::required_slot_elements(sub),
                                                    pressure::Solver::required_slot_elements(grid, sub)}));
    momentum::Solver momentum_solver(sub, workspace);
    // 접선 속도 벽 행 닫기를 ghost 정책과 일치 ([boundary] velocity_ghost: value → 0,
    // mirror → −1; slip → +1). 기본 value 는 1차(벽 마찰 절반) — 벤치마크는 mirror 권장.
    momentum::set_wall_closure(momentum_solver, bc);
    scalar::Solver scalar_solver(sub, workspace);
    // 에너지 벽 행 닫기를 ghost 정책과 일치시킨다 (Dirichlet 거울 → −1, 단열 → +1).
    // 기본 drop(PaScaL 충실)은 dt·α/dx² ≳ 3 에서 벽 셀 모드가 폭주 (scalar/solve.hpp).
    scalar::set_wall_closure(scalar_solver, bc, boundary::Field::T);
    pressure::Solver pressure_solver(grid, sub, workspace);

    long restart_step = 0;
    double restart_time = 0;
    if (!s.initial.restart_from.empty()) {
      mpmstd::post::read_checkpoint(s.initial.restart_from, restart_step, restart_time,
                                    {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T},
                                     {"P", &P}},
                                    {}, sub);
      if (mpi.is_root())
        std::printf("# restart: %s (step=%ld, t=%.4f)\n",
                    s.initial.restart_from.c_str(), restart_step, restart_time);
    } else {
      cold_start_initial_condition(T, bc, grid, sub, s.domain.lx, s.domain.lz,
                                   s.initial.perturb);
    }
    sync_velocity(U, V, W, bc, sub);
    sync_temperature(T, bc, sub);
    sync_pressure(P, bc, sub);
    if (mpi.is_root() && s.output.checkpoint_interval > 0)
      std::filesystem::create_directories(s.output.checkpoint_dir);

    if (mpi.is_root()) {
      std::printf("# cavity OB: Ra=%.3g Pr=%.3g -> nu=%.6e alpha=%.6e | %dx%dx%d\n",
                  s.physics.rayleigh, s.physics.prandtl, nu, Ct, s.mesh.nx, s.mesh.ny,
                  s.mesh.nz);
      std::printf("# step      time        dt     max|div|   Nu_hot   Nu_cold   "
                  "u_max(z)          w_max(x)        sec/step   T_center\n");
    }

    const auto write_statistics = [&](long step_now, double t_now) {
      // 중앙선 프로파일 (루트가 전역 z 를 모으기엔 분할이 끼므로, 각 랭크 블록을
      // 이어 쓰는 대신 npz = 1 전제로 루트만 쓴다 — 검증 런은 1 랭크)
      if (!mpi.is_root() || sub.topology().z().nprocs != 1) return;
      const int h = halo_width;
      std::vector<real_t> u, w, th;
      U.download(u);
      W.download(w);
      T.download(th);
      const auto at = [&](int i, int j, int k) {
        return (static_cast<std::size_t>(i) * ny + j) * nz + k;
      };
      char path[512];
      std::snprintf(path, sizeof(path), "%s/cavity_stats_%08ld.dat",
                    s.output.stats_dir.c_str(), step_now);
      std::FILE* f = std::fopen(path, "w");
      if (!f) return;
      const int i_mid = h + s.mesh.nx / 2, k_mid = h + s.mesh.nz / 2;
      std::fprintf(f, "# cavity midlines  step=%ld t=%.4f Ra=%g Pr=%g (velocity x sqrt(Ra Pr))\n",
                   step_now, t_now, s.physics.rayleigh, s.physics.prandtl);
      std::fprintf(f, "# section 1: z  u(x=Lx/2, z)  T(x=Lx/2, z)\n");
      for (int k = h; k < nz - h; ++k)
        std::fprintf(f, "%12.8f %14.8e %14.8e\n", grid.z.center[k],
                     u[at(i_mid, h, k)] * velocity_scale,
                     real_t(0.5) * (th[at(i_mid - 1, h, k)] + th[at(i_mid, h, k)]));
      std::fprintf(f, "\n\n# section 2: x  w(x, z=Lz/2)  T(x, z=Lz/2)\n");
      for (int i = h; i < nx - h; ++i)
        std::fprintf(f, "%12.8f %14.8e %14.8e\n", grid.x.center[i],
                     w[at(i, h, k_mid)] * velocity_scale,
                     real_t(0.5) * (th[at(i, h, k_mid - 1)] + th[at(i, h, k_mid)]));
      // 진단용: hot 벽 첫 셀 열(x = dx/2) 을 따라 T, w, 그리고 u(첫 내부 면 i=h+1)
      std::fprintf(f, "\n\n# section 3: z  T(x=dx/2, z)  w(x=dx/2, z)  u(x=dx, z)  [free-fall units]\n");
      for (int k = h; k < nz - h; ++k)
        std::fprintf(f, "%12.8f %14.8e %14.8e %14.8e\n", grid.z.center[k], th[at(h, h, k)],
                     w[at(h, h, k)], u[at(h + 1, h, k)]);
      std::fclose(f);
      std::printf("# stats written: %s\n", path);
    };

    // 최종 2D 장 덤프 (벤치마크 후처리 plot/benchmark.py 입력): U 는 x 면 (nx+1)×nz,
    // W 는 z 면 nx×(nz+1), T 는 셀중심 nx×nz. npz = 1 전제 (검증 런).
    const auto write_field_dump = [&](long step_now, double t_now) {
      if (!mpi.is_root() || sub.topology().z().nprocs != 1) return;
      const int h = halo_width;
      std::vector<real_t> u, w, th;
      U.download(u);
      W.download(w);
      T.download(th);
      const auto at = [&](int i, int j, int k) {
        return (static_cast<std::size_t>(i) * ny + j) * nz + k;
      };
      char path[512];
      std::snprintf(path, sizeof(path), "%s/cavity_field_%08ld.dat",
                    s.output.stats_dir.c_str(), step_now);
      std::FILE* f = std::fopen(path, "w");
      if (!f) return;
      std::fprintf(f, "# cavity field dump  step=%ld t=%.6f Ra=%g Pr=%g nx=%d nz=%d lx=%g lz=%g\n",
                   step_now, t_now, s.physics.rayleigh, s.physics.prandtl, s.mesh.nx,
                   s.mesh.nz, s.domain.lx, s.domain.lz);
      std::fprintf(f, "# velocities in free-fall units (multiply by sqrt(Ra Pr) for L/alpha)\n");
      std::fprintf(f, "# block U: (nx+1) x nz rows of  x_face z_center U\n");
      for (int i = h; i <= nx - h; ++i)
        for (int k = h; k < nz - h; ++k)
          std::fprintf(f, "%.10f %.10f %.12e\n", grid.x.face[i], grid.z.center[k],
                       u[at(i, h, k)]);
      std::fprintf(f, "# block W: nx x (nz+1) rows of  x_center z_face W\n");
      for (int i = h; i < nx - h; ++i)
        for (int k = h; k <= nz - h; ++k)
          std::fprintf(f, "%.10f %.10f %.12e\n", grid.x.center[i], grid.z.face[k],
                       w[at(i, h, k)]);
      std::fprintf(f, "# block T: nx x nz rows of  x_center z_center T\n");
      for (int i = h; i < nx - h; ++i)
        for (int k = h; k < nz - h; ++k)
          std::fprintf(f, "%.10f %.10f %.12e\n", grid.x.center[i], grid.z.center[k],
                       th[at(i, h, k)]);
      std::fclose(f);
      std::printf("# field dump written: %s\n", path);
    };

    const auto write_checkpoint = [&](long step_now, double t_now) {
      if (s.output.checkpoint_interval <= 0) return;
      char ck[512];
      std::snprintf(ck, sizeof(ck), "%s/cavity_ck_%08ld.bin",
                    s.output.checkpoint_dir.c_str(), step_now);
      mpmstd::post::write_checkpoint(ck, step_now, t_now,
                                     {{"U", &U}, {"V", &V}, {"W", &W}, {"T", &T},
                                      {"P", &P}},
                                     {}, sub);
      if (mpi.is_root()) std::printf("# checkpoint written: %s (t=%.4f)\n", ck, t_now);
    };

    real_t t = static_cast<real_t>(restart_time);
    long step = restart_step;
    double next_stats_write =
        s.output.stats_interval > 0 ? s.output.stats_start_time + s.output.stats_interval
                                    : 0;
    while (next_stats_write > 0 && next_stats_write <= t)
      next_stats_write += s.output.stats_interval;
    double wall_prev = MPI_Wtime();
    Diagnostics diag;

    while (t < s.time.end_time && (s.time.max_step == 0 || step < s.time.max_step)) {
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(
          U, V, W, grid, sub, s.time.max_cfl, s.time.dt_max);

      // ── 에너지 (uⁿ, 상수-α 융합 ADI) → Tⁿ⁺¹ ──
      scalar::solve_constant_property(scalar_solver, grid, sub, T, U, V, W, Ct, dt);
      sync_temperature(T, bc, sub);

      // ── 운동량 예측자 (상수 ν) + 부력_z(Tⁿ⁺¹) − ∇Pⁿ ──
      rhs_U.fill(0);
      rhs_V.fill(0);
      rhs_W.fill(0);
      momentum::add_advection_diffusion_u_constant_property(rhs_U, U, V, W, grid, sub, nu, dt);
      momentum::add_advection_diffusion_v_constant_property(rhs_V, U, V, W, grid, sub, nu, dt);
      momentum::add_advection_diffusion_w_constant_property(rhs_W, U, V, W, grid, sub, nu, dt);
      if (s.physics.gravity_axis == 2) physics::add_buoyancy_z_constant_property(rhs_W, T, grid, sub, dt);
      else physics::add_buoyancy_x_constant_property(rhs_U, T, grid, sub, dt);  // 축 교대 검증
      pressure::add_gradient_constant_property(rhs_U, rhs_V, rhs_W, P, grid, sub, dt);
      momentum::solve_constant_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W,
                                        U, V, W, nu, dt);
      sync_velocity(U, V, W, bc, sub);

      // ── 증분 투영 ──
      rhs_P.fill(0);
      pressure::add_divergence(rhs_P, U, V, W, grid, sub, dt);
      pressure::solve(pressure_solver, grid, sub, rhs_P, dP);
      pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, dt);
      pressure::accumulate(P, dP);
      sync_velocity(U, V, W, bc, sub);
      sync_pressure(P, bc, sub);

      t += dt;
      ++step;
      if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval == 0)
        write_checkpoint(step, t);
      if (s.output.stats_interval > 0 && t >= next_stats_write) {
        write_statistics(step, t);
        next_stats_write += s.output.stats_interval;
      }

      if (step % s.output.monitor_interval == 0 || step == 1) {
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub);
        diag = compute_diagnostics(U, V, W, T, grid, sub, bc, s.domain.lx, s.domain.lz,
                                   velocity_scale);
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %8.4f %8.4f %9.3f (%6.4f) %9.3f (%6.4f) %8.4f %12.5e\n",
                      step, t, dt, div, diag.nu_hot, diag.nu_cold, diag.u_max,
                      diag.z_of_u_max, diag.w_max, diag.x_of_w_max,
                      (wall_now - wall_prev) / s.output.monitor_interval, diag.t_center);
          std::printf("#   max|U| %.3e  max|V| %.3e  max|W| %.3e\n", diag.abs_u_max,
                      diag.abs_v_max, diag.abs_w_max);
          std::fflush(stdout);
        }
        wall_prev = wall_now;
        if (!(div == div) || div > 1e3)
          throw std::runtime_error("divergence blew up — aborting");
      }
    }

    write_statistics(step, t);
    write_field_dump(step, t);
    if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval != 0)
      write_checkpoint(step, t);
    diag = compute_diagnostics(U, V, W, T, grid, sub, bc, s.domain.lx, s.domain.lz,
                               velocity_scale);
    if (mpi.is_root())
      std::printf("# FINAL: Nu_hot = %.4f  Nu_cold = %.4f  Nu_hot_1st = %.4f  u_max = %.3f (z=%.4f)  "
                  "w_max = %.3f (x=%.4f)  (Ra = %g, t = %.1f, step = %ld)\n",
                  diag.nu_hot, diag.nu_cold, diag.nu_hot_1st, diag.u_max, diag.z_of_u_max,
                  diag.w_max, diag.x_of_w_max, s.physics.rayleigh, t, step);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return 0;
}
