// MPM-STD boundary_layer 드라이버 — ZPG 평판 경계층 (단계 A: 2D 층류 Blasius 검증).
//   docs/Design/20261006_boundary_layer_validation.md
//
// 축: x 유동(x− 유입 Blasius 프로파일, x+ 대류 유출), y 스팬(주기; ny=1 이면 2D),
//     z 벽수직(z=0 no-slip, z+ 자유류: ∂u/∂z=∂v/∂z=0, w 는 연속식).
// 무차원: δ*₀ = 1, U∞ = 1, ν = U∞/Re_δ*₀. 유입면 물리 위치 x₀ = (δ*₀/1.7208)² U∞/ν.
//
// 한 스텝: [CFL dt] → [유출면 대류 갱신(uⁿ)] → [운동량 rhs(상수 ν) − ∇Pⁿ → solve]
//   → [동기화: halo·벽 ghost·유입·상단 연속식] → [유출 유량 보정(∮u·n=0)] → [증분 투영]
//   → [P 누적·동기화]. 압력은 x-DCT(Neumann 양끝)·y FFT·z TDMA (기존 솔버 그대로).
#include <mpi.h>
#ifdef MPMSTD_GPU
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <memory>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "blasius.hpp"
#include "boundary/apply.hpp"
#include "boundary/open.hpp"
#include "boundary/recycling.hpp"
#include "equation/momentum/rhs/rhs.hpp"
#include "equation/momentum/solve.hpp"
#include "equation/pressure/correction.hpp"
#include "equation/pressure/rhs/rhs.hpp"
#include "equation/pressure/solve.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "grid/stretching.hpp"
#include "parallel/mpi_context.hpp"
#include "parallel/subdomain/subdomain.hpp"
#include "physics/LES/WALE/wale.hpp"
#include "physics/LES/constant_vreman/vreman.hpp"
#include "physics/LES/les.hpp"
#include "physics/separable_force.hpp"
#include "parallel/topology/topology.hpp"
#include "post/checkpoint.hpp"
#include "post/spatial_statistics.hpp"
#include "post/statistics.hpp"
#include "settings.hpp"
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

namespace {

constexpr double kBlasiusDisplacement = 1.7208;  // δ* = 1.7208 √(νx/U∞)

struct Flow {
  double u_inf, nu, x0;
  boundary_layer::Blasius blasius;
  double eta(double z, double x) const { return z * std::sqrt(u_inf / (nu * x)); }
  double u(double z, double x) const { return u_inf * blasius.f_prime(eta(z, x)); }
  double w(double z, double x) const {
    const double e = eta(z, x);
    return 0.5 * std::sqrt(nu * u_inf / x) * (e * blasius.f_prime(e) - blasius.f_value(e));
  }
};

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
    const boundary_layer::Settings s = boundary_layer::load_settings(argv[1]);
    const BoundaryCondition& bc = s.boundary.condition;
    if (mpi.is_root()) {
      std::printf("# settings loaded from %s  (%d ranks)\n", argv[1], mpi.world_size());
      std::fputs(boundary_layer::format_settings(s).c_str(), stdout);
      std::printf("# boundary condition table:\n%s", boundary::format_boundary_condition(bc).c_str());
      std::filesystem::create_directories(s.output.stats_dir);
    }

    const Topology topo(mpi, s.process.npx, s.process.npy, s.process.npz,
                        bc.is_periodic(0), bc.is_periodic(1), bc.is_periodic(2));
    const Subdomain sub(topo, s.mesh.nx, s.mesh.ny, s.mesh.nz);
    const auto faces_x = mpmstd::grid::make_uniform_faces(s.mesh.nx, s.domain.lx);
    const auto faces_y = mpmstd::grid::make_uniform_faces(s.mesh.ny, s.domain.ly);
    const auto faces_z = s.stretch.uniform_z
                             ? mpmstd::grid::make_uniform_faces(s.mesh.nz, s.domain.lz)
                             : mpmstd::grid::make_tanh_wall_faces(s.mesh.nz, s.domain.lz,
                                                                  s.stretch.gamma_z);
    const mpmstd::grid::Grid grid{
        mpmstd::grid::make_axis(faces_x, {sub.x().n_global, sub.x().offset, sub.x().n_local}, "x"),
        mpmstd::grid::make_axis(faces_y, {sub.y().n_global, sub.y().offset, sub.y().n_local}, "y"),
        mpmstd::grid::make_axis(faces_z, {sub.z().n_global, sub.z().offset, sub.z().n_local}, "z"),
    };

    Flow flow;
    flow.u_inf = s.physics.u_infinity;
    flow.nu = s.physics.u_infinity / s.physics.reynolds;  // 단위길이 1 (δ*₀ 또는 δ99,in)
    flow.x0 = std::pow(1.0 / kBlasiusDisplacement, 2) * flow.u_inf / flow.nu;
    flow.blasius = boundary_layer::Blasius::solve();
    const real_t convective_velocity =
        s.physics.convective_velocity > 0 ? s.physics.convective_velocity : flow.u_inf;

    const int nx = sub.x().n_total, ny = sub.y().n_total, nz = sub.z().n_total;
    Field U(nx, ny, nz, "U"), V(nx, ny, nz, "V"), W(nx, ny, nz, "W"), P(nx, ny, nz, "P");
    Field rhs_U(nx, ny, nz, "rhs_U"), rhs_V(nx, ny, nz, "rhs_V"), rhs_W(nx, ny, nz, "rhs_W"),
        rhs_P(nx, ny, nz, "rhs_P"), dP(nx, ny, nz, "dP");
  // [최적화 2026-10-23] 솔버 공용 작업 영역 — 세 솔버가 밴드·스펙트럼 버퍼를 빌려 쓴다 (docs/Memory/plan.md)
    mpmstd::equation::Workspace workspace(std::max(momentum::Solver::required_slot_elements(sub),
                                                   pressure::Solver::required_slot_elements(grid, sub)));
    momentum::Solver momentum_solver(sub, workspace);
    momentum::set_wall_closure(momentum_solver, bc);
    // ── LES (채널 앱과 같은 배선): μ̂ = 1 + ν_t/ν 를 가변물성 운동량의 viscosity 자리에, 1/ρ̂ ≡ 1.
    //    model = none 이면 기존 상수물성 경로 그대로 (DNS 결과와 비트 동일).
    const bool les = s.les.model != "none";
    const bool les_wale = s.les.model == "wale";
    std::unique_ptr<Field> mu_hat, unit_inverse_density;
    if (les) {
      mu_hat = std::make_unique<Field>(nx, ny, nz, "mu_hat");
      unit_inverse_density = std::make_unique<Field>(nx, ny, nz, "unit_inverse_density");
      mu_hat->fill(1.0);
      unit_inverse_density->fill(1.0);
    }
    pressure::Solver pressure_solver(grid, sub, workspace);

    // 유입 평면 (로컬 (y,z), ghost 포함): blasius = z 프로파일 복제, recycling = 매 스텝 갱신
    const bool recycling = s.inflow.method == "recycling";
    boundary::InflowPlane inflow;
    {
      std::vector<real_t> pu(nz), pv(nz, 0.0), pw(nz);
      for (int k = 0; k < nz; ++k) {
        pu[k] = flow.u(std::max(0.0, double(grid.z.center[k])), flow.x0);
        pw[k] = flow.w(std::max(0.0, double(grid.z.face[k])), flow.x0);
      }
      inflow = boundary::InflowPlane::from_profile(pu, pv, pw, ny);
    }
    // 재순환: 전역 z 좌표(ghost 제외)와 재순환면 로컬 i
    std::vector<real_t> z_center_global(s.mesh.nz), z_face_global(s.mesh.nz + 1);
    for (int k = 0; k <= s.mesh.nz; ++k) z_face_global[k] = faces_z[k];
    for (int k = 0; k < s.mesh.nz; ++k) z_center_global[k] = 0.5 * (faces_z[k] + faces_z[k + 1]);
    const double recycle_x = s.inflow.recycle_x > 0 ? s.inflow.recycle_x : 0.75 * s.domain.lx;
    int i_recycle = halo_width;
    for (int i = halo_width; i < nx - halo_width; ++i)
      if (grid.x.face[i] <= recycle_x) i_recycle = i;
    // 공간 발달 통계 (relaxed 평균은 상단 w 공급에도 쓴다)
    mpmstd::post::SpatialStatistics stats(grid, sub, z_center_global, z_face_global, flow.u_inf,
                                          flow.nu, s.inflow.relaxation_time);
    std::vector<real_t> station_re_theta;
    {
      std::stringstream ss(s.output.stations);
      std::string tok;
      while (std::getline(ss, tok, ',')) if (!tok.empty()) station_re_theta.push_back(std::stod(tok));
    }
    std::unique_ptr<boundary::Recycling> recycler;
    if (recycling)
      recycler = std::make_unique<boundary::Recycling>(
          grid, sub, z_center_global, z_face_global, i_recycle, flow.u_inf, flow.nu,
          s.inflow.delta99, s.inflow.relaxation_time);

    // 자유류 상단 w∞(x) (FreeStream kind): 층류 Blasius 0.8604 U∞/√Re_x = U∞ dδ*/dx
    const bool top_freestream = bc.at(boundary::Field::W, 2, 1).kind == boundary::BcKind::FreeStream;
    std::vector<real_t> w_top(nx, 0.0);
    for (int i = 0; i < nx; ++i) {
      const double x = flow.x0 + std::max(0.0, double(grid.x.center[i]));
      // 층류: Blasius w∞. 재순환(난류): 단계 ② 에서 U∞ dδ*/dx 로 교체 — 지금은 0 (작동 확인용)
      w_top[i] = recycling ? 0.0 : 0.8604 * std::sqrt(flow.nu * flow.u_inf / x);
    }
    const auto sync_velocity = [&]() {
      sub.exchange_halo(U.data());
      sub.exchange_halo(V.data());
      sub.exchange_halo(W.data());
      boundary::apply_velocity_wall_ghosts(U, V, W, bc, sub);
      boundary::set_inflow(U, V, W, inflow, bc.velocity_ghost, sub);
      if (top_freestream) boundary::set_top_normal_profile(W, w_top, sub);
      else boundary::set_top_normal_from_continuity(W, U, V, grid, sub);
    };
    const auto sync_pressure = [&]() {
      sub.exchange_halo(P.data());
      boundary::apply_pressure_wall_ghosts(P, bc, sub);
    };

    long restart_step = 0;
    double restart_time = 0;
    if (!s.initial.restart_from.empty()) {
      mpmstd::post::read_checkpoint(s.initial.restart_from, restart_step, restart_time,
                                    {{"U", &U}, {"V", &V}, {"W", &W}, {"P", &P}}, {}, sub);
      if (mpi.is_root())
        std::printf("# restart: %s (step=%ld, t=%.4f)\n", s.initial.restart_from.c_str(),
                    restart_step, restart_time);
    } else {
      // IC: 각 x 위치의 Blasius (또는 균일 U∞)
      std::vector<real_t> hu(U.size(), 0.0), hw(W.size(), 0.0);
      const auto at = [&](int i, int j, int k) {
        return (static_cast<std::size_t>(i) * ny + j) * nz + k;
      };
      std::vector<real_t> hv(V.size(), 0.0);
      std::mt19937 rng(12345u + 7919u * static_cast<unsigned>(mpi.world_rank()));
      std::uniform_real_distribution<double> uni(-1.0, 1.0);
      // 재순환 IC: 1/7 멱법칙 평균 (δ(x) = δ_in ((x_v+x)/x_v)^0.8) + 경계층 안 백색 섭동
      const double x_virtual = std::pow(s.inflow.delta99 / 0.37, 1.25) * std::pow(flow.u_inf / flow.nu, 0.25);
      for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
          for (int k = 0; k < nz; ++k) {
            const double xf = flow.x0 + std::max(0.0, double(grid.x.face[i]));
            const double xc = flow.x0 + std::max(0.0, double(grid.x.center[i]));
            const double zc = std::max(0.0, double(grid.z.center[k]));
            const double zf = std::max(0.0, double(grid.z.face[k]));
            if (recycling) {
              const double xl = std::max(0.0, double(grid.x.face[i]));
              const double delta = s.inflow.delta99 * std::pow((x_virtual + xl) / x_virtual, 0.8);
              const double um = flow.u_inf * std::min(1.0, std::pow(zc / delta, 1.0 / 7.0));
              const double inside = zc < delta ? 1.0 : 0.0;
              hu[at(i, j, k)] = um + s.inflow.noise * flow.u_inf * inside * uni(rng);
              hv[at(i, j, k)] = s.inflow.noise * flow.u_inf * inside * uni(rng);
              hw[at(i, j, k)] = s.inflow.noise * flow.u_inf * (zf < delta ? 1.0 : 0.0) * uni(rng);
            } else {
              hu[at(i, j, k)] = s.initial.profile == "blasius" ? flow.u(zc, xf) : flow.u_inf;
              hw[at(i, j, k)] = s.initial.profile == "blasius" ? flow.w(zf, xc) : 0.0;
            }
          }
      U.upload(hu);
      V.upload(hv);
      W.upload(hw);
    }
    sync_velocity();
    if (recycler) {
      recycler->initialize_running_mean(U, W);
      inflow = recycler->make_inflow_plane(U, V, W, real_t(0));
      sync_velocity();
    }
    sync_pressure();
    if (mpi.is_root() && s.output.checkpoint_interval > 0)
      std::filesystem::create_directories(s.output.checkpoint_dir);

    // IC 발산 청소 (유출 유량 보정 → Neumann Poisson 양립)
    boundary::correct_outflow_flux(U, W, grid, sub);
    rhs_P.fill(0);
    pressure::add_divergence(rhs_P, U, V, W, grid, sub, real_t(1));
    pressure::solve(pressure_solver, grid, sub, rhs_P, dP);
    pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, real_t(1));
    pressure::accumulate(P, dP);
    sync_velocity();
    sync_pressure();

    if (mpi.is_root()) {
      std::printf("# boundary layer: Re_dstar0=%g nu=%.6e x0=%.4f  f''(0)=%.6f | %dx%dx%d\n",
                  s.physics.reynolds, flow.nu, flow.x0, flow.blasius.shear,
                  s.mesh.nx, s.mesh.ny, s.mesh.nz);
      std::printf("# step      time        dt     max|div|   dU_out      Cf(x=Lx/2)  Cf_blasius   sec/step\n");
    }

    // 장 덤프 (2D: j = h 평면; U x면·W z면) — 후처리 plot/blasius_check.py
    const auto write_field_dump = [&](long step_now, double t_now) {
      if (!mpi.is_root() || sub.topology().z().nprocs != 1) return;
      const int h = halo_width;
      std::vector<real_t> u, w;
      U.download(u);
      W.download(w);
      const auto at = [&](int i, int j, int k) {
        return (static_cast<std::size_t>(i) * ny + j) * nz + k;
      };
      char path[512];
      std::snprintf(path, sizeof(path), "%s/bl_field_%08ld.dat", s.output.stats_dir.c_str(), step_now);
      std::FILE* f = std::fopen(path, "w");
      if (!f) return;
      std::fprintf(f, "# bl field dump  step=%ld t=%.6f nu=%.10e u_inf=%g x0=%.8f nx=%d nz=%d lx=%g lz=%g\n",
                   step_now, t_now, flow.nu, flow.u_inf, flow.x0, s.mesh.nx, s.mesh.nz,
                   s.domain.lx, s.domain.lz);
      std::fprintf(f, "# block U: (nx+1) x nz rows of  x_face z_center U\n");
      for (int i = h; i <= nx - h; ++i)
        for (int k = h; k < nz - h; ++k)
          std::fprintf(f, "%.10f %.10f %.12e\n", grid.x.face[i], grid.z.center[k], u[at(i, h, k)]);
      std::fprintf(f, "# block W: nx x (nz+1) rows of  x_center z_face W\n");
      for (int i = h; i < nx - h; ++i)
        for (int k = h; k <= nz - h; ++k)
          std::fprintf(f, "%.10f %.10f %.12e\n", grid.x.center[i], grid.z.face[k], w[at(i, h, k)]);
      std::fclose(f);
      std::printf("# field dump written: %s\n", path);
    };
    const auto write_checkpoint = [&](long step_now, double t_now) {
      if (s.output.checkpoint_interval <= 0) return;
      char ck[512];
      std::snprintf(ck, sizeof(ck), "%s/bl_ck_%08ld.bin", s.output.checkpoint_dir.c_str(), step_now);
      mpmstd::post::write_checkpoint(ck, step_now, t_now,
                                     {{"U", &U}, {"V", &V}, {"W", &W}, {"P", &P}}, {}, sub);
      if (mpi.is_root()) std::printf("# checkpoint written: %s (t=%.4f)\n", ck, t_now);
    };
    // 모니터용 C_f(x = Lx/2): 벽 두 셀 2점 Lagrange 편측 미분 (루트 랭크가 하단 벽 소유 시)
    const auto skin_friction_mid = [&]() -> real_t {
      if (sub.topology().z().minus_neighbor != MPI_PROC_NULL) return 0;
      const int h = halo_width;
      std::vector<real_t> u;
      U.download(u);
      const int i = h + s.mesh.nx / 2;
      const auto at = [&](int ii, int j, int k) {
        return (static_cast<std::size_t>(ii) * ny + j) * nz + k;
      };
      const double z1 = grid.z.center[h], z2 = grid.z.center[h + 1];
      const double u1 = u[at(i, h, h)], u2 = u[at(i, h, h + 1)];
      const double dudz = u1 * z2 / (z1 * (z2 - z1)) - u2 * z1 / (z2 * (z2 - z1));
      return static_cast<real_t>(2 * flow.nu * dudz / (flow.u_inf * flow.u_inf));
    };
    const double x_mid = flow.x0 + grid.x.face[halo_width + s.mesh.nx / 2];
    const double cf_blasius_mid = 0.664 / std::sqrt(flow.u_inf * x_mid / flow.nu);

    // ── 난류 트립 체적력 ([trip], 설계 §4-4; Schlatter & Örlü 2012) ──────────────────────────
    //   F_z = A·e_x(x_c)·e_z(z_f)·g(y_c,t) 를 rhs_W(z-면 변수)에. g 는 스팬 Fourier 랜덤(rms 1)을
    //   t_s 세그먼트마다 새로 뽑아 3차(3p²−2p³)로 완만히 섞는다. 세그먼트 번호로 시드하므로
    //   모든 랭크·재시작에서 같은 모양이다. 벽면(k = h) 은 W = 0 고정이라 포락선 0.
    const bool trip_on = s.trip.amplitude > 0;
    const int trip_nx = sub.x().n_total, trip_ny = sub.y().n_total, trip_nz = sub.z().n_total;
    mpmstd::field::Buffer trip_envelope_x, trip_shape_y, trip_envelope_z;
    std::vector<real_t> trip_shape_host(trip_ny, 0);
    std::vector<real_t> trip_segment_cache[2];
    long trip_segment_cached[2] = {-1, -1};
    const auto trip_segment_shape = [&](const long segment) -> const std::vector<real_t>& {
      const int slot = static_cast<int>(segment & 1);
      if (trip_segment_cached[slot] != segment) {
        std::mt19937 rng(20261008u + 1000003u * static_cast<unsigned>(segment));
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        std::vector<double> amplitude(s.trip.modes), phase(s.trip.modes);
        const double pi = std::acos(-1.0);
        for (int m = 0; m < s.trip.modes; ++m) {
          amplitude[m] = uni(rng);
          phase[m] = pi * uni(rng);
        }
        std::vector<real_t>& g = trip_segment_cache[slot];
        g.assign(trip_ny, 0);
        double sum_square = 0;
        for (int j = mpmstd::halo_width; j < trip_ny - mpmstd::halo_width; ++j) {
          const double y = grid.y.center[j];
          double value = 0;
          for (int m = 0; m < s.trip.modes; ++m)
            value += amplitude[m] * std::cos(2 * pi * (m + 1) * y / s.domain.ly + phase[m]);
          g[j] = static_cast<real_t>(value);
          sum_square += value * value;
        }
        const double rms = std::sqrt(sum_square / sub.y().n_local);
        if (rms > 0)
          for (real_t& v : g) v /= static_cast<real_t>(rms);
        trip_segment_cached[slot] = segment;
      }
      return trip_segment_cache[slot];
    };
    if (trip_on) {
      std::vector<real_t> envelope_x(trip_nx, 0), envelope_z(trip_nz, 0);
      for (int i = mpmstd::halo_width; i < trip_nx - mpmstd::halo_width; ++i) {
        const double xr = (grid.x.center[i] - s.trip.x_center) / s.trip.length_x;
        envelope_x[i] = static_cast<real_t>(std::exp(-xr * xr));
      }
      for (int k = mpmstd::halo_width + 1; k < trip_nz - mpmstd::halo_width; ++k) {
        const double zr = grid.z.face[k] / s.trip.length_z;
        envelope_z[k] = static_cast<real_t>(std::exp(-zr * zr));
      }
      trip_envelope_x.assign(trip_nx);
      trip_envelope_x.upload(envelope_x);
      trip_envelope_z.assign(trip_nz);
      trip_envelope_z.upload(envelope_z);
      trip_shape_y.assign(trip_ny);
      if (mpi.is_root())
        std::printf("# trip: A=%g x_t=%g l_x=%g l_z=%g t_s=%g modes=%d end=%g\n", s.trip.amplitude,
                    s.trip.x_center, s.trip.length_x, s.trip.length_z, s.trip.time_scale,
                    s.trip.modes, s.trip.end_time);
    }
    const auto apply_trip = [&](const real_t t_now, const real_t dt_now) {
      if (!trip_on || (s.trip.end_time > 0 && t_now >= s.trip.end_time)) return;
      const double segment_position = t_now / s.trip.time_scale;
      const long segment = static_cast<long>(std::floor(segment_position));
      const double p = segment_position - segment;
      const double blend = p * p * (3 - 2 * p);
      const std::vector<real_t>& g0 = trip_segment_shape(segment);
      const std::vector<real_t>& g1 = trip_segment_shape(segment + 1);
      for (int j = 0; j < trip_ny; ++j)
        trip_shape_host[j] = static_cast<real_t>((1 - blend) * g0[j] + blend * g1[j]);
      trip_shape_y.upload(trip_shape_host);
      mpmstd::physics::add_separable_force(rhs_W, trip_envelope_x, trip_shape_y, trip_envelope_z,
                                           static_cast<real_t>(s.trip.amplitude) * dt_now, sub);
    };

    real_t t = static_cast<real_t>(restart_time);
    long step = restart_step;
    double next_dump = s.output.stats_interval > 0 ? s.output.stats_interval : 0;
    while (next_dump > 0 && next_dump <= t) next_dump += s.output.stats_interval;
    double wall_prev = MPI_Wtime();

    while (t < s.time.end_time && (s.time.max_step == 0 || step < s.time.max_step)) {
      const real_t dt = mpmstd::time_integration::compute_cfl_dt(U, V, W, grid, sub,
                                                                   s.time.max_cfl, s.time.dt_max);
      // 유출면 대류 갱신 (uⁿ) — 음해 스윕은 이 값을 고정(δ=0)으로 본다
      boundary::update_outflow_convective(U, V, W, convective_velocity, dt, grid, sub);
      // 재순환 유입: 재순환면(uⁿ) → 재스케일 → 유입 평면 (유입면 값은 이 스텝 동안 고정)
      if (recycler) {
        inflow = recycler->make_inflow_plane(U, V, W, dt);
        boundary::set_inflow(U, V, W, inflow, bc.velocity_ghost, sub);
      }

      if (les) {
        if (les_wale)
          mpmstd::physics::les::calculate_eddy_viscosity_wale(*mu_hat, U, V, W, grid, sub, flow.nu,
                                                              static_cast<real_t>(s.les.c_w));
        else
          mpmstd::physics::les::calculate_eddy_viscosity_constant_vreman(
              *mu_hat, U, V, W, grid, sub, flow.nu,
              static_cast<real_t>(2.5 * s.les.c_s * s.les.c_s));
        mpmstd::physics::les::synchronize_viscosity_ratio(*mu_hat, sub);  // 비주기 축 ghost μ̂ = 1
        // [최적화 2026-10-18] 전용 RHS 빌드 — 이송확산 ×3 + ∇P 를 성분당 한 패스로 (트립은 그 뒤에 += ; 트립이 켜지면 덧셈 순서만 다름)
        momentum::build_variable_viscosity_rhs(rhs_U, rhs_V, rhs_W, U, V, W, P, *mu_hat,
                                               *unit_inverse_density, grid, sub, flow.nu, dt);
      } else {
        momentum::build_constant_property_rhs(rhs_U, rhs_V, rhs_W, U, V, W, P, grid, sub, flow.nu, dt);
      }
      apply_trip(t, dt);  // 난류 트립 (옵션)
      if (les)
        momentum::solve_variable_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V, W,
                                          *mu_hat, *unit_inverse_density, flow.nu, dt);
      else
        momentum::solve_constant_property(momentum_solver, grid, sub, rhs_U, rhs_V, rhs_W, U, V, W,
                                          flow.nu, dt);
      sync_velocity();
      const real_t du_out = boundary::correct_outflow_flux(U, W, grid, sub);

      rhs_P.fill(0);
      pressure::add_divergence(rhs_P, U, V, W, grid, sub, dt);
      pressure::solve(pressure_solver, grid, sub, rhs_P, dP);
      pressure::correct_velocity_constant_property(U, V, W, dP, grid, sub, dt);
      pressure::accumulate(P, dP);
      sync_velocity();
      sync_pressure();

      t += dt;
      ++step;
      stats.sample(U, V, W, dt, t >= s.output.stats_start_time);
      if ((recycling || trip_on) && step % s.output.monitor_interval == 0) {
        // 상단 자유류 w(x) = U∞ dδ*/dx — relaxed 평균의 δ*(x) 국소 기울기에서.
        // ⚠ 2026-10-12 이전엔 recycling 일 때만 갱신 → 층류 유입+트립 런은 상단 w 가 층류 Blasius
        // 값(난류 dδ*/dx 의 절반)에 묶여 자유류가 2% 가속(FPG)했다 (열 경계층 §5-3).
        w_top = stats.free_stream_w(stats.integrals(/*from_relaxed=*/true));
      }
      if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval == 0)
        write_checkpoint(step, t);
      if (next_dump > 0 && t >= next_dump) {
        if (s.mesh.ny == 1) write_field_dump(step, t);
        if (stats.accumulated_time() > 0) {
          char sp[512];
          std::snprintf(sp, sizeof(sp), "%s/bl_stats_%08ld.dat", s.output.stats_dir.c_str(), step);
          stats.write(sp, step, t, station_re_theta);
          if (mpi.is_root()) std::printf("# stats written: %s (T_avg=%.2f)\n", sp, stats.accumulated_time());
        }
        next_dump += s.output.stats_interval;
      }
      if (step % s.output.monitor_interval == 0 || step == 1) {
        const real_t div = mpmstd::post::max_divergence(U, V, W, grid, sub);
        const real_t cf = skin_friction_mid();
        const double wall_now = MPI_Wtime();
        if (mpi.is_root()) {
          std::printf("%7ld %9.3f %9.5f %10.2e %10.2e %11.6f %11.6f %8.4f\n", step, t, dt, div,
                      du_out, cf, cf_blasius_mid,
                      (wall_now - wall_prev) / s.output.monitor_interval);
          if (recycler) {
            real_t wmax = 0;
            for (const real_t v : w_top) wmax = std::max(wmax, v);
            std::printf("#   top w_max %.3e\n", wmax);
            const auto& sc = recycler->scales();
            std::printf("#   recycle: u_tau_r %.5f  delta99_r %.4f  theta_r %.4f  gamma %.4f  U_e_r %.4f | inflow: u_tau %.5f  Re_theta %.1f (Re_theta_r %.1f)\n",
                        sc.u_tau_recycle, sc.delta99_recycle, sc.theta_recycle, sc.gamma, sc.u_edge_recycle,
                        sc.u_tau_inflow, flow.u_inf * sc.theta_inflow / flow.nu,
                        flow.u_inf * sc.theta_recycle / flow.nu);
          }
          std::fflush(stdout);
        }
        wall_prev = wall_now;
        if (!(div == div) || div > 1e3) throw std::runtime_error("divergence blew up — aborting");
      }
    }
    if (s.mesh.ny == 1) write_field_dump(step, t);
    if (stats.accumulated_time() > 0) {
      char sp[512];
      std::snprintf(sp, sizeof(sp), "%s/bl_stats_%08ld.dat", s.output.stats_dir.c_str(), step);
      stats.write(sp, step, t, station_re_theta);
      if (mpi.is_root()) std::printf("# stats written: %s (T_avg=%.2f)\n", sp, stats.accumulated_time());
    }
    if (s.output.checkpoint_interval > 0 && step % s.output.checkpoint_interval != 0)
      write_checkpoint(step, t);
    if (mpi.is_root())
      std::printf("# FINAL: Cf(x=Lx/2) = %.6f (Blasius %.6f, %+.2f%%)  t = %.1f step = %ld\n",
                  skin_friction_mid(), cf_blasius_mid,
                  100 * (skin_friction_mid() / cf_blasius_mid - 1), t, step);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error (rank %d): %s\n", mpi.world_rank(), e.what());
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return 0;
}
