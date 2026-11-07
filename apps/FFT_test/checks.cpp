#include "checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "boundary/wall.hpp"

namespace fft_test {

using mpmstd::field::Field;
using mpmstd::grid::Grid;
using mpmstd::parallel::Subdomain;

namespace {

const double kTwoPi = 2 * std::acos(-1.0);

// 오차 누적용 최댓값. std::max(m, NaN) 은 m 을 돌려주므로 비유한 값을 +inf 로 바꾼다 (NaN 해가 시험을 통과하지 못하게).
double worst(const double m, const double v) {
  return std::max(m, std::isfinite(v) ? std::abs(v) : std::numeric_limits<double>::infinity());
}

double global_max(const double v, MPI_Comm comm) {
  double out = 0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
  return out;
}

double global_sum(const double v, MPI_Comm comm) {
  double out = 0;
  MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, comm);
  return out;
}

// ── Poisson 시험 ───────────────────────────────────────────────────────────────

std::size_t field_index(const Subdomain& sub, const int i, const int j, const int k) {
  return (static_cast<std::size_t>(i) * sub.y().n_total + j) * sub.z().n_total + k;
}

// 이산 라플라시안 (기존 솔버의 행렬과 같은 식): x·y 균일 3점, z 는 a = 1/(dz·dmz_k), c = 1/(dz·dmz_{k+1})
double laplacian(const std::vector<double>& f, const Grid& g, const Subdomain& sub, const int i, const int j,
                 const int k) {
  const int h = mpmstd::halo_width;
  const double dx = g.x.cell_width[h], dy = g.y.cell_width[h];
  const double c = f[field_index(sub, i, j, k)];
  const double lap_x = (f[field_index(sub, i + 1, j, k)] - 2 * c + f[field_index(sub, i - 1, j, k)]) / (dx * dx);
  const double lap_y = (f[field_index(sub, i, j + 1, k)] - 2 * c + f[field_index(sub, i, j - 1, k)]) / (dy * dy);
  const double dz = g.z.cell_width[k];
  const double lap_z = (f[field_index(sub, i, j, k + 1)] - c) / (dz * g.z.center_distance[k + 1])
                       - (c - f[field_index(sub, i, j, k - 1)]) / (dz * g.z.center_distance[k]);
  return lap_x + lap_y + lap_z;
}

// 장의 ghost 를 solve 의 후처리와 같은 규칙으로 채운다 (halo + 벽 zero_gradient)
void fill_ghosts(Field& f, const Subdomain& sub) {
  sub.exchange_halo(f.data());
  for (int axis = 0; axis < 3; ++axis) {
    if (sub.topology().axis(axis).periodic) continue;
    mpmstd::boundary::fill_wall_zero_gradient(f, sub, axis, 0);
    mpmstd::boundary::fill_wall_zero_gradient(f, sub, axis, 1);
  }
}

template <class Function>
void for_interior(const Subdomain& sub, const Function& function) {
  const int h = mpmstd::halo_width;
  for (int i = h; i < sub.x().n_total - h; ++i)
    for (int j = h; j < sub.y().n_total - h; ++j)
      for (int k = h; k < sub.z().n_total - h; ++k) function(i, j, k);
}

// SplitMix64 — 전역 셀 번호의 결정적 해시 → [−1, 1). 분할과 무관하게 같은 rhs.
double hashed_uniform(std::uint64_t key) {
  key += 0x9E3779B97F4A7C15ULL;
  key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9ULL;
  key = (key ^ (key >> 27)) * 0x94D049BB133111EBULL;
  key ^= key >> 31;
  return static_cast<double>(key >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

double max_abs_interior(const std::vector<double>& f, const Subdomain& sub) {
  double m = 0;
  for_interior(sub, [&](int i, int j, int k) { m = worst(m, f[field_index(sub, i, j, k)]); });
  return m;
}

double max_difference(const std::vector<double>& a, const std::vector<double>& b, const Subdomain& sub) {
  double m = 0;
  for_interior(sub, [&](int i, int j, int k) {
    const std::size_t q = field_index(sub, i, j, k);
    m = worst(m, a[q] - b[q]);
  });
  return m;
}

double max_residual(const std::vector<double>& phi, const std::vector<double>& rhs, const Grid& g,
                    const Subdomain& sub) {
  double m = 0;
  for_interior(sub, [&](int i, int j, int k) {
    m = worst(m, laplacian(phi, g, sub, i, j, k) - rhs[field_index(sub, i, j, k)]);
  });
  return m;
}

}  // namespace

// 실수 입력 f = (z+1)·(cos θ + sin θ), θ = 2π(b1·x/nx + b2·y/ny).
// 복소 스펙트럼: F(b) = (z+1)·nx·ny·(1 − i)/2, F(−b) = (z+1)·nx·ny·(1 + i)/2, 나머지 0 (b = −b 이면 합 = (z+1)·nx·ny).
// sin 항이 켤레 방향(부호 오류)을 잡고, 저장은 x 모드 절반이지만 위 식을 그대로 대조하면 된다.
FftCheck check_cyclic_fft(CyclicFft& fft, MPI_Comm comm) {
  const CyclicDims& d = fft.layout().dims();
  const double volume = static_cast<double>(d.n_x) * d.n_y;
  const int modes[][2] = {{0, 0},
                          {1, 0},
                          {0, 1},
                          {d.n_x - 1, d.n_y - 1},
                          {(d.m + 1) % d.n_x, (d.c + 2) % d.n_y},
                          {d.n_x / 2, (3 * d.n_y) / 4},
                          {(d.q + 3) % d.n_x, (d.m - 1 + d.n_y) % d.n_y}};
  FftCheck out;
  std::vector<double> original(fft.lines_buffer().size(), 0), host;
  double delta_error = 0, roundtrip_error = 0;
  const long long n_lines = d.line_elements();
  for (const auto& b : modes) {
    for (long long i = 0; i < n_lines; ++i) {
      const int x = static_cast<int>(i % d.n_x);
      const long long line = i / d.n_x;
      const int z = static_cast<int>(line % d.nz);
      const int k = static_cast<int>(line / d.nz);
      const int y = d.s + k * d.q;
      const double fraction = static_cast<double>((static_cast<long long>(b[0]) * x) % d.n_x) / d.n_x
                              + static_cast<double>((static_cast<long long>(b[1]) * y) % d.n_y) / d.n_y;
      const double theta = kTwoPi * fraction;
      original[i] = (z + 1.0) * (std::cos(theta) + std::sin(theta));
    }
    fft.lines_buffer().upload(original);
    fft.forward(nullptr);
    fft.spectrum_buffer().download(host);
    const int minus_x = (d.n_x - b[0]) % d.n_x, minus_y = (d.n_y - b[1]) % d.n_y;
    for (long long i = 0; i < d.spectrum_elements(); ++i) {
      const int mode = static_cast<int>(i % d.modes_local());
      const int z = static_cast<int>(i / d.modes_local());
      int mode_x = 0, mode_y = 0;
      d.global_mode(mode, mode_x, mode_y);
      double re = 0, im = 0;
      if (mode_x == b[0] && mode_y == b[1]) { re += 0.5; im -= 0.5; }
      if (mode_x == minus_x && mode_y == minus_y) { re += 0.5; im += 0.5; }
      const double err = std::hypot(host[2 * i] / ((z + 1.0) * volume) - re, host[2 * i + 1] / ((z + 1.0) * volume) - im);
      delta_error = worst(delta_error, err);
    }
    fft.backward(nullptr);
    fft.lines_buffer().download(host);
    for (long long i = 0; i < n_lines; ++i) {
      const int z = static_cast<int>((i / d.n_x) % d.nz);
      roundtrip_error = worst(roundtrip_error, (host[i] - volume * original[i]) / ((z + 1.0) * volume));
    }
    ++out.modes_tested;
  }
  out.delta_error = global_max(delta_error, comm);
  out.roundtrip_error = global_max(roundtrip_error, comm);
  return out;
}

void fill_random_rhs(Field& rhs, const Grid& g, const Subdomain& sub) {
  const int h = mpmstd::halo_width;
  std::vector<double> r(rhs.size(), 0);
  double weighted = 0, weight = 0;
  for_interior(sub, [&](int i, int j, int k) {
    const std::uint64_t gi = static_cast<std::uint64_t>(sub.x().global(i));
    const std::uint64_t gj = static_cast<std::uint64_t>(sub.y().global(j));
    const std::uint64_t gk = static_cast<std::uint64_t>(sub.z().global(k));
    const std::uint64_t key = (gi * static_cast<std::uint64_t>(sub.y().n_global) + gj)
                                  * static_cast<std::uint64_t>(sub.z().n_global) + gk;
    const double v = hashed_uniform(key);
    r[field_index(sub, i, j, k)] = v;
    weighted += v * g.z.cell_width[k];
    weight += g.z.cell_width[k];
  });
  (void)h;
  const MPI_Comm comm = sub.topology().cart_comm();
  const double mean = global_sum(weighted, comm) / global_sum(weight, comm);
  for_interior(sub, [&](int i, int j, int k) { r[field_index(sub, i, j, k)] -= mean; });
  rhs.upload(r);
}

PoissonCheck check_poisson(CyclicPoissonSolver& new_solver, mpmstd::equation::pressure::Solver& old_solver,
                           const Grid& g, const Subdomain& sub, const double lx, const double ly,
                           const double lz, Field& rhs, Field& phi_new, Field& phi_old) {
  const MPI_Comm comm = sub.topology().cart_comm();
  PoissonCheck out;
  std::vector<double> exact(rhs.size(), 0), r(rhs.size(), 0), p_new, p_old;

  // ① 제조해 — 이산 라플라시안 rhs → 두 솔버 모두 φ 를 반올림 수준으로 돌려줘야 한다
  for_interior(sub, [&](int i, int j, int k) {
    exact[field_index(sub, i, j, k)] = std::cos(kTwoPi * g.x.center[i] / lx) * std::cos(kTwoPi * g.y.center[j] / ly)
                                       * std::cos(0.5 * kTwoPi * g.z.center[k] / lz);
  });
  phi_new.upload(exact);   // ghost 채움에 장 하나를 빌린다
  fill_ghosts(phi_new, sub);
  phi_new.download(exact);
  for_interior(sub, [&](int i, int j, int k) { r[field_index(sub, i, j, k)] = laplacian(exact, g, sub, i, j, k); });
  // 정확해의 (z 산술평균 게이지) 평균을 빼 둔다 — 솔버 게이지와 맞추기 (대칭 tanh 격자에서는 이미 0)
  double sum = 0;
  for_interior(sub, [&](int i, int j, int k) { sum += exact[field_index(sub, i, j, k)]; });
  const double cells = static_cast<double>(sub.x().n_global) * sub.y().n_global * sub.z().n_global;
  const double mean = global_sum(sum, comm) / cells;
  for_interior(sub, [&](int i, int j, int k) { exact[field_index(sub, i, j, k)] -= mean; });
  rhs.upload(r);
  new_solver.solve(rhs, phi_new);
  mpmstd::equation::pressure::solve(old_solver, g, sub, rhs, phi_old);
  phi_new.download(p_new);
  phi_old.download(p_old);
  const double scale_exact = global_max(max_abs_interior(exact, sub), comm);
  out.manufactured_new = global_max(max_difference(p_new, exact, sub), comm) / scale_exact;
  out.manufactured_old = global_max(max_difference(p_old, exact, sub), comm) / scale_exact;

  // ② 난수 rhs — 잔차와 두 해의 차
  fill_random_rhs(rhs, g, sub);
  rhs.download(r);
  new_solver.solve(rhs, phi_new);
  mpmstd::equation::pressure::solve(old_solver, g, sub, rhs, phi_old);
  phi_new.download(p_new);
  phi_old.download(p_old);
  const double scale_rhs = global_max(max_abs_interior(r, sub), comm);
  const double scale_old = global_max(max_abs_interior(p_old, sub), comm);
  out.residual_new = global_max(max_residual(p_new, r, g, sub), comm) / scale_rhs;
  out.residual_old = global_max(max_residual(p_old, r, g, sub), comm) / scale_rhs;
  out.difference = global_max(max_difference(p_new, p_old, sub), comm) / scale_old;
  return out;
}

namespace {

double contract_solution(const Grid& g, const int i, const int j, const int k) {
  const double pi = 0.5 * kTwoPi;
  return std::cos(pi * g.x.center[i]) * std::cos(pi * g.y.center[j]) * std::cos(pi * g.z.center[k]);
}

}  // namespace

void fill_contract_rhs(Field& rhs, const Grid& g, const Subdomain& sub) {
  const double pi = 0.5 * kTwoPi;
  std::vector<double> r(rhs.size(), 0);
  for_interior(sub, [&](int i, int j, int k) { r[field_index(sub, i, j, k)] = -3 * pi * pi * contract_solution(g, i, j, k); });
  rhs.upload(r);
}

ContractError contract_error(const Field& phi, const Grid& g, const Subdomain& sub) {
  const MPI_Comm comm = sub.topology().cart_comm();
  std::vector<double> p;
  phi.download(p);
  double squared = 0, linf = 0;
  for_interior(sub, [&](int i, int j, int k) {
    const double e = p[field_index(sub, i, j, k)] - contract_solution(g, i, j, k);
    squared += e * e;
    linf = worst(linf, e);
  });
  const double cells = static_cast<double>(sub.x().n_global) * sub.y().n_global * sub.z().n_global;
  ContractError out;
  out.rms = std::sqrt(global_sum(squared, comm) / cells);
  out.linf = global_max(linf, comm);
  return out;
}

double contract_expected_rms(const Grid& g) {
  const double pi = 0.5 * kTwoPi;
  const int h = mpmstd::halo_width;
  const auto lambda = [&](const double spacing) { return 2 / (spacing * spacing) * (1 - std::cos(pi * spacing)); };
  const double c = 3 * pi * pi / (lambda(g.x.cell_width[h]) + lambda(g.y.cell_width[h]) + lambda(g.z.cell_width[h]));
  return std::abs(c - 1) / (2 * std::sqrt(2.0));
}

}  // namespace fft_test
