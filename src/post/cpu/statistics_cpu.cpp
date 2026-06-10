#include "post/statistics.hpp"
#include "common/macros.hpp"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdio>

// P1 channel z-profile statistics rewired to Domain + Fields. GLOBAL nx*ny
// xy-plane normalization (16x-bug guard).

namespace mpmstd::post {

void init_statistics_cpu(Stats& s, const core::Domain& domain) {
  s.nz_global = domain.sub.n_global()[2];
  s.nz_local  = domain.sub.n_interior()[2];
  s.kstart    = domain.sub.global_offset()[2];
  s.n         = 0;
  s.U_m.assign(s.nz_local, 0.0);  s.U2_m.assign(s.nz_local, 0.0);
  s.V_m.assign(s.nz_local, 0.0);  s.V2_m.assign(s.nz_local, 0.0);
  s.Wc_m.assign(s.nz_local, 0.0); s.Wc2_m.assign(s.nz_local, 0.0);
  s.UWc_m.assign(s.nz_local, 0.0); s.P_m.assign(s.nz_local, 0.0);

  const auto& zc = domain.grid.xc(Direction::Z);
  std::vector<double> tmp(s.nz_global, 0.0);
  for (int kl = 0; kl < s.nz_local; ++kl)
    tmp[s.kstart + kl] = static_cast<double>(zc[kHaloWidth + kl]);
  s.zc_global.assign(s.nz_global, 0.0);
  MPI_Allreduce(tmp.data(), s.zc_global.data(), s.nz_global, MPI_DOUBLE, MPI_MAX, domain.sub.topology().cart_comm());
}

void accumulate_statistics_cpu(Stats& s, const core::Domain& domain, const core::CpuFields& fields) {
  ++s.n;
  const double inv_n = 1.0 / static_cast<double>(s.n);
  const int h  = kHaloWidth;
  const auto nt = domain.sub.n_total(); const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int nx = domain.sub.n_global()[0], ny = domain.sub.n_global()[1];   // GLOBAL (16x-bug guard)
  const double inv_NxNy = 1.0 / static_cast<double>(nx * ny);

  const real_t* u = fields[core::Var::U].data(); const real_t* v = fields[core::Var::V].data();
  const real_t* w = fields[core::Var::W].data(); const real_t* p = fields[core::Var::P].data();

  for (int kl = 0; kl < s.nz_local; ++kl) {
    const int k = kl + h;
    double su = 0, su2 = 0, sv = 0, sv2 = 0, swc = 0, swc2 = 0, suwc = 0, sp = 0;
    for (int i = h; i < n1 - h; ++i)
      for (int j = h; j < n2 - h; ++j) {
        const double uc  = 0.5 * (double(u[(i * n2 + j) * n3 + k]) + double(u[((i + 1) * n2 + j) * n3 + k]));
        const double vc  = 0.5 * (double(v[(i * n2 + j) * n3 + k]) + double(v[(i * n2 + j + 1) * n3 + k]));
        const double wci = 0.5 * (double(w[(i * n2 + j) * n3 + k]) + double(w[(i * n2 + j) * n3 + k + 1]));
        const double pi  = double(p[(i * n2 + j) * n3 + k]);
        su += uc; su2 += uc * uc; sv += vc; sv2 += vc * vc;
        swc += wci; swc2 += wci * wci; suwc += uc * wci; sp += pi;
      }
    s.U_m[kl]   += (su   * inv_NxNy - s.U_m[kl])   * inv_n;
    s.U2_m[kl]  += (su2  * inv_NxNy - s.U2_m[kl])  * inv_n;
    s.V_m[kl]   += (sv   * inv_NxNy - s.V_m[kl])   * inv_n;
    s.V2_m[kl]  += (sv2  * inv_NxNy - s.V2_m[kl])  * inv_n;
    s.Wc_m[kl]  += (swc  * inv_NxNy - s.Wc_m[kl])  * inv_n;
    s.Wc2_m[kl] += (swc2 * inv_NxNy - s.Wc2_m[kl]) * inv_n;
    s.UWc_m[kl] += (suwc * inv_NxNy - s.UWc_m[kl]) * inv_n;
    s.P_m[kl]   += (sp   * inv_NxNy - s.P_m[kl])   * inv_n;
  }
}

void write_statistics_cpu(const Stats& s, const std::string& path, int step, double nu, const core::Domain& domain) {
  constexpr int NF = 8;
  std::vector<double> send(static_cast<std::size_t>(NF) * s.nz_global, 0.0);
  std::vector<double> recv(static_cast<std::size_t>(NF) * s.nz_global, 0.0);
  auto row = [&](int fi) { return send.data() + static_cast<std::size_t>(fi) * s.nz_global; };
  const std::vector<const std::vector<double>*> locals =
      {&s.U_m, &s.U2_m, &s.V_m, &s.V2_m, &s.Wc_m, &s.Wc2_m, &s.UWc_m, &s.P_m};
  for (int fi = 0; fi < NF; ++fi)
    for (int kl = 0; kl < s.nz_local; ++kl)
      row(fi)[s.kstart + kl] = (*locals[fi])[kl];

  MPI_Allreduce(send.data(), recv.data(), static_cast<int>(send.size()),
                MPI_DOUBLE, MPI_SUM, domain.sub.topology().cart_comm());

  int rank = 0;
  MPI_Comm_rank(domain.sub.topology().cart_comm(), &rank);
  if (rank != 0 || s.n == 0) return;

  auto col = [&](int fi, int k) { return recv[static_cast<std::size_t>(fi) * s.nz_global + k]; };
  const double tau_w = nu * std::fabs(col(0, 0)) / s.zc_global[0];
  const double u_tau = std::sqrt(std::max(tau_w, 0.0));
  const double inv_nu = 1.0 / nu;

  FILE* fp = std::fopen(path.c_str(), "w");
  if (!fp) { std::fprintf(stderr, "[stats] cannot open '%s'\n", path.c_str()); return; }
  std::fprintf(fp,
      "TITLE = \"Channel Statistics (step=%d, n=%ld)\"\n"
      "VARIABLES = \"Z\" \"Z_plus\" \"U_mean\" \"W_mean\" \"u_rms\" \"v_rms\" \"w_rms\" \"uw_stress\" \"P_mean\"\n"
      "ZONE T=\"Stats\", I=%d, J=1, K=1, DATAPACKING=POINT\n",
      step, s.n, s.nz_global);
  for (int k = 0; k < s.nz_global; ++k) {
    const double zp    = s.zc_global[k] * u_tau * inv_nu;
    const double u_rms = std::sqrt(std::max(col(1, k) - col(0, k) * col(0, k), 0.0));
    const double v_rms = std::sqrt(std::max(col(3, k) - col(2, k) * col(2, k), 0.0));
    const double w_rms = std::sqrt(std::max(col(5, k) - col(4, k) * col(4, k), 0.0));
    const double uw    = col(6, k) - col(0, k) * col(4, k);
    std::fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                 s.zc_global[k], zp, col(0, k), col(4, k), u_rms, v_rms, w_rms, uw, col(7, k));
  }
  std::fclose(fp);
}

double compute_re_delta_star_cpu(const Stats& s, double nu, const core::Domain& domain) {
  const int nz = s.nz_global;
  if (nz <= 0 || s.n == 0) return 0.0;

  // Gather the global mean streamwise profile U(z) = <u>_{x,y,t}.
  std::vector<double> send(nz, 0.0), U(nz, 0.0);
  for (int kl = 0; kl < s.nz_local; ++kl) send[s.kstart + kl] = s.U_m[kl];
  MPI_Allreduce(send.data(), U.data(), nz, MPI_DOUBLE, MPI_SUM, domain.sub.topology().cart_comm());

  // U_max = positive jet peak (near the hot wall) + its wall-normal index.
  int kmax = 0; double u_max = U[0];
  for (int k = 0; k < nz; ++k) if (U[k] > u_max) { u_max = U[k]; kmax = k; }
  if (u_max <= 0.0) return 0.0;

  // δ* = ∫₀^{z[kmax]} (1 − U/U_max) dz  (trapezoid incl. the no-slip wall z=0,U=0).
  double dstar = 0.0, z_prev = 0.0, f_prev = 1.0;   // wall: 1 − 0/U_max = 1
  for (int k = 0; k <= kmax; ++k) {
    const double z = s.zc_global[k];
    const double f = 1.0 - U[k] / u_max;
    dstar += 0.5 * (f_prev + f) * (z - z_prev);
    z_prev = z; f_prev = f;
  }
  return u_max * dstar / nu;
}

} // namespace mpmstd::post
