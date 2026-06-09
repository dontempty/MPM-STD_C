#pragma once

#include "core/domain.hpp"
#include "core/variables.hpp"

#include <string>
#include <vector>

namespace mpmstd::post {

// Time-averaged z-profile channel statistics (Welford). ⚠ xy-plane average uses
// GLOBAL nx*ny normalization (16x-bug guard). Reads velocities/P from Fields.
struct Stats {
  int  nz_global = 0, nz_local = 0, kstart = 0;
  long n = 0;
  std::vector<double> U_m, U2_m, V_m, V2_m, Wc_m, Wc2_m, UWc_m, P_m;
  std::vector<double> zc_global;
};

void init_statistics_cpu(Stats& s, const core::Domain& domain);
void accumulate_statistics_cpu(Stats& s, const core::Domain& domain, const core::CpuFields& fields);
void write_statistics_cpu(const Stats& s, const std::string& path, int step, double nu, const core::Domain& domain);

void accumulate_statistics_gpu(Stats& s, const core::Domain& domain, const core::GpuFields& fields);

// DHVC shear Reynolds number Re_δ* = U_max·δ*/ν (PaScaL_TCS Fig 7). Uses the
// mean streamwise profile U_m(z) (= <u>_{x,y,t}); δ* = ∫₀^δmax (1−U/U_max) dz
// with δmax the wall-normal distance to U_max. Gathers the global profile via
// MPI; returns the value on every rank. (streamwise = x, wall-normal = z.)
double compute_re_delta_star_cpu(const Stats& s, double nu, const core::Domain& domain);

} // namespace mpmstd::post
