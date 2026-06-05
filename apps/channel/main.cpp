// =============================================================================
//  apps/channel/main.cpp  —  Turbulent channel flow (Poiseuille / DNS)
//
//  Based on Filtered_TDMA/channel pattern.  Grid/mesh uses MPM-STD_C conventions.
//
//  Key config sections:
//    [physics]          nu = 1/Re_tau
//    [channel_forcing]  mode = "mass_flow" | "pressure_gradient"
//                       target_bulk_velocity = 1.0   (mass_flow)
//                       target_dPdx          = 1.0   (pressure_gradient, body-force magnitude)
//    [thermal]          enabled = 0
//    [bc.z.minus/plus]  ghost = "zero"  (prevents extra nu/dz^2 damping)
//    [init]             u_ic_poiseuille = 1
//                       perturbation = 0.05
//    [output]           nmonitor, nstat_start, nstat, nout_stats, nout
//                       out_stats, out_field, continue_in, continue_out
//                       dir_restart, dir_stats, dir_instant
//
//  Run:  mpirun -np N ./build/cpu/bin/channel apps/channel/input.toml
// =============================================================================

#include "channel_forcing.hpp"
#include "channel_stats.hpp"

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "config/main.hpp"
#include "equation/main.hpp"
#include "equation/momentum/kernels/kernels.hpp"
#include "equation/pressure/pressure_solver_factory.hpp"
#include "field/main.hpp"
#include "grid/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "parallel/main.hpp"
#include "post/main.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>

using namespace mpmstd;

// ── helpers ──────────────────────────────────────────────────────────────────

static double max_divergence(const field::ScalarField& U,
                              const field::ScalarField& V,
                              const field::ScalarField& W,
                              const grid::Grid& g,
                              const parallel::mpi::Subdomain& sub,
                              MPI_Comm cart) {
    const int h  = kHaloWidth;
    const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
    const real_t* u = U.host_ptr(), *v = V.host_ptr(), *w = W.host_ptr();
    const real_t* dx1 = g.dx_ptr(Direction::X);
    const real_t* dx2 = g.dx_ptr(Direction::Y);
    const real_t* dx3 = g.dx_ptr(Direction::Z);
    double local = 0.0;
    for (int i = h; i < n1-h; ++i)
        for (int j = h; j < n2-h; ++j)
            for (int k = h; k < n3-h; ++k) {
                const double d =
                    std::fabs((u[((i+1)*n2+j)*n3+k] - u[(i*n2+j)*n3+k])
                              / static_cast<double>(dx1[i])
                            + (v[(i*n2+(j+1))*n3+k] - v[(i*n2+j)*n3+k])
                              / static_cast<double>(dx2[j])
                            + (w[(i*n2+j)*n3+k+1]   - w[(i*n2+j)*n3+k])
                              / static_cast<double>(dx3[k]));
                if (d > local) local = d;
            }
    double global = local;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, cart);
    return global;
}

// MPI MAXLOC helper: returns {global_max, gi, gj, gk} (0-based global interior indices).
struct LocMax { double val; int gi, gj, gk; };

static LocMax allreduce_max_located(
    double local_val, int li, int lj, int lk,
    const parallel::mpi::Subdomain& sub, MPI_Comm cart)
{
    int my_rank;
    MPI_Comm_rank(cart, &my_rank);
    struct { double v; int r; } send = {local_val, my_rank}, recv = {0.0, 0};
    MPI_Allreduce(&send, &recv, 1, MPI_DOUBLE_INT, MPI_MAXLOC, cart);
    const int h = kHaloWidth;
    int gbuf[3] = {
        sub.global_offset(Direction::X) + (li - h),
        sub.global_offset(Direction::Y) + (lj - h),
        sub.global_offset(Direction::Z) + (lk - h)
    };
    MPI_Bcast(gbuf, 3, MPI_INT, recv.r, cart);
    return {recv.v, gbuf[0], gbuf[1], gbuf[2]};
}

// When div > threshold: print max-div location, max |U|/|V|/|W| locations, max W/dz.
static void divergence_abort_dump(
    const field::ScalarField& U, const field::ScalarField& V,
    const field::ScalarField& W,
    const grid::Grid& g, const parallel::mpi::Subdomain& sub,
    MPI_Comm cart, int step, double time, double dt)
{
    const int h  = kHaloWidth;
    const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
    const real_t* u  = U.host_ptr(), *v = V.host_ptr(), *w = W.host_ptr();
    const real_t* dx1 = g.dx_ptr(Direction::X);
    const real_t* dx2 = g.dx_ptr(Direction::Y);
    const real_t* dx3 = g.dx_ptr(Direction::Z);

    double div_loc=0, u_loc=0, v_loc=0, w_loc=0, cfw_loc=0;
    int di=h, dj=h, dk=h;
    int ui=h, uj=h, uk=h;
    int vi=h, vj=h, vk=h;
    int wi=h, wj=h, wk=h;
    int cfi=h, cfj=h, cfk=h;

    for (int i = h; i < n1-h; ++i)
        for (int j = h; j < n2-h; ++j)
            for (int k = h; k < n3-h; ++k) {
                const double d = std::fabs(
                    (u[((i+1)*n2+j)*n3+k] - u[(i*n2+j)*n3+k]) / static_cast<double>(dx1[i])
                   +(v[(i*n2+(j+1))*n3+k] - v[(i*n2+j)*n3+k]) / static_cast<double>(dx2[j])
                   +(w[(i*n2+j)*n3+k+1]   - w[(i*n2+j)*n3+k]) / static_cast<double>(dx3[k]));
                if (d > div_loc) { div_loc=d; di=i; dj=j; dk=k; }

                const double au = std::fabs(static_cast<double>(u[(i*n2+j)*n3+k]));
                const double av = std::fabs(static_cast<double>(v[(i*n2+j)*n3+k]));
                const double aw = std::fabs(static_cast<double>(w[(i*n2+j)*n3+k]));
                if (au > u_loc) { u_loc=au; ui=i; uj=j; uk=k; }
                if (av > v_loc) { v_loc=av; vi=i; vj=j; vk=k; }
                if (aw > w_loc) { w_loc=aw; wi=i; wj=j; wk=k; }

                const double cfw = aw / static_cast<double>(dx3[k]);
                if (cfw > cfw_loc) { cfw_loc=cfw; cfi=i; cfj=j; cfk=k; }
            }

    const auto div_g = allreduce_max_located(div_loc, di, dj, dk, sub, cart);
    const auto u_g   = allreduce_max_located(u_loc,   ui, uj, uk, sub, cart);
    const auto v_g   = allreduce_max_located(v_loc,   vi, vj, vk, sub, cart);
    const auto w_g   = allreduce_max_located(w_loc,   wi, wj, wk, sub, cart);
    const auto cfw_g = allreduce_max_located(cfw_loc, cfi,cfj,cfk,sub, cart);

    int my_rank; MPI_Comm_rank(cart, &my_rank);
    if (my_rank == 0) {
        std::printf("\n=== ABORT: maxDivU=%.3e > 1e-10  step=%d  t=%.5f  dt=%.3e ===\n",
                    div_g.val, step, time, dt);
        std::printf("  div_max     : %.3e  at global(i,j,k)=(%d,%d,%d)\n",
                    div_g.val, div_g.gi, div_g.gj, div_g.gk);
        std::printf("  |U|_max     : %.3e  at global(i,j,k)=(%d,%d,%d)\n",
                    u_g.val,   u_g.gi,   u_g.gj,   u_g.gk);
        std::printf("  |V|_max     : %.3e  at global(i,j,k)=(%d,%d,%d)\n",
                    v_g.val,   v_g.gi,   v_g.gj,   v_g.gk);
        std::printf("  |W|_max     : %.3e  at global(i,j,k)=(%d,%d,%d)\n",
                    w_g.val,   w_g.gi,   w_g.gj,   w_g.gk);
        std::printf("  |W|/dz_max  : %.3e  at global(i,j,k)=(%d,%d,%d)  [wall-normal CFL contrib]\n",
                    cfw_g.val, cfw_g.gi, cfw_g.gj, cfw_g.gk);
        std::fflush(stdout);
    }
}

static double max_cfl_impl(const field::ScalarField& U,
                            const field::ScalarField& V,
                            const field::ScalarField& W,
                            const grid::Grid& g,
                            const parallel::mpi::Subdomain& sub,
                            MPI_Comm cart,
                            real_t dt) {
    const int h  = kHaloWidth;
    const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
    const real_t* u = U.host_ptr(), *v = V.host_ptr(), *w = W.host_ptr();
    const real_t* dx1 = g.dx_ptr(Direction::X);
    const real_t* dx2 = g.dx_ptr(Direction::Y);
    const real_t* dx3 = g.dx_ptr(Direction::Z);
    double local = 1e-30;
    for (int i = h; i < n1-h; ++i)
        for (int j = h; j < n2-h; ++j)
            for (int k = h; k < n3-h; ++k) {
                const double c =
                    (std::fabs(static_cast<double>(u[(i*n2+j)*n3+k])) / static_cast<double>(dx1[i])
                   + std::fabs(static_cast<double>(v[(i*n2+j)*n3+k])) / static_cast<double>(dx2[j])
                   + std::fabs(static_cast<double>(w[(i*n2+j)*n3+k])) / static_cast<double>(dx3[k]));
                if (c > local) local = c;
            }
    double global = local;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, cart);
    return global * static_cast<double>(dt);
}

// Wall shear stress: ν * |<U_cc(z=zc[0])>| / (0.5*dz[0])
// U is x-face-staggered; cell-centre = 0.5*(U[i]+U[i+1])
static double wss_diagnostic(const field::ScalarField& U,
                              const grid::Grid& g,
                              const parallel::mpi::Subdomain& sub,
                              double nu, MPI_Comm cart) {
    const int h  = kHaloWidth;
    const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
    const real_t* u  = U.host_ptr();
    const real_t* dz = g.dx_ptr(Direction::Z);
    // Use GLOBAL interior cell counts — global is MPI_Allreduce(SUM) over all ranks,
    // so the denominator must be the total number of xy-wall-face cells globally.
    const int nx = sub.n_global()[0], ny = sub.n_global()[1];
    const bool lo = (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
    const bool hi = (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);
    double loc = 0.0;
    if (lo) {
        const int k = h;
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j) {
                const double uc = 0.5*(static_cast<double>(u[(i*n2+j)*n3+k])
                                     +static_cast<double>(u[((i+1)*n2+j)*n3+k]));
                loc += uc / (static_cast<double>(dz[k]) * 0.5);
            }
    }
    if (hi) {
        const int k = n3-h-1;
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j) {
                const double uc = 0.5*(static_cast<double>(u[(i*n2+j)*n3+k])
                                     +static_cast<double>(u[((i+1)*n2+j)*n3+k]));
                loc += uc / (static_cast<double>(dz[k]) * 0.5);
            }
    }
    double global = 0.0;
    MPI_Allreduce(&loc, &global, 1, MPI_DOUBLE, MPI_SUM, cart);
    return std::fabs(nu * global / (2.0 * nx * ny));
}

// Gershgorin spectral radius ρ for z-direction BW-TDMA (pure diffusion):
//   am = ν_h·dt / (dz[k]·dmz[k])   (backward)
//   ap = ν_h·dt / (dz[k]·dmz[k+1]) (forward)
//   ρ_k = max(am,ap)/(am+ap+1)
// Skip skip=2 wall-adjacent rows (same as Filtered_TDMA).
static std::pair<double,double> rho_diagnostic(
    const grid::Grid& g,
    const parallel::mpi::Subdomain& sub,
    double nu, double dt, MPI_Comm cart)
{
    const int h    = kHaloWidth;
    const int nz   = sub.n_interior(Direction::Z);
    const int n3m  = sub.n_global()[2];
    const int gz0  = sub.global_offset(Direction::Z);
    const int skip = 2;
    const double nu_h = 0.5 * nu;

    const real_t* dz  = g.dx_ptr(Direction::Z);
    const real_t* dmz = g.dmx_ptr(Direction::Z);

    double rmax = 0.0, rmin = 1.0;
    for (int k = h; k < h + nz; ++k) {
        const int gk = gz0 + (k - h) + 1;
        if (gk < skip + 1 || gk > n3m - skip) continue;
        const double am = nu_h * dt / (static_cast<double>(dz[k]) * static_cast<double>(dmz[k]));
        const double ap = nu_h * dt / (static_cast<double>(dz[k]) * static_cast<double>(dmz[k+1]));
        const double rk = std::max(am, ap) / (am + ap + 1.0);
        if (rk > rmax) rmax = rk;
        if (rk < rmin) rmin = rk;
    }
    double rho_max = 0.0, rho_min = 1.0;
    MPI_Allreduce(&rmax, &rho_max, 1, MPI_DOUBLE, MPI_MAX, cart);
    MPI_Allreduce(&rmin, &rho_min, 1, MPI_DOUBLE, MPI_MIN, cart);
    return { rho_max, rho_min };
}

// 3-component Poiseuille + perturbation IC (Filtered_TDMA laminar_init pattern):
//   1. Fill U,V,W with uniform[-0.5, 0.5] noise
//   2. Subtract volume-weighted mean from each
//   3. Add Poiseuille parabola to U; scale all by pert*Umax
//   4. Re-normalize bulk to (Ub_target, 0, 0)
static void laminar_init(field::ScalarField& U,
                          field::ScalarField& V,
                          field::ScalarField& W,
                          const parallel::mpi::Subdomain& sub,
                          const grid::Grid& g,
                          double Lz, double Ly, double Ub_target, double pert,
                          double roll_amp, int n_roll,
                          MPI_Comm cart, int world_rank) {
    const int h  = kHaloWidth;
    const int n1 = sub.n_total()[0], n2 = sub.n_total()[1], n3 = sub.n_total()[2];
    real_t* u = U.host_ptr(), *v = V.host_ptr(), *w = W.host_ptr();
    const real_t* dz  = g.dx_ptr(Direction::Z);
    const real_t* dx1 = g.dx_ptr(Direction::X);
    const real_t* dx2 = g.dx_ptr(Direction::Y);
    const auto& zc = g.xc(Direction::Z);
    const auto& zf = g.xf(Direction::Z);   // W lives at z-faces
    const auto& yc = g.xc(Direction::Y);   // U,W cell-centred in y
    const auto& yf = g.xf(Direction::Y);   // V lives at y-faces

    std::random_device rd;
    const std::uint64_t seed = static_cast<std::uint64_t>(rd())
                             ^ (static_cast<std::uint64_t>(world_rank) * 0x9E3779B97F4A7C15ULL);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> rnd(-0.5, 0.5);

    // Step 1: random noise
    for (int i = h; i < n1-h; ++i)
        for (int j = h; j < n2-h; ++j)
            for (int k = h; k < n3-h; ++k) {
                u[(i*n2+j)*n3+k] = static_cast<real_t>(rnd(rng));
                v[(i*n2+j)*n3+k] = static_cast<real_t>(rnd(rng));
                w[(i*n2+j)*n3+k] = static_cast<real_t>(rnd(rng));
            }

    // Step 2: subtract volume-mean
    auto bulk_mean_f = [&](const real_t* f) {
        double s = 0.0, vol = 0.0;
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j)
                for (int k = h; k < n3-h; ++k) {
                    double dV = static_cast<double>(dx1[i])
                              * static_cast<double>(dx2[j])
                              * static_cast<double>(dz[k]);
                    s   += static_cast<double>(f[(i*n2+j)*n3+k]) * dV;
                    vol += dV;
                }
        double ps[2] = {s, vol}, gs[2] = {0,0};
        MPI_Allreduce(ps, gs, 2, MPI_DOUBLE, MPI_SUM, cart);
        return (gs[1] > 0) ? gs[0]/gs[1] : 0.0;
    };
    const double Um0 = bulk_mean_f(u), Vm0 = bulk_mean_f(v), Wm0 = bulk_mean_f(w);
    for (int i = h; i < n1-h; ++i)
        for (int j = h; j < n2-h; ++j)
            for (int k = h; k < n3-h; ++k) {
                u[(i*n2+j)*n3+k] -= static_cast<real_t>(Um0);
                v[(i*n2+j)*n3+k] -= static_cast<real_t>(Vm0);
                w[(i*n2+j)*n3+k] -= static_cast<real_t>(Wm0);
            }

    // Step 3: add Poiseuille + perturbation.
    //
    // Two modes:
    //   roll_amp == 0 : pure white-noise perturbation scaled by pert*Umax
    //                   (the MPM-STD / Filtered_TDMA IC; relaminarizes at Re_b=2857
    //                    because white noise is grid-scale and viscously dissipates).
    //   roll_amp >  0 : STRUCTURED bypass-transition IC —
    //                   streamwise counter-rotating vortex rolls (x-invariant,
    //                   spanwise wavenumber beta) + a streak seed, plus a small
    //                   white-noise symmetry-breaker (noise_amp) to trigger the
    //                   secondary (streak) instability.  The rolls survive viscous
    //                   decay (low wavenumber) and pump the mean shear into streaks
    //                   via lift-up → bypass transition.
    //
    // Rolls (divergence-free in the y-z plane, no-slip at z-walls):
    //   W = A * sin^2(pi*z/Lz) * cos(beta*y)
    //   V = -(A/beta) * (pi/Lz) * sin(2*pi*z/Lz) * sin(beta*y)
    //   (g=sin^2 has g=g'=0 at both walls → V=W=0 there)
    // Streak seed (lift-up-consistent sign): U += -A * sin(pi*z/Lz) * cos(beta*y)
    const double Umax = 1.5 * Ub_target;
    const double half = 0.5 * Lz;
    const bool   structured = (roll_amp > 0.0);
    const double noise_amp  = structured ? 0.05 : pert;   // symmetry-breaker vs full noise
    const double beta       = 2.0 * M_PI * static_cast<double>(n_roll) / Ly;
    const double A_roll     = roll_amp * Ub_target;
    for (int k = h; k < n3-h; ++k) {
        const double z   = static_cast<double>(zc[k]);
        const double zr  = (z - half) / half;
        const double Up  = Umax * std::max(0.0, 1.0 - zr*zr);
        // roll z-shapes at the staggered z-locations
        const double sf      = std::sin(M_PI * static_cast<double>(zf[k]) / Lz);
        const double W_shape = sf * sf;                                   // sin^2(pi*zf/Lz)  (W at z-face)
        const double V_shape = (M_PI / Lz) * std::sin(2.0*M_PI*z / Lz);   // g'(zc)           (V at z-centre)
        const double U_shape = std::sin(M_PI * z / Lz);                   // streak shape     (U at z-centre)
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j) {
                const int p = (i*n2+j)*n3+k;
                double uu = Up + noise_amp * Umax * static_cast<double>(u[p]);
                double vv =      noise_amp * Umax * static_cast<double>(v[p]);
                double ww =      noise_amp * Umax * static_cast<double>(w[p]);
                if (structured) {
                    const double cy = std::cos(beta * static_cast<double>(yc[j]));  // W,U at yc
                    const double sy = std::sin(beta * static_cast<double>(yf[j]));  // V at yf
                    ww +=  A_roll * W_shape * cy;
                    vv += -(A_roll / beta) * V_shape * sy;
                    uu += -A_roll * U_shape * cy;        // streak (lift-up-consistent)
                }
                u[p] = static_cast<real_t>(uu);
                v[p] = static_cast<real_t>(vv);
                w[p] = static_cast<real_t>(ww);
            }
    }

    // Step 4: renormalize bulk (staggered U)
    // Fill the right x-halo face from the left interior (periodic) before computing
    // bulk so that the last interior cell uc = 0.5*(U[i]+U[i+1]) is correct.
    // Without this, u[n1-h] is still 0 (from fill_host) → bulk is underestimated
    // by O(1/n1_int), leading to a spurious dU ≈ 0.03 shift that pushes U[wall] up.
    for (int j = h; j < n2-h; ++j)
        for (int k = h; k < n3-h; ++k)
            u[(( n1-h)*n2+j)*n3+k] = u[((h)*n2+j)*n3+k];   // periodic: right halo = left interior

    auto bulk_u_stag = [&]() {
        double s = 0.0, vol = 0.0;
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j)
                for (int k = h; k < n3-h; ++k) {
                    double dV = static_cast<double>(dx1[i])
                              * static_cast<double>(dx2[j])
                              * static_cast<double>(dz[k]);
                    double uc = 0.5*(static_cast<double>(u[(i*n2+j)*n3+k])
                                   +static_cast<double>(u[((i+1)*n2+j)*n3+k]));
                    s += uc*dV; vol += dV;
                }
        double ps[2] = {s, vol}, gs[2] = {0,0};
        MPI_Allreduce(ps, gs, 2, MPI_DOUBLE, MPI_SUM, cart);
        return (gs[1] > 0) ? gs[0]/gs[1] : 0.0;
    };
    const double Um1 = bulk_u_stag(), Vm1 = bulk_mean_f(v), Wm1 = bulk_mean_f(w);
    const double dU = Ub_target - Um1;
    for (int i = h; i < n1-h; ++i)
        for (int j = h; j < n2-h; ++j)
            for (int k = h; k < n3-h; ++k) {
                u[(i*n2+j)*n3+k] += static_cast<real_t>(dU);
                v[(i*n2+j)*n3+k] -= static_cast<real_t>(Vm1);
                w[(i*n2+j)*n3+k] -= static_cast<real_t>(Wm1);
            }

    if (world_rank == 0)
        std::printf("[laminar_init] pert=%.3g  pre=(%.3e,%.3e,%.3e) -> (%.3e,0,0)\n",
                    pert, Um1, Vm1, Wm1, Ub_target);
}


// =============================================================================
int main(int argc, char** argv) {

  // ── (1) MPI startup ─────────────────────────────────────────────────────────
  parallel::mpi::MpiContext mpi(&argc, &argv);

  // ── (2) Logger + config ──────────────────────────────────────────────────────
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);
  const std::string input_path = (argc >= 2) ? argv[1] : "apps/channel/input.toml";
  auto cfg = config::Config::load(input_path);

  // ── (3) MPI topology ─────────────────────────────────────────────────────────
  std::array<int,3> dims = {
      cfg.get<int>("mpi","np1"), cfg.get<int>("mpi","np2"), cfg.get<int>("mpi","np3")};
  auto axis_periodic = [](const std::string& s){ return s == "periodic"; };
  std::array<bool,3> periodic = {
      axis_periodic(cfg.get_or<std::string>("topology","x","periodic")),
      axis_periodic(cfg.get_or<std::string>("topology","y","periodic")),
      axis_periodic(cfg.get_or<std::string>("topology","z","periodic"))};
  parallel::mpi::MpiTopology topo(mpi, dims, periodic);

  // ── (4) Subdomain ────────────────────────────────────────────────────────────
  std::array<int,3> n_global = {
      cfg.get<int>("mesh","n1m"), cfg.get<int>("mesh","n2m"), cfg.get<int>("mesh","n3m")};
  parallel::mpi::Subdomain sub(topo, n_global);

  // ── (5) Grid ─────────────────────────────────────────────────────────────────
  std::array<grid::AxisConfig,3> axes;
  for (int a = 0; a < 3; ++a) {
      const std::string gamma_key = "gamma"   + std::to_string(a+1);
      const std::string uni_key   = "uniform" + std::to_string(a+1);
      const std::string len_key   = std::string("L") + char('x'+a);
      const int    is_uni = cfg.get_or<int>   ("uniform_mesh",  uni_key,   1);
      const double gamma  = cfg.get_or<double>("mesh_stretch",  gamma_key, 0.0);
      const double L      = cfg.get<double>("domain", len_key);
      axes[a] = { n_global[a], L,
                  is_uni ? grid::StretchKind::Uniform : grid::StretchKind::Tanh, gamma };
  }
  grid::Grid g(sub, axes);

  // ── (6) Problem + BC ─────────────────────────────────────────────────────────
  boundary::Problem problem = boundary::load_problem_from_config(cfg);
  problem.validate();
  boundary::BoundaryApplier bc(problem);

  // ── (7) Backend + TDMA ───────────────────────────────────────────────────────
  auto backend = parallel::make_default_backend();
  auto tdma    = linear_solver::tdma::TdmaRegistry::make_default(topo);

  // ── (8) Fields ───────────────────────────────────────────────────────────────
  field::FieldRegistry fields(sub, *backend);
  auto& U  = fields.add_scalar("U");
  auto& V  = fields.add_scalar("V");
  auto& W  = fields.add_scalar("W");
  auto& P  = fields.add_scalar("P");
  auto& T  = fields.add_scalar("T");
  fields.add_scalar("dU"); fields.add_scalar("dV");
  fields.add_scalar("dW"); fields.add_scalar("dP");

  // ── (9) Physics ──────────────────────────────────────────────────────────────
  const bool thermal_enabled = cfg.get_or<int>("thermal","enabled",1) != 0;
  const double nu = [&]() -> double {
      if (cfg.has("physics","nu")) return cfg.get<double>("physics","nu");
      return std::sqrt(cfg.get<double>("physics","Pr") / cfg.get<double>("physics","Ra"));
  }();
  const double alpha_T = [&]() -> double {
      if (!thermal_enabled) return 0.0;
      if (cfg.has("physics","Ra") && cfg.has("physics","Pr"))
          return std::sqrt(1.0/(cfg.get<double>("physics","Ra")*cfg.get<double>("physics","Pr")));
      return cfg.get_or<double>("physics","alpha_T",0.0);
  }();
  const double Lz = cfg.get<double>("domain","Lz");

  // ── (10) Channel forcing ──────────────────────────────────────────────────────
  const std::string fmode_str =
      cfg.get_or<std::string>("channel_forcing","mode","pressure_gradient");
  const double target_Ub =
      cfg.get_or<double>("channel_forcing","target_bulk_velocity",1.0);
  const double target_dPdx =
      cfg.get_or<double>("channel_forcing","target_dPdx",
          cfg.get_or<double>("source","U_force",1.0));

  const channel::ForcingMode fmode = (fmode_str == "mass_flow")
      ? channel::ForcingMode::MassFlow
      : channel::ForcingMode::PressureGradient;
  const double forcing_target = (fmode == channel::ForcingMode::MassFlow)
      ? target_Ub : target_dPdx;

  channel::ChannelForcing forcing(fmode, forcing_target, topo.cart_comm(), sub, g, problem);

  // Ub used for Poiseuille IC
  const double Ub_init = (fmode == channel::ForcingMode::MassFlow)
      ? target_Ub
      : (target_dPdx / (2.0 * nu)) * (Lz*Lz / 6.0);

  // ── (11) I/O config ───────────────────────────────────────────────────────────
  const std::string dir_restart = cfg.get_or<std::string>("output","dir_restart","./restart_out/");
  const std::string dir_stats   = cfg.get_or<std::string>("output","dir_stats",  "./statistics/");
  const std::string dir_instant = cfg.get_or<std::string>("output","dir_instant","./instant/");
  const int  nmonitor    = cfg.get_or<int>("output","nmonitor",    100);
  const int  nstat_start = cfg.get_or<int>("output","nstat_start", 10000);
  const int  nstat_int   = std::max(1, cfg.get_or<int>("output","nstat", 1));
  const int  nout_stats  = cfg.get_or<int>("output","nout_stats",  10000);
  const int  nout        = cfg.get_or<int>("output","nout",        50000);
  const bool out_stats   = cfg.get_or<int>("output","out_stats",   1) != 0;
  const bool out_field   = cfg.get_or<int>("output","out_field",   0) != 0;
  const bool continue_in = cfg.get_or<int>("output","continue_in", 0) != 0;
  const bool continue_out= cfg.get_or<int>("output","continue_out",1) != 0;

  if (mpi.is_root()) {
      namespace fs = std::filesystem;
      for (const auto& d : {dir_restart, dir_stats, dir_instant}) {
          std::error_code ec;
          fs::create_directories(d, ec);
      }
  }
  MPI_Barrier(topo.cart_comm());

  // ── (12) Initial conditions ───────────────────────────────────────────────────
  U.fill_host(0.0); V.fill_host(0.0); W.fill_host(0.0);
  P.fill_host(0.0); T.fill_host(0.0);

  double restart_time = cfg.get_or<double>("time","t_start",0.0);
  double restart_dPdx = forcing.mean_dPdx();
  int    restart_step = 0;
  real_t restart_dt   = static_cast<real_t>(cfg.get_or<double>("time","dt_start",0.005));

  if (continue_in) {
      if (mpi.is_root())
          std::printf("[channel] reading restart from '%s'\n", dir_restart.c_str());
      post::read_scalar(U, dir_restart + "U.bin");
      post::read_scalar(V, dir_restart + "V.bin");
      post::read_scalar(W, dir_restart + "W.bin");
      post::read_scalar(P, dir_restart + "P.bin");
      if (mpi.is_root()) {
          FILE* mf = std::fopen((dir_restart + "meta.txt").c_str(), "r");
          if (mf) {
              double dt_d;
              if (std::fscanf(mf, "%d %lf %lf %lf",
                          &restart_step, &restart_time, &dt_d, &restart_dPdx) != 4)
                  std::fprintf(stderr, "[channel] warning: meta.txt read failed\n");
              restart_dt = static_cast<real_t>(dt_d);
              std::fclose(mf);
          }
      }
      MPI_Bcast(&restart_step,  1, MPI_INT,    0, topo.cart_comm());
      MPI_Bcast(&restart_time,  1, MPI_DOUBLE, 0, topo.cart_comm());
      {
          double dt_d = static_cast<double>(restart_dt);
          MPI_Bcast(&dt_d, 1, MPI_DOUBLE, 0, topo.cart_comm());
          restart_dt = static_cast<real_t>(dt_d);
      }
      MPI_Bcast(&restart_dPdx, 1, MPI_DOUBLE, 0, topo.cart_comm());
      forcing.set_mean_dPdx(restart_dPdx);
  } else {
      const int    u_ic_poi = cfg.get_or<int>("init","u_ic_poiseuille",0);
      const double pert     = cfg.get_or<double>("init","perturbation",0.0);
      const double roll_amp = cfg.get_or<double>("init","roll_amp",0.0);
      const int    n_roll   = cfg.get_or<int>("init","n_roll",10);
      const double Ly       = cfg.get<double>("domain","Ly");
      if (mpi.is_root() && roll_amp > 0.0)
          std::printf("[laminar_init] STRUCTURED IC: roll_amp=%.3g n_roll=%d (beta=%.3f), noise=0.05\n",
                      roll_amp, n_roll, 2.0*M_PI*n_roll/Ly);
      if (u_ic_poi || pert > 0.0 || roll_amp > 0.0)
          laminar_init(U, V, W, sub, g, Lz, Ly, Ub_init, pert, roll_amp, n_roll,
                       topo.cart_comm(), mpi.world_rank());

      // Initialize body force to laminar Poiseuille value (F_lam = 3*nu for Ub=1, h=1, Lz=2).
      // Without this, dPdx starts at 0 and the initial perturbation drives it to a
      // spuriously large value (~16*F_lam) that locks the system into a wrong fixed point.
      const double F_lam = 3.0 * nu;
      forcing.set_mean_dPdx(-F_lam);   // dPdx < 0 → body force in +x direction
  }

  // Zero W at wall faces before halo exchange (same pattern as time loop step 5).
  // Prevents large dW/dz at wall in IC from creating spurious near-wall U via pressure.
  {
      const int n1t = sub.n_total()[0], n2t = sub.n_total()[1], n3t = sub.n_total()[2];
      real_t* wp = fields.scalar("W").host_ptr();
      const bool lo = (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
      const bool hi = (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);
      if (lo)
          for (int i = 0; i < n1t; ++i)
              for (int j = 0; j < n2t; ++j)
                  wp[(i*n2t+j)*n3t + kHaloWidth] = real_t{0};
      if (hi)
          for (int i = 0; i < n1t; ++i)
              for (int j = 0; j < n2t; ++j)
                  wp[(i*n2t+j)*n3t + (n3t-kHaloWidth)] = real_t{0};
  }

  // Initial halo + ghost
  U.exchange_halo(); bc.apply_ghost(U, problem.U);
  V.exchange_halo(); bc.apply_ghost(V, problem.V);
  W.exchange_halo(); bc.apply_ghost(W, problem.W);
  P.exchange_halo(); bc.apply_ghost(P, problem.P);
  T.exchange_halo(); bc.apply_ghost(T, problem.T);

  // ── (13) Equations ────────────────────────────────────────────────────────────
  std::unique_ptr<equation::scalar::ScalarEquation> thermal_eq;
  if (thermal_enabled) {
      equation::scalar::ScalarTraits T_traits;
      T_traits.name        = "T";
      T_traits.diffusivity = static_cast<real_t>(alpha_T);
      T_traits.with_convection = true;
      thermal_eq = std::make_unique<equation::scalar::ScalarEquation>(
          T_traits, g, sub, fields, problem, problem.T, *tdma, bc);
  }

  // Momentum — constant_source = 0: body force applied explicitly via ChannelForcing.
  equation::momentum::MomentumTraits U_traits, V_traits, W_traits;
  U_traits.name = "U"; U_traits.viscosity = static_cast<real_t>(nu);
  U_traits.with_convection = true;
  U_traits.source_name = (thermal_enabled
      ? cfg.get_or<std::string>("source","U","T") : std::string(""));
  U_traits.constant_source = real_t{0};

  V_traits.name = "V"; V_traits.viscosity = static_cast<real_t>(nu);
  V_traits.with_convection = true;

  W_traits.name = "W"; W_traits.viscosity = static_cast<real_t>(nu);
  W_traits.with_convection = true;

  equation::momentum::MomentumEquation mom_U(
      U_traits, g, sub, fields, problem, problem.U, *tdma, bc);
  equation::momentum::MomentumEquation mom_V(
      V_traits, g, sub, fields, problem, problem.V, *tdma, bc);
  equation::momentum::MomentumEquation mom_W(
      W_traits, g, sub, fields, problem, problem.W, *tdma, bc);

  auto pressure_solver = equation::pressure::make_pressure_solver(
      g, sub, fields, problem, *tdma, bc);

  // ── (14) Statistics ───────────────────────────────────────────────────────────
  channel::ChannelStats stats(sub, g, topo.cart_comm());

  // ── (15) Setup summary ────────────────────────────────────────────────────────
  if (mpi.is_root()) {
      std::cout << "\n=== MPM-STD channel setup ===\n"
                << "  nu         : " << nu
                << "  (Re_b~" << static_cast<int>(std::round(1.0/nu)) << ")\n"
                << "  forcing    : " << fmode_str << "  target=" << forcing_target << "\n"
                << "  thermal    : " << (thermal_enabled ? "on" : "off") << "\n"
                << "  U.ghost ±z : "
                << boundary::ghost_policy_name(problem.U.face(Direction::Z, Side::Minus).ghost_policy)
                << " / "
                << boundary::ghost_policy_name(problem.U.face(Direction::Z, Side::Plus).ghost_policy)
                << "\n=== time loop  t_end=" << cfg.get<double>("time","t_end")
                << "  n_steps="  << cfg.get_or<int>("time","n_steps",std::numeric_limits<int>::max())
                << " ===\n\n";
  }

  // ── (16) Time loop ────────────────────────────────────────────────────────────
  const double t_end   = cfg.get<double>("time","t_end");
  const double MaxCFL  = cfg.get_or<double>("time","MaxCFL",1.0);
  const int    n_steps = cfg.get_or<int>("time","n_steps", std::numeric_limits<int>::max());
  const real_t dt_cap  = static_cast<real_t>(cfg.get_or<double>("time","dt_start",0.05));

  real_t dt   = restart_dt;
  double time = restart_time;
  int    step = restart_step;

  FILE* mon_fp = nullptr;
  FILE* wss_fp = nullptr;
  if (mpi.is_root()) {
      std::string mon_path = dir_instant + "Monitor_Channel.plt";
      mon_fp = std::fopen(mon_path.c_str(), "a");
      if (mon_fp && std::ftell(mon_fp) == 0)
          std::fprintf(mon_fp,
              "VARIABLES=\"Timestep\" \"Time\" \"dt\""
              " \"CFL\" \"maxDivU\" \"WSS\" \"u_tau\" \"U_b\""
              " \"rho_max\" \"rho_min\"\n");

      std::string wss_path = dir_stats + "wss_history.dat";
      wss_fp = std::fopen(wss_path.c_str(), "a");
      if (wss_fp && std::ftell(wss_fp) == 0)
          std::fprintf(wss_fp,
              "# %10s %12s %12s %12s %12s %12s %12s %12s %12s\n",
              "step", "time", "dt", "wss", "u_tau", "div_max", "U_b",
              "rho_max", "rho_min");
  }

  while (time < t_end && step < n_steps) {

      // 0. CFL-based dt update — compute BEFORE the step (matches Filtered_TDMA)
      //    First step uses initial velocities (Poiseuille+perturbation) to pick safe dt.
      {
          const double speed = max_cfl_impl(U, V, W, g, sub, topo.cart_comm(), real_t{1.0});
          if (speed > 1e-14) {
              const real_t dt_cfl = static_cast<real_t>(MaxCFL / speed);
              dt = std::min(dt_cfl, dt_cap);
          }
      }

      // 1. Thermal (optional)
      if (thermal_eq) thermal_eq->step(dt);

      // 2. Momentum — MPM-STD fully-coupled BW-ADI.
      //    (a) predict dU, dV, dW from the SAME u^n (do NOT apply yet),
      //    (b) block lower-triangular coupling: correct dV by dW, then dU by dV,dW,
      //    (c) apply all three increments simultaneously.
      {
          const int n1t = sub.n_total()[0];
          const int n2t = sub.n_total()[1];
          const int n3t = sub.n_total()[2];
          const std::size_t n_full =
              static_cast<std::size_t>(n1t) * n2t * n3t;

          mom_U.predict(dt);
          mom_V.predict(dt);
          mom_W.predict(dt);

          // Move increments into the registered dU/dV/dW fields so we can
          // halo-exchange them for the block-coupling stencil.
          auto& dUf = fields.scalar("dU");
          auto& dVf = fields.scalar("dV");
          auto& dWf = fields.scalar("dW");
          std::copy(mom_U.delta_ptr(), mom_U.delta_ptr() + n_full, dUf.host_ptr());
          std::copy(mom_V.delta_ptr(), mom_V.delta_ptr() + n_full, dVf.host_ptr());
          std::copy(mom_W.delta_ptr(), mom_W.delta_ptr() + n_full, dWf.host_ptr());
          dUf.exchange_halo(); bc.apply_ghost(dUf, problem.U);
          dVf.exchange_halo(); bc.apply_ghost(dVf, problem.V);
          dWf.exchange_halo(); bc.apply_ghost(dWf, problem.W);

          // (b1) blockLdV: dV -= dt*0.25*(dW·∂V/∂z)
          equation::momentum::kernels::block_couple_dV(
              dVf.host_ptr(), dWf.host_ptr(), V.host_ptr(),
              g.dx_ptr(Direction::Y), g.dmx_ptr(Direction::Y),
              g.dx_ptr(Direction::Z), g.dmx_ptr(Direction::Z),
              n1t, n2t, n3t, dt);
          dVf.exchange_halo(); bc.apply_ghost(dVf, problem.V);

          // (b2) blockLdU: dU -= dt*0.25*(dV·∂U/∂y + dW·∂U/∂z)  (uses corrected dV)
          equation::momentum::kernels::block_couple_dU(
              dUf.host_ptr(), dVf.host_ptr(), dWf.host_ptr(), U.host_ptr(),
              g.dx_ptr(Direction::X), g.dmx_ptr(Direction::X),
              g.dmx_ptr(Direction::Y),
              g.dmx_ptr(Direction::Z),
              n1t, n2t, n3t, dt);

          // Copy corrected increments back and apply (dW unchanged).
          std::copy(dUf.host_ptr(), dUf.host_ptr() + n_full, mom_U.delta_ptr());
          std::copy(dVf.host_ptr(), dVf.host_ptr() + n_full, mom_V.delta_ptr());
          mom_U.apply_increment();
          mom_V.apply_increment();
          mom_W.apply_increment();
      }

      // 3. Explicit body force: U += -dt * dPdx
      forcing.apply_body_force(U, dt);

      // 4. Mass-flow correction (no-op for PressureGradient)
      forcing.correct(U, dt);

      // Refresh halo after uniform shift (needed for divergence computation)
      U.exchange_halo();
      bc.apply_ghost(U, problem.U);

      // 5. W wall face: enforce no-penetration (MAC stagger workaround)
      {
          const int n1t = sub.n_total()[0], n2t = sub.n_total()[1], n3t = sub.n_total()[2];
          real_t* wp = fields.scalar("W").host_ptr();
          const bool lo = (sub.topology().axis(Direction::Z).west_rank == MPI_PROC_NULL);
          const bool hi = (sub.topology().axis(Direction::Z).east_rank == MPI_PROC_NULL);
          if (lo)
              for (int i = 0; i < n1t; ++i)
                  for (int j = 0; j < n2t; ++j)
                      wp[(i*n2t+j)*n3t + kHaloWidth] = real_t{0};
          if (hi)
              for (int i = 0; i < n1t; ++i)
                  for (int j = 0; j < n2t; ++j)
                      wp[(i*n2t+j)*n3t + (n3t-kHaloWidth)] = real_t{0};
      }

      // 6. Pressure solve + projection (handles halo internally)
      pressure_solver->solve(dt, U, V, W, P);

      time += static_cast<double>(dt);
      ++step;

      // 7. Statistics accumulation
      if (out_stats && step >= nstat_start && ((step - nstat_start) % nstat_int == 0))
          stats.accumulate(U, V, W, P);

      // 8. Divergence check every step + monitor every nmonitor steps
      {
          const double div = max_divergence(U, V, W, g, sub, topo.cart_comm());

          // Safety abort: print debug dump and exit if div > 1e-10
          if (div > 1e-10) {
              divergence_abort_dump(U, V, W, g, sub, topo.cart_comm(),
                                    step, time, static_cast<double>(dt));
              if (mpi.is_root()) { if (mon_fp) std::fclose(mon_fp); if (wss_fp) std::fclose(wss_fp); }
              MPI_Abort(topo.cart_comm(), 1);
          }

          if (step % nmonitor == 0) {
              const auto [rho_max, rho_min] = rho_diagnostic(g, sub, nu,
                                                              static_cast<double>(dt),
                                                              topo.cart_comm());
              const double wss  = wss_diagnostic(U, g, sub, nu, topo.cart_comm());
              const double utau = std::sqrt(wss > 0 ? wss : 0.0);
              const double Ub   = forcing.bulk_velocity(U);
              if (mpi.is_root()) {
                  const int mstr = nmonitor > 0 ? nmonitor : 1;
                  if (((step / mstr) % 10) == 1) {
                      std::printf("\n");
                      std::printf("%12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
                                  "Timestep","Time","dt",
                                  "maxDivU","WSS","u_tau","U_b","rho_max","rho_min");
                  }
                  std::printf(
                      "%12d %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e %12.5e\n",
                      step, time, static_cast<double>(dt),
                      div, wss, utau, Ub, rho_max, rho_min);
                  std::fflush(stdout);
                  if (mon_fp) {
                      std::fprintf(mon_fp,
                          "%d %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                          step, time, static_cast<double>(dt),
                          div, wss, utau, Ub, rho_max, rho_min);
                      std::fflush(mon_fp);
                  }
                  if (wss_fp) {
                      std::fprintf(wss_fp,
                          "  %10d %12.6e %12.4e %12.6e %12.6e %12.4e %12.6e %12.6e %12.6e\n",
                          step, time, static_cast<double>(dt), wss, utau, div, Ub, rho_max, rho_min);
                      std::fflush(wss_fp);
                  }
              }
          }
      }

      // 9. Stats write
      if (out_stats && step >= nstat_start && stats.samples() > 0
              && ((step - nstat_start) % nout_stats == 0)) {
          char sname[512];
          std::snprintf(sname, sizeof(sname), "%s/stats_%08d.dat", dir_stats.c_str(), step);
          stats.write(sname, step, nu);
          if (mpi.is_root())
              std::printf("  [stats] %s (n=%ld)\n", sname, stats.samples());
      }

      // 10. 3D field output
      if (out_field && nout > 0 && step % nout == 0) {
          char tag[32];
          std::snprintf(tag, sizeof(tag), "%08d", step);
          post::write_scalar(U, dir_instant + "U_" + tag + ".bin");
          post::write_scalar(V, dir_instant + "V_" + tag + ".bin");
          post::write_scalar(W, dir_instant + "W_" + tag + ".bin");
          post::write_scalar(P, dir_instant + "P_" + tag + ".bin");
      }

      // 11. Restart write
      if (continue_out && step >= nstat_start && ((step - nstat_start) % nout_stats == 0)) {
          post::write_scalar(U, dir_restart + "U.bin");
          post::write_scalar(V, dir_restart + "V.bin");
          post::write_scalar(W, dir_restart + "W.bin");
          post::write_scalar(P, dir_restart + "P.bin");
          if (mpi.is_root()) {
              FILE* mf = std::fopen((dir_restart + "meta.txt").c_str(), "w");
              if (mf) {
                  std::fprintf(mf, "%d %lf %lf %lf\n",
                               step, time, static_cast<double>(dt), forcing.mean_dPdx());
                  std::fclose(mf);
              }
          }
      }

  }

  // Final stats
  if (out_stats && stats.samples() > 0) {
      char sname[512];
      std::snprintf(sname, sizeof(sname), "%s/stats_final_%08d.dat", dir_stats.c_str(), step);
      stats.write(sname, step, nu);
      if (mpi.is_root())
          std::printf("[channel] final stats: %s (n=%ld)\n", sname, stats.samples());
  }

  // Final restart
  if (continue_out) {
      post::write_scalar(U, dir_restart + "U.bin");
      post::write_scalar(V, dir_restart + "V.bin");
      post::write_scalar(W, dir_restart + "W.bin");
      post::write_scalar(P, dir_restart + "P.bin");
      if (mpi.is_root()) {
          FILE* mf = std::fopen((dir_restart + "meta.txt").c_str(), "w");
          if (mf) {
              std::fprintf(mf, "%d %lf %lf %lf\n",
                           step, time, static_cast<double>(dt), forcing.mean_dPdx());
              std::fclose(mf);
          }
      }
  }

  if (mpi.is_root()) {
      std::printf("\n=== channel done: step=%d  t=%.5f  dPdx=%.4e ===\n\n",
                  step, time, forcing.mean_dPdx());
      if (mon_fp) std::fclose(mon_fp);
      if (wss_fp) std::fclose(wss_fp);
  }
  return 0;
}
