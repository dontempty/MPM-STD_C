// apps/channel/channel_forcing.hpp
//
// Channel-flow body-force forcing.
//
// Two modes (matching Filtered_TDMA/channel/ChannelForcing):
//   PRESSURE_GRADIENT  — constant dPdx each step (equivalent to old constant_source)
//   MASS_FLOW          — adjusts dPdx each step to maintain target bulk velocity Ub
//
// Usage (time loop):
//   1. forcing.apply_body_force(U, dt);   // U += -dt * dPdx
//   2. forcing.correct(U, dt);            // MASS_FLOW: shift U → Ub_target, update dPdx
//   3. U.exchange_halo(); bc.apply_ghost(U, problem.U);   // refresh ghost after shift
//   4. pressure_solver->solve(dt, U, V, W, P);

#pragma once

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "field/main.hpp"
#include "grid/main.hpp"
#include "parallel/main.hpp"

#include <mpi.h>
#include <cmath>
#include <cstdio>

namespace mpmstd::channel {

enum class ForcingMode { PressureGradient, MassFlow };

class ChannelForcing {
public:
    // target: Ub for MassFlow, |dPdx| (positive body-force magnitude) for PressureGradient.
    ChannelForcing(ForcingMode mode, double target,
                   MPI_Comm cart,
                   const parallel::mpi::Subdomain& sub,
                   const grid::Grid& g,
                   const boundary::Problem& problem)
        : mode_(mode), target_(target), cart_(cart), sub_(sub), g_(g), problem_(problem)
    {
        // PRESSURE_GRADIENT: dPdx_ stored as negative (pressure gradient sign).
        // Body force on U: U += -dt * dPdx_ = dt * target  (positive, drives +x flow).
        if (mode_ == ForcingMode::PressureGradient)
            dPdx_ = -target_;

        // Precompute total volume (once, in constructor).
        const int h  = kHaloWidth;
        const int n1 = sub_.n_total()[0];
        const int n2 = sub_.n_total()[1];
        const int n3 = sub_.n_total()[2];
        const real_t* dx1 = g_.dx_ptr(Direction::X);
        const real_t* dx2 = g_.dx_ptr(Direction::Y);
        const real_t* dx3 = g_.dx_ptr(Direction::Z);

        double local_vol = 0.0;
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j)
                for (int k = h; k < n3-h; ++k)
                    local_vol += static_cast<double>(dx1[i])
                               * static_cast<double>(dx2[j])
                               * static_cast<double>(dx3[k]);
        MPI_Allreduce(&local_vol, &total_volume_, 1, MPI_DOUBLE, MPI_SUM, cart_);
    }

    double mean_dPdx() const { return dPdx_; }
    void   set_mean_dPdx(double v) { dPdx_ = v; }

    // Add body-force increment to U: U_interior += -dt * dPdx_.
    // For PressureGradient: dPdx_ < 0 → U += dt * |target|  (drives +x).
    // For MassFlow:         dPdx_ evolves toward equilibrium.
    void apply_body_force(field::ScalarField& U, double dt) const {
        const int h  = kHaloWidth;
        const int n1 = sub_.n_total()[0];
        const int n2 = sub_.n_total()[1];
        const int n3 = sub_.n_total()[2];
        real_t* u   = U.host_ptr();
        const real_t val = static_cast<real_t>(-dt * dPdx_);

        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j)
                for (int k = h; k < n3-h; ++k)
                    u[(i*n2+j)*n3+k] += val;
    }

    // PRESSURE_GRADIENT: no-op, returns stored dPdx_.
    // MASS_FLOW: shift U interior so bulk(U) = Ub_target; update dPdx_.
    // Returns updated dPdx_.
    double correct(field::ScalarField& U, double dt) {
        if (mode_ == ForcingMode::PressureGradient)
            return dPdx_;

        const double Ub      = bulk_velocity_(U);
        const double DMpresg = (dt > 1.0e-15) ? (Ub - target_) / dt : 0.0;
        const real_t shift   = static_cast<real_t>(-dt * DMpresg);  // = target - Ub

        const int h  = kHaloWidth;
        const int n1 = sub_.n_total()[0];
        const int n2 = sub_.n_total()[1];
        const int n3 = sub_.n_total()[2];
        real_t* u = U.host_ptr();
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j)
                for (int k = h; k < n3-h; ++k)
                    u[(i*n2+j)*n3+k] += shift;

        dPdx_ += DMpresg;
        return dPdx_;
    }

    // Volume-averaged bulk velocity (cell-center U = 0.5*(U[i]+U[i+1]) for x-staggered MAC).
    double bulk_velocity(const field::ScalarField& U) const {
        return bulk_velocity_(U);
    }

private:
    double bulk_velocity_(const field::ScalarField& U) const {
        const int h  = kHaloWidth;
        const int n1 = sub_.n_total()[0];
        const int n2 = sub_.n_total()[1];
        const int n3 = sub_.n_total()[2];
        const real_t* u   = U.host_ptr();
        const real_t* dx1 = g_.dx_ptr(Direction::X);
        const real_t* dx2 = g_.dx_ptr(Direction::Y);
        const real_t* dx3 = g_.dx_ptr(Direction::Z);

        double local = 0.0;
        for (int i = h; i < n1-h; ++i)
            for (int j = h; j < n2-h; ++j)
                for (int k = h; k < n3-h; ++k) {
                    // U is x-face-staggered: cell-center = 0.5*(U[i]+U[i+1])
                    const double uc = 0.5 * (static_cast<double>(u[(i*n2+j)*n3+k])
                                           + static_cast<double>(u[((i+1)*n2+j)*n3+k]));
                    local += uc * static_cast<double>(dx1[i])
                               * static_cast<double>(dx2[j])
                               * static_cast<double>(dx3[k]);
                }
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, cart_);
        return global / total_volume_;
    }

    ForcingMode mode_;
    double      target_;
    double      dPdx_         = 0.0;
    double      total_volume_ = 0.0;
    MPI_Comm    cart_;
    const parallel::mpi::Subdomain& sub_;
    const grid::Grid&               g_;
    const boundary::Problem&        problem_;
};

} // namespace mpmstd::channel
