// apps/channel/channel_stats.hpp
//
// Time-averaged z-profile statistics for channel flow.
// Matches Filtered_TDMA/channel/Statistics logic, adapted for MPM-STD_C APIs.
//
// Grid assumptions (MAC staggered):
//   U at x-faces: cell-center U = 0.5*(U[i]+U[i+1])
//   V at y-faces: cell-center V = 0.5*(V[j]+V[j+1])
//   W at z-faces: cell-center W = 0.5*(W[k]+W[k+1])
//   P at cell-center (collocated)
//
// accumulate() — local Welford incremental mean update (no MPI).
// write()      — gathers via MPI_Allreduce, rank-0 writes Tecplot ASCII.

#pragma once

#include "common/main.hpp"
#include "field/main.hpp"
#include "grid/main.hpp"
#include "parallel/main.hpp"

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace mpmstd::channel {

class ChannelStats {
public:
    ChannelStats(const parallel::mpi::Subdomain& sub,
                 const grid::Grid& g,
                 MPI_Comm cart)
        : sub_(sub), g_(g), cart_(cart)
    {
        nz_global_ = sub_.n_global ()[2];
        nz_local_  = sub_.n_interior()[2];
        kstart_    = sub_.global_offset()[2];   // 0-based global z offset

        U_m_  .assign(nz_local_, 0.0);  U2_m_ .assign(nz_local_, 0.0);
        V_m_  .assign(nz_local_, 0.0);  V2_m_ .assign(nz_local_, 0.0);
        Wc_m_ .assign(nz_local_, 0.0);  Wc2_m_.assign(nz_local_, 0.0);
        UWc_m_.assign(nz_local_, 0.0);  P_m_  .assign(nz_local_, 0.0);

        // Build global z-center array via MPI_Allreduce MAX.
        // Each rank contributes its local z values; non-owned slots stay 0.
        const auto& zc = g_.xc(Direction::Z);  // local, includes halo
        std::vector<double> tmp(nz_global_, 0.0);
        for (int kl = 0; kl < nz_local_; ++kl)
            tmp[kstart_ + kl] = static_cast<double>(zc[kHaloWidth + kl]);
        zc_global_.resize(nz_global_);
        MPI_Allreduce(tmp.data(), zc_global_.data(), nz_global_,
                      MPI_DOUBLE, MPI_MAX, cart_);
    }

    void reset() {
        n_ = 0;
        std::fill(U_m_  .begin(), U_m_  .end(), 0.0);
        std::fill(U2_m_ .begin(), U2_m_ .end(), 0.0);
        std::fill(V_m_  .begin(), V_m_  .end(), 0.0);
        std::fill(V2_m_ .begin(), V2_m_ .end(), 0.0);
        std::fill(Wc_m_ .begin(), Wc_m_ .end(), 0.0);
        std::fill(Wc2_m_.begin(), Wc2_m_.end(), 0.0);
        std::fill(UWc_m_.begin(), UWc_m_.end(), 0.0);
        std::fill(P_m_  .begin(), P_m_  .end(), 0.0);
    }

    long samples() const { return n_; }

    // No MPI: local Welford mean update per z-slab.
    void accumulate(const field::ScalarField& U,
                    const field::ScalarField& V,
                    const field::ScalarField& W,
                    const field::ScalarField& P) {
        ++n_;
        const double inv_n = 1.0 / static_cast<double>(n_);

        const int h  = kHaloWidth;
        const int n1 = sub_.n_total()[0];
        const int n2 = sub_.n_total()[1];
        const int n3 = sub_.n_total()[2];
        // GLOBAL xy cell count: write() does MPI_Allreduce(SUM) over all xy-ranks
        // (np1*np2 of them share each z-slab), so the spatial xy-average must be
        // normalized by the GLOBAL nx*ny, not the per-rank local count — otherwise
        // every mean/rms is inflated by np1*np2 (the same bug as wss_diagnostic).
        const int nx = sub_.n_global()[0];
        const int ny = sub_.n_global()[1];
        const double inv_NxNy = 1.0 / static_cast<double>(nx * ny);

        const real_t* u = U.host_ptr();
        const real_t* v = V.host_ptr();
        const real_t* w = W.host_ptr();
        const real_t* p = P.host_ptr();

        for (int kl = 0; kl < nz_local_; ++kl) {
            const int k = kl + h;   // total (halo) index
            double su=0, su2=0, sv=0, sv2=0;
            double swc=0, swc2=0, suwc=0, sp=0;

            for (int i = h; i < n1-h; ++i)
                for (int j = h; j < n2-h; ++j) {
                    // Cell-centre interpolation (MAC grid)
                    const double uc  = 0.5 * (static_cast<double>(u[(i*n2+j)*n3+k])
                                            + static_cast<double>(u[((i+1)*n2+j)*n3+k]));
                    const double vc  = 0.5 * (static_cast<double>(v[(i*n2+j)*n3+k])
                                            + static_cast<double>(v[(i*n2+j+1)*n3+k]));
                    const double wci = 0.5 * (static_cast<double>(w[(i*n2+j)*n3+k])
                                            + static_cast<double>(w[(i*n2+j)*n3+k+1]));
                    const double pi  = static_cast<double>(p[(i*n2+j)*n3+k]);

                    su   += uc;    su2  += uc  * uc;
                    sv   += vc;    sv2  += vc  * vc;
                    swc  += wci;   swc2 += wci * wci;
                    suwc += uc * wci;
                    sp   += pi;
                }

            const double um   = su   * inv_NxNy;
            const double u2m  = su2  * inv_NxNy;
            const double vm   = sv   * inv_NxNy;
            const double v2m  = sv2  * inv_NxNy;
            const double wcm  = swc  * inv_NxNy;
            const double wc2m = swc2 * inv_NxNy;
            const double uwcm = suwc * inv_NxNy;
            const double pm   = sp   * inv_NxNy;

            // Welford incremental update
            U_m_  [kl] += (um   - U_m_  [kl]) * inv_n;
            U2_m_ [kl] += (u2m  - U2_m_ [kl]) * inv_n;
            V_m_  [kl] += (vm   - V_m_  [kl]) * inv_n;
            V2_m_ [kl] += (v2m  - V2_m_ [kl]) * inv_n;
            Wc_m_ [kl] += (wcm  - Wc_m_ [kl]) * inv_n;
            Wc2_m_[kl] += (wc2m - Wc2_m_[kl]) * inv_n;
            UWc_m_[kl] += (uwcm - UWc_m_[kl]) * inv_n;
            P_m_  [kl] += (pm   - P_m_  [kl]) * inv_n;
        }
    }

    // All ranks must call (MPI_Allreduce inside). Rank-0 writes Tecplot ASCII.
    // nu used to compute u_tau and z+.
    void write(const std::string& path, int step, double nu) {
        // Gather 8 stat fields in one coalesced Allreduce.
        constexpr int NFIELDS = 8;
        std::vector<double> buf_send(static_cast<std::size_t>(NFIELDS) * nz_global_, 0.0);
        std::vector<double> buf_recv(static_cast<std::size_t>(NFIELDS) * nz_global_, 0.0);

        auto row = [&](int f) { return buf_send.data() + static_cast<std::size_t>(f) * nz_global_; };

        const std::vector<const std::vector<double>*> locals =
            { &U_m_, &U2_m_, &V_m_, &V2_m_, &Wc_m_, &Wc2_m_, &UWc_m_, &P_m_ };
        for (int f = 0; f < NFIELDS; ++f)
            for (int kl = 0; kl < nz_local_; ++kl)
                row(f)[kstart_ + kl] = (*locals[f])[kl];

        MPI_Allreduce(buf_send.data(), buf_recv.data(),
                      static_cast<int>(buf_send.size()),
                      MPI_DOUBLE, MPI_SUM, cart_);

        // The sum counts each xy-rank once (stats are LOCAL — each rank owns distinct z-slabs).
        // So no division needed.

        int rank = 0;
        MPI_Comm_rank(cart_, &rank);
        if (rank != 0 || n_ == 0) return;

        auto col = [&](int f, int k) { return buf_recv[static_cast<std::size_t>(f)*nz_global_ + k]; };

        // u_tau from bottom-wall gradient: τ_w = ν * |<U(z≈0)>| / z_c[0]
        const double tau_w = nu * std::fabs(col(0, 0)) / zc_global_[0];
        const double u_tau = std::sqrt(std::max(tau_w, 0.0));
        const double inv_nu = 1.0 / nu;

        FILE* fp = std::fopen(path.c_str(), "w");
        if (!fp) {
            std::fprintf(stderr, "[ChannelStats] cannot open '%s'\n", path.c_str());
            return;
        }

        std::fprintf(fp,
            "TITLE = \"Channel Statistics (step=%d, n=%ld)\"\n"
            "VARIABLES = \"Z\" \"Z_plus\" \"U_mean\" \"W_mean\""
            " \"u_rms\" \"v_rms\" \"w_rms\" \"uw_stress\" \"P_mean\"\n"
            "ZONE T=\"Stats\", I=%d, J=1, K=1, DATAPACKING=POINT\n",
            step, n_, nz_global_);

        for (int k = 0; k < nz_global_; ++k) {
            const double zp    = zc_global_[k] * u_tau * inv_nu;
            const double u_rms = std::sqrt(std::max(col(1,k) - col(0,k)*col(0,k), 0.0));
            const double v_rms = std::sqrt(std::max(col(3,k) - col(2,k)*col(2,k), 0.0));
            const double w_rms = std::sqrt(std::max(col(5,k) - col(4,k)*col(4,k), 0.0));
            const double uw    = col(6,k) - col(0,k)*col(4,k);

            std::fprintf(fp, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                zc_global_[k], zp,
                col(0,k), col(4,k),
                u_rms, v_rms, w_rms,
                uw, col(7,k));
        }
        std::fclose(fp);
    }

private:
    const parallel::mpi::Subdomain& sub_;
    const grid::Grid& g_;
    MPI_Comm cart_;

    int  nz_global_ = 0, nz_local_ = 0, kstart_ = 0;
    long n_ = 0;

    std::vector<double> U_m_, U2_m_;
    std::vector<double> V_m_, V2_m_;
    std::vector<double> Wc_m_, Wc2_m_;
    std::vector<double> UWc_m_, P_m_;
    std::vector<double> zc_global_;
};

} // namespace mpmstd::channel
