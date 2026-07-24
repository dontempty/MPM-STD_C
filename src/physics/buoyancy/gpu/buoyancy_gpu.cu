#include "physics/buoyancy/buoyancy.hpp"
#include "common/macros.hpp"
#include "common/direction.hpp"
#include "core/gpu/gpu_grid.hpp"

#ifdef MPMSTD_BACKEND_CUDA

using mpmstd::real_t;

namespace {

// OB buoyancy with PaScaL_TCS face-interpolated temperature (f90:348):
//   Tc = 0.5*(dxg[ig-1]*T[c] + dxg[ig]*T[c-gstride])/dmxg[ig]   (own = gravity axis)
// One thread per INTERIOR cell; the gravity-minus ghost (T[c-gstride]) must be
// halo-current (DHVC: periodic x; the time loop syncs T before this call).
__global__ void k_add_buoyancy(real_t* rhs, const real_t* T,
                                const double* dxg, const double* dmxg,
                                int n1t, int n2t, int n3t, int h, int gaxis,
                                real_t coeff, real_t t_ref, real_t dt) {
  const int ni = n1t - 2*h, nj = n2t - 2*h, nk = n3t - 2*h;
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int kk = blockIdx.z * blockDim.z + threadIdx.z;
  if (ii >= ni || jj >= nj || kk >= nk) return;
  const int i = ii + h, j = jj + h, k = kk + h;
  const int c = (i * n2t + j) * n3t + k;
  const int gstride = (gaxis == 0) ? n2t * n3t : (gaxis == 1) ? n3t : 1;
  const int ig      = (gaxis == 0) ? i         : (gaxis == 1) ? j   : k;
  const double Tc = 0.5*(dxg[ig-1]*(double)T[c] + dxg[ig]*(double)T[c-gstride])/dmxg[ig];
  rhs[c] += dt * coeff * ((real_t)Tc - t_ref);
}

// NOB buoyancy with face-interpolated T and invRho (PaScaL_TCS mom.f90:711/731,
// solvedV) — mirrors k_add_buoyancy above, extended to also face-interpolate invRho.
__global__ void k_add_buoyancy_nob(real_t* rhs, const real_t* T, const real_t* ir,
                                    const double* dxg, const double* dmxg,
                                    int n1t, int n2t, int n3t, int h, int gaxis,
                                    real_t a12pera11, real_t dT, real_t dt) {
  const int ni = n1t - 2*h, nj = n2t - 2*h, nk = n3t - 2*h;
  const int ii = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int kk = blockIdx.z * blockDim.z + threadIdx.z;
  if (ii >= ni || jj >= nj || kk >= nk) return;
  const int i = ii + h, j = jj + h, k = kk + h;
  const int c = (i * n2t + j) * n3t + k;
  const int gstride = (gaxis == 0) ? n2t * n3t : (gaxis == 1) ? n3t : 1;
  const int ig      = (gaxis == 0) ? i         : (gaxis == 1) ? j   : k;
  const double Tc  = 0.5*(dxg[ig-1]*(double)T[c]  + dxg[ig]*(double)T[c-gstride])/dmxg[ig];
  const double irc = 0.5*(dxg[ig-1]*(double)ir[c] + dxg[ig]*(double)ir[c-gstride])/dmxg[ig];
  rhs[c] += dt * (real_t)irc * ((real_t)Tc + a12pera11 * (real_t)(Tc * Tc) * dT);
}

} // anonymous namespace

namespace mpmstd::physics::buoyancy {

void add_buoyancy_force_constant_property_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom,
                             const core::GpuField& T, const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.dU.n_total();
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2], h = kHaloWidth;
  real_t* rhs = (p.gravity_axis == 0) ? mom.dU.data()
              : (p.gravity_axis == 1) ? mom.dV.data() : mom.dW.data();
  // gravity-axis metrics (device) for the face interpolation of T
  const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
  const double* dxg  = (p.gravity_axis == 0) ? dg->d_dx1()  : (p.gravity_axis == 1) ? dg->d_dx2()  : dg->d_dx3();
  const double* dmxg = (p.gravity_axis == 0) ? dg->d_dmx1() : (p.gravity_axis == 1) ? dg->d_dmx2() : dg->d_dmx3();
  const int ni = n1t - 2*h, nj = n2t - 2*h, nk = n3t - 2*h;
  const dim3 blk(8, 8, 8);
  const dim3 grd((ni + 7) / 8, (nj + 7) / 8, (nk + 7) / 8);
  // Note: GPU rhs is in dU/dV/dW device fields (different from CPU rhs_u/v/w vectors)
  k_add_buoyancy<<<grd, blk>>>(rhs, T.data(), dxg, dmxg, n1t, n2t, n3t, h, p.gravity_axis,
                               p.coeff, p.T_ref, dt);
}

void add_buoyancy_force_variable_property_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom,
                                 const core::GpuField& T, const core::GpuField& inv_rho,
                                 const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.dU.n_total();
  const int n1t = nt[0], n2t = nt[1], n3t = nt[2], h = kHaloWidth;
  real_t* rhs = (p.gravity_axis == 0) ? mom.dU.data()
              : (p.gravity_axis == 1) ? mom.dV.data() : mom.dW.data();
  const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
  const double* dxg  = (p.gravity_axis == 0) ? dg->d_dx1()  : (p.gravity_axis == 1) ? dg->d_dx2()  : dg->d_dx3();
  const double* dmxg = (p.gravity_axis == 0) ? dg->d_dmx1() : (p.gravity_axis == 1) ? dg->d_dmx2() : dg->d_dmx3();
  const int ni = n1t - 2*h, nj = n2t - 2*h, nk = n3t - 2*h;
  const dim3 blk(8, 8, 8);
  const dim3 grd((ni + 7) / 8, (nj + 7) / 8, (nk + 7) / 8);
  k_add_buoyancy_nob<<<grd, blk>>>(rhs, T.data(), inv_rho.data(), dxg, dmxg,
                                   n1t, n2t, n3t, h, p.gravity_axis,
                                   p.a12pera11, p.delta_T, dt);
}

} // namespace mpmstd::physics::buoyancy

#else
namespace mpmstd::physics::buoyancy {
void add_buoyancy_force_constant_property_gpu(const core::Domain&, equation::momentum::GpuLinearSystem&, const core::GpuField&, const BuoyancyParams&, real_t) {}
void add_buoyancy_force_variable_property_gpu(const core::Domain&, equation::momentum::GpuLinearSystem&, const core::GpuField&, const core::GpuField&, const BuoyancyParams&, real_t) {}
} // namespace mpmstd::physics::buoyancy
#endif
