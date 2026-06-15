#include "physics/properties/properties.hpp"
#include "common/macros.hpp"

#ifdef MPMSTD_BACKEND_CUDA

namespace {

// Horner evaluation: k[0]*(1 + k[1]*t + k[2]*t^2 + ... + k[5]*t^5)
__device__ static double poly_d(const double k[6], double t) {
  return k[0] * (1.0 + t * (k[1] + t * (k[2] + t * (k[3] + t * (k[4] + t * k[5])))));
}
__device__ static double npoly_d(const double k[6], double t) {
  return 1.0 + t * (k[1] + t * (k[2] + t * (k[3] + t * (k[4] + t * k[5]))));
}

struct PropertyModelFlat {
  double a[6], b[6], c[6], d[6];
  double Dt;
  double Mu0;  // a[0]*d[0]
};

__global__ void k_update_properties(
    double* mu, double* irho, double* kpp, double* dkpp, double* ircp, double* dircp,
    const double* T, int n, const PropertyModelFlat pm) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const double t = pm.Dt * T[idx];
  mu[idx]   = poly_d(pm.a, t) * poly_d(pm.d, t) / pm.Mu0;
  irho[idx] = 1.0 / npoly_d(pm.a, t);
  kpp[idx]  = npoly_d(pm.c, t);
  dkpp[idx] = pm.c[1] + t * (2.0*pm.c[2] + t * (3.0*pm.c[3] + t * (4.0*pm.c[4] + t * 5.0*pm.c[5])));
  const double ia = 1.0 + t * (pm.a[1] + t * (pm.a[2] + t * pm.a[3]));
  const double ib = 1.0 + t * (pm.b[1] + t * (pm.b[2] + t * pm.b[3]));
  ircp[idx] = 1.0 / (ia * ib);
  const double t2 = t * t, t3 = t2 * t;
  const double num = pm.b[1] * (1.0 + 3.0*pm.a[2]*t2 + 4.0*pm.a[3]*t3)
                   + pm.a[1] * (1.0 + 2.0*pm.b[1]*t + 3.0*pm.b[2]*t2 + 4.0*pm.b[3]*t3)
                   + t * ( pm.b[2] * (2.0 + 5.0*pm.a[3]*t3)
                         + pm.a[2] * (2.0 + 4.0*pm.b[2]*t2 + 5.0*pm.b[3]*t3)
                         + 3.0*t * (pm.a[3] + pm.b[3] + 2.0*pm.a[3]*pm.b[3]*t3) );
  dircp[idx] = -num / (ia * ia * ib * ib);
}

} // anonymous namespace

namespace mpmstd::physics {

void update_properties_gpu(GpuPropertyFields& p, const core::GpuField& T, const PropertyModel& m) {
  const auto nt = T.n_total();
  const int n = nt[0] * nt[1] * nt[2];
  PropertyModelFlat pm;
  for (int i = 0; i < 6; ++i) { pm.a[i]=m.a[i]; pm.b[i]=m.b[i]; pm.c[i]=m.c[i]; pm.d[i]=m.d[i]; }
  pm.Dt  = m.DeltaT;
  pm.Mu0 = double(m.a[0]) * m.d[0];
  const int block = 256;
  const int grid  = (n + block - 1) / block;
  k_update_properties<<<grid, block>>>(
    p.mu.data(), p.inv_rho.data(), p.kappa.data(), p.dkappa.data(),
    p.inv_rhocp.data(), p.dinv_rhocp.data(), T.data(), n, pm);
}

} // namespace mpmstd::physics

#else
namespace mpmstd::physics {
void update_properties_gpu(GpuPropertyFields&, const core::GpuField&, const PropertyModel&) {}
} // namespace mpmstd::physics
#endif
