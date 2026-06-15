#include "physics/buoyancy/buoyancy.hpp"
#include "common/macros.hpp"

#ifdef MPMSTD_BACKEND_CUDA

using mpmstd::real_t;

namespace {

__global__ void k_add_buoyancy(real_t* rhs, const real_t* T, int n,
                                real_t coeff, real_t t_ref, real_t dt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) rhs[c] += dt * coeff * (T[c] - t_ref);
}

__global__ void k_add_buoyancy_nob(real_t* rhs, const real_t* T, const real_t* ir,
                                    int n, real_t a12pera11, real_t dT, real_t dt) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) {
    const real_t th = T[c];
    rhs[c] += dt * ir[c] * (th + a12pera11 * th * th * dT);
  }
}

} // anonymous namespace

namespace mpmstd::physics {

void add_buoyancy_force_gpu(core::GpuMomentumSystem& mom, const core::GpuField& T,
                             const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.dU.n_total(); const int n = nt[0] * nt[1] * nt[2];
  real_t* rhs = (p.gravity_axis == 0) ? mom.dU.data()
              : (p.gravity_axis == 1) ? mom.dV.data() : mom.dW.data();
  const int block = 256, grid = (n + block - 1) / block;
  // Note: for GPU path rhs is in dU/dV/dW device fields (different from CPU rhs_u/v/w vectors)
  k_add_buoyancy<<<grid, block>>>(rhs, T.data(), n, p.coeff, p.T_ref, dt);
}

void add_buoyancy_force_nob_gpu(core::GpuMomentumSystem& mom, const core::GpuField& T,
                                 const core::GpuField& inv_rho, const BuoyancyParams& p, real_t dt) {
  const auto nt = mom.dU.n_total(); const int n = nt[0] * nt[1] * nt[2];
  real_t* rhs = (p.gravity_axis == 0) ? mom.dU.data()
              : (p.gravity_axis == 1) ? mom.dV.data() : mom.dW.data();
  const int block = 256, grid = (n + block - 1) / block;
  k_add_buoyancy_nob<<<grid, block>>>(rhs, T.data(), inv_rho.data(), n, p.a12pera11, p.delta_T, dt);
}

} // namespace mpmstd::physics

#else
namespace mpmstd::physics {
void add_buoyancy_force_gpu(core::GpuMomentumSystem&, const core::GpuField&, const BuoyancyParams&, real_t) {}
void add_buoyancy_force_nob_gpu(core::GpuMomentumSystem&, const core::GpuField&, const core::GpuField&, const BuoyancyParams&, real_t) {}
} // namespace mpmstd::physics
#endif
