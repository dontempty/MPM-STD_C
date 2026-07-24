#include "physics/properties/properties.hpp"
#include "common/macros.hpp"

#ifdef MPMSTD_BACKEND_CUDA

using mpmstd::real_t;

namespace {

// The property formulas themselves live in properties.hpp
// (evaluate_properties, MPMSTD_HD — shared with the cpu/ backend); this kernel
// only maps it over the field. PropertyModel is trivially copyable, so it is
// passed to the kernel by value directly.
__global__ void k_update_properties(
    real_t* Mu, real_t* invRho, real_t* KPP, real_t* dKPP, real_t* invRhoCp, real_t* dinvRhoCp,
    const real_t* T, int n, const mpmstd::physics::properties::PropertyModel pm) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const mpmstd::physics::properties::PropertyValues v = mpmstd::physics::properties::evaluate_properties(pm, T[idx]);
  Mu[idx] = v.Mu;             invRho[idx] = v.invRho;
  KPP[idx] = v.KPP;           dKPP[idx] = v.dKPP;
  invRhoCp[idx] = v.invRhoCp; dinvRhoCp[idx] = v.dinvRhoCp;
}

} // anonymous namespace

namespace mpmstd::physics::properties {

void update_properties_gpu(GpuPropertyFields& p, const core::GpuField& T, const PropertyModel& m) {
  const auto nt = T.n_total();
  const int n = nt[0] * nt[1] * nt[2];
  const int block = 256;
  const int grid  = (n + block - 1) / block;
  k_update_properties<<<grid, block>>>(
    p.Mu.data(), p.invRho.data(), p.KPP.data(), p.dKPP.data(),
    p.invRhoCp.data(), p.dinvRhoCp.data(), T.data(), n, m);
}

} // namespace mpmstd::physics::properties

#else
namespace mpmstd::physics::properties {
void update_properties_gpu(GpuPropertyFields&, const core::GpuField&, const PropertyModel&) {}
} // namespace mpmstd::physics::properties
#endif
