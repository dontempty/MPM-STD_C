#include "driver/cfl.hpp"

// P0 skeleton stub. Returns dt_cap so a future time loop is well-defined; P1
// computes the true convective/viscous CFL min.
namespace mpmstd::driver {
real_t compute_cfl_dt_cpu(const core::CpuField&, const core::CpuField&, const core::CpuField&,
                          const core::Grid&, const core::Subdomain&, real_t dt_cap) {
  return dt_cap;  // TODO(P1)
}
} // namespace mpmstd::driver
