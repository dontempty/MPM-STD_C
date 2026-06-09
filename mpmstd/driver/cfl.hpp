#pragma once

#include "core/domain.hpp"
#include "core/variables.hpp"

namespace mpmstd::driver {

// Case-shared CFL time step (rev.2 M5). speed = global-max over the domain of
// (|U|/dx + |V|/dy + |W|/dz); dt = min(max_cfl/speed, dt_cap).
real_t compute_cfl_dt_cpu(const core::Domain& domain, const core::CpuFields& fields, real_t max_cfl, real_t dt_cap);
real_t compute_cfl_dt_gpu(const core::Domain& domain, const core::GpuFields& fields, real_t max_cfl, real_t dt_cap);

} // namespace mpmstd::driver
