#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"
#include "core/mpi_topology.hpp"

namespace mpmstd::driver {

// Case-shared CFL time-step (rev.2 M5 driver layer). speed = global-max over the
// domain of (|U|/dx + |V|/dy + |W|/dz); dt = min(max_cfl/speed, dt_cap).
real_t compute_cfl_dt_cpu(const core::CpuField& U, const core::CpuField& V, const core::CpuField& W,
                          const core::Grid& grid, const core::Subdomain& sub, real_t max_cfl, real_t dt_cap);
real_t compute_cfl_dt_gpu(const core::GpuField& U, const core::GpuField& V, const core::GpuField& W,
                          const core::Grid& grid, const core::Subdomain& sub, real_t max_cfl, real_t dt_cap);

} // namespace mpmstd::driver
