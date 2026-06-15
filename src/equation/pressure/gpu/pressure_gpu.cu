#include "equation/pressure/solve.hpp"
#include "equation/pressure/gpu/pressure_engine_gpu.cuh"  // PressureSolverGpu definition

#include "core/domain.hpp"
#include "core/field_gpu.hpp"
#include "core/system.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace mpmstd::equation {

// ── OB GPU pressure solve ─────────────────────────────────────────────────────

void solve_pressure_gpu(const core::Domain&             domain,
                        const core::BoundaryCondition&  bc,
                        core::GpuFields&                gpu_fields,
                        core::PressureSystem&           poi,
                        real_t                          dt)
{
  using Eng = pressure::PressureSolverGpu;

  if (!poi.engine_gpu) {
    poi.engine_gpu = std::make_shared<Eng>(domain.grid, domain.sub, bc, domain.tdma);
  }
  auto& eng = *std::static_pointer_cast<Eng>(poi.engine_gpu);
  eng.solve(dt, gpu_fields, bc, domain.sub);
}


// ── NOB GPU pressure solve ────────────────────────────────────────────────────

void solve_pressure_nob_gpu(const core::Domain&             domain,
                             const core::BoundaryCondition&  bc,
                             core::GpuFields&                gpu_fields,
                             core::PressureSystem&           poi,
                             real_t                          dt,
                             const core::GpuField&           inv_rho)
{
  using Eng = pressure::PressureSolverGpu;

  if (!poi.engine_gpu) {
    poi.engine_gpu = std::make_shared<Eng>(domain.grid, domain.sub, bc, domain.tdma);
  }

  // Lazy-init dPhat GPU field (zero-initialized)
  if (!poi.dPhat_gpu_field) {
    auto* f = new core::GpuField(domain.sub, "dPhat_gpu");
    CUDA_CHK(cudaMemset(f->data(), 0, f->n_elements() * sizeof(real_t)));
    poi.dPhat_gpu_field = std::shared_ptr<void>(f,
        [](void* p){ delete static_cast<core::GpuField*>(p); });
  }

  auto& eng   = *std::static_pointer_cast<Eng>(poi.engine_gpu);
  auto& dPhat = *static_cast<core::GpuField*>(poi.dPhat_gpu_field.get());
  eng.solve_nob(dt, gpu_fields, bc, domain.sub, inv_rho, dPhat);
}

} // namespace mpmstd::equation
