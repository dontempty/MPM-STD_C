#pragma once

#include "core/system.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/grid.hpp"

namespace mpmstd::physics {

// Immersed-boundary mask + forcing (rev.2 §9, P10). The `is_fluid_face` masking
// (RHS=0, diag=1 on solid faces) modifies A·b AFTER assemble. Composed in main;
// omit ⇒ no IBM.
struct IbmMask {
  // device-/host-resident fluid-fraction or flag field filled in P10
};

void build_ibm_mask_cpu  (IbmMask& mask, const core::Grid& grid);
void apply_ibm_forcing_cpu(core::CpuMomentumSystem& mom, const IbmMask& mask);
void build_ibm_mask_gpu  (IbmMask& mask, const core::Grid& grid);
void apply_ibm_forcing_gpu(core::GpuMomentumSystem& mom, const IbmMask& mask);

} // namespace mpmstd::physics
