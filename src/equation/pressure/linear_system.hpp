#pragma once

#include "core/field_cpu.hpp"   // core::CpuField (dPhat/dP0)

#include <memory>

namespace mpmstd::equation::pressure {

class PressureSolver;   // defined in pressure_engine.hpp (same module)

// Pressure-Poisson linear-system workspace. Unlike the momentum/scalar workspaces
// (pure per-step scratch), this holds the heavy solver engine (FFTW plans, pencil
// buffers, distributed z-TDMA) built lazily on first solve and reused, plus the
// NOB pressure-predictor extrapolation buffers. (The accumulated pressure Pⁿ is
// still carried by the app as a separate field; folding it in here is stage (b).)
struct LinearSystem {
  std::shared_ptr<PressureSolver> pressure_solver;   // lazy (CPU engine)
  std::unique_ptr<core::CpuField> dP;                // per-step pressure increment δp (owned here,
                                                     // like momentum's dU). The Poisson solve writes it;
                                                     // the accumulator Pⁿ lives OUTSIDE in Var::P.
  std::unique_ptr<core::CpuField> dPhat;             // prev-step dP predictor for NOB; null = OB
  std::unique_ptr<core::CpuField> dP0;               // running half-average of dPhat for the 2nd-order
                                                     // extrapolation dPhat=2*dP-dP0 (PaScaL_TCS §820-821)

  // GPU engine + fields: type-erased (shared_ptr<void>) to avoid GPU headers here.
  // Holds a PressureSolverGpu* (GPU build only); null until first GPU solve.
  std::shared_ptr<void> pressure_solver_gpu;
  // Holds a GpuField* for the increment δp (owned; Poisson writes it, Var::P accumulates it).
  std::shared_ptr<void> dP_gpu_field;
  // Holds a GpuField* for dPhat (prev-step dP predictor for GPU NOB correction); null = OB.
  std::shared_ptr<void> dPhat_gpu_field;
  // Holds a GpuField* for dP0 (companion of dPhat for the GPU 2nd-order extrapolation).
  std::shared_ptr<void> dP0_gpu_field;
};

} // namespace mpmstd::equation::pressure
