#pragma once

#include "core/bands.hpp"
#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"   // Subdomain
#include "common/direction.hpp"    // Direction, Component, to_int

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

// The pressure-Poisson engine (FFT/transpose/TDMA state) lives in the pressure
// layer; PressureSystem just holds it (rev.2 §4: System holds solver state).
namespace mpmstd::equation::pressure { class PressureSolver; }

namespace mpmstd::core {

// =============================================================================
// *System  — the Ax=b bundle for one equation. assemble_* fills it; solve_*
//            consumes it. (rev.2 §4)
// =============================================================================

// One scalar transport equation: a Bands per sweep direction (3-stage ADI).
struct ScalarSystem {
  Bands x, y, z;
  Bands&       along(Direction d)       { return d == Direction::X ? x : (d == Direction::Y ? y : z); }
  const Bands& along(Direction d) const { return d == Direction::X ? x : (d == Direction::Y ? y : z); }
};

// Momentum (rev.2 M2 + structural redesign): solve_momentum does U,V,W + the
// block lower-triangular coupling in one. The system OWNS the increments
// dU,dV,dW (created inside momentum — NOT user variables) plus the per-component
// explicit RHS / ADI ping-pong / tridiagonal band workspace, all sized from the
// subdomain at construction. Templated on the field type so the GPU path gets
// device-resident increments (GpuMomentumSystem). [Workspace vectors are host
// for now; the device workspace lands with the P4/P5 GPU kernels.]
template <class FieldT>
struct MomentumSystemT {
  FieldT dU, dV, dW;                                  // increments (owned here)
  std::vector<real_t> rhs_u, rhs_v, rhs_w, stage;     // explicit RHS → running increment, ADI ping-pong
  std::vector<real_t> A, B, C, D;                     // tridiagonal bands [n_row × n_sys]
  std::array<int, 3>  n_total{}, n_interior{};

  explicit MomentumSystemT(const Subdomain& sub)
    : dU(sub, "dU"), dV(sub, "dV"), dW(sub, "dW"),
      n_total(sub.n_total()), n_interior(sub.n_interior()) {
    const std::size_t nf  = static_cast<std::size_t>(n_total[0]) * n_total[1] * n_total[2];
    const std::size_t nin = static_cast<std::size_t>(n_interior[0]) * n_interior[1] * n_interior[2];
    rhs_u.assign(nf, 0); rhs_v.assign(nf, 0); rhs_w.assign(nf, 0); stage.assign(nf, 0);
    A.assign(nin, 0); B.assign(nin, 0); C.assign(nin, 0); D.assign(nin, 0);
  }

  std::vector<real_t>& rhs(Component c) {
    return c == Component::U ? rhs_u : (c == Component::V ? rhs_v : rhs_w);
  }
};

using CpuMomentumSystem = MomentumSystemT<CpuField>;
using GpuMomentumSystem = MomentumSystemT<GpuField>;

// Per-axis spectral transform for the pressure Poisson (chosen from BC, §9c).
enum class Transform { Fft, Dct, Tdma };

// Pressure-Poisson system: per-axis transform tags + the heavy solver engine
// (FFTW plans, pencil buffers, wavenumbers, distributed z-TDMA), built lazily on
// first solve and reused. (The engine is the CPU FFTW one; a GPU engine lands at P4.)
struct PressureSystem {
  std::array<Transform, 3> transform{Transform::Tdma, Transform::Tdma, Transform::Tdma};
  std::shared_ptr<equation::pressure::PressureSolver> engine;   // lazy
};

} // namespace mpmstd::core
