#pragma once

#include "core/bands.hpp"
#include "common/direction.hpp"   // Direction, Component, to_int

#include <array>
#include <memory>

// The pressure-Poisson engine (FFT/transpose/TDMA state) lives in the pressure
// layer; PressureSystem just holds it (rev.2 §4: System holds solver state).
namespace mpmstd::equation::pressure { class PressureSolver; }

namespace mpmstd::core {

// =============================================================================
// *System  — the Ax=b bundle for one equation. assemble_* fills it; solve_*
//            consumes it. One per equation type (rev.2 §4). For the spike we
//            pin the SHAPE / signature; full per-field content lands in P1.
// =============================================================================

// One scalar transport equation: a Bands per sweep direction (3-stage ADI).
struct ScalarSystem {
  Bands x, y, z;

  Bands& along(Direction d) {
    return d == Direction::X ? x : (d == Direction::Y ? y : z);
  }
  const Bands& along(Direction d) const {
    return d == Direction::X ? x : (d == Direction::Y ? y : z);
  }
};

// Momentum (rev.2 M2): solve_momentum does U,V,W + the block lower-triangular
// velocity coupling (blockLdV/blockLdU) in one. For the CPU port (P1) the system
// carries the per-component explicit RHS / running increment, the ADI ping-pong
// buffer, and the shared tridiagonal band workspace — allocated once from the
// grid extents. [Device-resident bands for the GPU land in P4/P5.]
struct MomentumSystem {
  std::vector<real_t> rhs_u, rhs_v, rhs_w;   // explicit RHS → running increment (n_total³)
  std::vector<real_t> stage;                 // ADI ping-pong buffer (n_total³)
  std::vector<real_t> A, B, C, D;            // tridiagonal bands (n_interior³, [n_row×n_sys])
  std::array<int, 3>  n_total{}, n_interior{};

  void allocate(std::array<int, 3> nt, std::array<int, 3> ni) {
    if (n_total == nt && n_interior == ni && !rhs_u.empty()) return;
    n_total = nt; n_interior = ni;
    const std::size_t nf  = static_cast<std::size_t>(nt[0]) * nt[1] * nt[2];
    const std::size_t nin = static_cast<std::size_t>(ni[0]) * ni[1] * ni[2];
    rhs_u.assign(nf, 0); rhs_v.assign(nf, 0); rhs_w.assign(nf, 0); stage.assign(nf, 0);
    A.assign(nin, 0); B.assign(nin, 0); C.assign(nin, 0); D.assign(nin, 0);
  }

  std::vector<real_t>& rhs(Component c) {
    return c == Component::U ? rhs_u : (c == Component::V ? rhs_v : rhs_w);
  }
};

// Per-axis spectral transform used by the pressure Poisson solve. Chosen from
// the BC (rev.2 §9c): periodic -> Fft, Neumann/wall on a uniform axis -> Dct,
// the remaining axis -> Tdma.
enum class Transform { Fft, Dct, Tdma };

// Pressure-Poisson system. Holds the per-axis transform tags + the heavy solver
// engine (FFTW plans, pencil buffers, wavenumbers, distributed z-TDMA), built
// lazily on the first solve_pressure_cpu and reused thereafter.
struct PressureSystem {
  std::array<Transform, 3> transform{Transform::Tdma, Transform::Tdma, Transform::Tdma};
  std::shared_ptr<equation::pressure::PressureSolver> engine;   // lazy; built by solve_pressure_cpu
};

} // namespace mpmstd::core
