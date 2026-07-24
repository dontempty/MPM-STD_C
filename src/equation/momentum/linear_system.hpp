#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "parallel/mpi/subdomain.hpp"
#include "common/direction.hpp"   // Component
#include "common/types.hpp"       // real_t

#include <array>
#include <cstddef>
#include <vector>

namespace mpmstd::equation::momentum {

// Momentum linear-system workspace (the Ax=b scratch for solving U,V,W in one
// pass). OWNS the increments dU,dV,dW (delta-form output) plus the per-component
// explicit RHS / ADI ping-pong / tridiagonal band scratch, all sized once from
// the subdomain and reused every step. assemble_* fills it, solve_* consumes it;
// it is NOT physical state (that lives in core::FieldStore). Templated on the
// field type so the GPU path gets device-resident increments (GpuLinearSystem).
template <class FieldT>
struct LinearSystemT {
  FieldT dU, dV, dW;                                  // increments (owned here)
  std::vector<real_t> rhs_u, rhs_v, rhs_w, stage;     // explicit RHS → running increment, ADI ping-pong
  std::vector<real_t> A, B, C, D;                     // tridiagonal bands [n_row × n_sys]
  std::array<int, 3>  n_total{}, n_interior{};

  explicit LinearSystemT(const parallel::mpi::Subdomain& sub)
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

using CpuLinearSystem = LinearSystemT<core::CpuField>;
using GpuLinearSystem = LinearSystemT<core::GpuField>;

} // namespace mpmstd::equation::momentum
