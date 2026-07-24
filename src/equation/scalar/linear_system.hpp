#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "parallel/mpi/subdomain.hpp"
#include "common/types.hpp"       // real_t

#include <array>
#include <cstddef>
#include <vector>

namespace mpmstd::equation::scalar {

// Scalar (temperature) linear-system workspace (the Ax=b scratch). Explicit RHS →
// running increment, ADI ping-pong, tridiagonal band scratch, sized once from the
// subdomain and reused every step. assemble_* fills it, solve_* consumes it; the
// increment is added straight back into T. Single-component and no block coupling,
// unlike the momentum workspace. Templated on FieldT only so the Cpu/Gpu variants
// are distinct overload types (the struct body itself is host scratch for now).
template <class FieldT>
struct LinearSystemT {
  std::vector<real_t> rhs, stage;   // explicit RHS → running increment, ADI ping-pong
  std::vector<real_t> A, B, C, D;   // tridiagonal bands [n_row × n_sys]
  std::array<int, 3>  n_total{}, n_interior{};

  explicit LinearSystemT(const parallel::mpi::Subdomain& sub)
    : n_total(sub.n_total()), n_interior(sub.n_interior()) {
    const std::size_t nf  = static_cast<std::size_t>(n_total[0]) * n_total[1] * n_total[2];
    const std::size_t nin = static_cast<std::size_t>(n_interior[0]) * n_interior[1] * n_interior[2];
    rhs.assign(nf, 0); stage.assign(nf, 0);
    A.assign(nin, 0); B.assign(nin, 0); C.assign(nin, 0); D.assign(nin, 0);
  }
};

using CpuLinearSystem = LinearSystemT<core::CpuField>;
using GpuLinearSystem = LinearSystemT<core::GpuField>;

} // namespace mpmstd::equation::scalar
