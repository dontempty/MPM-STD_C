#pragma once

#include "core/field_cpu.hpp"
#include "core/field_gpu.hpp"
#include "core/mpi_topology.hpp"   // Subdomain
#include "common/types.hpp"

#include <array>
#include <memory>
#include <string>

namespace mpmstd::core {

// =============================================================================
// Variable container (rev.2 structural redesign — see docs/STRUCTURE_REDESIGN.md)
// -----------------------------------------------------------------------------
// The user registers, BEFORE the time loop, exactly which PHYSICAL variables and
// constants the case uses; all are passed to the solver as one `Fields` object
// instead of one-by-one. Access is by a TYPE-SAFE enum key (Var/Const) — the
// lightweight, compile-checked successor to the dropped string-keyed
// FieldRegistry. Solver-internal scratch (increments dU,dV,dW; band/FFT systems)
// is NOT here — momentum owns its own increments.
//
// FieldT is CpuField or GpuField (rev.2 C1: separate types). In GPU mode the
// whole store is GpuFields → every variable is device-resident; data returns to
// the host only at fileout (GpuField::to_host).
// =============================================================================

enum class Var   : int { U, V, W, P, T, Count };   // spatially-varying fields
enum class Const : int { nu, alpha_T, Count };      // scalar constants

constexpr int to_idx(Var v)   { return static_cast<int>(v); }
constexpr int to_idx(Const c) { return static_cast<int>(c); }

constexpr const char* var_name(Var v) {
  switch (v) {
    case Var::U: return "U"; case Var::V: return "V"; case Var::W: return "W";
    case Var::P: return "P"; case Var::T: return "T"; default: return "?";
  }
}

// Thin scalar parameter (host). Uniform with Field so the container holds both;
// a Constant may later become a Field (NOB: nu → mu(T)). A scalar needs no device
// storage — it is passed to kernels by value.
struct Constant {
  real_t value = real_t{0};
};

// Container of the registered fields + constants, keyed by Var/Const. Fields are
// heap-owned (unique_ptr) so the move-only GpuField works uniformly. Only what
// is registered exists (rev.2 "allocate only what you use").
template <class FieldT>
class FieldStore {
public:
  FieldT& add(Var v, const Subdomain& sub) {
    auto& slot = fields_[to_idx(v)];
    if (!slot) slot = std::make_unique<FieldT>(sub, std::string(var_name(v)));
    return *slot;
  }
  void add_constant(Const c, real_t value) { consts_[to_idx(c)].value = value; }

  bool has(Var v) const { return static_cast<bool>(fields_[to_idx(v)]); }

  FieldT&       operator[](Var v)       { return *fields_[to_idx(v)]; }
  const FieldT& operator[](Var v) const { return *fields_[to_idx(v)]; }

  real_t constant(Const c) const { return consts_[to_idx(c)].value; }

private:
  std::array<std::unique_ptr<FieldT>, static_cast<int>(Var::Count)> fields_{};
  std::array<Constant, static_cast<int>(Const::Count)>              consts_{};
};

using CpuFields = FieldStore<CpuField>;
using GpuFields = FieldStore<GpuField>;

} // namespace mpmstd::core
