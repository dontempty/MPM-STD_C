#include "equation/pressure/solve.hpp"
#include "equation/pressure/pressure_engine.hpp"   // PressureSolver (complete type)
#include "common/direction.hpp"
#include "common/macros.hpp"

#include <cstddef>
#include <memory>

// P1 pressure step rewired to Domain + Fields. Lazily builds the validated
// pencil-FFT engine (from Domain+BC) and runs the full step on the Fields.

namespace mpmstd::equation::pressure {

void add_pressure_gradient_constant_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom,
                               const core::CpuField& P_acc, real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* p    = P_acc.data();
  const real_t* dmx1 = domain.grid.dmx_ptr(Direction::X);
  const real_t* dmx2 = domain.grid.dmx_ptr(Direction::Y);
  const real_t* dmx3 = domain.grid.dmx_ptr(Direction::Z);
  real_t* du = mom.rhs_u.data();
  real_t* dv = mom.rhs_v.data();
  real_t* dw = mom.rhs_w.data();
  // dU/dV/dW -= dt·∇P_acc  (same staggered-face gradient as project_; ref mom.f90:344-345)
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c = (i * n2 + j) * n3 + k;
        du[c] -= dt * (p[c] - p[((i-1)*n2+j)*n3+k]) / dmx1[i];
        dv[c] -= dt * (p[c] - p[(i*n2+j-1)*n3+k])   / dmx2[j];
        dw[c] -= dt * (p[c] - p[(i*n2+j)*n3+k-1])   / dmx3[k];
      }
}

// NOB incremental-projection predictor: dU/dV/dW -= dt·(1/ρ̂)_face·∇P_acc.
// PaScaL_TCS mom.f90:331/343 — invRhoc is the dx-weighted face mean of invRho:
//   invRhoc = 0.5·(dx(im)·invRho(i) + dx(i)·invRho(im))/dmx(i)
// Mirrors the GPU kernel k_pred_pgrad_var (pressure_gpu.cu) — see Report/32.
void add_pressure_gradient_variable_property_cpu(const core::Domain& domain, equation::momentum::CpuLinearSystem& mom,
                                   const core::CpuField& P_acc, const core::CpuField& inv_rho,
                                   real_t dt) {
  const auto nt = mom.n_total; const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h = kHaloWidth;
  const real_t* p  = P_acc.data();
  const real_t* ir = inv_rho.data();
  const real_t* dx1  = domain.grid.dx_ptr(Direction::X);
  const real_t* dx2  = domain.grid.dx_ptr(Direction::Y);
  const real_t* dx3  = domain.grid.dx_ptr(Direction::Z);
  const real_t* dmx1 = domain.grid.dmx_ptr(Direction::X);
  const real_t* dmx2 = domain.grid.dmx_ptr(Direction::Y);
  const real_t* dmx3 = domain.grid.dmx_ptr(Direction::Z);
  real_t* du = mom.rhs_u.data();
  real_t* dv = mom.rhs_v.data();
  real_t* dw = mom.rhs_w.data();
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int c   = (i * n2 + j) * n3 + k;
        const int cxm = ((i-1) * n2 + j) * n3 + k;
        const int cym = (i * n2 + (j-1)) * n3 + k;
        const int czm = (i * n2 + j) * n3 + (k-1);
        const real_t irx = real_t(0.5) * (dx1[i-1] * ir[c] + dx1[i] * ir[cxm]) / dmx1[i];
        const real_t iry = real_t(0.5) * (dx2[j-1] * ir[c] + dx2[j] * ir[cym]) / dmx2[j];
        const real_t irz = real_t(0.5) * (dx3[k-1] * ir[c] + dx3[k] * ir[czm]) / dmx3[k];
        du[c] -= dt * irx * (p[c] - p[cxm]) / dmx1[i];
        dv[c] -= dt * iry * (p[c] - p[cym]) / dmx2[j];
        dw[c] -= dt * irz * (p[c] - p[czm]) / dmx3[k];
      }
}

// accumulate: Pⁿ⁺¹ = Pⁿ + δp — accumulator Var::P += the system's increment poi.dP.
void accumulate_pressure_cpu(core::CpuFields& fields, const equation::pressure::LinearSystem& poi) {
  real_t*       pa = fields[core::Var::P].data();
  const real_t* dp = poi.dP->data();
  const std::size_t n = fields[core::Var::P].n_elements();
  for (std::size_t q = 0; q < n; ++q) pa[q] += dp[q];   // ref pre.f90:818 P=P+dP
}

void solve_pressure_cpu(const core::Domain& domain, const boundary::BoundaryCondition& bc,
                        core::CpuFields& fields, equation::pressure::LinearSystem& poi, real_t dt,
                        const real_t* inv_rho) {
  if (!poi.pressure_solver)
    poi.pressure_solver = std::make_shared<pressure::PressureSolver>(domain.grid, domain.sub, bc, domain.tdma);
  if (!poi.dP)
    poi.dP = std::make_unique<core::CpuField>(domain.sub, "dP");          // δp scratch (owned by the system)
  if (inv_rho) {
    if (!poi.dPhat)
      poi.dPhat = std::make_unique<core::CpuField>(domain.sub, "dPhat");  // zero-init
    if (!poi.dP0)
      poi.dP0 = std::make_unique<core::CpuField>(domain.sub, "dP0");      // zero-init
    poi.pressure_solver->solve_nob(dt,
                          fields[core::Var::U], fields[core::Var::V],
                          fields[core::Var::W], *poi.dP,
                          inv_rho, *poi.dPhat, *poi.dP0);
  } else {
    poi.pressure_solver->solve(dt, fields[core::Var::U], fields[core::Var::V], fields[core::Var::W], *poi.dP);
  }
}

} // namespace mpmstd::equation::pressure
