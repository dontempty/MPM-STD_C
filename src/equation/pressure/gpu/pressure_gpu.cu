#include "equation/pressure/solve.hpp"
#include "equation/pressure/gpu/pressure_engine_gpu.cuh"  // PressureSolverGpu definition

#include "core/domain.hpp"
#include "core/field_gpu.hpp"
#include "equation/momentum/linear_system.hpp"
#include "equation/pressure/linear_system.hpp"
#include "core/gpu/gpu_grid.hpp"
#include "common/macros.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace mpmstd::equation::pressure {

namespace {

// Incremental-projection predictor: dU/dV/dW -= dt·∇P_acc on the staggered faces
// (identical gradient to k_project_ob_pres). Matches PaScaL_TCS mom.f90:344-345.
__global__ void k_pred_pgrad(double* dU, double* dV, double* dW, const double* P,
    const double* dmx1, const double* dmx2, const double* dmx3,
    int n2t, int n3t, int nx, int ny, int nz, int h, double dt) {
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int ii = blockIdx.z;
  if (kk >= nz || jj >= ny) return;
  const int i = ii + h, j = jj + h, k = kk + h;
  const int c = (i * n2t + j) * n3t + k;
  const double Pc = P[c];
  dU[c] -= dt * (Pc - P[((i-1)*n2t+j)*n3t+k]) / dmx1[i];
  dV[c] -= dt * (Pc - P[(i*n2t+j-1)*n3t+k])   / dmx2[j];
  dW[c] -= dt * (Pc - P[(i*n2t+j)*n3t+k-1])   / dmx3[k];
}

// NOB incremental-projection predictor: dU/dV/dW -= dt·(1/ρ̂)_face·∇P_acc.
// PaScaL_TCS mom.f90:331/343 — invRhoc is the dx-weighted face mean of invRho:
//   invRhoc = 0.5·(dx(im)·invRho(i) + dx(i)·invRho(im))/dmx(i)
// The OB kernel above omits invRhoc (≡1 there); using it for NOB drops the
// (1/ρ̂) weight and degrades the projection to first order in time (Report/32).
__global__ void k_pred_pgrad_var(double* dU, double* dV, double* dW,
    const double* P, const double* inv_rho,
    const double* dx1, const double* dx2, const double* dx3,
    const double* dmx1, const double* dmx2, const double* dmx3,
    int n2t, int n3t, int nx, int ny, int nz, int h, double dt) {
  const int kk = blockIdx.x * blockDim.x + threadIdx.x;
  const int jj = blockIdx.y * blockDim.y + threadIdx.y;
  const int ii = blockIdx.z;
  if (kk >= nz || jj >= ny) return;
  const int i = ii + h, j = jj + h, k = kk + h;
  const int c   = (i * n2t + j) * n3t + k;
  const int cxm = ((i-1) * n2t + j) * n3t + k;
  const int cym = (i * n2t + (j-1)) * n3t + k;
  const int czm = (i * n2t + j) * n3t + (k-1);
  const double Pc  = P[c];
  const double irx = 0.5 * (dx1[i-1] * inv_rho[c] + dx1[i] * inv_rho[cxm]) / dmx1[i];
  const double iry = 0.5 * (dx2[j-1] * inv_rho[c] + dx2[j] * inv_rho[cym]) / dmx2[j];
  const double irz = 0.5 * (dx3[k-1] * inv_rho[c] + dx3[k] * inv_rho[czm]) / dmx3[k];
  dU[c] -= dt * irx * (Pc - P[cxm]) / dmx1[i];
  dV[c] -= dt * iry * (Pc - P[cym]) / dmx2[j];
  dW[c] -= dt * irz * (Pc - P[czm]) / dmx3[k];
}

// P_acc += δp over all cells (ghost/halo included — linear, stays consistent).
__global__ void k_pacc_add(double* Pacc, const double* dP, long n) {
  const long q = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (q < n) Pacc[q] += dP[q];
}

} // anonymous namespace

void add_pressure_gradient_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom,
                               const core::GpuField& P_acc, real_t dt) {
  const auto ni = domain.sub.n_interior();
  const auto nt = domain.sub.n_total();
  const int nx = ni[0], ny = ni[1], nz = ni[2];
  const int n2t = nt[1], n3t = nt[2], h = kHaloWidth;
  const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
  const dim3 blk(64, 4, 1);
  const dim3 grd((nz + 63) / 64, (ny + 3) / 4, nx);
  k_pred_pgrad<<<grd, blk>>>(mom.dU.data(), mom.dV.data(), mom.dW.data(), P_acc.data(),
      dg->d_dmx1(), dg->d_dmx2(), dg->d_dmx3(), n2t, n3t, nx, ny, nz, h,
      static_cast<double>(dt));
}

void add_pressure_gradient_nob_gpu(const core::Domain& domain, equation::momentum::GpuLinearSystem& mom,
                                   const core::GpuField& P_acc, const core::GpuField& inv_rho,
                                   real_t dt) {
  const auto ni = domain.sub.n_interior();
  const auto nt = domain.sub.n_total();
  const int nx = ni[0], ny = ni[1], nz = ni[2];
  const int n2t = nt[1], n3t = nt[2], h = kHaloWidth;
  const auto* dg = static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
  const dim3 blk(64, 4, 1);
  const dim3 grd((nz + 63) / 64, (ny + 3) / 4, nx);
  k_pred_pgrad_var<<<grd, blk>>>(mom.dU.data(), mom.dV.data(), mom.dW.data(),
      P_acc.data(), inv_rho.data(),
      dg->d_dx1(), dg->d_dx2(), dg->d_dx3(),
      dg->d_dmx1(), dg->d_dmx2(), dg->d_dmx3(),
      n2t, n3t, nx, ny, nz, h, static_cast<double>(dt));
}

// accumulate: Pⁿ⁺¹ = Pⁿ + δp, i.e. the accumulator Var::P += the system's increment poi.dP.
void accumulate_pressure_gpu(core::GpuFields& fields, const equation::pressure::LinearSystem& poi) {
  core::GpuField& P_acc = fields[core::Var::P];
  const auto& dP = *static_cast<const core::GpuField*>(poi.dP_gpu_field.get());
  const long n = static_cast<long>(P_acc.n_elements());
  const int blk = 256;
  const int grd = static_cast<int>((n + blk - 1) / blk);
  k_pacc_add<<<grd, blk>>>(P_acc.data(), dP.data(), n);
}

// Lazy-allocate the system-owned δp GPU field (zero-init), like dPhat/dP0.
static core::GpuField& ensure_dP_gpu(equation::pressure::LinearSystem& poi,
                                     const parallel::mpi::Subdomain& sub) {
  if (!poi.dP_gpu_field) {
    auto* f = new core::GpuField(sub, "dP_gpu");
    CUDA_CHK(cudaMemset(f->data(), 0, f->n_elements() * sizeof(real_t)));
    poi.dP_gpu_field = std::shared_ptr<void>(f,
        [](void* p){ delete static_cast<core::GpuField*>(p); });
  }
  return *static_cast<core::GpuField*>(poi.dP_gpu_field.get());
}

// ── OB GPU pressure solve ─────────────────────────────────────────────────────

void solve_pressure_gpu(const core::Domain&             domain,
                        const boundary::BoundaryCondition&  bc,
                        core::GpuFields&                gpu_fields,
                        equation::pressure::LinearSystem&           poi,
                        real_t                          dt)
{
  using Eng = pressure::PressureSolverGpu;

  if (!poi.pressure_solver_gpu) {
    poi.pressure_solver_gpu = std::make_shared<Eng>(domain.grid, domain.sub, bc, domain.tdma);
  }
  auto& eng = *std::static_pointer_cast<Eng>(poi.pressure_solver_gpu);
  core::GpuField& dP = ensure_dP_gpu(poi, domain.sub);
  eng.solve(dt, gpu_fields, bc, domain.sub, dP);
}


// ── NOB GPU pressure solve ────────────────────────────────────────────────────

void solve_pressure_nob_gpu(const core::Domain&             domain,
                             const boundary::BoundaryCondition&  bc,
                             core::GpuFields&                gpu_fields,
                             equation::pressure::LinearSystem&           poi,
                             real_t                          dt,
                             const core::GpuField&           inv_rho)
{
  using Eng = pressure::PressureSolverGpu;

  if (!poi.pressure_solver_gpu) {
    poi.pressure_solver_gpu = std::make_shared<Eng>(domain.grid, domain.sub, bc, domain.tdma);
  }

  // Lazy-init dPhat + dP0 GPU fields (zero-initialized)
  if (!poi.dPhat_gpu_field) {
    auto* f = new core::GpuField(domain.sub, "dPhat_gpu");
    CUDA_CHK(cudaMemset(f->data(), 0, f->n_elements() * sizeof(real_t)));
    poi.dPhat_gpu_field = std::shared_ptr<void>(f,
        [](void* p){ delete static_cast<core::GpuField*>(p); });
  }
  if (!poi.dP0_gpu_field) {
    auto* f = new core::GpuField(domain.sub, "dP0_gpu");
    CUDA_CHK(cudaMemset(f->data(), 0, f->n_elements() * sizeof(real_t)));
    poi.dP0_gpu_field = std::shared_ptr<void>(f,
        [](void* p){ delete static_cast<core::GpuField*>(p); });
  }

  auto& eng   = *std::static_pointer_cast<Eng>(poi.pressure_solver_gpu);
  auto& dPhat = *static_cast<core::GpuField*>(poi.dPhat_gpu_field.get());
  auto& dP0   = *static_cast<core::GpuField*>(poi.dP0_gpu_field.get());
  core::GpuField& dP = ensure_dP_gpu(poi, domain.sub);
  eng.solve_nob(dt, gpu_fields, bc, domain.sub, inv_rho, dPhat, dP0, dP);
}

} // namespace mpmstd::equation::pressure
