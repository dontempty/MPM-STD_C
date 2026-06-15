#include "core/boundary_ops.hpp"
#include "core/halo.hpp"          // exchange_halo_gpu
#include "common/macros.hpp"      // kHaloWidth

#include <cuda_runtime.h>
#include <mpi.h>

// GPU ghost-cell BC application (rev.2 §6 GPU build only).
// Mirrors boundary_ops_cpu.cpp logic: host determines which faces are global
// boundaries, then launches a 2-D CUDA kernel to fill the ghost plane in-place.

namespace mpmstd::core {

using boundary::BcKind;
using boundary::GhostPolicy;
using boundary::FaceBc;

namespace {

bool is_global_boundary(const Subdomain& sub, Direction d, Side s) {
  const auto& axis = sub.topology().axis(d);
  return (s == Side::Minus) ? axis.west_rank == MPI_PROC_NULL
                            : axis.east_rank == MPI_PROC_NULL;
}

// ---------------------------------------------------------------------------
// Kernels: fill one ghost plane in the linearised C-order buffer
// (i*n2 + j)*n3 + k.  Each kernel is launched with a 2-D grid covering the
// two free dimensions of the plane.
// mode: 0 = ZeroGhost (phi = vw), 1 = Antisymmetric (phi = 2*vw - phi_in),
//        2 = Neumann   (phi = phi_in + vw, caller pre-multiplied sign into vw)
// ---------------------------------------------------------------------------

__global__ void k_fill_x_ghost(real_t* phi, int n2, int n3,
                                int i_ghost, int i_inner,
                                real_t vw, int mode) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= n2 || k >= n3) return;
  const real_t phi_in = phi[(i_inner * n2 + j) * n3 + k];
  real_t val;
  if (mode == 1) val = real_t(2) * vw - phi_in;
  else if (mode == 2) val = phi_in + vw;
  else val = vw;
  phi[(i_ghost * n2 + j) * n3 + k] = val;
}

__global__ void k_fill_y_ghost(real_t* phi, int n1, int n2, int n3,
                                int j_ghost, int j_inner,
                                real_t vw, int mode) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n1 || k >= n3) return;
  const real_t phi_in = phi[(i * n2 + j_inner) * n3 + k];
  real_t val;
  if (mode == 1) val = real_t(2) * vw - phi_in;
  else if (mode == 2) val = phi_in + vw;
  else val = vw;
  phi[(i * n2 + j_ghost) * n3 + k] = val;
}

__global__ void k_fill_z_ghost(real_t* phi, int n1, int n2, int n3,
                                int k_ghost, int k_inner,
                                real_t vw, int mode) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n1 || j >= n2) return;
  const real_t phi_in = phi[(i * n2 + j) * n3 + k_inner];
  real_t val;
  if (mode == 1) val = real_t(2) * vw - phi_in;
  else if (mode == 2) val = phi_in + vw;
  else val = vw;
  phi[(i * n2 + j) * n3 + k_ghost] = val;
}

// Determine fill mode + effective wall value from FaceBc.
int ghost_mode(const FaceBc& bc, Side s, real_t& vw_out) {
  if (bc.kind == BcKind::Neumann) {
    const real_t sgn = (s == Side::Minus) ? real_t{-1} : real_t{1};
    vw_out = sgn * bc.value;
    return 2;
  }
  vw_out = bc.value;
  return (bc.ghost_policy == GhostPolicy::Antisymmetric) ? 1 : 0;
}

void launch_ghost_kernel(real_t* phi, Direction d, Side s, const FaceBc& bc,
                         int n1, int n2, int n3) {
  real_t vw = real_t{0};
  const int mode = ghost_mode(bc, s, vw);
  const dim3 blk(16, 16);

  if (d == Direction::X) {
    const int g  = (s == Side::Minus) ? 0 : n1 - 1;
    const int in = (s == Side::Minus) ? kHaloWidth : n1 - 1 - kHaloWidth;
    const dim3 grd((n2 + blk.x - 1)/blk.x, (n3 + blk.y - 1)/blk.y);
    k_fill_x_ghost<<<grd, blk>>>(phi, n2, n3, g, in, vw, mode);
  } else if (d == Direction::Y) {
    const int g  = (s == Side::Minus) ? 0 : n2 - 1;
    const int in = (s == Side::Minus) ? kHaloWidth : n2 - 1 - kHaloWidth;
    const dim3 grd((n1 + blk.x - 1)/blk.x, (n3 + blk.y - 1)/blk.y);
    k_fill_y_ghost<<<grd, blk>>>(phi, n1, n2, n3, g, in, vw, mode);
  } else {
    const int g  = (s == Side::Minus) ? 0 : n3 - 1;
    const int in = (s == Side::Minus) ? kHaloWidth : n3 - 1 - kHaloWidth;
    const dim3 grd((n1 + blk.x - 1)/blk.x, (n2 + blk.y - 1)/blk.y);
    k_fill_z_ghost<<<grd, blk>>>(phi, n1, n2, n3, g, in, vw, mode);
  }
}

} // anonymous namespace

void apply_ghost_gpu(GpuField& phi, const FieldBoundary& fbc, const Subdomain& sub, real_t /*t*/) {
  const int n1 = phi.n_total(Direction::X);
  const int n2 = phi.n_total(Direction::Y);
  const int n3 = phi.n_total(Direction::Z);
  real_t* d = phi.data();

  for (int a = 0; a < 3; ++a) {
    const Direction dir = static_cast<Direction>(a);
    for (Side s : {Side::Minus, Side::Plus}) {
      if (!is_global_boundary(sub, dir, s)) continue;
      const FaceBc& bc = fbc.face(dir, s);
      if (bc.kind == BcKind::Periodic) continue;
      launch_ghost_kernel(d, dir, s, bc, n1, n2, n3);
    }
  }
  cudaDeviceSynchronize();
}

void sync_field_gpu(GpuField& field, const FieldBoundary& fbc, const Subdomain& sub, real_t t) {
  exchange_halo_gpu(field, sub);
  apply_ghost_gpu(field, fbc, sub, t);
}

void sync_field_gpu(GpuFields& fields, Var v, const BoundaryCondition& bc, const Subdomain& sub) {
  sync_field_gpu(fields[v], face_bc_for(bc, v), sub);
}

} // namespace mpmstd::core
