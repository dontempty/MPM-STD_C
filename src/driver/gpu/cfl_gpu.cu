#include "driver/cfl.hpp"
#include "core/gpu/gpu_grid.hpp"
#include "common/macros.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>

// GPU CFL time-step computation.
// Mirrors compute_cfl_dt_cpu: per-cell speed = |u|/dx + |v|/dy + |w|/dz,
// reduced to global max via shared-memory tree + atomicCAS, then MPI_Allreduce.

namespace mpmstd::driver {

namespace {

__device__ void atomic_max_double(double* addr, double val) {
  unsigned long long* addr_ull = (unsigned long long*)addr;
  unsigned long long old = *addr_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) break;
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
}

// One kernel: flat interior-cell loop → shared-memory tree → atomicMax into d_max.
// Grid-stride loop so the launch can use a fixed grid (≤ 512 blocks).
__global__ void k_cfl_reduce(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w,
    const double* __restrict__ dx1,
    const double* __restrict__ dx2,
    const double* __restrict__ dx3,
    int n1, int n2, int n3,
    int n1i, int n2i, int n3i, int h,
    double* d_max) {

  extern __shared__ double sdata[];
  const int tid  = threadIdx.x;
  const int ntot = n1i * n2i * n3i;

  double lmax = 0.0;
  for (int idx = blockIdx.x * blockDim.x + tid; idx < ntot; idx += gridDim.x * blockDim.x) {
    const int ii = idx / (n2i * n3i);
    const int jj = (idx / n3i) % n2i;
    const int kk = idx % n3i;
    const int p  = ((h + ii) * n2 + (h + jj)) * n3 + (h + kk);
    const double c = fabs(double(u[p])) / dx1[h + ii]
                   + fabs(double(v[p])) / dx2[h + jj]
                   + fabs(double(w[p])) / dx3[h + kk];
    if (c > lmax) lmax = c;
  }

  sdata[tid] = lmax;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }

  if (tid == 0) atomic_max_double(d_max, sdata[0]);
}

// MPMSTD_CFLDBG=1 diagnostic: given the already-known global max value, find
// (any) one cell attaining it and record its indices — "first writer wins"
// via atomicCAS on a found-flag, no ordering guarantee across ties but good
// enough to answer "is host_max near a wall (small kk/large kk) or bulk".
__global__ void k_cfl_find_argmax(
    const real_t* __restrict__ u,
    const real_t* __restrict__ v,
    const real_t* __restrict__ w,
    const double* __restrict__ dx1,
    const double* __restrict__ dx2,
    const double* __restrict__ dx3,
    int n1, int n2, int n3,
    int n1i, int n2i, int n3i, int h,
    double maxval, int* d_found, int* d_idx3) {
  const int ntot = n1i * n2i * n3i;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ntot; idx += gridDim.x * blockDim.x) {
    const int ii = idx / (n2i * n3i);
    const int jj = (idx / n3i) % n2i;
    const int kk = idx % n3i;
    const int p  = ((h + ii) * n2 + (h + jj)) * n3 + (h + kk);
    const double c = fabs(double(u[p])) / dx1[h + ii]
                   + fabs(double(v[p])) / dx2[h + jj]
                   + fabs(double(w[p])) / dx3[h + kk];
    if (c >= maxval * 0.999999 && atomicCAS(d_found, 0, 1) == 0) {
      d_idx3[0] = ii; d_idx3[1] = jj; d_idx3[2] = kk;
    }
  }
}

// Lazy persistent device scalar (one per rank, allocated once on first call).
double* d_cfl_out = nullptr;

} // anonymous namespace

real_t compute_cfl_dt_gpu(const core::Domain& domain, const core::GpuFields& fields,
                           real_t max_cfl, real_t dt_cap) {
  const auto nt = domain.sub.n_total();
  const int n1 = nt[0], n2 = nt[1], n3 = nt[2];
  const int h   = kHaloWidth;
  const int n1i = n1 - 2*h, n2i = n2 - 2*h, n3i = n3 - 2*h;

  const real_t* u = fields[core::Var::U].data();
  const real_t* v = fields[core::Var::V].data();
  const real_t* w = fields[core::Var::W].data();

  const auto* gg = static_cast<const core::gpu::GpuGridArrays*>(domain.gpu_grid);
  const double* dx1 = gg->d_dx1();
  const double* dx2 = gg->d_dx2();
  const double* dx3 = gg->d_dx3();

  if (!d_cfl_out) cudaMalloc(&d_cfl_out, sizeof(double));
  cudaMemset(d_cfl_out, 0, sizeof(double));

  const int block = 256;
  const int grid  = std::min(512, (n1i * n2i * n3i + block - 1) / block);
  k_cfl_reduce<<<grid, block, block * sizeof(double)>>>(
      u, v, w, dx1, dx2, dx3, n1, n2, n3, n1i, n2i, n3i, h, d_cfl_out);
  cudaDeviceSynchronize();

  double host_max = 0.0;
  cudaMemcpy(&host_max, d_cfl_out, sizeof(double), cudaMemcpyDeviceToHost);
  MPI_Allreduce(MPI_IN_PLACE, &host_max, 1, MPI_DOUBLE, MPI_MAX,
                domain.sub.topology().cart_comm());

  static const bool cfl_dbg = std::getenv("MPMSTD_CFLDBG") != nullptr;
  if (cfl_dbg && host_max > 0.0) {
    static int* d_found = nullptr;
    static int* d_idx3 = nullptr;
    if (!d_found) { cudaMalloc(&d_found, sizeof(int)); cudaMalloc(&d_idx3, 3*sizeof(int)); }
    cudaMemset(d_found, 0, sizeof(int));
    k_cfl_find_argmax<<<grid, block>>>(u, v, w, dx1, dx2, dx3, n1, n2, n3,
                                        n1i, n2i, n3i, h, host_max, d_found, d_idx3);
    cudaDeviceSynchronize();
    int found = 0, idx3[3] = {-1,-1,-1};
    cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(idx3, d_idx3, 3*sizeof(int), cudaMemcpyDeviceToHost);
    int rank; MPI_Comm_rank(domain.sub.topology().cart_comm(), &rank);
    if (found)
      std::fprintf(stderr, "[CFLDBG] rank=%d host_max=%.4f at local(i,j,k)=(%d,%d,%d) of (n1i,n2i,n3i)=(%d,%d,%d) [k near wall if k<<n3i or k>>n3i]\n",
                   rank, host_max, idx3[0], idx3[1], idx3[2], n1i, n2i, n3i);
  }

  const real_t dt_cfl = (host_max > 0.0) ? real_t(double(max_cfl) / host_max) : dt_cap;
  // CFL-limited, capped at dt_cap. Convection is explicit (OB) so dt MUST follow
  // the advective CFL — pinning dt at dt_cap blows up once velocities grow. For a
  // large fixed dt, make convection implicit instead (see Report 18), not a floor here.
  return std::min(dt_cfl, dt_cap);
}

} // namespace mpmstd::driver
