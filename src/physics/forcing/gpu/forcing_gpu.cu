#include "physics/forcing/forcing.hpp"
#include "core/domain.hpp"
#include "core/field_gpu.hpp"
#include "core/gpu/gpu_grid.hpp"
#include "common/macros.hpp"

#include <mpi.h>

#ifdef MPMSTD_BACKEND_CUDA
#include <cuda_runtime.h>

namespace {

// Sum cell-centered U × dx1×dx2×dx3 over all interior cells (block reduction).
__global__ void k_reduce_bulk_u(const mpmstd::real_t* U,
                                 const double* dx1, const double* dx2, const double* dx3,
                                 int n1, int n2, int n3, int h, double* d_sum) {
    extern __shared__ double shm[];
    const int tid = threadIdx.x;
    const int ni = n1 - 2*h, nj = n2 - 2*h, nk = n3 - 2*h;
    const int total = ni * nj * nk;
    double local = 0.0;
    for (int idx = (int)(blockIdx.x * blockDim.x) + tid; idx < total;
             idx += (int)(blockDim.x * gridDim.x)) {
        const int ki = idx / (nj * nk), kj = (idx / nk) % nj, kk = idx % nk;
        const int i = ki + h, j = kj + h, k = kk + h;
        const int c = (i * n2 + j) * n3 + k;
        // MAC face-centered U → cell center average
        const double uc = 0.5 * (double(U[c]) + double(U[((i + 1) * n2 + j) * n3 + k]));
        local += uc * dx1[i] * dx2[j] * dx3[k];
    }
    shm[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shm[tid] += shm[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_sum, shm[0]);
}

// Add a constant shift to all interior U face values.
__global__ void k_shift_u(mpmstd::real_t* U, int n1, int n2, int n3, int h,
                           mpmstd::real_t shift) {
    const int ni = n1 - 2*h, nj = n2 - 2*h, nk = n3 - 2*h;
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= ni * nj * nk) return;
    const int ki = idx / (nj * nk), kj = (idx / nk) % nj, kk = idx % nk;
    U[((ki + h) * n2 + (kj + h)) * n3 + (kk + h)] += shift;
}

} // anonymous namespace

namespace mpmstd::physics::forcing {

double channel_bulk_velocity_gpu(const core::Domain& domain,
                                  const core::GpuFields& fields,
                                  double total_vol) {
    const auto& ga = *static_cast<const core::GpuGridArrays*>(domain.gpu_grid);
    const auto nt = domain.sub.n_total();
    const int n1 = nt[0], n2 = nt[1], n3 = nt[2], h = kHaloWidth;

    static double* d_sum = nullptr;
    if (!d_sum) cudaMalloc(&d_sum, sizeof(double));
    cudaMemset(d_sum, 0, sizeof(double));

    const int ni = n1 - 2*h, nj = n2 - 2*h, nk = n3 - 2*h;
    const int block = 256, grid = (ni * nj * nk + block - 1) / block;
    k_reduce_bulk_u<<<grid, block, block * sizeof(double)>>>(
        fields[core::Var::U].data(), ga.d_dx1(), ga.d_dx2(), ga.d_dx3(),
        n1, n2, n3, h, d_sum);

    double local = 0.0;
    cudaMemcpy(&local, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                  domain.sub.topology().cart_comm());
    return global / total_vol;
}

double apply_mass_flow_correction_gpu(const core::Domain& domain,
                                       core::GpuFields& fields,
                                       double Ub_target, double dt,
                                       double total_vol, double& dpdx) {
    const double Ub      = channel_bulk_velocity_gpu(domain, fields, total_vol);
    const double DMpresg = (dt > 1e-15) ? (Ub - Ub_target) / dt : 0.0;
    const real_t shift   = static_cast<real_t>(-dt * DMpresg);

    const auto nt = domain.sub.n_total();
    const int n1 = nt[0], n2 = nt[1], n3 = nt[2], h = kHaloWidth;
    const int ni = n1 - 2*h, nj = n2 - 2*h, nk = n3 - 2*h;
    const int block = 256, grid = (ni * nj * nk + block - 1) / block;
    k_shift_u<<<grid, block>>>(fields[core::Var::U].data(), n1, n2, n3, h, shift);

    dpdx += DMpresg;
    return dpdx;
}

void apply_body_force_gpu(core::GpuFields&, real_t, real_t) {}

} // namespace mpmstd::physics::forcing

#else  // CPU build stubs
namespace mpmstd::physics::forcing {
void apply_body_force_gpu(core::GpuFields&, real_t, real_t) {}
double channel_bulk_velocity_gpu(const core::Domain&, const core::GpuFields&, double) { return 0.0; }
double apply_mass_flow_correction_gpu(const core::Domain&, core::GpuFields&,
                                       double, double, double, double& dpdx) { return dpdx; }
} // namespace mpmstd::physics::forcing
#endif
