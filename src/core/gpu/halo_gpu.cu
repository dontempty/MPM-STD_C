#include "core/halo.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdlib>
#include <cstddef>
#include <algorithm>

// ---------------------------------------------------------------------------
//  GPU halo exchange — CUDA-aware MPI throughout.
//
//  Device pointers are passed directly to MPI_Sendrecv; no host staging.
//  Requires HPC-X MPI built with CUDA support (nvhpc-hpcx-cuda12/23.7).
//
//  Field layout: f[i,j,k] = f[i*n2t*n3t + j*n3t + k]  (k fastest).
//
//  Per-axis strategy:
//   X  single-rank periodic : D2D cudaMemcpy  (x-slabs are contiguous)
//   X  multi-rank periodic  : CUDA-aware MPI direct  (no packing, contiguous)
//   Y  single-rank periodic : GPU kernel  (strided y-planes)
//   Y  multi-rank periodic  : D2D cudaMemcpy3D pack → CUDA-aware MPI → D2D unpack
//   Z  single-rank wall     : no exchange
//   Z  MPI wall/periodic    : D2D cudaMemcpy3D pack → CUDA-aware MPI → D2D unpack
// ---------------------------------------------------------------------------

namespace {

__global__ void k_y_periodic_wrap(double* f, int n1t, int n2t, int n3t)
{
    const int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int i = (int)blockIdx.y;
    if (k >= n3t || i >= n1t) return;
    const int base = i * n2t * n3t;
    f[base + 0       * n3t + k] = f[base + (n2t-2) * n3t + k];
    f[base + (n2t-1) * n3t + k] = f[base + 1       * n3t + k];
}

// Lazy device staging buffers (one set per axis, allocated once on first use).
struct DevHaloBufs {
    double *send_lo = nullptr, *send_hi = nullptr;
    double *recv_lo = nullptr, *recv_hi = nullptr;
    std::size_t cap = 0;
    void ensure(std::size_t n) {
        if (n <= cap) return;
        cudaFree(send_lo); cudaFree(send_hi);
        cudaFree(recv_lo); cudaFree(recv_hi);
        cudaMalloc(&send_lo, n * sizeof(double));
        cudaMalloc(&send_hi, n * sizeof(double));
        cudaMalloc(&recv_lo, n * sizeof(double));
        cudaMalloc(&recv_hi, n * sizeof(double));
        cap = n;
    }
};

DevHaloBufs g_y_bufs, g_z_bufs;

} // anonymous namespace

namespace mpmstd::core {

void exchange_halo_gpu(GpuField& field, const Subdomain& sub)
{
    using Direction = mpmstd::Direction;

    const auto nt  = sub.n_total();
    const int n1t  = nt[0], n2t = nt[1], n3t = nt[2];
    double*    f   = field.data();
    const auto& topo = sub.topology();
    static constexpr int kH = 1;

    // ── X axis ───────────────────────────────────────────────────────────────
    // x-slabs (all j,k for fixed i) are contiguous → no packing needed.
    {
        const auto& ax = topo.axis(Direction::X);
        int sz = 1;
        if (ax.comm != MPI_COMM_NULL) MPI_Comm_size(ax.comm, &sz);
        const std::size_t slab = (std::size_t)n2t * n3t;

        if (sz == 1) {
            // Single-rank periodic: local D2D copy
            cudaMemcpy(f,
                       f + (std::size_t)(n1t-2) * slab,
                       slab * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(f + (std::size_t)(n1t-1) * slab,
                       f + slab,
                       slab * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            // Multi-rank: pass device pointers directly (contiguous, no staging)
            MPI_Sendrecv(f + (std::size_t)(n1t-2) * slab, (int)slab, MPI_DOUBLE,
                         ax.east_rank, 10,
                         f,                                (int)slab, MPI_DOUBLE,
                         ax.west_rank, 10,
                         ax.comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(f + slab,                         (int)slab, MPI_DOUBLE,
                         ax.west_rank, 11,
                         f + (std::size_t)(n1t-1) * slab, (int)slab, MPI_DOUBLE,
                         ax.east_rank, 11,
                         ax.comm, MPI_STATUS_IGNORE);
        }
    }

    // ── Y axis ───────────────────────────────────────────────────────────────
    // y-planes (fixed j, all i×k) are strided → pack into contiguous device bufs.
    // cudaMemcpy3D with kind=D2D: slice stride = pitch × height = n3t × n2t per i.
    {
        const auto& ax = topo.axis(Direction::Y);
        int sz = 1;
        if (ax.comm != MPI_COMM_NULL) MPI_Comm_size(ax.comm, &sz);

        if (sz == 1) {
            const dim3 blk(128);
            const dim3 grd((n3t + 127) / 128, n1t);
            k_y_periodic_wrap<<<grd, blk>>>(f, n1t, n2t, n3t);
            cudaDeviceSynchronize();
        } else {
            const std::size_t halo_elems = (std::size_t)n1t * n3t;
            g_y_bufs.ensure(halo_elems);

            // Pack j=jj plane (all i,k) into compact device buffer
            auto pack_y = [&](int jj, double* dst) {
                cudaMemcpy3DParms p = {};
                // src: f[i*n2t*n3t + jj*n3t + k], pitch = n3t (j-row stride), height = n2t
                p.srcPtr = make_cudaPitchedPtr(f + (std::size_t)jj * n3t,
                                               (std::size_t)n3t * sizeof(double), n3t, n2t);
                // dst: compact buffer, 1 row per i-slice
                p.dstPtr = make_cudaPitchedPtr(dst,
                                               (std::size_t)n3t * sizeof(double), n3t, 1);
                p.extent = make_cudaExtent((std::size_t)n3t * sizeof(double), 1, n1t);
                p.kind   = cudaMemcpyDeviceToDevice;
                cudaMemcpy3D(&p);
            };
            pack_y(kH,         g_y_bufs.send_lo);   // j=1       → send to west
            pack_y(n2t-kH-1,   g_y_bufs.send_hi);   // j=n2t-2   → send to east

            const int count = (int)halo_elems;
            // CUDA-aware MPI: device pointers passed directly
            MPI_Sendrecv(g_y_bufs.send_hi, count, MPI_DOUBLE, ax.east_rank, 20,
                         g_y_bufs.recv_lo, count, MPI_DOUBLE, ax.west_rank, 20,
                         ax.comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(g_y_bufs.send_lo, count, MPI_DOUBLE, ax.west_rank, 21,
                         g_y_bufs.recv_hi, count, MPI_DOUBLE, ax.east_rank, 21,
                         ax.comm, MPI_STATUS_IGNORE);

            auto unpack_y = [&](double* src, int jdst, int nbr) {
                if (nbr == MPI_PROC_NULL) return;
                cudaMemcpy3DParms p = {};
                p.srcPtr = make_cudaPitchedPtr(src,
                                               (std::size_t)n3t * sizeof(double), n3t, 1);
                p.dstPtr = make_cudaPitchedPtr(f + (std::size_t)jdst * n3t,
                                               (std::size_t)n3t * sizeof(double), n3t, n2t);
                p.extent = make_cudaExtent((std::size_t)n3t * sizeof(double), 1, n1t);
                p.kind   = cudaMemcpyDeviceToDevice;
                cudaMemcpy3D(&p);
            };
            unpack_y(g_y_bufs.recv_lo, 0,        ax.west_rank);  // → ghost j=0
            unpack_y(g_y_bufs.recv_hi, n2t-kH,   ax.east_rank);  // → ghost j=n2t-1
        }
    }

    // ── Z axis ───────────────────────────────────────────────────────────────
    // z-planes (fixed k, all i×j) are strided → pack into contiguous device bufs.
    {
        const auto& ax = topo.axis(Direction::Z);
        int sz = 1;
        if (ax.comm != MPI_COMM_NULL) MPI_Comm_size(ax.comm, &sz);

        const bool wall_no_mpi = (sz == 1 &&
                                   ax.west_rank == MPI_PROC_NULL &&
                                   ax.east_rank == MPI_PROC_NULL);
        if (!wall_no_mpi) {
            const std::size_t halo_elems = (std::size_t)n1t * n2t * kH;
            g_z_bufs.ensure(halo_elems);

            // Pack k=ksrc plane into compact device buffer
            auto pack_z = [&](int ksrc, double* dst) {
                cudaMemcpy3DParms p = {};
                p.srcPtr = make_cudaPitchedPtr(f + ksrc,
                                               (std::size_t)n3t * sizeof(double), n3t, n2t);
                p.dstPtr = make_cudaPitchedPtr(dst, kH * sizeof(double), kH, n2t);
                p.extent = make_cudaExtent(kH * sizeof(double), n2t, n1t);
                p.kind   = cudaMemcpyDeviceToDevice;
                cudaMemcpy3D(&p);
            };
            pack_z(kH,         g_z_bufs.send_lo);   // k=1       → send to west
            pack_z(n3t-kH-1,   g_z_bufs.send_hi);   // k=n3t-2   → send to east

            const int count = (int)halo_elems;
            // CUDA-aware MPI: device pointers passed directly
            MPI_Sendrecv(g_z_bufs.send_hi, count, MPI_DOUBLE, ax.east_rank, 30,
                         g_z_bufs.recv_lo, count, MPI_DOUBLE, ax.west_rank, 30,
                         ax.comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(g_z_bufs.send_lo, count, MPI_DOUBLE, ax.west_rank, 31,
                         g_z_bufs.recv_hi, count, MPI_DOUBLE, ax.east_rank, 31,
                         ax.comm, MPI_STATUS_IGNORE);

            auto unpack_z = [&](double* src, int kdst, int nbr) {
                if (nbr == MPI_PROC_NULL) return;
                cudaMemcpy3DParms p = {};
                p.srcPtr = make_cudaPitchedPtr(src, kH * sizeof(double), kH, n2t);
                p.dstPtr = make_cudaPitchedPtr(f + kdst,
                                               (std::size_t)n3t * sizeof(double), n3t, n2t);
                p.extent = make_cudaExtent(kH * sizeof(double), n2t, n1t);
                p.kind   = cudaMemcpyDeviceToDevice;
                cudaMemcpy3D(&p);
            };
            unpack_z(g_z_bufs.recv_lo, 0,       ax.west_rank);  // → ghost k=0
            unpack_z(g_z_bufs.recv_hi, n3t-kH,  ax.east_rank);  // → ghost k=n3t-1
        }
    }
}

void bind_gpu_to_local_rank_gpu(const MpiContext& ctx) {
    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err == cudaSuccess && ndev > 0) {
        cudaSetDevice(ctx.node_rank() % ndev);
    }
}

} // namespace mpmstd::core
