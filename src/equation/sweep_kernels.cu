#include "equation/sweep_kernels.hpp"

#include <cstddef>

#include "common/cuda_check.hpp"

// GPU 백엔드 — sweep_kernels.cpp 의 쌍둥이.

namespace mpmstd::equation {

namespace {

enum : int { kThreads = 256 };

__device__ inline std::size_t at(const int i, const int j, const int k,
                                 const int ny_total, const int nz_total) {
  return (static_cast<std::size_t>(i) * ny_total + j) * nz_total + k;
}

struct InteriorIndex {
  int i, j, k;
  bool valid;
};

__device__ inline InteriorIndex decode(const long long idx, const int h, const int i1,
                                       const int i2, const int i3) {
  InteriorIndex out{0, 0, 0, false};
  if (idx >= static_cast<long long>(i1) * i2 * i3) return out;
  out.k = h + static_cast<int>(idx % i3);
  out.j = h + static_cast<int>((idx / i3) % i2);
  out.i = h + static_cast<int>(idx / (static_cast<long long>(i3) * i2));
  out.valid = true;
  return out;
}

__global__ void scatter_x_kernel(real_t* out, const real_t* D, const int h,
                                 const int i1, const int i2, const int i3,
                                 const int ny_total, const int nz_total) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const InteriorIndex c = decode(idx, h, i1, i2, i3);
  if (!c.valid) return;
  const int nsys = i2 * i3;
  out[at(c.i, c.j, c.k, ny_total, nz_total)] =
      D[static_cast<std::size_t>(c.i - h) * nsys + (c.j - h) * i3 + (c.k - h)];
}

__global__ void scatter_y_kernel(real_t* out, const real_t* D, const int h,
                                 const int i1, const int i2, const int i3,
                                 const int ny_total, const int nz_total) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const InteriorIndex c = decode(idx, h, i1, i2, i3);
  if (!c.valid) return;
  const int nsys = i1 * i3;
  out[at(c.i, c.j, c.k, ny_total, nz_total)] =
      D[static_cast<std::size_t>(c.j - h) * nsys + (c.i - h) * i3 + (c.k - h)];
}

__global__ void scatter_z_kernel(real_t* out, const real_t* D, const int h,
                                 const int i1, const int i2, const int i3,
                                 const int ny_total, const int nz_total) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const InteriorIndex c = decode(idx, h, i1, i2, i3);
  if (!c.valid) return;
  const int nsys = i1 * i2;
  out[at(c.i, c.j, c.k, ny_total, nz_total)] =
      D[static_cast<std::size_t>(c.k - h) * nsys + (c.i - h) * i2 + (c.j - h)];
}

__global__ void add_increment_kernel(const field::View q, const real_t* delta,
                                     const int h, const int i1, const int i2,
                                     const int i3, const int ny_total,
                                     const int nz_total) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const InteriorIndex c = decode(idx, h, i1, i2, i3);
  if (!c.valid) return;
  q(c.i, c.j, c.k) += delta[at(c.i, c.j, c.k, ny_total, nz_total)];
}

inline int blocks_for(const parallel::Subdomain& sub) {
  const long long n = static_cast<long long>(sub.x().n_local) * sub.y().n_local
                      * sub.z().n_local;
  return static_cast<int>((n + kThreads - 1) / kThreads);
}

}  // namespace

void scatter_x(real_t* out, const real_t* D, const parallel::Subdomain& sub) {
  scatter_x_kernel<<<blocks_for(sub), kThreads>>>(
      out, D, halo_width, sub.x().n_local, sub.y().n_local, sub.z().n_local,
      sub.y().n_total, sub.z().n_total);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void scatter_y(real_t* out, const real_t* D, const parallel::Subdomain& sub) {
  scatter_y_kernel<<<blocks_for(sub), kThreads>>>(
      out, D, halo_width, sub.x().n_local, sub.y().n_local, sub.z().n_local,
      sub.y().n_total, sub.z().n_total);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void scatter_z(real_t* out, const real_t* D, const parallel::Subdomain& sub) {
  scatter_z_kernel<<<blocks_for(sub), kThreads>>>(
      out, D, halo_width, sub.x().n_local, sub.y().n_local, sub.z().n_local,
      sub.y().n_total, sub.z().n_total);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

void add_increment(field::Field& q, const real_t* delta,
                   const parallel::Subdomain& sub) {
  add_increment_kernel<<<blocks_for(sub), kThreads>>>(
      q.view(), delta, halo_width, sub.x().n_local, sub.y().n_local,
      sub.z().n_local, sub.y().n_total, sub.z().n_total);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

namespace {

__global__ void clip_interior_kernel(const field::View f, const int h, const int ni,
                                     const int nj, const int nk, const real_t lower,
                                     const real_t upper) {
  const long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (idx >= static_cast<long long>(ni) * nj * nk) return;
  const int k = h + static_cast<int>(idx % nk);
  const int j = h + static_cast<int>((idx / nk) % nj);
  const int i = h + static_cast<int>(idx / (static_cast<long long>(nk) * nj));
  if (f(i, j, k) < lower) f(i, j, k) = lower;
  if (f(i, j, k) > upper) f(i, j, k) = upper;
}

}  // namespace

void clip_interior(field::Field& field_to_clip, const parallel::Subdomain& sub,
                   const real_t lower, const real_t upper) {
  const int ni = sub.x().n_local, nj = sub.y().n_local, nk = sub.z().n_local;
  const long long n = static_cast<long long>(ni) * nj * nk;
  const int threads = 256;
  clip_interior_kernel<<<static_cast<int>((n + threads - 1) / threads), threads>>>(
      field_to_clip.view(), halo_width, ni, nj, nk, lower, upper);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::equation
