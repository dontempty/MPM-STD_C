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

// [최적화 2026-10-18] scatter_z 32×32 타일 전치(z 밴드 j 연속 ↔ 필드 k 연속; 1차원 매핑은 보폭 읽기로 scatter_x 의 3.7 배).
//   accumulate 가 있으면 q += δ 도 같이(별도 add_increment 패스 제거). 비트 동일. 필드당 1.36 → 0.39 ms.
constexpr int kTileZ = 32;
constexpr int kRowsZ = 8;

__global__ void scatter_z_kernel(real_t* out, real_t* accumulate, const real_t* D,
                                 const int h, const int i1, const int i2, const int i3,
                                 const int ny_total, const int nz_total) {
  __shared__ real_t tile[kTileZ][kTileZ + 1];
  const int ii = static_cast<int>(blockIdx.z);
  const int j0 = static_cast<int>(blockIdx.y) * kTileZ;
  const int k0 = static_cast<int>(blockIdx.x) * kTileZ;
  const int tx = static_cast<int>(threadIdx.x), ty = static_cast<int>(threadIdx.y);
  const std::size_t nsys = static_cast<std::size_t>(i1) * i2;
  for (int r = 0; r < kTileZ; r += kRowsZ) {          // ① 읽기: tx 가 j 를 걷는다
    const int jj = j0 + tx, kk = k0 + ty + r;
    if (jj < i2 && kk < i3) tile[ty + r][tx] = D[kk * nsys + ii * i2 + jj];
  }
  __syncthreads();
  for (int r = 0; r < kTileZ; r += kRowsZ) {          // ② 쓰기: tx 가 k 를 걷는다
    const int kk = k0 + tx, jj = j0 + ty + r;
    if (jj < i2 && kk < i3) {
      const real_t v = tile[tx][ty + r];
      const std::size_t g = at(ii + h, jj + h, kk + h, ny_total, nz_total);
      if (out) out[g] = v;
      if (accumulate) accumulate[g] += v;
    }
  }
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

void scatter_z(real_t* out, const real_t* D, const parallel::Subdomain& sub,
               real_t* accumulate) {
  const int i1 = sub.x().n_local, i2 = sub.y().n_local, i3 = sub.z().n_local;
  const dim3 threads(kTileZ, kRowsZ);
  const dim3 blocks((i3 + kTileZ - 1) / kTileZ, (i2 + kTileZ - 1) / kTileZ, i1);
  scatter_z_kernel<<<blocks, threads>>>(out, accumulate, D, halo_width, i1, i2, i3,
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
