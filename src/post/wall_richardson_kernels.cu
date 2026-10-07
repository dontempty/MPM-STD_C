#include <cmath>
#include <limits>

#include "common/cuda_check.hpp"
#include "post/wall_richardson.hpp"

// GPU 백엔드 — wall_richardson_kernels.cpp 의 쌍둥이. 열별 |Ri_f| 는 디바이스에서 구해 호스트로 내린다
// (벽 랭크만, nx·ny 개 — 분류·Bcast 는 호스트).

namespace mpmstd::post {

namespace {

__global__ void richardson_kernel(const field::ConstView u, const field::ConstView t,
                                  real_t* out, const int h, const int i1, const int i2,
                                  const int k_first, const int k_ghost, const real_t gap,
                                  const real_t nu, const real_t alpha,
                                  const real_t z_plus_reference) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= i1 * i2) return;
  const int i = h + c / i2, j = h + c % i2;
  const real_t dudz = (u(i, j, k_first) - u(i, j, k_ghost)) / gap;
  const real_t dtdz = (t(i, j, k_first) - t(i, j, k_ghost)) / gap;
  const real_t u_tau_sq = nu * fabs(dudz);
  const real_t heat_flux = alpha * fabs(dtdz);
  const real_t denominator = u_tau_sq * u_tau_sq;
  out[c] = denominator > real_t(0) ? z_plus_reference * nu * heat_flux / denominator
                                   : HUGE_VAL;
}

__global__ void complement_kernel(real_t* o, const real_t* w, const std::size_t n) {
  const std::size_t c = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (c < n) o[c] = 1 - w[c];
}

}  // namespace

void local_flux_richardson(std::vector<real_t>& richardson, const field::Field& U,
                           const field::Field& T, const grid::Grid& grid,
                           const parallel::Subdomain& sub, const real_t nu, const real_t alpha,
                           const int side, const real_t z_plus_reference) {
  const int h = halo_width;
  const int nz = sub.z().n_total;
  const int i1 = sub.x().n_local, i2 = sub.y().n_local;
  const int k_first = side == 0 ? h : nz - h - 1;
  const int k_ghost = side == 0 ? h - 1 : nz - h;
  const real_t gap = grid.z.center[k_first] - grid.z.center[k_ghost];
  const int n = i1 * i2;
  real_t* d = nullptr;
  MPMSTD_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d), n * sizeof(real_t)));
  const int threads = 256;
  richardson_kernel<<<(n + threads - 1) / threads, threads>>>(
      U.view(), T.view(), d, h, i1, i2, k_first, k_ghost, gap, nu, alpha, z_plus_reference);
  MPMSTD_CUDA_CHECK_LAUNCH();
  richardson.resize(n);
  MPMSTD_CUDA_CHECK(cudaMemcpy(richardson.data(), d, n * sizeof(real_t), cudaMemcpyDeviceToHost));
  cudaFree(d);
}

void complement_weights(field::Buffer& other, const field::Buffer& indicator) {
  if (other.size() != indicator.size()) other.assign(indicator.size());
  const std::size_t n = indicator.size();
  const int threads = 256;
  complement_kernel<<<static_cast<int>((n + threads - 1) / threads), threads>>>(
      other.data(), indicator.data(), n);
  MPMSTD_CUDA_CHECK_LAUNCH();
}

}  // namespace mpmstd::post
