#include "equation/momentum/kernels.hpp"
#include "common/macros.hpp"

// NOB transpose viscous stress (cross-stress) — the ∇μ-proportional piece that
// the diagonal-only variable-μ RHS drops. For momentum component i:
//     T_i = Σ_j ∂_j( μ ∂_i u_j )        (the ∇u^T half of ∇·(μ(∇u+∇u^T)))
// At μ=const this equals ∂_i(∇·u) ≈ 0 (discretely ~1e-14 after projection), so
// adding it leaves the μ≡1 / OB validation intact while supplying the stress
// that stabilises the strong glycerol viscosity contrast (NOB_RBC_SPEC.md §3).
//
// Combined with the diagonal own-direction term already in compute_mpmstd_rhs_var,
// the own direction is effectively doubled (2·∂_i(μ∂_i u_i)), matching the
// reference 2× factor. Ported from PaScaL_TCS module_solve_momentum.f90
// (dU :334-339, dV :690-695, dW) into the mpmstd staggering with z = wall-normal:
//   reference x1→my x (stream, U), x2(wall-normal)→my z (W), x3(span)→my y (V).
// μ is volume-weighted to the edge between the component face and the cross face.

namespace mpmstd::equation::momentum::kernels {

namespace {
MPMSTD_HD int ix(int i, int j, int k, int n2, int n3) { return (i * n2 + j) * n3 + k; }
constexpr real_t kHalf = real_t{0.5};
} // anonymous namespace

// ---- U-momentum (u on x-face i; cross y span, z wall-normal) ----------------
//   T_u = ∂x(μ∂xu) + ∂y(μ∂xv) + ∂z(μ∂xw)
void add_transpose_stress_u_cpu(real_t* rhs_u, const real_t* U, const real_t* V, const real_t* W,
    const real_t* mu, const real_t* invrho,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t* dx3, const real_t* dmx3, int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int im = i-1, ip = i+1, jm = j-1, jp = j+1, km = k-1, kp = k+1;
        const int c = ix(i, j, k, n2, n3);
        // own-x: ∂x(μ∂xu), face-μ (→ doubles the diagonal own term)
        const real_t mxp = kHalf * (mu[c] + mu[ix(ip,j,k,n2,n3)]);
        const real_t mxm = kHalf * (mu[ix(im,j,k,n2,n3)] + mu[c]);
        const real_t ox = ( mxp * (U[ix(ip,j,k,n2,n3)] - U[c]) / dmx1[i+1]
                          - mxm * (U[c] - U[ix(im,j,k,n2,n3)]) / dmx1[i] ) / dx1[i];
        // ∂y(μ∂xv): ∂xv at (x-face i, y-face j/jp); μ x-interp→face i then y-interp
        const real_t dvdx_j  = (V[ix(i,j ,k,n2,n3)] - V[ix(im,j ,k,n2,n3)]) / dmx1[i];
        const real_t dvdx_jp = (V[ix(i,jp,k,n2,n3)] - V[ix(im,jp,k,n2,n3)]) / dmx1[i];
        const real_t mxa_y = kHalf * (dx1[im]*mu[ix(i,jm,k,n2,n3)] + dx1[i]*mu[ix(im,jm,k,n2,n3)]) / dmx1[i];
        const real_t mxc_y = kHalf * (dx1[im]*mu[ix(i,j ,k,n2,n3)] + dx1[i]*mu[ix(im,j ,k,n2,n3)]) / dmx1[i];
        const real_t mxb_y = kHalf * (dx1[im]*mu[ix(i,jp,k,n2,n3)] + dx1[i]*mu[ix(im,jp,k,n2,n3)]) / dmx1[i];
        const real_t mu_yj  = kHalf * (dx2[jm]*mxc_y + dx2[j]*mxa_y) / dmx2[j];
        const real_t mu_yjp = kHalf * (dx2[jp]*mxc_y + dx2[j]*mxb_y) / dmx2[jp];
        const real_t oy = (mu_yjp * dvdx_jp - mu_yj * dvdx_j) / dx2[j];
        // ∂z(μ∂xw): ∂xw at (x-face i, z-face k/kp); μ x-interp→face i then z-interp
        const real_t dwdx_k  = (W[ix(i,j,k ,n2,n3)] - W[ix(im,j,k ,n2,n3)]) / dmx1[i];
        const real_t dwdx_kp = (W[ix(i,j,kp,n2,n3)] - W[ix(im,j,kp,n2,n3)]) / dmx1[i];
        const real_t mxa_z = kHalf * (dx1[im]*mu[ix(i,j,km,n2,n3)] + dx1[i]*mu[ix(im,j,km,n2,n3)]) / dmx1[i];
        const real_t mxc_z = kHalf * (dx1[im]*mu[ix(i,j,k ,n2,n3)] + dx1[i]*mu[ix(im,j,k ,n2,n3)]) / dmx1[i];
        const real_t mxb_z = kHalf * (dx1[im]*mu[ix(i,j,kp,n2,n3)] + dx1[i]*mu[ix(im,j,kp,n2,n3)]) / dmx1[i];
        const real_t mu_zk  = kHalf * (dx3[km]*mxc_z + dx3[k]*mxa_z) / dmx3[k];
        const real_t mu_zkp = kHalf * (dx3[kp]*mxc_z + dx3[k]*mxb_z) / dmx3[kp];
        const real_t oz = (mu_zkp * dwdx_kp - mu_zk * dwdx_k) / dx3[k];
        const real_t irc = kHalf * (dx1[im]*invrho[c] + dx1[i]*invrho[ix(im,j,k,n2,n3)]) / dmx1[i];
        rhs_u[c] += dt * Cmu * irc * (ox + oy + oz);
      }
}

// ---- V-momentum (v on y-face j; cross x stream, z wall-normal) --------------
//   T_v = ∂x(μ∂yu) + ∂y(μ∂yv) + ∂z(μ∂yw)
void add_transpose_stress_v_cpu(real_t* rhs_v, const real_t* U, const real_t* V, const real_t* W,
    const real_t* mu, const real_t* invrho,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t* dx3, const real_t* dmx3, int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int im = i-1, ip = i+1, jm = j-1, jp = j+1, km = k-1, kp = k+1;
        const int c = ix(i, j, k, n2, n3);
        // own-y: ∂y(μ∂yv), face-μ
        const real_t myp = kHalf * (mu[c] + mu[ix(i,jp,k,n2,n3)]);
        const real_t mym = kHalf * (mu[ix(i,jm,k,n2,n3)] + mu[c]);
        const real_t oy = ( myp * (V[ix(i,jp,k,n2,n3)] - V[c]) / dmx2[j+1]
                          - mym * (V[c] - V[ix(i,jm,k,n2,n3)]) / dmx2[j] ) / dx2[j];
        // ∂x(μ∂yu): ∂yu at (y-face j, x-face i/ip); μ y-interp→face j then x-interp
        const real_t dudy_i  = (U[ix(i ,j,k,n2,n3)] - U[ix(i ,jm,k,n2,n3)]) / dmx2[j];
        const real_t dudy_ip = (U[ix(ip,j,k,n2,n3)] - U[ix(ip,jm,k,n2,n3)]) / dmx2[j];
        const real_t mya_x = kHalf * (dx2[jm]*mu[ix(im,j,k,n2,n3)] + dx2[j]*mu[ix(im,jm,k,n2,n3)]) / dmx2[j];
        const real_t myc_x = kHalf * (dx2[jm]*mu[ix(i ,j,k,n2,n3)] + dx2[j]*mu[ix(i ,jm,k,n2,n3)]) / dmx2[j];
        const real_t myb_x = kHalf * (dx2[jm]*mu[ix(ip,j,k,n2,n3)] + dx2[j]*mu[ix(ip,jm,k,n2,n3)]) / dmx2[j];
        const real_t mu_xi  = kHalf * (dx1[im]*myc_x + dx1[i]*mya_x) / dmx1[i];
        const real_t mu_xip = kHalf * (dx1[ip]*myc_x + dx1[i]*myb_x) / dmx1[ip];
        const real_t ox = (mu_xip * dudy_ip - mu_xi * dudy_i) / dx1[i];
        // ∂z(μ∂yw): ∂yw at (y-face j, z-face k/kp); μ y-interp→face j then z-interp
        const real_t dwdy_k  = (W[ix(i,j,k ,n2,n3)] - W[ix(i,jm,k ,n2,n3)]) / dmx2[j];
        const real_t dwdy_kp = (W[ix(i,j,kp,n2,n3)] - W[ix(i,jm,kp,n2,n3)]) / dmx2[j];
        const real_t mya_z = kHalf * (dx2[jm]*mu[ix(i,j,km,n2,n3)] + dx2[j]*mu[ix(i,jm,km,n2,n3)]) / dmx2[j];
        const real_t myc_z = kHalf * (dx2[jm]*mu[ix(i,j,k ,n2,n3)] + dx2[j]*mu[ix(i,jm,k ,n2,n3)]) / dmx2[j];
        const real_t myb_z = kHalf * (dx2[jm]*mu[ix(i,j,kp,n2,n3)] + dx2[j]*mu[ix(i,jm,kp,n2,n3)]) / dmx2[j];
        const real_t mu_zk  = kHalf * (dx3[km]*myc_z + dx3[k]*mya_z) / dmx3[k];
        const real_t mu_zkp = kHalf * (dx3[kp]*myc_z + dx3[k]*myb_z) / dmx3[kp];
        const real_t oz = (mu_zkp * dwdy_kp - mu_zk * dwdy_k) / dx3[k];
        const real_t irc = kHalf * (dx2[jm]*invrho[c] + dx2[j]*invrho[ix(i,jm,k,n2,n3)]) / dmx2[j];
        rhs_v[c] += dt * Cmu * irc * (ox + oy + oz);
      }
}

// ---- W-momentum (w on z-face k, wall-normal STAGGERED; cross x, y) ----------
//   T_w = ∂x(μ∂zu) + ∂y(μ∂zv) + ∂z(μ∂zw)
void add_transpose_stress_w_cpu(real_t* rhs_w, const real_t* U, const real_t* V, const real_t* W,
    const real_t* mu, const real_t* invrho,
    const real_t* dx1, const real_t* dmx1, const real_t* dx2, const real_t* dmx2,
    const real_t* dx3, const real_t* dmx3, int n1, int n2, int n3, real_t Cmu, real_t dt) {
  const int h = kHaloWidth;
  for (int i = h; i < n1 - h; ++i)
    for (int j = h; j < n2 - h; ++j)
      for (int k = h; k < n3 - h; ++k) {
        const int im = i-1, ip = i+1, jm = j-1, jp = j+1, km = k-1, kp = k+1;
        const int c = ix(i, j, k, n2, n3);
        // own-z: ∂z(μ∂zw), face-μ, STAGGERED (w on z-faces, like z_staggered Lz)
        const real_t mzp = kHalf * (mu[c] + mu[ix(i,j,kp,n2,n3)]);
        const real_t mzm = kHalf * (mu[ix(i,j,km,n2,n3)] + mu[c]);
        const real_t oz = ( mzp * (W[ix(i,j,kp,n2,n3)] - W[c]) / dx3[k]
                          - mzm * (W[c] - W[ix(i,j,km,n2,n3)]) / dx3[k-1] ) / dmx3[k];
        // ∂x(μ∂zu): ∂zu at (x-face i/ip, z-face k); μ z-interp→face k then x-interp
        const real_t dudz_i  = (U[ix(i ,j,k,n2,n3)] - U[ix(i ,j,km,n2,n3)]) / dmx3[k];
        const real_t dudz_ip = (U[ix(ip,j,k,n2,n3)] - U[ix(ip,j,km,n2,n3)]) / dmx3[k];
        const real_t mza_x = kHalf * (dx3[km]*mu[ix(im,j,k,n2,n3)] + dx3[k]*mu[ix(im,j,km,n2,n3)]) / dmx3[k];
        const real_t mzc_x = kHalf * (dx3[km]*mu[ix(i ,j,k,n2,n3)] + dx3[k]*mu[ix(i ,j,km,n2,n3)]) / dmx3[k];
        const real_t mzb_x = kHalf * (dx3[km]*mu[ix(ip,j,k,n2,n3)] + dx3[k]*mu[ix(ip,j,km,n2,n3)]) / dmx3[k];
        const real_t mu_xi  = kHalf * (dx1[im]*mzc_x + dx1[i]*mza_x) / dmx1[i];
        const real_t mu_xip = kHalf * (dx1[ip]*mzc_x + dx1[i]*mzb_x) / dmx1[ip];
        const real_t ox = (mu_xip * dudz_ip - mu_xi * dudz_i) / dx1[i];
        // ∂y(μ∂zv): ∂zv at (y-face j/jp, z-face k); μ z-interp→face k then y-interp
        const real_t dvdz_j  = (V[ix(i,j ,k,n2,n3)] - V[ix(i,j ,km,n2,n3)]) / dmx3[k];
        const real_t dvdz_jp = (V[ix(i,jp,k,n2,n3)] - V[ix(i,jp,km,n2,n3)]) / dmx3[k];
        const real_t mza_y = kHalf * (dx3[km]*mu[ix(i,jm,k,n2,n3)] + dx3[k]*mu[ix(i,jm,km,n2,n3)]) / dmx3[k];
        const real_t mzc_y = kHalf * (dx3[km]*mu[ix(i,j ,k,n2,n3)] + dx3[k]*mu[ix(i,j ,km,n2,n3)]) / dmx3[k];
        const real_t mzb_y = kHalf * (dx3[km]*mu[ix(i,jp,k,n2,n3)] + dx3[k]*mu[ix(i,jp,km,n2,n3)]) / dmx3[k];
        const real_t mu_yj  = kHalf * (dx2[jm]*mzc_y + dx2[j]*mza_y) / dmx2[j];
        const real_t mu_yjp = kHalf * (dx2[jp]*mzc_y + dx2[j]*mzb_y) / dmx2[jp];
        const real_t oy = (mu_yjp * dvdz_jp - mu_yj * dvdz_j) / dx2[j];
        const real_t irc = kHalf * (dx3[km]*invrho[c] + dx3[k]*invrho[ix(i,j,km,n2,n3)]) / dmx3[k];
        rhs_w[c] += dt * Cmu * irc * (ox + oy + oz);
      }
}

} // namespace mpmstd::equation::momentum::kernels
