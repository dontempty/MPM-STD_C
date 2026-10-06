#include "boundary/recycling.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mpmstd::boundary {

namespace {

// 선형 보간 (coord 오름차순; 범위 밖은 끝값)
real_t interp(const std::vector<real_t>& coord, const std::vector<real_t>& val, const real_t z) {
  const int n = static_cast<int>(coord.size());
  if (z <= coord[0]) return val[0];
  if (z >= coord[n - 1]) return val[n - 1];
  const int k = static_cast<int>(std::upper_bound(coord.begin(), coord.end(), z) - coord.begin()) - 1;
  const real_t w = (z - coord[k]) / (coord[k + 1] - coord[k]);
  return (1 - w) * val[k] + w * val[k + 1];
}

real_t blend_weight(const real_t eta) {  // Lund W(η), α = 4, b = 0.2
  const real_t alpha = 4, b = 0.2;
  if (eta <= 0) return 0;
  if (eta >= 1) return 1;
  return real_t(0.5) * (1 + std::tanh(alpha * (eta - b) / ((1 - 2 * b) * eta + b)) / std::tanh(alpha));
}

}  // namespace

Recycling::Recycling(const grid::Grid& grid, const parallel::Subdomain& sub,
                     const std::vector<real_t>& z_center_global,
                     const std::vector<real_t>& z_face_global, const int i_recycle_local,
                     const real_t u_infinity, const real_t nu, const real_t delta99_inflow,
                     const real_t relaxation_time)
    : grid_(grid), sub_(sub), zc_(z_center_global), zf_(z_face_global),
      i_recycle_(i_recycle_local), u_inf_(u_infinity), nu_(nu), delta_in_(delta99_inflow),
      tau_(relaxation_time), nz_global_(sub.z().n_global), ny_(sub.y().n_total) {
  if (sub.topology().x().nprocs != 1 || sub.topology().y().nprocs != 1)
    throw std::invalid_argument("Recycling: z-slab decomposition only (npx = npy = 1)");
  if (static_cast<int>(zc_.size()) != nz_global_ || static_cast<int>(zf_.size()) != nz_global_ + 1)
    throw std::invalid_argument("Recycling: global z coordinate sizes");
  mean_u_run_.assign(nz_global_, u_inf_);
  mean_w_run_.assign(nz_global_ + 1, 0);
}

void Recycling::gather_plane(const field::Field& F, std::vector<real_t>& out,
                             const bool z_face) const {
  // 로컬 평면(ghost 포함) → 내부만 [j_local·nz_local(+1) + k] 로 패킹 → Allgatherv(z) → 재배열
  std::vector<real_t> local;
  extract_x_plane(F, i_recycle_, local, sub_);
  const int h = halo_width;
  const int nyl = sub_.y().n_local, nzl = sub_.z().n_local + (z_face ? 1 : 0);
  const int nzt = sub_.z().n_total;
  std::vector<real_t> packed(static_cast<std::size_t>(nyl) * nzl);
  for (int j = 0; j < nyl; ++j)
    for (int k = 0; k < nzl; ++k)
      packed[static_cast<std::size_t>(j) * nzl + k] = local[static_cast<std::size_t>(j + h) * nzt + (k + h)];
  // z 분할 랭크들의 조각 모으기
  MPI_Comm comm = sub_.topology().z().comm;
  int nprocs = 0, rank = 0;
  MPI_Comm_size(comm, &nprocs);
  MPI_Comm_rank(comm, &rank);
  std::vector<int> counts(nprocs), displs(nprocs), nz_each(nprocs), off_each(nprocs);
  const int my_count = static_cast<int>(packed.size());
  MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  const int my_nz = nzl, my_off = sub_.z().offset;
  MPI_Allgather(&my_nz, 1, MPI_INT, nz_each.data(), 1, MPI_INT, comm);
  MPI_Allgather(&my_off, 1, MPI_INT, off_each.data(), 1, MPI_INT, comm);
  int total = 0;
  for (int r = 0; r < nprocs; ++r) { displs[r] = total; total += counts[r]; }
  std::vector<real_t> all(total);
  MPI_Allgatherv(packed.data(), my_count, MPI_DOUBLE, all.data(), counts.data(), displs.data(),
                 MPI_DOUBLE, comm);
  const int nzg = nz_global_ + (z_face ? 1 : 0);
  out.assign(static_cast<std::size_t>(nyl) * nzg, 0);
  for (int r = 0; r < nprocs; ++r)
    for (int j = 0; j < nyl; ++j)
      for (int k = 0; k < nz_each[r]; ++k) {
        const int kg = off_each[r] + k;
        if (kg >= nzg) continue;  // 면 배열: 랭크 경계 면이 두 번 — 아래 랭크 값 유지
        if (z_face && r > 0 && k == 0) continue;
        out[static_cast<std::size_t>(j) * nzg + kg] = all[displs[r] + static_cast<std::size_t>(j) * nz_each[r] + k];
      }
}

void Recycling::initialize_running_mean(const field::Field& U, const field::Field& W) {
  std::vector<real_t> pu, pw;
  gather_plane(U, pu, false);
  gather_plane(W, pw, true);
  const int nyl = sub_.y().n_local;
  for (int k = 0; k < nz_global_; ++k) {
    real_t s = 0;
    for (int j = 0; j < nyl; ++j) s += pu[static_cast<std::size_t>(j) * nz_global_ + k];
    mean_u_run_[k] = s / nyl;
  }
  for (int k = 0; k <= nz_global_; ++k) {
    real_t s = 0;
    for (int j = 0; j < nyl; ++j) s += pw[static_cast<std::size_t>(j) * (nz_global_ + 1) + k];
    mean_w_run_[k] = s / nyl;
  }
  mean_ready_ = true;
}

InflowPlane Recycling::make_inflow_plane(const field::Field& U, const field::Field& V,
                                         const field::Field& W, const real_t dt) {
  if (!mean_ready_) initialize_running_mean(U, W);
  const int nyl = sub_.y().n_local, nzg = nz_global_;
  std::vector<real_t> pu, pv, pw;
  gather_plane(U, pu, false);
  gather_plane(V, pv, false);
  gather_plane(W, pw, true);

  // ① 스팬 평균(순간)과 변동
  std::vector<real_t> mu(nzg, 0), mw(nzg + 1, 0);
  for (int k = 0; k < nzg; ++k) {
    real_t s = 0;
    for (int j = 0; j < nyl; ++j) s += pu[static_cast<std::size_t>(j) * nzg + k];
    mu[k] = s / nyl;
  }
  for (int k = 0; k <= nzg; ++k) {
    real_t s = 0;
    for (int j = 0; j < nyl; ++j) s += pw[static_cast<std::size_t>(j) * (nzg + 1) + k];
    mw[k] = s / nyl;
  }
  // ② 완화 시간평균 → 스케일
  const real_t a = std::min(real_t(1), dt / tau_);
  for (int k = 0; k < nzg; ++k) mean_u_run_[k] = (1 - a) * mean_u_run_[k] + a * mu[k];
  for (int k = 0; k <= nzg; ++k) mean_w_run_[k] = (1 - a) * mean_w_run_[k] + a * mw[k];
  const std::vector<real_t>& Ur = mean_u_run_;
  const real_t z1 = zc_[0], z2 = zc_[1];
  const real_t dudz_w = Ur[0] * z2 / (z1 * (z2 - z1)) - Ur[1] * z1 / (z2 * (z2 - z1));
  const real_t u_tau_r = std::sqrt(std::max(real_t(0), nu_ * dudz_w));
  real_t delta_r = zc_.back();
  for (int k = 0; k < nzg; ++k)
    if (Ur[k] >= real_t(0.99) * u_inf_) {
      if (k == 0) { delta_r = zc_[0]; break; }
      const real_t w = (real_t(0.99) * u_inf_ - Ur[k - 1]) / (Ur[k] - Ur[k - 1]);
      delta_r = zc_[k - 1] + w * (zc_[k] - zc_[k - 1]);
      break;
    }
  real_t theta_r = 0;
  for (int k = 0; k < nzg; ++k) {
    const real_t f = std::min(real_t(1), std::max(real_t(0), Ur[k] / u_inf_));
    theta_r += f * (1 - f) * (zf_[k + 1] - zf_[k]);
  }
  // ③ γ
  const real_t gamma = std::pow(std::max(real_t(1e-6), delta_r / delta_in_), real_t(0.125));
  scales_.u_tau_recycle = u_tau_r;
  scales_.delta99_recycle = delta_r;
  scales_.theta_recycle = theta_r;
  scales_.gamma = gamma;
  scales_.u_tau_inflow = gamma * u_tau_r;

  // ④⑤ 유입 평면 (로컬 ghost 포함 색인) — 중심 z: U, V / 면 z: W
  InflowPlane plane;
  plane.ny_total = sub_.y().n_total;
  plane.nz_total = sub_.z().n_total;
  const std::size_t n = static_cast<std::size_t>(plane.ny_total) * plane.nz_total;
  plane.u.assign(n, u_inf_); plane.v.assign(n, 0); plane.w.assign(n, 0);
  const int h = halo_width;
  const real_t ratio = delta_r / delta_in_;
  std::vector<real_t> fluct_u(nzg), fluct_v(nzg), fluct_w(nzg + 1);
  real_t theta_in = 0;
  for (int jl = 0; jl < nyl; ++jl) {
    for (int k = 0; k < nzg; ++k) {
      fluct_u[k] = pu[static_cast<std::size_t>(jl) * nzg + k] - mu[k];
      fluct_v[k] = pv[static_cast<std::size_t>(jl) * nzg + k];  // V̄ = 0
    }
    for (int k = 0; k <= nzg; ++k) fluct_w[k] = pw[static_cast<std::size_t>(jl) * (nzg + 1) + k] - mw[k];
    for (int kl = 0; kl < sub_.z().n_local; ++kl) {
      const int kg = sub_.z().offset + kl;
      const real_t z = zc_[kg], zf = zf_[kg];
      const real_t eta = z / delta_in_, etaf = zf / delta_in_;
      const real_t Wb = blend_weight(eta), Wbf = blend_weight(etaf);
      // 내부(z⁺ 보존): z_r = γ z ; 외부(η 보존): z_r = z·δ_r/δ_in
      const real_t zi = gamma * z, zo = ratio * z, zfi = gamma * zf, zfo = ratio * zf;
      const real_t Ui = gamma * interp(zc_, Ur, zi);
      const real_t Uo = u_inf_ - gamma * (u_inf_ - interp(zc_, Ur, zo));
      const real_t ui = gamma * interp(zc_, fluct_u, zi), uo = gamma * interp(zc_, fluct_u, zo);
      const real_t vi = gamma * interp(zc_, fluct_v, zi), vo = gamma * interp(zc_, fluct_v, zo);
      const real_t Wi = interp(zf_, mean_w_run_, zfi), Wo = interp(zf_, mean_w_run_, zfo);
      const real_t wi = gamma * interp(zf_, fluct_w, zfi), wo = gamma * interp(zf_, fluct_w, zfo);
      const std::size_t q = static_cast<std::size_t>(jl + h) * plane.nz_total + (kl + h);
      plane.u[q] = (1 - Wb) * (Ui + ui) + Wb * (Uo + uo);
      plane.v[q] = (1 - Wb) * vi + Wb * vo;
      plane.w[q] = (1 - Wbf) * (Wi + wi) + Wbf * (Wo + wo);
      if (jl == 0) {
        const real_t Um = (1 - Wb) * Ui + Wb * Uo;
        const real_t f = std::min(real_t(1), std::max(real_t(0), Um / u_inf_));
        theta_in += f * (1 - f) * (zf_[kg + 1] - zf_[kg]);
      }
    }
  }
  // 상단 ghost(k = nz_total − h): 자유류
  // (유입 θ 는 z 분할이면 로컬 합 → 전체 합)
  real_t theta_in_g = 0;
  MPI_Allreduce(&theta_in, &theta_in_g, 1, MPI_DOUBLE, MPI_SUM, sub_.topology().z().comm);
  scales_.theta_inflow = theta_in_g;
  // y ghost: 주기 — 교환이 덮지 않는 면(x− 슬롯)이므로 직접 복제
  for (int k = 0; k < plane.nz_total; ++k) {
    plane.u[static_cast<std::size_t>(0) * plane.nz_total + k] = plane.u[static_cast<std::size_t>(nyl) * plane.nz_total + k];
    plane.u[static_cast<std::size_t>(nyl + h) * plane.nz_total + k] = plane.u[static_cast<std::size_t>(h) * plane.nz_total + k];
    plane.v[static_cast<std::size_t>(0) * plane.nz_total + k] = plane.v[static_cast<std::size_t>(nyl) * plane.nz_total + k];
    plane.v[static_cast<std::size_t>(nyl + h) * plane.nz_total + k] = plane.v[static_cast<std::size_t>(h) * plane.nz_total + k];
    plane.w[static_cast<std::size_t>(0) * plane.nz_total + k] = plane.w[static_cast<std::size_t>(nyl) * plane.nz_total + k];
    plane.w[static_cast<std::size_t>(nyl + h) * plane.nz_total + k] = plane.w[static_cast<std::size_t>(h) * plane.nz_total + k];
  }
  return plane;
}

}  // namespace mpmstd::boundary
