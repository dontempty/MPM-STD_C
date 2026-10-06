#pragma once
// 재순환-재스케일 유입 (Lund, Wu & Squires 1998, JCP 140:233) — 난류 경계층 유입 생성.
// (docs/Design/20261006_boundary_layer_validation.md §3·§8; 2026-10-06)
//
// 매 스텝: 재순환면 x_r 의 (u,v,w)(y,z) 를 전 랭크로 모아(z 분할 → Allgatherv)
//   ① 스팬 평균 Ū_r(z), W̄_r(z) 와 변동 u′ = u − Ū_r (V̄ 는 0 으로 둔다)
//   ② 완화 시간평균 ⟨Ū_r⟩ 로 스케일: u_τ,r = √(ν ∂⟨Ū⟩/∂z|w), δ_r (⟨Ū⟩ = 0.99 U∞), θ_r
//   ③ γ = u_τ,in/u_τ,r = (δ_r/δ_in)^{1/8}  (1/7 멱법칙 C_f ∝ Re_δ^{−1/4})
//   ④ 내부: Ū_in(z) = γ Ū_r(γ z),  u′_in(y,z) = γ u′_r(y, γ z)          (z⁺ 보존)
//      외부: U∞ − Ū_in(z) = γ (U∞ − Ū_r(z δ_r/δ_in)),  u′_in = γ u′_r(y, z δ_r/δ_in)   (η 보존)
//      W̄ 는 γ 없이 위치만 재배치, 변동은 γ 배 (Lund)
//   ⑤ 혼합 W(η) = ½[1 + tanh(α(η−b)/((1−2b)η+b))/tanh α], α = 4, b = 0.2 (η = z/δ_in)
//      → 유입 평면 u = (1−W)·내부 + W·외부
// 호스트 로직(평면 크기 ny×nz 뿐). 디바이스 접근은 open.hpp 의 extract_x_plane / set_inflow.

#include <vector>

#include "boundary/open.hpp"
#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::boundary {

struct RecyclingScales {  // 모니터용
  real_t u_tau_recycle = 0, delta99_recycle = 0, theta_recycle = 0, gamma = 1;
  real_t u_tau_inflow = 0, theta_inflow = 0;
};

class Recycling {
 public:
  // z_center_global/z_face_global: 전역 z 좌표 (ghost 제외, 크기 nz_global / nz_global+1).
  // delta99_inflow: 유입 목표 δ99 (길이 단위), relaxation_time: ⟨Ū⟩ 완화 시간.
  Recycling(const grid::Grid& grid, const parallel::Subdomain& sub,
            const std::vector<real_t>& z_center_global, const std::vector<real_t>& z_face_global,
            int i_recycle_local, real_t u_infinity, real_t nu, real_t delta99_inflow,
            real_t relaxation_time);

  // 현재 장에서 유입 평면을 만든다 (집합 통신 포함 — 전 랭크가 같이 불러야 함).
  InflowPlane make_inflow_plane(const field::Field& U, const field::Field& V,
                                const field::Field& W, real_t dt);
  const RecyclingScales& scales() const { return scales_; }
  // 완화 평균 초기화 (IC 의 스팬 평균으로) — 첫 호출 전에
  void initialize_running_mean(const field::Field& U, const field::Field& W);

 private:
  const grid::Grid& grid_;
  const parallel::Subdomain& sub_;
  std::vector<real_t> zc_, zf_;      // 전역 z 중심/면
  int i_recycle_;
  real_t u_inf_, nu_, delta_in_, tau_;
  int nz_global_, ny_;                 // ny_ = 로컬 ny_total (npy = 1)
  std::vector<real_t> mean_u_run_, mean_w_run_;  // ⟨Ū_r⟩(k_global), ⟨W̄_r⟩(k_global+면)
  bool mean_ready_ = false;
  RecyclingScales scales_;

  // 재순환면 전 랭크 모으기: out[j·nz_global + k_global] (ghost 제외; U/V 는 중심, W 는 면)
  void gather_plane(const field::Field& F, std::vector<real_t>& out, bool z_face) const;
};

}  // namespace mpmstd::boundary
