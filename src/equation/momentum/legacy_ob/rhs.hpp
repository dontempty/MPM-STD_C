#pragma once
// ⚠ legacy OB 경로 — 간이 스킴(보존형 RHS + 짝 없는 이류형 음해항). 블록
// 커플링의 평균이 0 이 아니어서 운동량이 dt 에 비례해 샌다(2026-10-04 수지
// 계측). **새 코드에서 쓰지 말 것** — 충실 경로(상위 디렉터리)의
// *_variable_property / *_constant_property 를 쓴다. rbc(OB) 앱 하위호환으로만
// 남아 있고, rbc 전환(Nu 회귀 재검증) 후 디렉터리째 삭제 예정.
// 운동량 방정식의 명시(explicit) RHS — 전부 += (계획 §5.5). 성분마다 rhs 필드를
// 호출자가 선언하고 fill(0) 후 쌓는다. 압력 구배는 pressure::add_gradient 가,
// 채널 평균 구배 강제는 physics::channel_forcing 이 **그 뒤에** 더한다.
//
//   rhs_q += dt·ν·∇²qⁿ − dt·∇·(u qⁿ)
//
// 수식은 MPM-STD_C OB 경로 1:1 (momentum_kernels_cpu.cpp 의 compute_mpmstd_rhs):
// * 확산: 라플라시안 세 방향 각 1회 (델타형 — 음해 밴드와의 쌍은 solve.hpp 참조)
// * 대류: 보존형 면 플럭스 (면 속도 × 2점 평균), 전체 계수 1
// * W 는 z-면 변수라 z 라플라시안의 메트릭이 다르다 (제어체적 = dmz, 이웃 = dz):
//     셀 중심(U,V):  ((q_{k+1}−q_k)/dmz_{k+1} − (q_k−q_{k−1})/dmz_k)/dz_k
//     z-면   (W) :  ((q_{k+1}−q_k)/dz_k     − (q_k−q_{k−1})/dz_{k−1})/dmz_k
//   그래서 함수가 둘이다 — 이름이 슬롯 해석을 말한다 (enum·플래그 없음).
//
// ⚠ 이 보존형 RHS + 이류형 음해 밴드/블록 커플링 조합은 증분 짝이 없어
//   운동량이 dt 에 비례해 샌다 (CFR 과구동; 2026-10-04 조사). 원본 충실
//   M-form 은 **가변물성 경로(rhs_variable + solve_variable_property, μ̂≡1)** 가 담당
//   한다 — 채널은 그쪽을 쓴다. 이 OB 경로는 rbc(OB) 하위호환용.
//
// 호출 전 q 와 U, V, W 의 ghost 가 최신이어야 한다 (교환 + 벽 처리).

#include "common/types.hpp"
#include "field/field.hpp"
#include "grid/grid.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::equation::momentum {

// 셀 중심 z 슬롯 성분 (U, V): rhs += dt·ν∇²q − dt·∇·(u q)
void add_advection_diffusion_cell(field::Field& rhs, const field::Field& q,
                                  const field::Field& U, const field::Field& V,
                                  const field::Field& W, const grid::Grid& grid,
                                  const parallel::Subdomain& sub, real_t viscosity,
                                  real_t dt);

// z-면 슬롯 성분 (W): 같은 식, z 메트릭만 면 기준
void add_advection_diffusion_z_face(field::Field& rhs, const field::Field& w,
                                    const field::Field& U, const field::Field& V,
                                    const field::Field& W, const grid::Grid& grid,
                                    const parallel::Subdomain& sub, real_t viscosity,
                                    real_t dt);

}  // namespace mpmstd::equation::momentum
