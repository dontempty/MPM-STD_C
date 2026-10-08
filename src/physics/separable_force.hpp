#pragma once
// 분리형 체적력 — rhs(i,j,k) += scale · e_x[i] · g_y[j] · e_z[k]  (내부 셀만).
// 용도: 경계층 난류 트립(Schlatter & Örlü 2012: F_z = A·exp[−((x−x_t)/ℓ_x)² − (z/ℓ_z)²]·g(y,t)) 처럼
// 공간 포락선 × 스팬 랜덤 모양으로 분리되는 힘. 세 배열은 **로컬 n_total 길이(ghost 포함)** 의 백엔드
// 버퍼이고, 어느 면 변수(U·V·W)에 더하느냐에 맞는 좌표(면/중심)로 호출자가 채운다.
// scale 에 dt 를 포함시키면 rhs 관례(dt·항)와 맞는다. CPU/GPU 쌍둥이(separable_force.{cpp,cu}).
#include "common/types.hpp"
#include "field/buffer.hpp"
#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd::physics {

void add_separable_force(field::Field& rhs, const field::Buffer& envelope_x,
                         const field::Buffer& shape_y, const field::Buffer& envelope_z,
                         real_t scale, const parallel::Subdomain& sub);

}  // namespace mpmstd::physics
