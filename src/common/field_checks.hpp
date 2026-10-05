#pragma once
// (필드, 분할) 짝 검사 — 모듈 공용 (조용한 실패 금지, D10).

#include <stdexcept>
#include <string>

#include "field/field.hpp"
#include "parallel/subdomain/subdomain.hpp"

namespace mpmstd {

inline void check_field_matches(const char* who, const field::Field& f,
                                const parallel::Subdomain& sub) {
  if (f.nx_total() != sub.x().n_total || f.ny_total() != sub.y().n_total
      || f.nz_total() != sub.z().n_total)
    throw std::invalid_argument(std::string(who) + ": field '" + f.name()
                                + "' does not match the subdomain");
}

}  // namespace mpmstd
