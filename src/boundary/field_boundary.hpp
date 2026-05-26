#pragma once

#include "boundary/face_bc.hpp"
#include "common/direction.hpp"

#include <array>

namespace mpmstd::boundary {

// FieldBoundary: per-field BC descriptor over all 6 domain faces.
//
// Indexing: faces[2*axis + side], where side==0 is the minus face and side==1
// is the plus face on that axis.

class FieldBoundary {
public:
  FaceBc& face(Direction d, Side s) {
    return faces_[2 * to_int(d) + to_int(s)];
  }
  const FaceBc& face(Direction d, Side s) const {
    return faces_[2 * to_int(d) + to_int(s)];
  }

  // All 6 faces in a fixed order (-x, +x, -y, +y, -z, +z).
  const std::array<FaceBc, 6>& faces() const { return faces_; }

private:
  std::array<FaceBc, 6> faces_{};
};

} // namespace mpmstd::boundary
