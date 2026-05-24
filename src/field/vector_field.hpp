#pragma once

#include "field/scalar_field.hpp"
#include "common/direction.hpp"

#include <string>

namespace mpmstd::field {

// Vector field = 3 ScalarFields, one per velocity component.
//
// Each component is stored in a same-shape array (matching PaScaL_TCS); the
// staggered MAC interpretation is encoded by which stencil functions are used,
// not by the storage layout. Component names .x() / .y() / .z() refer to the
// **face-centered** locations (FaceX, FaceY, FaceZ respectively).

class VectorField {
public:
  VectorField(const parallel::mpi::Subdomain& sub,
              parallel::Backend& backend,
              std::string name);

  VectorField(const VectorField&) = delete;
  VectorField& operator=(const VectorField&) = delete;

  ScalarField&       x()       { return x_; }
  const ScalarField& x() const { return x_; }
  ScalarField&       y()       { return y_; }
  const ScalarField& y() const { return y_; }
  ScalarField&       z()       { return z_; }
  const ScalarField& z() const { return z_; }

  ScalarField&       component(Component c);
  const ScalarField& component(Component c) const;

  const std::string& name() const { return name_; }

  // Convenience: halo exchange on all three components.
  void exchange_halo();

  // Convenience: host<->device sync on all three components.
  void to_device();
  void to_host();

private:
  std::string  name_;
  ScalarField  x_;
  ScalarField  y_;
  ScalarField  z_;
};

} // namespace mpmstd::field
