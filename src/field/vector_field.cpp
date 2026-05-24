#include "field/vector_field.hpp"

namespace mpmstd::field {

VectorField::VectorField(const parallel::mpi::Subdomain& sub,
                          parallel::Backend& backend,
                          std::string name)
  : name_(std::move(name)),
    x_(sub, backend, name_ + ".x"),
    y_(sub, backend, name_ + ".y"),
    z_(sub, backend, name_ + ".z") {}

ScalarField& VectorField::component(Component c) {
  switch (c) {
    case Component::U: return x_;
    case Component::V: return y_;
    case Component::W: return z_;
  }
  return x_;  // unreachable
}
const ScalarField& VectorField::component(Component c) const {
  switch (c) {
    case Component::U: return x_;
    case Component::V: return y_;
    case Component::W: return z_;
  }
  return x_;  // unreachable
}

void VectorField::exchange_halo() {
  x_.exchange_halo();
  y_.exchange_halo();
  z_.exchange_halo();
}

void VectorField::to_device() {
  x_.to_device();
  y_.to_device();
  z_.to_device();
}

void VectorField::to_host() {
  x_.to_host();
  y_.to_host();
  z_.to_host();
}

} // namespace mpmstd::field
