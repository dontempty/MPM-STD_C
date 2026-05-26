#pragma once

#include "field/scalar_field.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace mpmstd::field {

// Named storage for all 3-D fields used by a simulation.
//
// We follow the PaScaL_TCS convention: each velocity component (U, V, W) is
// registered as its own ScalarField — the staggered MAC interpretation
// (U on x-face, V on y-face, W on z-face) is encoded by the stencil helpers
// that consume them, not by the storage type.
//
// Re-adding the same name throws.

class FieldRegistry {
public:
  FieldRegistry(const parallel::mpi::Subdomain& sub,
                parallel::Backend& backend);

  FieldRegistry(const FieldRegistry&) = delete;
  FieldRegistry& operator=(const FieldRegistry&) = delete;

  ScalarField& add_scalar(const std::string& name);

  bool has_scalar(const std::string& name) const;

  ScalarField&       scalar(const std::string& name);
  const ScalarField& scalar(const std::string& name) const;

  // Iteration helpers (for IO / debug dumps).
  std::vector<std::string> scalar_names() const;

private:
  const parallel::mpi::Subdomain& sub_;
  parallel::Backend&              backend_;
  std::unordered_map<std::string, std::unique_ptr<ScalarField>> scalars_;
};

} // namespace mpmstd::field
