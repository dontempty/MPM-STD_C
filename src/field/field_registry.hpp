#pragma once

#include "field/scalar_field.hpp"
#include "field/vector_field.hpp"

#include <memory>
#include <string>
#include <unordered_map>

namespace mpmstd::field {

// Named storage for ScalarField / VectorField instances.
// Each app's main.cpp adds the fields it needs and looks them up by name.
//
// add_*("name") returns a reference to the newly created field. Re-adding
// the same name throws.

class FieldRegistry {
public:
  FieldRegistry(const parallel::mpi::Subdomain& sub,
                parallel::Backend& backend);

  FieldRegistry(const FieldRegistry&) = delete;
  FieldRegistry& operator=(const FieldRegistry&) = delete;

  ScalarField& add_scalar(const std::string& name);
  VectorField& add_vector(const std::string& name);

  bool has_scalar(const std::string& name) const;
  bool has_vector(const std::string& name) const;

  ScalarField&       scalar(const std::string& name);
  const ScalarField& scalar(const std::string& name) const;
  VectorField&       vector(const std::string& name);
  const VectorField& vector(const std::string& name) const;

  // Iteration helpers (for IO / debug dumps).
  std::vector<std::string> scalar_names() const;
  std::vector<std::string> vector_names() const;

private:
  const parallel::mpi::Subdomain& sub_;
  parallel::Backend&              backend_;
  std::unordered_map<std::string, std::unique_ptr<ScalarField>> scalars_;
  std::unordered_map<std::string, std::unique_ptr<VectorField>> vectors_;
};

} // namespace mpmstd::field
