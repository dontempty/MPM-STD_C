#include "field/field_registry.hpp"

#include <algorithm>
#include <stdexcept>

namespace mpmstd::field {

FieldRegistry::FieldRegistry(const parallel::mpi::Subdomain& sub,
                              parallel::Backend& backend)
  : sub_(sub), backend_(backend) {}

ScalarField& FieldRegistry::add_scalar(const std::string& name) {
  if (scalars_.count(name)) {
    throw std::runtime_error("FieldRegistry: scalar '" + name + "' already exists");
  }
  auto field = std::make_unique<ScalarField>(sub_, backend_, name);
  ScalarField& ref = *field;
  scalars_.emplace(name, std::move(field));
  return ref;
}

VectorField& FieldRegistry::add_vector(const std::string& name) {
  if (vectors_.count(name)) {
    throw std::runtime_error("FieldRegistry: vector '" + name + "' already exists");
  }
  auto field = std::make_unique<VectorField>(sub_, backend_, name);
  VectorField& ref = *field;
  vectors_.emplace(name, std::move(field));
  return ref;
}

bool FieldRegistry::has_scalar(const std::string& n) const { return scalars_.count(n) != 0; }
bool FieldRegistry::has_vector(const std::string& n) const { return vectors_.count(n) != 0; }

ScalarField& FieldRegistry::scalar(const std::string& name) {
  auto it = scalars_.find(name);
  if (it == scalars_.end()) {
    throw std::runtime_error("FieldRegistry: no scalar named '" + name + "'");
  }
  return *it->second;
}
const ScalarField& FieldRegistry::scalar(const std::string& name) const {
  auto it = scalars_.find(name);
  if (it == scalars_.end()) {
    throw std::runtime_error("FieldRegistry: no scalar named '" + name + "'");
  }
  return *it->second;
}

VectorField& FieldRegistry::vector(const std::string& name) {
  auto it = vectors_.find(name);
  if (it == vectors_.end()) {
    throw std::runtime_error("FieldRegistry: no vector named '" + name + "'");
  }
  return *it->second;
}
const VectorField& FieldRegistry::vector(const std::string& name) const {
  auto it = vectors_.find(name);
  if (it == vectors_.end()) {
    throw std::runtime_error("FieldRegistry: no vector named '" + name + "'");
  }
  return *it->second;
}

std::vector<std::string> FieldRegistry::scalar_names() const {
  std::vector<std::string> out;
  out.reserve(scalars_.size());
  for (auto& kv : scalars_) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}
std::vector<std::string> FieldRegistry::vector_names() const {
  std::vector<std::string> out;
  out.reserve(vectors_.size());
  for (auto& kv : vectors_) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

} // namespace mpmstd::field
