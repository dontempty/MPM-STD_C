#pragma once

// Host-single run configuration (TOML-parsed). Re-exported from the existing
// tree (rev.2: host metadata not duplicated per backend).
#include "config/config.hpp"

namespace mpmstd::core {
using Config = config::Config;
} // namespace mpmstd::core
