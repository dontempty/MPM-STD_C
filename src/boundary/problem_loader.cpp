#include "boundary/problem_loader.hpp"

#include <stdexcept>
#include <string>

namespace mpmstd::boundary {

namespace {

constexpr const char* kAxisNames[3] = { "x", "y", "z" };
constexpr const char* kSideNames[2] = { "minus", "plus" };

// Internal classification of an axis's BC kind.  Maps four user strings onto
// an enum so the rest of the loader stays branch-friendly.
enum class AxisKind {
  Periodic,
  Dirichlet,
  Neumann,
  Wall,
};

AxisKind parse_axis_kind(const std::string& s, const char* axis_name) {
  if (s == "periodic")  return AxisKind::Periodic;
  if (s == "dirichlet") return AxisKind::Dirichlet;
  if (s == "neumann")   return AxisKind::Neumann;
  if (s == "wall")      return AxisKind::Wall;
  throw std::runtime_error(
    std::string("load_problem_from_config: unknown topology kind '") + s +
    "' on axis '" + axis_name +
    "' (expected exactly one of: 'periodic', 'dirichlet', 'neumann', 'wall')");
}

// Set every field face on (d, s) according to the axis kind. Called after
// the topology has been read so that downstream override logic can rely on
// each face holding the correct default.
void apply_axis_defaults(Problem& p, AxisKind k, Direction d, Side s) {
  switch (k) {
    case AxisKind::Periodic:
      p.U.face(d, s) = FaceBc::periodic();
      p.V.face(d, s) = FaceBc::periodic();
      p.W.face(d, s) = FaceBc::periodic();
      p.P.face(d, s) = FaceBc::periodic();
      p.T.face(d, s) = FaceBc::periodic();
      return;
    case AxisKind::Dirichlet:
      p.U.face(d, s) = FaceBc::dirichlet(0.0);
      p.V.face(d, s) = FaceBc::dirichlet(0.0);
      p.W.face(d, s) = FaceBc::dirichlet(0.0);
      p.P.face(d, s) = FaceBc::dirichlet(0.0);
      p.T.face(d, s) = FaceBc::dirichlet(0.0);
      return;
    case AxisKind::Neumann:
      p.U.face(d, s) = FaceBc::neumann(0.0);
      p.V.face(d, s) = FaceBc::neumann(0.0);
      p.W.face(d, s) = FaceBc::neumann(0.0);
      p.P.face(d, s) = FaceBc::neumann(0.0);
      p.T.face(d, s) = FaceBc::neumann(0.0);
      return;
    case AxisKind::Wall:
      // No-slip wall: velocities Dirichlet 0, T Dirichlet 0, P Neumann 0.
      p.U.face(d, s) = FaceBc::dirichlet(0.0);
      p.V.face(d, s) = FaceBc::dirichlet(0.0);
      p.W.face(d, s) = FaceBc::dirichlet(0.0);
      p.P.face(d, s) = FaceBc::neumann (0.0);
      p.T.face(d, s) = FaceBc::dirichlet(0.0);
      return;
  }
}

// Override a single FaceBc value from the config, preserving its kind.
// Used for both Dirichlet and Neumann overrides — the existing FaceBc kind
// stays put, only the value function gets refreshed.
void try_override_value(const config::Config& cfg,
                          const std::string& section,
                          const char* key,
                          FieldBoundary& fb,
                          Direction d, Side s) {
  if (!cfg.has(section, key)) return;
  const real_t v = cfg.get<real_t>(section, key);

  // Re-wrap with a constant value function while keeping the kind.
  FaceBc& face = fb.face(d, s);
  switch (face.kind) {
    case BcKind::Dirichlet:
      face = FaceBc::dirichlet(v); break;
    case BcKind::Neumann:
      face = FaceBc::neumann(v); break;
    default:
      // Periodic / Wall / Inflow / Outflow are never combined with explicit
      // numeric overrides here — silently keep the existing FaceBc.  (Wall
      // is decomposed into per-field Dirichlet/Neumann by apply_axis_defaults,
      // so this branch is only reached if a user fed a constant into a face
      // that the loader chose not to override.)
      break;
  }
}

} // anonymous namespace

Problem load_problem_from_config(const config::Config& cfg) {
  Problem p;   // The default ctor seeds an RBC-like state; we overwrite below.

  // -----------------------------------------------------------------------
  // (1) Read [topology] for each axis.
  // -----------------------------------------------------------------------
  AxisKind kinds[3]{};
  for (int a = 0; a < 3; ++a) {
    const std::string kind_str = cfg.get<std::string>("topology", kAxisNames[a]);
    kinds[a] = parse_axis_kind(kind_str, kAxisNames[a]);
    p.topology.axis[a] = (kinds[a] == AxisKind::Periodic)
                          ? AxisTopology::Periodic
                          : AxisTopology::NonPeriodic;
  }

  // -----------------------------------------------------------------------
  // (2) Apply per-axis defaults to every (axis, side).
  // -----------------------------------------------------------------------
  for (int a = 0; a < 3; ++a) {
    for (int s = 0; s < 2; ++s) {
      apply_axis_defaults(p, kinds[a],
                            static_cast<Direction>(a),
                            static_cast<Side>(s));
    }
  }

  // -----------------------------------------------------------------------
  // (3) Per-face overrides from [bc.<axis>.<side>].
  //
  //     The set of overridable fields depends on the axis kind:
  //       periodic   → none (sections ignored entirely)
  //       dirichlet  → U, V, W, P, T
  //       neumann    → none (always 0)
  //       wall       → U, V, W, T   (P stays Neumann 0)
  // -----------------------------------------------------------------------
  for (int a = 0; a < 3; ++a) {
    const AxisKind k = kinds[a];
    if (k == AxisKind::Periodic || k == AxisKind::Neumann) continue;

    for (int sidx = 0; sidx < 2; ++sidx) {
      const Direction d  = static_cast<Direction>(a);
      const Side      s  = static_cast<Side>(sidx);
      const std::string section =
        std::string("bc.") + kAxisNames[a] + "." + kSideNames[sidx];

      try_override_value(cfg, section, "U", p.U, d, s);
      try_override_value(cfg, section, "V", p.V, d, s);
      try_override_value(cfg, section, "W", p.W, d, s);
      try_override_value(cfg, section, "T", p.T, d, s);
      // P is only override-able under the "dirichlet" axis kind.
      if (k == AxisKind::Dirichlet) {
        try_override_value(cfg, section, "P", p.P, d, s);
      }
    }
  }

  return p;
}

} // namespace mpmstd::boundary
