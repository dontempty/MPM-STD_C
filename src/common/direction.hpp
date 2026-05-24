#pragma once

namespace mpmstd {

enum class Direction : int { X = 0, Y = 1, Z = 2 };
enum class Side      : int { Minus = 0, Plus = 1 };
enum class Component : int { U = 0, V = 1, W = 2 };

constexpr int to_int(Direction d) { return static_cast<int>(d); }
constexpr int to_int(Side s)      { return static_cast<int>(s); }
constexpr int to_int(Component c) { return static_cast<int>(c); }

constexpr Direction component_to_direction(Component c) {
  return static_cast<Direction>(static_cast<int>(c));
}

constexpr const char* direction_name(Direction d) {
  switch (d) {
    case Direction::X: return "X";
    case Direction::Y: return "Y";
    case Direction::Z: return "Z";
  }
  return "?";
}

constexpr const char* side_name(Side s) {
  return s == Side::Minus ? "-" : "+";
}

constexpr const char* component_name(Component c) {
  switch (c) {
    case Component::U: return "U";
    case Component::V: return "V";
    case Component::W: return "W";
  }
  return "?";
}

} // namespace mpmstd
