#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

// Minimal assertion helpers; no external dependency (Catch2 / GoogleTest).
// Each test executable just has a `main()` returning 0 on success.

#define MPMSTD_TEST_CHECK(cond)                                                \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s\n  at %s:%d\n", #cond, __FILE__, __LINE__); \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define MPMSTD_TEST_NEAR(a, b, tol)                                            \
  do {                                                                         \
    auto _a = (a);                                                             \
    auto _b = (b);                                                             \
    auto _t = (tol);                                                           \
    if (std::abs(_a - _b) > _t) {                                              \
      std::fprintf(stderr,                                                     \
        "FAIL: |%s - %s| = %g  >  %g\n  at %s:%d\n",                           \
        #a, #b, double(std::abs(_a - _b)), double(_t),                         \
        __FILE__, __LINE__);                                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

inline void mpmstd_test_pass(const std::string& name) {
  std::cout << "[PASS] " << name << "\n";
}
