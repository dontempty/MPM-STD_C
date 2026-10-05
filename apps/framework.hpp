#pragma once
// 최소 테스트 프레임워크 — 외부 의존 없음, 등록 매직 없음.
// 각 테스트 파일이 자기 main() 을 갖고, 끝에서 summary("이름") 을 반환한다.
//
//   #include "framework.hpp"
//   int main() {
//     CHECK(1 + 1 == 2);
//     CHECK_THROWS_MSG(parse("bad"), "t.ini:1");
//     return mpmstd::test::summary("test_example");
//   }
#include <cstdio>
#include <exception>
#include <string>

namespace mpmstd::test {

inline int failures = 0;

inline void check(bool ok, const std::string& what, const char* file, int line) {
  if (!ok) {
    ++failures;
    std::printf("  FAIL %s:%d  %s\n", file, line, what.c_str());
  }
}

inline int summary(const char* name) {
  if (failures == 0) {
    std::printf("PASS %s\n", name);
    return 0;
  }
  std::printf("FAIL %s (%d failure%s)\n", name, failures, failures == 1 ? "" : "s");
  return 1;
}

}  // namespace mpmstd::test

#define CHECK(cond) ::mpmstd::test::check((cond), #cond, __FILE__, __LINE__)

// expr 이 std::exception 을 던지고, 그 메시지에 substr 이 들어있는지 확인한다.
#define CHECK_THROWS_MSG(expr, substr)                                         \
  do {                                                                         \
    bool threw_ = false;                                                       \
    std::string msg_;                                                          \
    try {                                                                      \
      (void)(expr);                                                            \
    } catch (const std::exception& e_) {                                       \
      threw_ = true;                                                           \
      msg_ = e_.what();                                                        \
    }                                                                          \
    ::mpmstd::test::check(threw_, "expected throw: " #expr, __FILE__, __LINE__); \
    if (threw_)                                                                \
      ::mpmstd::test::check(msg_.find(substr) != std::string::npos,            \
                            std::string("message should contain \"") + substr  \
                                + "\", got: " + msg_,                          \
                            __FILE__, __LINE__);                               \
  } while (0)
