// rbc_nob 은 rbc 와 같은 설정을 읽는다 (같은 키, 같은 검증) — 설정 코드는
// rbc 앱이 소유하고, 이 1줄 TU 가 그 소유를 명시적으로 빌려온다
// (test_rbc_settings 와 같은 패턴). NOB 전용 요구(delta_temperature > 0)는
// main 이 fail-loud 로 검사한다.
#include "../rbc/settings.cpp"
