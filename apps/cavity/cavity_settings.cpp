// cavity 는 rbc 와 같은 설정 스키마([boundary] 포함)를 읽는다 — 설정 코드는 rbc
// 앱이 소유하고 이 1줄 TU 가 빌려온다 (dhvc/rbc_nob 와 같은 패턴). cavity 전용
// 요구(gravity_axis == 2, x 벽·y 주기)는 main 이 fail-loud 로 검사한다.
#include "../rbc/settings.cpp"
