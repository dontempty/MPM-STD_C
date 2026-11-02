// 모니터 진단(θ_c·벽 Nu)은 rbc_nob 의 Statistics 를 그대로 쓴다 — 모니터 줄이 rbc_nob 과 같아야
// 잡 스크립트의 게이트(check_gate.py)가 두 앱을 비교할 수 있다. 소유는 rbc_nob, 이 1줄 TU 가 빌려온다
// (rbc_settings.cpp 와 같은 패턴). 누적(accumulate)은 측정 런에서 부르지 않는다.
#include "../rbc_nob/statistics.cpp"
