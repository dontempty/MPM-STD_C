# RBC 결과·restart 백업 인덱스 (2026-07-21 정리)

현역(apps/rbc/ 최상위)은 P8(ΔT=20)·P9(ΔT=60) 256³ CFL=0.3 검증 체인만 유지.
아래는 백업된 주요 결과 — 자세한 맥락은 Report/RBC/ 4부작 참조.

## 핵심 결과 (Fig9용)
- **p7b_deltat40_z256_resume/** — ★ ΔT=40 최종: 통계 θ_c=0.13827 (5.531K, Hormes +0.05%).
  z256(128×128×256), 냉시작(전반은 p7_deltat40_z256_cold_cfl03/monitor.csv), CFL=0.3, θ=0.5,
  수정 바이너리(7/16 21:51). restart_p7b_* = t=25,000 평형장 (웜체인 출발점으로 사용 가능).
- **runb_e22_newcode/** — E22 restart + 신코드 + CFL=0.3 (np=1): θ_c 0.1297↑ (code 무죄 판정 근거)
- **runc_e22_cfl1/** — 같은 restart + CFL=1.0: θ_c 0.1258↓ (CFL=1.0 유죄 판정 근거, 클립 140셀/스텝)
- p2/p3(+b,c)/p5 — 신코드 냉시작 시리즈 (P2 128³ ΔT20: 0.0707; P3 128³ ΔT40: 0.106→0.110;
  P5 256³ ΔT40 CFL1.0: 0.118, 타임아웃) — CFL·해상도·심버그 교란 포함, 해석 주의
- e18/e19 (ΔT=20 256³ 웜체인, 구코드: 0.0647), e20/e22 (ΔT=40 구코드 체인: 0.128 미수렴),
  e21/e23 (ΔT=60 구코드)

## restart 자산 (웜스타트 가능)
- restart_p7b_deltat40_z256_resume/ — ΔT=40 z256 t=25,000 평형 ★
- restart_runb_e22_newcode/ — ΔT=40 256³ t=10,600 (신코드 CFL0.3 상승 중)
- restart_e19_stats_reset_h256/ — ΔT=20 256³ t=20,168 평형 (구코드 이력)
- restart_e22_deltat40_settled/ — ΔT=40 256³ t=8,644 (Run B/C의 공통 출발점)

## 진단·검증 런 (참고용)
- diag_* (E19 CFL/argmax/블록수정 판정), verify_* (스모크), p1(dt 프로브), p4(취소됨),
  fig9_nob_gpu_out_40_x (구세대 ΔT=40, θ_c~0.069 시절)
