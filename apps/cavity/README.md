# cavity — 입력 `[boundary]` 로 축별 BC 를 받는 첫 앱 (de Vahl Davis 2D 검증)

설계·단계별 검증 기록 정본: `docs/Design/20261005_boundary_condition_framework.md`.

## 문제
de Vahl Davis (1983) 차등가열 정방 공동, Pr = 0.71, Ra 1e3~1e6.
x = 0 벽 T = +0.5 (hot), x = 1 벽 T = −0.5 (cold), 상하벽 단열, 전 벽 no-slip.
**2D 는 ny = 1 축 붕괴** (y 주기, 셀 1개) — 별도 2D 코드 없이 진짜 2D 비용.

무차원: 자유낙하 스케일 (ν = √(Pr/Ra), α = 1/√(Ra·Pr), 부력 계수 1).
벤치마크의 속도(u·L/α)는 출력에서 이미 √(Ra·Pr) 를 곱해 둔 값이다.

## 실행
```
make cavity
sbatch apps/cavity/sbatch_cavity_ladder.sh      # 1e3~1e6 을 cpu04 에서 병렬 (각 1 랭크)
```
로그 `logs/cavity_<Ra>.log` 의 모니터 열: step, t, dt, max|div|, Nu_hot, Nu_cold,
u_max(z), w_max(x), sec/step, T_center. 프로파일 `stats/<Ra>/cavity_stats_<step>.dat`
(section 1: 연직 중앙선 u·T, 2: 수평 중앙선 w·T, 3: hot 벽 첫 셀 열 진단).
시각화: `plot/plot_cavity.ipynb`.

## 벤치마크 (de Vahl Davis 1983 Table V, Richardson 외삽값; 논문 `apps/Ref [4].pdf`)
| Ra | \|ψ\|_mid | \|ψ\|_max (x,z) | u_max (z) | w_max (x) | Nu̅ | Nu_½ | Nu_0 | Nu_max (z) | Nu_min (z=1) |
|---|---|---|---|---|---|---|---|---|---|
| 1e3 | 1.174 | — | 3.649 (0.813) | 3.697 (0.178) | 1.118 | 1.118 | 1.117 | 1.505 (0.092) | 0.692 |
| 1e4 | 5.071 | — | 16.178 (0.823) | 19.617 (0.119) | 2.243 | 2.243 | 2.238 | 3.528 (0.143) | 0.586 |
| 1e5 | 9.111 | 9.612 (0.285, 0.601) | 34.73 (0.855) | 68.59 (0.066) | 4.519 | 4.519 | 4.509 | 7.717 (0.081) | 0.729 |
| 1e6 | 16.32 | 16.750 (0.151, 0.547) | 64.63 (0.850) | 219.36 (0.0379) | 8.800 | 8.799 | 8.817 | 17.925 (0.0378) | 0.989 |
(u_max = 연직 중앙선 x=½ 최대 |u|, w_max = 수평 중앙선 z=½ 최대 |w|; 흐름 방향이 반대로
잡히면 위치는 z↔1−z, x↔1−x 대칭. 전 항목은 `plot/benchmark.py <field dump>` 가 계산.)

## 검증 결과 (2026-10-06, `velocity_ghost = mirror`; 1e3~1e5 128², 1e6 256²)
1e3~1e5 **전 10항목 ≤ 0.7%** (대부분 ≤ 0.3%), 1e6 은 8항목 ≤ 0.6% + Nu_max −1.4%·Nu_min
−1.0%(논문이 외삽에 가장 민감하다고 밝힌 항목), 극값 위치 전 Ra 3자리 일치 — 상세 표는
`docs/Design/20261005_boundary_condition_framework.md` 단계 4b.
`velocity_ghost = value`(기본, 1차 벽: 전단 V₁/dx = 마찰 절반)는 +1~8% 로 Ra 와 함께
악화하며 256² 에서 오차 반감(1차 수렴). 벤치마크는 반드시 mirror 로.

## 입력 요점 (`input_cavity_<Ra>.ini`)
```ini
[boundary]
x = wall                ; periodic | wall | dirichlet | neumann
y = periodic
z = wall
t_x_minus = 0.5         ; 숫자 → Dirichlet
t_x_plus  = -0.5
t_z_minus = adiabatic   ; adiabatic | slip | neumann → Neumann 0
t_z_plus  = adiabatic
; 접선 속도 slip 은 예: w_x_minus = slip,  lid 는 예: u_z_plus = 1.0
```
`gravity_axis = 2` (부력은 W), `delta_temperature = 0` (상수물성 OB).
`velocity_ghost = mirror` — 접선 속도 2차 벽 (운동량 닫기 B−=A 와 자동으로 짝:
`momentum::set_wall_closure`). 기본 `value` 는 채널/RBC 관례(1차)라 벤치마크엔 부적합.

## ★ 벽 행 음해 닫기 (안정성 — 2026-10-05 실측)
에너지 방정식의 벽 셀 음해 닫기가 PaScaL 충실 "drop"(δθ_ghost = 0)이면 거울 ghost
(Dirichlet)와 어긋나 **dt·α/dx² ≳ 3 에서 벽 셀 모드가 폭주**한다 (증폭률 → −2;
Ra=1e3 128² 에서 Nu 가 1e5 까지 치솟는 것으로 드러남). RBC/DHVC 는 확산수 ≪ 1 이라
무해했다. cavity 는 `scalar::set_wall_closure(solver, bc, Field::T)` 로 ghost 정책과
일치하는 닫기(Dirichlet −1 / Neumann +1)를 쓴다 — 기존 앱은 기본 0 (비트 불변).

## BC 조합 커버리지 (2026-10-06, `sbatch_cavity_extra.sh`)
- lid-driven Re=100 mirror (`input_lid_re100.ini`): Marchi 2009 1024² 대비 ≤0.17% (비영 접선 Dirichlet).
- 축 교대 (`input_cavity_<Ra>_rot.ini`, 중력 x·hot/cold z 벽): `plot/benchmark.py --swap-xz` 가 비교대와 4자리 동일.
- 혼합 면(x− Dirichlet/x+ adiabatic) 강성 전도: 안정, T → 0.5.
지원 범위: 조건은 면 단위(한 면 안은 균일 상수), 접선 성분끼리 같은 종류, 법선 속도 0, P Neumann 고정, y 벽 미지원.
