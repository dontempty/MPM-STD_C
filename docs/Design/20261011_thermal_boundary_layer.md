# 열 경계층(수동 스칼라) 검증 — apps/thermal_boundary_layer (2026-10-11)

사용자 요청: "열도 같이 푸는 경계층을 새 app 으로 만들어 ① 검증 자료 선정 ② 자료가 비교하는 통계량 분석 ③ 같은 통계량을 논문과 동일 조건에서 해석·검증".

## 1. 검증 자료 선정

| 후보 | 조건 | 공개 데이터 | 판단 |
|---|---|---|---|
| **Li, Schlatter, Brandt & Henningson (2009)** IJHFF 30, 916 — "DNS of a spatially developing TBL with passive scalar transport" | Re_δ*₀ 450, Re_θ 175→830, Pr 0.2/0.71/2.0, isoscalar + isoflux 벽, 스펙트럴(SIMSON) 1024×289×128, 상자 750×40×34 δ*₀ | 논문 그림(디지타이즈) | **주 참조**. 우리 boundary_layer 앱(Blasius 유입 + 트립, KTH 와 같은 틀)과 조건이 1:1 로 맞고, 두 벽 조건·세 Pr 을 한 런에서 다룬다. |
| Kong, Choi & Lee (2000) Phys. Fluids 12, 2555 | Re_θ 300, Pr 0.71, 재순환 유입 | 없음 | Li 2009 가 비교 대상으로 인용 — 값은 Li 본문을 통해 간접 사용 |
| Kasagi/Iida (THTLAB CH122_PG.WL4, 2001) | 채널 Re_τ 150, Pr 0.71, 등온 벽 | **수치표 공개** (y⁺, u⁺, t⁺, rms, ut⁺, −vt⁺) | **벽 근처(y⁺ ≲ 100) 보조 참조** — Li 2009 fig.9 도 이 자료(Kasagi 1992)를 겹쳐 쓴다 |
| Wu & Moin (2010) JFM 630 | Re_θ ≤ 1950, Pr 1, 등온 | 없음 | Pr 1 이라 Li 와 직접 비교 불가; 보류 |
| Araya & Castillo (2012) IJHMT 55 | Re_θ ≤ 2300, Pr 0.71 | 없음 | 보류 |

파일: `docs/Paper/TBL_scalar_Li2009_IJHFF.pdf`, `apps/thermal_boundary_layer/plot/ref/li2009_*.csv`(fig.4·8·9·10·Table 1 판독),
`plot/ref/kasagi_iida2001_ch150_pr071.txt`(THTLAB 원본).

## 2. 참조가 비교하는 통계량 (Li 2009 §3) 과 우리 측정

| 논문 항목 | 정의 | 그림 | 우리 출력 (`bl_scalar<n>_stats_*.dat`) |
|---|---|---|---|
| Stanton 수 St(Re_θ) | q_w/(ρ c_p U∞ (θ_w−θ∞)) ; Kays–Crawford 0.0125 Pr^−0.4 Re_θ^−0.25 | fig.4 | section 1 `St` = α (∂θ/∂z)_w/(u_e (θ∞−θ_w)) (isoflux 는 θ_w 외삽) |
| 레이놀즈 상사 2St/c_f | | (본문) | section 1 `2St_over_Cf` |
| 평균 스칼라 θ⁺, 로그법칙 κ_θ 0.41, 진단함수 Ξ_θ = y⁺ dθ⁺/dy⁺ | θ⁺ = (θ_w−θ)/θ_τ | fig.8a | section 2 `theta+` → 노트북에서 Ξ_θ |
| 스칼라 변동 θ_rms⁺ (isoflux 벽값 ~2.1, Pr 0.71 정점 2.4 @y⁺17) | | fig.8b | section 2 `theta_rms+` |
| 스칼라 플럭스 ⟨u′θ′⟩⁺, −⟨v′θ′⟩⁺ (Pr 스케일링 Pr·y⁺², Pr^0.6) | | fig.9 | section 2 `ut+`, `wt+` (우리 w = 벽수직) |
| 난류 Prandtl 수 Pr_t = ν_t/α_t | (⟨u′v′⟩ dθ/dy)/(⟨v′θ′⟩ dU/dy) | fig.10 | 노트북: 속도 스테이션(uw⁺, U⁺)과 스칼라 스테이션(wt⁺, θ⁺) 합성 |
| 열 경계층 두께 δ⁺99,θ (340 @Pr 0.71, 365 @0.2, 320 @2.0 vs 속도 315) | | Table 1 | section 1 `delta99_theta` |
| 수지·skewness/flatness·PDF·줄무늬 간격·스펙트럼 | | fig.11–18 | 범위 밖 (이번 검증 제외) |
| 속도장 c_f·H·U⁺ | Österlund 상관식, Spalart 670 | fig.3·6·7 | 기존 `bl_stats` (단계 B 통과한 것) |

## 3. 앱 구성 (apps/thermal_boundary_layer)
- boundary_layer 앱 복제 + `[scalars]`(prandtl 목록, wall 목록 isoscalar|isoflux, θ∞, θ_w, ∂θ/∂z|w).
- 스칼라 전진: `scalar::solve_constant_property`(융합 상수물성 ADI, uⁿ) 를 운동량 앞에서 스칼라마다; 벽 닫기 isoscalar −1 / isoflux +1, 상단 −1, 유입 0, 유출 +1.
- 경계: z− isoscalar `fill_wall_mirror(θ_w)` / isoflux `fill_wall_gradient`(신설, CPU/GPU 쌍둥이), z+ `fill_wall_mirror(θ∞)`, x+ 영구배, x− 층류 상사 프로파일(초기 1회 ghost 기입; 비주기 halo 는 no-op).
- 층류 상사해 `thermal_similarity.hpp`: isoscalar Pohlhausen 적분식, isoflux g″+(Pr/2)(f g′−f′ g)=0 두 기본해 결합. IC = 각 x 의 상사 프로파일.
- 통계 `post/scalar_spatial_statistics`(신설) + `post::span_average_scalar_moments`(쌍둥이): (x,z) 6 모멘트 dt 가중 → section 1 (St, θ_w, 구배, δ99θ, θ_τ, 2St/c_f), section 2 (속도 통계와 같은 스테이션 i).
- 체크포인트에 T1..Tn 포함.

## 4. 런 계획
- 층류 검증(스모크): 트립 없음, St(x) vs 상사해 (`plot/laminar_check.py`).
- 층류 검증 결과(GPU, 잡 36010): 벽 계수 θ′(0)=0.2942(Pr 0.71)·0.4223(Pr 2), g(0)=2.4516 (근사 이론 ±1%); **St(x) 상사해 대비 평균 −0.04%, 최대 0.3%** (세 스칼라) → 스칼라 이송·벽 조건·St 평가 정합.
- 본런 `input_tbl_li2009.ini`: 1600×160×192, 750×34×40, Re_δ*₀ 450, 트립 A 0.04 @x 10, 스칼라 θ1..θ5, t 0→9000(통계 2250~, 750 마다 파일), 잡 36011 gpu02 (0.26 s/step).
- 격자 민감도 `input_tbl_li2009_coarse.ini`: 1024×112×144 (논문 명목 Δx⁺ 17·Δz⁺ 7 을 2차 FD 로), 잡 36012 gpu01 (0.09 s/step).
- 노트북 `plot/plot_tbl.ipynb`: §1 모니터 · §2 St(Re_θ)·2St/c_f · §3 c_f·H · §4 Re_θ 830 θ⁺/Ξ_θ/θ_rms/플럭스 · §5 Pr_t · §6 표.

## 5. 결과

### 5-1. 격자 민감도 런(coarse 1024×112×144, 잡 36012) — 중간 (t 6750, 통계 4500 time), 스테이션 Re_θ 830
| 항목 | 우리(coarse) | Li 2009 | 비고 |
|---|---|---|---|
| c_f / H (속도) | 4.70e-3 / 1.478 | Österlund 4.43e-3 / 1.49 | c_f +6% — 경계층 LES 연구에서 본 2차 FD 성긴 격자의 c_f 과대와 같은 종류 |
| St Pr 0.71 iso / isoflux | 2.78e-3 / 2.93e-3 | 2.58e-3 / 2.66e-3 (KC 2.67e-3) | +8% / +10% — c_f 과대와 같이 움직임 |
| 2St/c_f Pr 0.71 | 1.18 | 1.165 (Li fig.4 ÷ Österlund) | **+1%** → 열 수송은 속도장과 정합 |
| St Pr 0.2 / Pr 2 | 4.02e-3 / 1.82e-3 | 3.76e-3 / 1.71e-3 | +7% / +6% |
| θ_rms⁺ 정점 (Pr 0.2 / 0.71 / 2) | 1.29@45 / 2.31@17 / 4.03@9 | 1.35@60 / 2.40@17 / 4.15@11 | −4~−5% |
| θ_rms⁺ 벽값 isoflux (Pr 0.71 / 2) | 1.95 / 3.66 | 2.10 / 3.95 | −7% |
| −⟨w′θ′⟩⁺/Pr 평탄값 (Pr 0.2 / 0.71 / 2) | 3.7 / 1.22 / 0.45 | ~3.5 / ~1.2 / ~0.5 (fig.9b) | ✓ |
| Pr_t Pr 0.71: 벽 / y⁺45 / y⁺220 | 1.04 / 0.95 / 0.54 | 1.05 / 1.07 / 0.72 | 벽 ✓, 바깥층 −11~−25% (coarse·창 4500) |
| δ⁺99,θ (Pr 0.2/0.71/2) · 속도 δ⁺99 | 403/382/363 · 341 | 365/340/320 · 315 | 비 δθ/δu 1.12 vs 1.08 |
- 평균 θ⁺ 는 Kader 법칙과 세 Pr 에서 겹치고, 진단함수 Ξ_θ 는 fig.8a 의 모양(Pr 2 정점 8@y⁺7, Pr 0.71 정점 4.3@y⁺11, y⁺ 60~100 에서 1/κ_θ 근방, 가장자리 2차 정점)을 재현.
- 플럭스 ⟨u′θ′⟩⁺/Pr·−⟨w′θ′⟩⁺/Pr 의 벽 근처 멱(y⁺², y⁺³)과 Kasagi 채널 자료가 y⁺ ≲ 30 에서 겹침.

### 5-2. 본런 (1600×160×192, 잡 36011) — 진행 중 (완주 ~02:00)
