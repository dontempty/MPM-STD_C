# DHVC 통계 설계 — Ng, Ooi, Lohse & Chung (2017) JFM 825 전 항목 검증용 (2026-10-07)

사용자 지시: "나중에 확장해서 돌릴 건데 논문에 있는 통계량 다 포함시키고, 확장 통계에서는 모두 검증할 수 있게
통계 누적하는 양을 다시 확인하고 수정". 논문(`docs/Paper/DHVC.pdf`)의 그림·식마다 우리가 누적하는 양과 파일·노트북 패널을
1:1 로 적는다. 구현: `apps/dhvc/statistics.{hpp,cpp}` (앱 소유), 공용 축소는 `src/post/{statistics,wall_richardson,spectra}`.

## 1. 논문 통계량 ↔ 누적량 ↔ 파일

| 논문 | 정의 | 누적량 (dt 가중) | 파일/열 | 노트북 |
|---|---|---|---|---|
| fig.7a ū⁺(z⁺) | ū/u_τ, u_τ = √(ν dū/dz|_w) | ⟨u⟩(z); 벽 전단은 ghost 정책 자(mirror 벽면↔첫 중심 / value ghost↔첫 중심) | `dhvc_stats`: U, 헤더 u_tau_hot/cold | §2·§5a |
| fig.7b Θ̄⁺(z⁺) | (Θ̄−Θ_h)/θ_τ, θ_τ = −f_w/u_τ | ⟨T⟩(z), f_w = Nu·α·ΔT/H | `dhvc_stats`: T, 헤더 Nu_*, theta_tau_* | §4·§5b |
| Nu (fig.8a 검은 점) | f_w H/(Δκ) | 양 벽 (T_wall − T₁)/z₁ (T ghost 거울) | 헤더 Nu_hot, Nu_cold | §3 Nu 패널 |
| Re (fig.8b) | U_rms(δ_max)·H/ν, δ_max = z(max ū) | ⟨u²⟩(z) → rms at k(u_max) | 헤더 U_rms_delta_max, Re_rms, delta_max | 요약 |
| Re_δ* (fig.10 검은 점) | max(ū)·δ*/ν | ⟨u⟩(z) | 헤더 Re_dstar, u_max, delta_star | §3 |
| 2차 통계 (Ng 2015) | u′u′ v′v′ w′w′ T′T′ u′w′ u′T′ w′T′ | ⟨u²⟩⟨v²⟩⟨w²⟩⟨T²⟩⟨uw⟩⟨uT⟩⟨wT⟩ (슬롯곱) | urms vrms wrms trms uw uT wT | §4 |
| 열·운동량 수지 (검산) | q = ⟨wT⟩−α dT/dz = Nu·α; τ = ν dU/dz − ⟨uw⟩ + ∫(T−dPdx)dz | 위 + mean_dPdx(스텝별 적용분 창 평균) | 헤더 mean_dPdx | §4 |
| §4 |Ri_f| 정의 | 15/L⁺, L = −u_τ³/(gβ f_w) | 열별 국소 u_τ(x,y)·f_w(x,y) 로 |Ri_f| = 15·ν·f_w/u_τ⁴ (gβ=1) | — | — |
| fig.4 PDF P(|Ri_f|) | 열·시간 분포 | log10 히스토그램 60 bins(1e-4~1e2) + under/overflow, dt 가중 | `dhvc_richardson_pdf`: count_hot/cold | §5c (premultiplied) |
| fig.5·6 면적비 ρ(Ra) | low 열 수/(nx ny) | 매 표본 ρ 의 dt 평균 (양 벽 각각) | `dhvc_conditional` 헤더 rho_hot/cold | §5d (+ 피팅 (5.2)(5.3)) |
| fig.7c–f 조건평균 ū_l ū_h Θ̄_l Θ̄_h | (6.2) 조건면적 정규화 | I 가중 평면평균 ⟨u⟩_c ⟨T⟩_c (c = low/high, 양 벽) | `U_{low,high}_{hot,cold}`, `T_…` | §5a,b |
| (6.3)(6.4) 조건 분산 (u′u′)_l | ⟨u²⟩_l − ρ(ū_l)² | ⟨u²⟩_c ⟨T²⟩_c 원시값 (ρ 적용은 노트북) | `uu_…`, `tt_…` (+ `uw_…`, `wT_…`) | 요약 |
| fig.8 Nu_l Nu_h (7.1) | f_w,l = −κ dΘ̄_l/dz|_w | 조건 T 프로파일 벽 기울기 | 헤더 Nu_low/high_hot/cold (빈 class = nan) | §5e |
| fig.8 Re_l (7.1) | U_rms,l(δ_max,l) H/ν | ⟨u²⟩_l, ū_l | 노트북 계산 | 요약 |
| fig.10 Re_δ*,l | max(ū_l)·δ*_l/ν | ū_l | 노트북 계산 vs 1.3·Ra^0.20 | 요약 |
| fig.11 premultiplied 스펙트럼 k_x E_uu/u_τ²(z⁺, λ_x⁺), k_y E_uu | 1-D 에너지 스펙트럼 | E(k_x; z) (y 라인 평균), E(k_y; z) (x 라인 평균), 한쪽, Parseval Σ_m E = ⟨u²⟩_line | `dhvc_spectra` 블록 x/y | §5f |
| fig.12 조건 스펙트럼 | low/high 영역 스펙트럼 | **미구현** — 라인(x)이 열 지시자(i,j)를 가로질러 단순 가중 불가(패치 FFT 필요) | — | — |

## 2. 표본 간격·비용
- `[output] stats_sample_interval`(평면 프로파일 11 축소), `conditional_interval`(분류 2 벽 + 가중 축소 24), `spectra_interval`(u 스펙트럼).
  간격 사이 dt 합을 그 표본의 가중으로 쓰므로 dt 가중 평균이 유지된다. 1e8(1536×340×768, 4 GPU) 은 축소 1회 ≈ 10~20 ms
  (atomicAdd) 라 매 스텝 35 회는 +50% 이상 → **10 스텝(0.006 time)** 권장(덱에 기록). 1e5·1e6 은 매 스텝, 1e7 은 5.
- 스펙트럼: 라인 FFT 를 필드 메모리 strided 로 직접(복사 없음), j 또는 i 마다 nz_local 배치 → 1e8 에서 호출 340+1536 회/표본,
  500 스텝마다. 출력 nz×(nx/2+1) ≈ 8 MB 텍스트/파일.
- 조건 분류는 벽 랭크가 계산(nx·ny 열)해 지시자(8 B×nx·ny ≈ 4 MB)를 z 랭크에 Bcast — 10 스텝마다라 무시.

## 3. 검증 (CPU 스모크 32×16×32, np=2 — 두 벽이 다른 랭크)
mirror·value 둘 다 120 스텝 완주·NaN 없음, 파일 4종 생성. Parseval: E_x[m=0] = Ū² (0.0417 = 0.204²) ✓.
열 수지 q/α 1.000~1.004(전도 Nu=1), 운동량 수지 τ ≈ u_τ,hot² ✓ (§4). 라미나 스모크라 ρ = 1(전 열 |Ri_f| ≈ 0.02 ≤ 0.1).
GPU 쌍둥이(statistics.cu·wall_richardson_kernels.cu·spectra_kernels.cu, cuFFT 경로)는 빌드 잡으로 확인(gpu02).

## 4. 남은 것
- fig.12 조건 스펙트럼(패치 FFT) — 필요 시.
- 조건평균의 통계 수렴은 ρ 가 작은 Ra(1e5 ρ≈0.06)에서 low 표본이 적어 느리다 — 창 100 time 이상 권장.
- 돌고 있는 1e8 mirror(35834) 는 구형 통계 — 확장 통계가 필요하면 새 바이너리(build_gpu, 잡 35929 빌드)로 재시작.

## 추가 2026-10-09 — 스펙트럼 cuFFT 레이아웃 버그 수정, 1e7 mirror 재검증 런
- 1e7 mirror + 확장 통계 런(잡 35986, 1e6 warm start)이 통계 시작 t=100 의 첫 스펙트럼 표본에서 `cuFFT failed (spectra): ExecD2Z x` 로 중단.
  원인: `LineSpectrumPlan` 이 cufftPlanMany 의 advanced layout 으로 필드 메모리를 **원소 보폭 ny·nz, 배치 간격 1** 로 직접 읽게 했는데
  cuFFT 는 이 (보폭 > 배치 간격) 레이아웃을 실행 시 거부한다. FFTW 경로도 같은 전제라 한 번도 검증된 적이 없었다(스펙트럼 파일이 어느 out_* 에도 없음).
- 수정: `gather_lines`(spectra_kernels 쌍둥이)로 nz_local 라인을 연속 [line][i] 버퍼에 모은 뒤 cufftPlan1d / fftw_plan_many(연속) 배치 변환.
  스모크(48×24×32, np 2, `input_dhvc_smoke.ini` 에 stats_sample 2·conditional 2·spectra 10 추가): 4종 파일 전부 생성, **CPU = GPU 비트 동일**
  (spectra rel diff 0, stats 7e-19, conditional·PDF 0), **Parseval Σ_m E(k;z) = U̅² + u_rms²** (같은 표본 간격일 때 1.000000, 간격이 다르면 ±4% 는 표본 차).
- 아울러 GPU 빌드에 .d 의존 파일이 없던 문제(Makefile.inc) 때문에 이전 GPU dhvc 바이너리가 새 통계 키를 몰랐던 것도 2026-10-09 수정(커밋 b579987).
- 재시작: `input_dhvc_1e7_mirror_restart.ini`(restart_1e7_mirror/dhvc_ck_00020000.bin, t=52.35) 잡 35992(gpu02, 05:00) — 통계 100~200, 완주 ~07:00.
