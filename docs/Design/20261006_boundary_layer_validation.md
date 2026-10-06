# 경계층(ZPG 평판 boundary layer) 검증 설계 (2026-10-06, 초안 — 결정 대기 항목 §6)

> BC 프레임워크(`docs/Design/20261005_boundary_condition_framework.md`) 다음 단계. cavity 가
> "벽 Dirichlet/Neumann" 을 닫았다면, 경계층은 **유입(inflow)·유출(outflow)·자유류(free-stream)**
> 경계와 **공간 발달(spatially developing) 통계**를 처음으로 연다.

## 1. 축 관례와 문제 설정
- x = 유동방향(비주기: 유입 x−, 유출 x+ → 압력은 x-DCT, Neumann 양끝 — 이미 있음),
  y = 스팬(주기, FFT), z = 벽수직(z=0 벽, z=Lz 자유류; TDMA, tanh 스트레칭 가능).
  → 현 압력 솔버 제약(x·y 균일, z 만 비균일)과 **정확히 맞는 배치**다.
- 무차원: 길이 δ*₀(유입 배제두께), 속도 U∞, ν = 1/Re_δ*₀. Schlatter & Örlü 와 같은 Re_δ*₀ = 450
  (Re_θ,0 ≈ 174) 유입 Blasius 프로파일.

## 2. 참조 자료
| 단계 | 자료 | 비교 항목 |
|---|---|---|
| A. 층류 (2D, ny=1) | **Blasius 상사해** (f″(0)=0.33206): C_f = 0.664/√Re_x, δ* = 1.7208 x/√Re_x, θ = 0.66412 x/√Re_x, H = 2.5911, δ₉₉ = 4.91 x/√Re_x, 자유류 w∞ = 0.8604 U∞/√Re_x | C_f(x), δ*(x), θ(x), H, u/U∞(η) 프로파일, 상단 w; 격자 EOC ≈ 2 |
| B. 난류 ZPG (3D) | **Schlatter & Örlü 2010 JFM 659:116** DNS (KTH, 8192×513×768 스펙트럴), 공개 파일 `vel_XXXX_dns.prof`, Re_θ = 670·1000·1410·2000(·2540~4060). 헤더: Re_θ, Re_δ*, Re_τ, H₁₂, c_f. 열: y/δ₉₉, y⁺, U⁺, u′v′w′ rms⁺, uv⁺, p rms⁺, S(u), F(u), dU⁺/dy⁺, V⁺, ω rms⁺. 예: Re_θ 1410 → Re_τ 492.2, c_f 0.003885, H 1.429 | C_f, H, Re_τ(Re_θ) 성장; U⁺(y⁺), rms⁺, uv⁺ 프로파일; 로그법칙(κ 0.41, B 5.2) |
| B 보조 | Spalart 1988 (Re_θ 300/670/1410), Jiménez et al. 2010 (≤1968), Sillero et al. 2013 (Torroja, ≤6500) | 교차 확인용 |
저장: `apps/boundary_layer/plot/ref/vel_{0670,1000,1410,2000}_dns.prof` (클러스터에서 curl 가능 여부는 §7).

## 3. 새로 필요한 코드 (전부 BC 프레임워크 확장)
1. **면 kind 추가** (`[boundary]`):
   - `x = inflow/outflow` 축 kind → x−: `inflow` (U = 프로파일(z) Dirichlet — 법선 성분에 **비영·공간 가변** 값: 지금은 법선 0 고정이라 확장 필요; V=W=0; P Neumann),
     x+: `outflow` (대류 BC ∂φ/∂t + U_c ∂φ/∂x = 0, U,V,W 전부; U_c = U∞ 또는 국소 평균; P Neumann).
   - z+: `freestream` (∂U/∂z = ∂V/∂z = 0; **W 는 연속식**으로 ∂W/∂z = −∂U/∂x − ∂V/∂y → 경계층 배제효과가 유도하는 유출을 허용; P Neumann).
     대안: U = U∞ Dirichlet + W = 0(높은 도메인), 또는 층류 전용 w∞(Blasius) Dirichlet — §6 결정.
   - `FaceBc` 를 스칼라 값 → **면 배열(profile)** 로 확장 (`inflow_profile = blasius | file:<경로>`). apply 층은 배열로 ghost/면 채움.
2. **Poisson 양립성**: 전 경계 Neumann 이므로 투영 전 유출면 유량을 스케일해 ∮u·n = 0 강제
   (`correct_bulk_velocity` 의 유출면 판: 유입 − 상단유출 = 유출). 없으면 DCT 해가 평균 모드에서 어긋남.
3. **운동량 닫기**: 유입 법선 행 = 항등(δU = ΔU_in, 정상 유입이면 0); 유출면은 대류 BC 로 먼저 명시 갱신 후 항등행(δ = δ_out); 접선 성분은 유입 mirror(프로파일값)·유출 Dirichlet(갱신값).
4. **난류 트립** (단계 B): Schlatter & Örlü 2012 식 — 벽 근처 국소 체적력 F_z = A·exp[−((x−x_t)/ℓ_x)² − (z/ℓ_z)²]·g(y,t), g 는 스팬 랜덤·시간 저역통과(컷오프 t_s). `[trip]` 섹션. Re_θ ≈ 180 에서 트립 → Re_θ ≳ 600 에서 트립 무관(그들의 결론).
5. **공간 발달 통계** (`post/`): 스팬+시간 평균 2D 장 ⟨U⟩(x,z), ⟨uu,vv,ww,uv⟩(x,z) → C_f(x), δ₉₉, δ*, θ, H, Re_θ(x); 목표 Re_θ 스테이션은 x 보간. 출력 간격별 파일.
6. **앱 `apps/boundary_layer/`** + 전용 settings(U∞, Re_δ*₀, 트립, 스테이션; rbc 스키마 아님) + `plot/compare_schlatter.py`.

## 4. 격자·비용 추정 (δ*₀ 단위, Re_δ*₀=450)
| | 단계 A 층류 | 단계 B-1 Re_θ ≤ 1000 | 단계 B-2 Re_θ ≤ 1410 (+2000) |
|---|---|---|---|
| 도메인 Lx×Ly×Lz | 1000 × (1셀) × 60 | 1000 × 60 × 80 | 2000 × 80 × 100 |
| 격자 | 1000 × 1 × 128 (z tanh) → 1.3e5 | 2000 × 256 × 192 → 9.8e7 | 4000 × 320 × 256 → 3.3e8 |
| 해상도 | Δx = δ*₀, 1차 셀 z = 0.02 | Δx⁺≈10, Δy⁺≈5, z⁺₁≈0.5 | 〃 |
| 자원/시간 | CPU 1랭크, 분 단위 | 4 GPU ~0.15 s/step × 3e4 → **~1.5 h** | 4 GPU ~0.5 s/step × 5.5e4 → **~8 h** |
(u_τ/U∞ ≈ 0.044 @Re_θ 1410 → ν/u_τ ≈ 0.05 δ*₀; δ₉₉(1410) ≈ 25 δ*₀; 유동 통과 Lx/U∞ 당 Δx/U∞ 스텝.
B 는 전이 3 통과 + 통계 8 통과 가정. 메모리 3.3e8 셀 × ~20 배열 × 8 B ≈ 53 GB → 4 GPU 13 GB/장 OK.)

## 5. 합격 기준
- A: 0.2 < x/Lx < 0.8 에서 C_f·δ*·θ 가 Blasius 대비 **≤ 1%**, H = 2.591 ± 0.5%, 격자 EOC ≈ 2,
  유출 영향이 마지막 ~10% 구간에 국한.
- B: Re_θ = 670·1000(·1410) 에서 C_f ±3%, H ±1%, U⁺ 로그 영역 ±2%, rms⁺ 피크 ±5%, uv⁺ ±5%;
  Re_θ ≥ 600 에서 트립 무관성(트립 세기 2종 비교).

## 6. 결정 (2026-10-06 사용자)
1. 범위: **A 층류 → B 난류 연속**.  2. 난류 유입: **재순환-재스케일(Lund 1998) 또는 SEM/XSEM** — 계산 시간
   고려; 참조 데이터 유무 확인 후 선택(§8).  3. 상단: **∂u/∂z = ∂v/∂z = 0 + w 연속식**.
4. B 목표: **Re_θ ≤ 1100** (2026-10-06 확정 — Torroja 1100 교차 포함).
   실행 계획: 지금은 **CPU 로 작동 확인만**(GPU 는 DHVC 1e8 점유); 4 GPU 가 풀리면 **Re 2종을 2 GPU 씩 병렬**.
5. "DNS 로 충분하면 LES 선행은 불필요" — 비용 판정 §8 → **DNS 로 간다**. (LES 는 나중에 독립 기능으로:
   가변물성 경로에 ν_eff = ν + ν_sgs 를 넣으면 되고, 참조 Fortran 은 Const-Vreman Cs=0.12·Pr_t=0.7.)

## 7. 자료 확보 (완료)
- KTH Schlatter & Örlü 2010: `plot/ref/vel_{0670,1000,1410,2000}_dns.prof` (527행; 로그인 노드 curl OK).
- Torroja (Jiménez 그룹, **재순환 유입** DNS, Simens et al. 2009 JCP 228:4218 / Jiménez et al. 2010 JFM 657:335):
  `plot/ref/torroja/Re_theta.{1100,1551,1968}.prof` + `bl2000.tar`(적분량). 공개 스테이션은 1100 부터 —
  600~1000 구간은 적분량(bl2000.tar)만. 인증서 문제로 `curl -k`.
- SEM/XSEM 으로 만든 **DNS 급 공개 참조 데이터는 없음** — SEM 은 LES 유입 도구(Jarrin 2006/2009, Pamiès 2009:
  회복 길이 ~10–20 δ). 재순환은 Torroja 가 DNS 정본이고 Simens 2009 는 "유입 뒤 300 θ 는 버려라"고 명시.

## 8. 비용 판정과 선택
Re_θ ≤ 1100, 재순환 유입 Re_θ,in ≈ 600: Lx ≈ 300 θ_in(버림) + ΔRe_x 2.5e5(600→1100, dRe_θ/dRe_x ≈ C_f/2)
+ 유출 버퍼 ≈ 4.8e5 ν/U∞ → Lx⁺ ≈ 2.2e4; Ly ≈ 3 δ₉₉(출구) → Ly⁺ ≈ 1200; Lz ≈ 4 δ₉₉.
격자 Δx⁺ 10 · Δy⁺ 5 · z⁺₁ 0.5(tanh, nz≈200) → **2200 × 240 × 200 ≈ 1.1e8 셀**. 스텝당 4 A100 ≈ 0.15 s
(DHVC 4e8 셀 0.57 s 비례), 통과당 ~4400 스텝, 버림 3 + 평균 8~10 통과 → ~5.5e4 스텝 → **≈ 2.5 h (4 GPU)
/ ≈ 5 h (2 GPU)**. 메모리 ~18 GB. → **DNS 로 충분**; LES 선행 불필요.
유입 방식: 세 방식 모두 회복 길이가 같은 급(트립 Re_θ 180→600 ≈ 500 δ*₀, 재순환 300 θ ≈ 35 δ, SEM 10–20 δ)이라
비용 차이는 작고, **DNS 참조와 같은 방식인 재순환**을 1차로 택한다(`inflow = recycling`). SEM/XSEM 은 같은
훅(`inflow = sem`)으로 뒤에 붙일 수 있게 설계(유입면 프로파일+섭동 공급자 인터페이스).
비교: Re_θ 670·1000 은 Schlatter & Örlü(트립 — 평형 경계층은 유입 방식과 무관), 1100 은 Torroja(재순환)로
교차 → 목표를 **Re_θ ≤ 1100** 으로 잡으면 두 DB 를 모두 쓴다.

## 9. 단계 A — 2D 층류 Blasius ✅ (2026-10-06, CPU)

**만든 것**
- `grid::make_tanh_wall_faces` (한쪽 벽 tanh), `boundary/open.{hpp,cpp,cu}`:
  `set_inflow`(U 면 = u(z) 프로파일, V·W ghost 거울/값), `update_outflow_convective`(∂φ/∂t + U_c∂φ/∂x = 0,
  uⁿ 으로 명시 갱신 → 음해 스윕은 δ=0), `set_top_normal_from_continuity`, `set_top_normal_profile`,
  `correct_outflow_flux`(∮u·n = 0, Neumann Poisson 양립). BC kind `Inflow/Outflow/Continuity/FreeStream`,
  축 kind `inflow_outflow`, `inflow_profile = blasius`, 면 키워드 `continuity | freestream`.
- `apps/boundary_layer/` (settings·main·blasius.hpp·`plot/blasius_check.py`), 덱 `input_bl_blasius.ini`
  (600×1×128, Lx 600, Lz 40, γ_z 2.5, Re_δ*₀ 450 → x₀ 152, dt 0.5, t 1200 = 2 통과; CPU 1랭크 2.5 분).

**★ 발견 — 자유류 상단 BC**: `∂u/∂z = 0 + w 연속식`(또는 u = U∞ 거울 Dirichlet)은 **ZPG 를 유지하지 못한다**:
유입 상단으로 들어온 Blasius w∞(x₀) = 0.0033 이 비점성 자유류를 따라 그대로 이류되어(참값은
0.86/√Re_x 로 감소) 질량이 과다 유출 → 자유류 1.0 → 0.977 감속(역압력구배) → θ +55%, H 2.1, C_f −6~11%.
u = U∞ Dirichlet 은 점성 앵커일 뿐이라 무효(결과 4자리 동일). 처방 = Schlatter/SIMSON 식
**상단 w = Blasius w∞(x) 프로파일 처방**(`w_z_plus = freestream`, 난류 단계는 U∞·dδ*/dx 공급).

**결과 (`w_z_plus = freestream`, 600×128)**
| x/Lx | Re_x | C_f err | δ* err | θ err | H |
|---|---|---|---|---|---|
| 0.1~0.8 | 9.5e4~2.8e5 | −0.08 … −0.01% | −0.02 … −0.01% | −0.07 … −0.02% | 2.5926 … 2.5914 (Blasius 2.5911) |
| 0.9 / 0.95 | 3.1e5 / 3.2e5 | +0.10 / +0.60% | −0.01% | +0.04 / +0.28% | 2.590 / 2.584 (유출 영향 ≲ 마지막 10%) |
프로파일 max|u/U∞ − f′(η)| (η<8) = 2.0~2.3e-4 (3 스테이션); 자유류 u = 1.000 ± 1.5e-4; max|div| 1e-16;
유량 보정 −1.9e-5 (정상, Blasius w∞ 처방과 이산 유량의 상수 차). 합격 기준(≤1%) 대비 **10배 여유**.
- 격자 반감(300×64, `input_bl_blasius_coarse.ini`): C_f 0.10% / δ* 0.06% / θ 0.05%, 프로파일 4.4~5.6e-4 →
  600×128 의 0.07/0.01/0.07%, 2.0~2.3e-4. 두 격자 모두 Blasius 와의 차이가 **Blasius 근사 자체의
  O(Re_x^{-1/2}) 오차·유출 영향 수준**이라 EOC 를 Blasius 대비로는 못 재며(수렴 완료 판정으로 충분),
  엄밀 EOC 는 `test_pressure_wall` 류 제조해로만 의미가 있다.
- np = 2 (z 분할): FINAL C_f(Lx/2) 동일(−0.04%) — 열린 경계 함수의 랭크 소유 판정·Allreduce 정상.
  (장 덤프는 npz = 1 전제라 np=2 에선 생략 — 단계 B 통계 모듈이 대체.)

## 10. 단계 B-① 재순환-재스케일 유입 (2026-10-06, CPU 작동 확인)
- `boundary/recycling.{hpp,cpp}`: Lund 1998 — 재순환면 평면 Allgatherv(z 분할) → 스팬 평균/변동 →
  완화 시간평균 ⟨Ū_r⟩ 로 u_τ,r·δ_r·θ_r → γ = (δ_r/δ_in)^{1/8} → 내부(γz)/외부(z δ_r/δ_in) 재배치 + W(η) 혼합
  → 유입 평면. `open.*` 의 유입이 z 프로파일 → **(y,z) 평면**(`InflowPlane`)으로 일반화, `extract_x_plane`.
- 설정 `[inflow] method = blasius | recycling, recycle_x, delta99, relaxation_time, noise`; `[physics] reynolds`
  (단위길이 1 = blasius 면 δ*₀, recycling 이면 δ99,in). 재순환 IC = 1/7 멱법칙 + 경계층 안 백색 섭동.
- 작동 확인 (`input_bl_recycling_cpu.ini`, 192×32×64, Re_δ99 2000 ≈ Re_θ 230, np=2, 0.16 s/step):
  안정(div 1e-14, 유량 보정 1e-5), 스케일 산출 정상(u_τ,r·δ_r·θ_r·γ·Re_θ 모니터). 저Re 라 층류화(γ→1)
  — 물리 검증은 GPU 급(Re_θ,in ≈ 600, Δx⁺ 10) 에서. ⚠ 재순환 모드의 상단 w_top 은 아직 0 (단계 ②).

## 11. 단계 B-②③ 상단 자유류 w + 공간 발달 통계 (2026-10-06, CPU 작동 확인)
- `post/statistics`: `span_average_center_moments` (셀중심 보간 uc·vc·wc 의 7 모멘트, 스팬 평균; CPU/GPU,
  (i,k) 당 1 스레드라 비결정 없음). `post/spatial_statistics`: relaxed 평균(τ) + dt 가중 누적, z Allgatherv
  로 전역 (x,z) 장을 만들어 C_f·δ99·δ*·θ·H·Re_θ·Re_τ(x) 와 스테이션(목표 Re_θ 최근접 i) U⁺·rms⁺·uw⁺ 를 쓴다
  (`bl_stats_<step>.dat`: section 1 적분량, section 2 스테이션). `[output] stats_start_time, stations`.
- ② 상단 w(x) = U∞ dδ*/dx: relaxed δ*(x) 를 0.1~0.9 Lx 2차 최소제곱 → 도함수(음수 0 클램프), 모니터 간격 갱신
  (`free_stream_w`). 재순환 모드에서만; blasius 모드는 해석 w∞.
- 소형 CPU 케이스에서 파일 생성·z 분할 gather·스테이션 선택 확인 (저Re 라 물리 의미 없음).

## 12. 단계 B-④⑤ 비교 스크립트 + GPU 덱 준비 (2026-10-06)
- `plot/compare_schlatter.py`: `bl_stats` 파일 ↔ KTH(`vel_*_dns.prof`: 헤더 Re_θ·Re_τ·H₁₂·c_f, 열 y/δ99,y⁺,U⁺,rms⁺,uv⁺)
  / Torroja(`Re_theta.*.prof`: 헤더 u_τ·Re_θ·Re_τ·δ99·δ*·θ → C_f = 2u_τ², H = δ*/θ; 열 y/δ99,y⁺,u/v/w rms⁺,uv⁺,U⁺).
  출력: 참조 Re_θ 에서의 C_f·H·Re_τ 오차(우리 Re_θ(x) 보간) + 상관식(Smits 0.024 Re_θ^{-1/4},
  Coles–Fernholz κ 0.384·C 4.127) + 스테이션별 로그영역 U⁺ RMS 차·rms⁺ 피크·uw⁺ 최소. `--png` 그림.
- GPU 덱 2종 (아직 **미제출** — 4 GPU 가 풀리면 2 GPU 씩 동시):
  | 케이스 | Re_δ99,in | Re_θ,in≈ | 격자 | 셀 | 스테이션 | 노드 | 예상 |
  |---|---|---|---|---|---|---|---|
  | A `input_bl_re4500.ini` | 4500 | 600 | 2048×192×192, Lx 100·Ly 5·Lz 6 | 75M | 670, 1000 | gpu01 (A100 40GB×2) | ~0.22 s/step × 3.3e4 ≈ 2 h |
  | B `input_bl_re6000.ini` | 6000 | 800 | 2560×224×192 | 110M | 1000, 1100 | gpu02 (80GB×2) | ≈ 3 h |
  두 케이스가 Re_θ 1000 을 공유 → 상호 일관성 확인. 유입 뒤 300 θ(≈40 δ99,in) 버림, 재순환면 x = 80,
  통계 t 300~1300 (10 통과). Δx⁺ 10~13, Δy⁺ 5, z⁺₁ ≈ 0.4.
- ⚠ 남은 위험: (i) 재순환 IC(멱법칙+백색 섭동)가 저Re 소형 CPU 케이스에선 층류화 — GPU 급 Re_θ 600 에선
  전이 예상이나 실패 시 트립 체적력 추가; (ii) 재순환면-유입 상관(가짜 주기) → 통계 창에서 스펙트럼 확인;
  (iii) 상단 w = U∞dδ*/dx 의 2차 적합이 초기 과도기에 흔들릴 수 있음(모니터 top w_max 로 감시).
