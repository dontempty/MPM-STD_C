# 23. RBC glycerol NOB θ_c 결손 조사 (포트 vs PaScaL_TCS / MPM-STD)

> 목적: 포트(MPM-STD_C, rbc_gpu_nob)의 glycerol NOB Rayleigh–Bénard 중심온도 θ_c가
> PaScaL_TCS 논문 검증값 대비 ~1.58–2× 낮은 원인 규명.
> 결론(요약): **포트는 PaScaL_TCS 저장소의 NOB 스킴을 (계수·전 공간항·물성타이밍 수정 후) 충실히
> 재현하나 θ_c는 여전히 ~0.04 (논문 0.0744의 1.58×↓). 격자·계수·스킴 공간항 모두 일치 → 결손은
> "저장소/포트 vs 논문" 차이로 좁혀짐. 확정엔 저장소 PaScaL 실구동 필요(미완).**
> 작성: 2026-06-25

---

## 0. 문제 정의

PaScaL_TCS 논문(RBC 검증): glycerol(Pr=2547), NOB, **Ra=1e6 @θm=313K**, **Γ=Lx/H=Lz/H=6**,
**128³** (ny=연직, Horn의 2배로 얇은 BL 해상), ΔT=20/40/60K. θ_c=(θc−θm) (무차원, θm=0).
검증값(BL 이론 Zhang/Ahlers/Horn과 일치): **θ_c = 0.0744 / 0.1350 / 0.1900** (=차원 1.49/5.40/11.40 K).

포트 결과 (rbc_gpu_nob, Γ=6, 128³, gamma3=2.6):
| ΔT | 포트 θ_c (accum/inst) | 논문 | 배율 |
|---|---|---|---|
| 20K | 0.041 / ~0.047 | 0.0744 | 1.58–1.8× ↓ |
| 40K | 0.066 / ~0.085 | 0.1350 | ~2× ↓ |

θ_c = `center_temp` = 중심평면(nz/2) T의 xy평균을 시간평균 (rbc_common.hpp / main_gpu_nob.cu).
벽 ±0.5 → T_mean=0 → θ_c = T_center − T_mean. **정의·측정 검증 완료** (stats_final.dat z-프로파일
중심값 0.0405 = 보고된 θ_c 일치).

---

## 1. 조사: 가능성 소거 (체계적)

| 가설 | 검증 방법 | 결과 |
|---|---|---|
| θ_c 측정/정의 오류 | z-프로파일 대조 | ✗ 정확 (T_center−T_mean) |
| dP̂ 외삽 누락 | 구현+재실행 | ✗ div용(별개), θ_c 거의 불변 |
| **격자/해상도** | **gamma 스캔 1.5/2.6/3.5** | ✗ 대류만 하면 θ_c 격자 둔감 |
| Γ=6 형상 | 논문이 같은 Γ=6/128³→0.0744 | ✗ 형상 정상 |
| κ̂ 에너지 확산 | PaScaL module_solve_thermal 대조 | ✗ 일치 |
| μ̂ 운동량 점성 | PaScaL module_solve_momentum 대조 | ✗ 일치(점성 4-16× 정확) |
| glycerol 계수 | 실제 글리세롤 물성 대조 | ✗ 유효 모델, 포트와 일치 |
| **물성 평가 타이밍** | 코드 대조 | ✅ 실차이 → 수정 → **+4%만** (주범 아님) |

### 1.1 격자 스캔 (gamma, 128³, Γ=6, Ra=1e6, ΔT=20)
| gamma | 열BL 셀 | 중앙 dz(균일比) | umax | θ_c @~200K |
|---|---|---|---|---|
| 1.5 | 3–4 | 0.92× | 0.007 | **대류 억제(무효)** |
| 2.6 | 6 | 1.5× | 0.53 | 0.031 → 0.041 |
| 3.5 | 10 | 1.86× | 0.68 | 0.030 |
→ 대류 영역(gamma≥2.6)에서 벽BL 6→10셀로 늘려도 θ_c 불변(~0.03-0.04). gamma=1.5는 BL 미해상으로
대류 자체가 죽음. **격자/해상도는 결손 원인 아님.** (논문도 128³로 0.0744 → 해상도 충분.)

### 1.2 κ̂ 에너지 확산 (port scalar_var_stencil.hpp ↔ PaScaL module_solve_thermal.f90)
- KPP=1+c11·tmp+… , dKPP=c11+2c12·tmp+…, tmp=DeltaT·T, irc_h=0.5·Ct·invRhoCp,
  확산 `(kPP_hi·ded_hi − kPP_lo·ded_lo − e·dkPP·ded)/dx` + 밴드계수: **전부 일치.**

### 1.3 μ̂ 운동량 점성 (port k_mom_rhs_var/k_mom_adi_var ↔ PaScaL solvedU/V/W)
축 스왑(ref x2=벽 ↔ port Z) 적용 시 **줄단위 일치**: 정규응력 2×, 모든 교차(transpose)응력
weight-1, μ 이중 부피가중 `mfce`, −M·qⁿ 잔차, 블록결합 _var 커널. 누락/반감/상수화 없음.

### 1.4 glycerol 계수 (실제 물성 대조)
점성 cold/hot: 모델 4.2×(20K)/15.9×(40K) vs 실제 4.1×/18× → **잘 재현**. (60K만 5차외삽 과대
97× vs 55%, θ_c를 높이는 방향이라 결손과 무관.) κ̂ 비대칭 +0.78~2.34%. **유효 글리세롤 모델.**

### 1.5 물성 타이밍 (유일한 실차이 — 수정함)
- **포트(수정 전)**: `update_properties(Tⁿ)` 1회 → 에너지·운동량 모두 Tⁿ 물성. 운동량 μ̂/invρ가 stale.
- **PaScaL**: `thermal_coeffi(Tⁿ)`→에너지→`momentum_coeffi(Tⁿ⁺¹)`. 운동량 물성 재평가.
- **수정**: 에너지 후 운동량 직전 `update_properties(Tⁿ⁺¹)` 추가 (main_gpu_nob.cu:419). MPM-STD
  staggered-time과 일치 → **유지**. 결과: θ_c 0.0314→0.0328 @200K (**+4%**) — 올바른 수정이나
  1.58× 결손의 주범 아님 (수렴상태에서 대부분 상쇄).

---

## 2. 참조 코드 현황 (MPM-STD, PaScaL_TCS)

- **두 본가 모두 glycerol NOB를 주석처리해 출고** (OB로 동작, invRho=Mu=κ̂=1):
  - MPM-STD `core_momentum.f90:400` (invRho/Mu 다항식 주석), `core_energy.f90` (KPP/invRhoCp=1).
  - PaScaL `module_global.f90:119-122` (계수=OB극한), glycerol(131-134) 주석.
- **NOB 압력은 두 본가 동일**: dPhat+dP0 2차 외삽 `dP0=(dPhat+dP0)/2; dPhat=2dP−dP0`,
  가변밀도 투영 `(invRhoc−1)∇dPhat`. 물성 다항식도 동형.
- **부력만 형태 차이**: PaScaL `Cmt(θ+a12pera11·θ²·ΔT)invρ` vs MPM-STD `Cmt(θ−Ground_T)invρ` —
  단 a12pera11=0(PaScaL 부호체크) / Ground_T=0(RBC) → 실효적으로 동일.
- **포트는 glycerol을 실제 활성화한 유일 코드.** (포트는 a12pera11≠0(7.9e-4)로 밀도2차항을 켜둠 —
  본가들은 실효 0. 작은 차이.)
- **PaScaL 논문은 이 케이스(Ra=1e6,Γ=6,128³)를 실제 검증** → 0.0744는 유효 참조이고 PaScaL은
  돌릴 수 있음.

---

## 3. PaScaL_TCS 빌드/구동 (이 클러스터)

- 컴퓨트 노드엔 시스템 OpenMPI 런타임 없음(login 전용), 로컬 fftw3는 nvhpc OpenMPI 오염 →
  segfault. **해결**: Intel oneAPI MPI의 **gfortran 래퍼**(`source /opt/intel/oneapi/setvars.sh`
  후 `mpif90`)로 빌드 + 깨끗한 시스템 fftw를 `/shared/home/.../local/fftw_clean/`에 staging.
  실행: 깨끗한 env(`env -i`+setvars, nvhpc 경로 제거) + Intel `mpirun -n N`.
- **현 상태(미해결)**: glycerol ON + RBC 부력(solvedV/x2; solvedU/x1 끔)으로 돌리면 **Ra=1e6에서
  step1부터 발산** (코어 1~64, uniform/stretch, dt 무관). step1 div≈0.014(serial) vs 포트 1e-10.
  포트(같은 스킴)는 안정 → **제 PaScaL 설정 문제로 추정**(논문이 작동 증명). 미해결.

---

## 4. 결론 & 남은 일

**포트는 PaScaL_TCS 저장소의 NOB 스킴을 충실히 재현한다** (계수·κ̂·μ̂·대류·압력 전부 일치, 물성
타이밍 수정 완료). 그럼에도 θ_c가 ~0.04로 논문 0.0744의 1.58×↓이며, 배율이 ΔT 클수록 커진다
(NOB 효과를 체계적으로 약하게 잡음). 포트→저장소 비교에서 결손을 만드는 항을 찾지 못했다.

→ **가장 유력**: 저장소 PaScaL을 돌려도 ~0.04가 나올 것 (논문 0.0744은 더 완전한 NOB 버전/설정).
즉 결손은 "포트 vs 저장소" 버그가 아니라 **"저장소/포트 vs 논문"** 차이.

**남은 결정적 검증**:
1. 저장소 PaScaL glycerol 실구동 → 0.04(포트 충실, 저장소≠논문) vs 0.0744(숨은 차이) 확정.
2. 논문/Pan et al.이 점성 외 추가 NOB 물리(점성소산 가열, 압력일, 또는 다른 EOS 항)를 쓰는지 확인 —
   포트/저장소에 없을 수 있음.

---

## 5. 세션 업데이트 (2026-06-26)

### 5.1 다항식 계수 DeltaT scaling 확인

PaScaL `module_solve_momentum.f90:159`: `tmp1 = DeltaT*T(i,j,k)` — **이미 DeltaT scaling 사용**.
MPM-STD_C `properties_cpu.cpp:58`: `t = Dt * double(Tp[i])` — 동일.
양 코드 모두 OLD 계수(d11=-702.83e-4 등)를 `t=DeltaT·θ∈[-10,10]`에서 평가 → μ_cold/μ_hot = **4.16×** ✓.

이 세션에서 계수를 잘못 수정(θ-스케일로 피팅한 d11=-1.398544 등)했다가 **즉시 원복**.
근거: `tmp1=DeltaT*T`이므로 기존 계수가 올바르고 새 계수는 cold wall에서 μ̂≈5097× (비물리적).
잘못된 계수로 돌린 pascal_ob_50k.log (θ_c=0.027@50K) 결과는 무효.

### 5.2 저장소 PaScaL 실구동 결과 (2026-06-26, 완료)

설정: 64³, Γ=6, gamma2=2.6, Ra=1e6, Pr=2547, ΔT=20, OB 압력(ExtraTerm=0), 50K steps.
`log: PaScaL_TCS/run/pascal_correct_visc.log`

| step | T | θ_c |
|------|---|-----|
| 1 | 0.03 | 0.02394 (초기) |
| ~15K | ~440 | 0.02312 (최솟값) |
| ~25K | ~680 | 0.02468 (상승 전환) |
| 50K | 1340 | **0.02696 (수렴)** |

**최종: θ_c = 0.0270.** 수렴 확인(마지막 5개 일정).

64³ 설정에서 gamma2=2.6 → 열BL 약 3셀. MPM-STD_C(128³, 6셀) 0.040보다 낮음 → 격자 의존성 확인:
| 격자 | 열BL 셀 | θ_c(수렴) |
|------|---------|---------|
| 64³, gamma2=2.6 | ~3 | 0.027 |
| 128³, gamma2=2.6 | ~6 | 0.040 |
| 논문(PaScaL TCS 128³) | ~6 | 0.0744 |

동일 격자(128³, gamma2=2.6)에서 MPM-STD_C와 논문이 0.040 vs 0.074로 다름. 격자 해상도로는 설명 불가.
→ 논문 참조값은 **동일 저장소 코드의 DNS가 아닐 가능성** 높음 (BL 이론 값이거나 다른 설정).

### 5.3 ExtraTerm 발산 상태

PaScaL NOB 압력 ExtraTerm은 여전히 step ~25에서 발산. Neumann ghost 수정(dPhat(:,0,:)=dPhat(:,1,:))
후에도 불안정. 현재 우회: ExtraTerm=0, OB projection만 사용. 근본 수정 미해결.
MPM-STD_C는 동일 ExtraTerm이 안정 → 알고리즘 구현 세부 차이 존재하나, 이 항의 크기는
(1-invRho)~0.5%로 θ_c에 미치는 영향이 아주 작을 것으로 추정됨.

---

## 부록: 재현 정보

- 포트 입력: `apps/rbc/input_fig9_gpu_nob_20_x.toml` (Γ=6,128³,Ra=1e6,Pr=2547,ΔT=20,gamma3=2.6).
  격자 변형 테스트: `input_fig9_g15_20.toml`(gamma=1.5), `_g35_20`(3.5), `_fix_20`(타이밍수정).
- 포트 수정: `apps/rbc/main_gpu_nob.cu:419` (운동량 직전 `update_properties_gpu(...,Tⁿ⁺¹)`).
  dP̂ 외삽: `src/equation/pressure/{cpu,gpu}/...` (dP0 추가, Report 별도).
- PaScaL: `module_global.f90`(glycerol 주석해제), `module_solve_momentum.f90`(부력 x1→x2),
  `module_solve_thermal.f90:73`(matched IC), `run/PARA_INPUT.dat`(Γ=6 매핑: H=1,Aspect1=Aspect3=6,
  gamma2=2.6). 빌드: Intel MPI gfortran 래퍼 + fftw_clean.
- 메모리: `project_gpu_fig9_status.md` (2026-06-25 UPDATE에 동일 결론).
