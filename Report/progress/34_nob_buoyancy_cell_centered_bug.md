# 34. RBC NOB 부력 — 셀중심 버그 발견·수정 (CPU+GPU, A/B 검증)

> **작성**: 2026-07-10
> **동기**: PaScaL_TCS의 부력 방향(Report/33) 조사 중, MPM-STD_C 쪽 부력 구현을 점검하다가
> RBC **NOB** 경로(`add_buoyancy_force_nob_{cpu,gpu}`)만 T·invRho를 면보간 없이
> **셀중심 값 그대로** 쓰고 있음을 발견.

## 1. 발견

- **OB 부력** (`add_buoyancy_force_{cpu,gpu}`): Report 19(2026-06-15)에서 셀중심→면보간으로
  이미 수정됨, PaScaL `mom.f90:348` 충실 (`Tc = 0.5*(dxg[ig-1]*T[c]+dxg[ig]*T[c-gstride])/dmxg[ig]`).
- **NOB 부력** (`add_buoyancy_force_nob_{cpu,gpu}`): Tc·invRhoc 둘 다 면보간 없이
  `T[c]`, `inv_rho[c]` 그대로 사용. 코드 주석 `"Cell-centred (matches the OB code's
  cell-T-on-W convention)"`은 Report 19 수정 **이전**의 옛 OB 관례를 가리키는 것으로 보이며,
  그 수정이 NOB 쪽엔 전파되지 않았던 것으로 판단됨.
- **대조 1 — 자체 설계문서** [docs/NOB_RBC_SPEC.md:49-54](../docs/NOB_RBC_SPEC.md):
  `Tc = θ interpolated to the (wall-normal) face; F = invRhoc·Cmt·(Tc+a12pera11·Tc²·DeltaT)`
  — 면보간을 명시적으로 요구.
- **대조 2 — PaScaL_TCS 원본** (`module_solve_momentum.f90` solvedV, Report/33에서
  재활성화한 그 부력 블록): Tc·invRhoc 둘 다 dx-가중 면보간.
- GPU 커널(`k_add_buoyancy_nob`)은 한술 더 떠 `gravity_axis` 인자조차 없이 flat 배열
  **전체(halo 포함)**에 그대로 적용 — OB 커널(`k_add_buoyancy`, interior-only, gaxis-aware)과
  인터페이스도 불일치.

## 2. 수정

- [buoyancy.hpp](../src/physics/buoyancy/buoyancy.hpp): `add_buoyancy_force_nob_{cpu,gpu}`
  시그니처에 `const core::Domain& domain` 추가 (면보간 메트릭 접근용).
- [buoyancy_cpu.cpp](../src/physics/buoyancy/cpu/buoyancy_cpu.cpp): `add_buoyancy_force_nob_cpu`
  — OB 함수와 동일한 gravity_axis 제네릭 면보간을 Tc·invRhoc 둘 다에 적용.
- [buoyancy_gpu.cu](../src/physics/buoyancy/gpu/buoyancy_gpu.cu): `k_add_buoyancy_nob` 커널을
  `k_add_buoyancy`와 동일한 인터페이스(gaxis/dxg/dmxg, interior-only grid)로 재작성,
  invRhoc 면보간 추가.
- 호출부 4곳 시그니처 갱신: `apps/rbc/main_cpu_nob.cpp`, `apps/rbc/main_gpu_nob.cu`,
  `apps/rbc/main.cpp` (CPU/GPU 분기 둘 다).

## 3. 빌드 검증

- **CPU**: `make BACKEND=cpu apps` — 전체 재빌드 성공 (`rbc`, `rbc_cpu_nob`, `rbc_cpu_ob` 포함).
- **GPU**: gpu01에서 `sbatch --wait scripts/compile_rbc_gpu_nob.sh` (nvhpc/23.7, job 30123) —
  `dhvc_gpu`, `rbc_gpu_nob`, `rbc_gpu_ob` 포함 전체 재빌드 성공.

## 4. 런타임 A/B 검증 (CPU, 로그인 노드)

**격리 방법**: `buoyancy_cpu.cpp`의 NOB 함수만 일시적으로 옛 셀중심 코드로 되돌려 별도
바이너리를 빌드·비교 실행한 뒤, 수정본으로 복원·재빌드 (다른 파일은 전혀 건드리지 않음 —
git 워킹트리에 잔여 diff 없음, 원래 있던 다른 uncommitted 변경들과도 무관).

**구성** ([input_verify_buoy_ab.toml](../apps/rbc/input_verify_buoy_ab.toml)): 32×32×48,
Lx=Ly=6/Lz=1, γ₃=2.6 tanh, Ra=1e6, Pr=2547, ΔT=20K(NOB), dt=0.02, MaxCFL=0.3, 3000스텝(t=60),
단일 랭크.

**결과**:

| t | Θ_c BEFORE(셀중심) | Θ_c AFTER(면보간) | 상대차 |
|---|---|---|---|
| 20 | 0.0118 | 0.0118 | 0% |
| 24 | 0.0143 | 0.0182 | +27% |
| 28 | 0.0102 | 0.0165 | +62% |
| 40 | 0.0150 | 0.0336 | +124% |
| 52 | 0.0184 | 0.0274 | +49% |
| 60 | 0.0185 | 0.0208 | +12% |

- **div**: 두 버전 모두 1e-17~1e-15 (머신 정밀도) — 수정으로 안정성·질량보존 저하 없음.
- **umax/vmax/wmax**: 초반(t≤20)은 거의 동일 (격자가 성기고 T가 아직 매끈해 면보간≈셀중심),
  t≥24부터 유의미하게 갈라짐 — 대류가 발달하며 T 구배가 커질수록 면보간 O(dx) 보정이
  커지는 것과 정합.
- **결론**: 수정이 실제로 다른 수치를 만들고(과도구간에서 12~124% 편차), 안정성엔 영향
  없음 — 코드 레벨 버그 존재 확인 + 수정 유효성 확인.

## 5. 남은 일

- 이 격리 A/B는 초기 과도구간(t≤60)만 봄 — **θ_c 평형값(장시간 정상통계)에 대한 영향은
  별도 프로덕션 런(h-256, Report 31/32 계열) 재측정 필요**. Report 23-32의 θ_c −25% 잔여
  갭 조사에 이 수정을 반영해 재검토 권장 — 지금까지 한 번도 검토되지 않았던 새 후보.
- GPU 경로는 빌드만 확인, 런타임 A/B는 미실행 (CPU와 로직 동일하므로 우선순위 낮음).
- RBC OB 경로는 원래부터 면보간 정상 — 무영향.
- DHVC는 OB 부력만 쓰므로(NOB 함수 미호출) 무영향.
