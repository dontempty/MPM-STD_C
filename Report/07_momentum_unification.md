# 운동량 방정식 통합 설계 — PaScaL_TCS vs MPM-STD Channel 실제 코드 비교 + 통합안

> 동기: [06_design_critique_and_revision.md](06_design_critique_and_revision.md) 에서 "단일 `MomentumEquation` 클래스" 를 주장했으나, PaScaL_TCS (NOB Rayleigh-Bénard) 와 MPM-STD Fortran (Channel forced convection) 의 운동량 방정식 차이가 크다는 지적. 본 보고서는 **두 코드의 실제 ADI 1st stage RHS 어셈블리를 줄 단위로 대조** 한 후, 통합이 가능한지·어디까지 가능한지·어떻게 해야 하는지 정직하게 분석.

---

## Part A. 실제 코드 비교 — 두 코드 안에서 무엇이 다른가

### A.1 PaScaL_TCS U-momentum 1st stage (z-direction)

[module_solve_momentum.f90:225–416](../../PaScaL_TCS/src/module_solve_momentum.f90):

```fortran
subroutine mpi_momentum_solvedU(T)
  ...
  do k = 1, n3msub
    do j = 1, n2msub
      jum = jC_BC(jm); jup = jC_BC(jp)     ! y wall flag
      do i = i_indexS, n1msub
        ! (1) Convection — skew-symmetric averaging
        u1 = 0.5*(U(im,j,k)+U(i,j,k));  u2 = 0.5*(U(ip,j,k)+U(i,j,k))
        v3 = 0.5*(dx1(im)*V(i,j,k) + dx1(i)*V(im,j,k))/dmx1(i)
        v4 = ...
        w5 = ...; w6 = ...
        dudx1, dudx2, dudy3, dudy4, dudz5, dudz6, dvdx3, dvdx4, dwdx5, dwdx6 = ...

        ! (2) Variable viscosity — harmonic mean across faces
        muc = 0.5*(dx1(im)*Mu(i,j,k) + dx1(i)*Mu(im,j,k))/dmx1(i)
        mua = ...;  mub = ...
        mu3 = 0.5*(dx2(jm)*muc + dx2(j)*mua)/dmx2(j)        ! μ at y-faces
        mu4 = ...; mu5 = ...; mu6 = ...                     ! μ at z-faces

        invRhoc         = 0.5*(dx1(im)*invRho(i,j,k)+dx1(i)*invRho(im,j,k))/dmx1(i)
        invRhocCmu_half = 0.5 * Cmu * invRhoc               ! ← 1/ρ(T) 매번 보간

        ! (3) Viscous + cross-direction viscous
        viscous_u1  = (Mu(i,j,k)*dudx2 - Mu(im,j,k)*dudx1)/dmx1(i)
        viscous_u2  = (mu4*dudy4 - mu3*dudy3)/dx2(j)
        viscous_u3  = (mu6*dudz6 - mu5*dudz5)/dx3(k)
        viscous_u12 = (mu4*dvdx4 - mu3*dvdx3)/dx2(j)        ! ← μ·∂v/∂x cross
        viscous_u13 = (mu6*dwdx6 - mu5*dwdx5)/dx3(k)        ! ← μ·∂w/∂x cross
        RHS(i,j,k)  = invRhocCmu_half*(2*viscous_u1 + viscous_u2 + viscous_u3
                                       + viscous_u12 + viscous_u13)

        ! (4) Pressure — 1/ρ 곱
        RHS = RHS - Cmp*invRhoc*(P(i,j,k)-P(im,j,k))/dmx1(i)

        ! (5) Buoyancy — NOB 비선형, U 컴포넌트에 들어감 (PaScaL_TCS 의 좌표축 컨벤션)
        Tc  = 0.5*(dx1(im)*T(i,j,k) + dx1(i)*T(im,j,k))/dmx1(i)
        RHS = RHS + Cmt*(Tc + a12pera11*Tc²*DeltaT)*invRhoc  ! ← 비선형 NOB

        ! (6) y-wall BC: convection + diffusion contribution at walls
        ubc_down = ...VBCbt_sub...;  ubc_up = ...VBCup_sub...
        RHS = RHS + (1-jum)*ubc_down + (1-jup)*ubc_up

        ! (7) LHS tridiagonal in z (자기-방향)
        mACK = invRhocCmu_half/dx3(k)*(mu6/dmx3(kp)+mu5/dmx3(k))
             + 0.25*(-w6/dmx3(kp)+w5/dmx3(k))
        mAPK = -invRhocCmu_half/dx3(k)*mu6/dmx3(kp) + 0.25*w6/dmx3(kp)
        mAMK = -invRhocCmu_half/dx3(k)*mu5/dmx3(k)  - 0.25*w5/dmx3(k)
        ac(i,j,k) = mACK*dt + 1.0
        ap(i,j,k) = mAPK*dt
        am(i,j,k) = mAMK*dt
        RHS = RHS*dt
      enddo
    enddo
  enddo
  call PaScaL_TDMA_many_solve_cycle(..., am, ac, ap, RHS, ...)
```

**핵심 특징**:
- `Mu(i,j,k), invRho(i,j,k)` 3D 필드 (매 step T 에서 갱신)
- `mu3, mu4, mu5, mu6` face-staggered harmonic mean (변동 μ 때문에 필요)
- `viscous_u12, viscous_u13` cross-derivative viscous (변동 μ 때문에 비대칭)
- `invRhocCmu_half = 0.5·Cmu·invRho` — 모든 항이 1/ρ 로 스케일
- Pressure gradient 에도 `Cmp·invRhoc` 곱
- Buoyancy: `Cmt·(Tc + a12pera11·Tc²·DeltaT)·invRhoc` — 비선형 NOB
- Wall BC (`ubc_down/up`) 가 convection + diffusion 양쪽 모두에서 변동 V, μ 와 연결

### A.2 MPM-STD Channel U-momentum 1st stage (z-direction)

[core_momentum.f90:660–954](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90):

```fortran
attributes(global) subroutine cuda_momentum_solvedU_Amatrix_kernel(...,
                                          presgrad1, ...)
  ! (0) GPU shared memory cache for U
  U(ti,tj,tk) = u_d(i,j,k)   ! ... + halo loads

  Vijk = v_d(i,j,k); ...    ! V, W, P 캐싱
  Muijk = Mu_d(i,j,k); ...  ! ← Mu_d, invRho_d 는 = 1.0 상수 (line 401-402)

  ! (1) 동일한 convection / 좌표 메트릭
  u1 = 0.5*(U(tim,tj,tk)+U(ti,tj,tk)); u2 = ...
  v3 = 0.5*(dx1_d(im)*Vijk + dx1_d(i)*Vim)/dmx1_d(i)
  ... 동일

  ! (2) "변동" viscosity 코드는 그대로 있음 — 단지 Mu = 1 이라 값만 다름
  muc = 0.5*(dx1_d(im)*Muijk + dx1_d(i)*Muim)/dmx1_d(i)
  mu3 = ...; mu4 = ...; mu5 = ...; mu6 = ...

  invRhoc = 0.5*(dx1_d(im)*invRho_d(i,j,k)+dx1_d(i)*invRho_d(im,j,k))/dmx1_d(i)
  invRhocCmu_half = 0.5 * Cmu * invRhoc         ! ← 같은 식, 단 invRhoc=1

  ! (3) Viscous + cross (PaScaL_TCS 와 식 동일)
  viscous_u1, viscous_u2, viscous_u3, viscous_u12, viscous_u13  = ...

  RHS = invRhocCmu_half * ( (viscous_u1+viscous_u2+viscous_u3)
                          + (viscous_u1+viscous_u12+viscous_u13) )
  ! ↑ PaScaL_TCS 는 (2*v1+v2+v3+v12+v13). 본 식은 (2*v1+v2+v3+v12+v13) 와 동치 (수학적으로)

  ! (4) LHS implicit 의 명시적 컨트리뷰션 — RHS 에서 빼내기
  RHS = RHS - (M11MI*ium*U(tim) + M11MJ*jum*U(tjm) + M11MK*kum*U(tkm)
             + M11CI*U(ti) + M11CJ*U(ti) + M11CK*U(ti)
             + M11PI*iup*U(tip) + M11PJ*jup*U(tjp) + M11PK*kup*U(tkp))
  RHS = RHS - M12Vn - M13Wn         ! cross-velocity 결합 (V, W 와의)

  ! (5) Pressure (PaScaL_TCS 동일) — 단, channel 의 forced 항 추가
  RHS = RHS - Cmp*invRhoc*(Pijk - Pim)/dmx1_d(i) - presgrad1
                                                  ! ↑ forced bulk dp/dx

  ! (6) Buoyancy — U 에서는 비활성 (주석)
  !RHS = RHS + Cmt*(Tc+Tc^2*DeltaT)*invRhoc

  ! (7) BC contribution — PaScaL_TCS 와 동형
  RHS = RHS + (1-ium)*ubc_x_down + (1-iup)*ubc_x_up   ! ← x 도 wall 가능
  RHS = RHS + (1-jum)*ubc_y_down + (1-jup)*ubc_y_up
  RHS = RHS + (1-kum)*ubc_z_down + (1-kup)*ubc_z_up   ! ← z 가 주 wall

  ! (8) LHS 3 방향 모두 한 번에 저장 (전치 후 stage 2, 3 에서 사용)
  AMK_d(i,j,k) = M11MK*kum*dt*iuc;  ACK_d(i,j,k) = M11CK*dt*iuc + 1
  AMI_d(j,k,i) = M11MI*ium*dt*iuc;  ACI_d(j,k,i) = M11CI*dt*iuc + 1
  AMJ_d(k,i,j) = M11MJ*jum*dt*iuc;  ACJ_d(k,i,j) = M11CJ*dt*iuc + 1
  RHS_d(i,j,k) = RHS * dt * iuc

  ! (9) IBM forcing — 고체 셀이면 행렬을 identity 로 치환
  H_x = H_d(i,j,k) + H_d(i,jp,k) + H_d(i,j,kp) + H_d(i,jp,kp)
  if(.not. is_fluid_face(H_x)) then
    RHS_d = 0;  ACK_d = 1; AMK_d=0; APK_d=0
                ACI_d = 1; AMI_d=0; API_d=0
                ACJ_d = 1; AMJ_d=0; APJ_d=0
  endif
```

**MPM-STD W-momentum** (z 가 wall-normal 이라 buoyancy 가 W 에 들어감, [core_momentum.f90:1482–1487](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90)):

```fortran
! Buoyancy
selectcase(Buoyancy)
case(.true.)
  Tc  = 0.5*(dx3_d(km)*T_d(i,j,k) + dx3_d(k)*T_d(i,j,km))/dmx3_d(k)
  RHS = RHS + Cmt*(Tc - Ground_T)*invRhoc     ! ← 선형 (Tc - Ground_T)
end select
```

→ PaScaL_TCS 의 `Cmt·(Tc + a12pera11·Tc²·DeltaT)·invRhoc` (비선형 NOB) 와 다름.

### A.3 줄 단위 차이 분석

| 항목 | PaScaL_TCS (RBC NOB) | MPM-STD (Channel forced) |
|---|---|---|
| **계수 필드 `Mu, invRho`** | T-의존 3D 필드 (`mpi_momentum_coeffi(T)` 가 매 step 갱신) | 현재 = 1.0 상수 (코드는 변동 지원, 다항식 부분 주석) |
| **Viscous 어셈블리 식** | 동일 (`viscous_u1..u3`, `u12, u13`) | **수학적으로 동일** (총합 = `2·v1+v2+v3+v12+v13`) |
| **Face-staggered μ** | `mu3,4,5,6` 모두 harmonic mean | **같은 코드** (단 결과 = 1) |
| **invRho 곱** | `invRhocCmu_half = 0.5·Cmu·invRho` | 동일. 값만 1 |
| **Pressure** | `-Cmp·invRhoc·∂P/∂x` | `-Cmp·invRhoc·∂P/∂x − presgrad1` ← bulk forcing 추가 |
| **Buoyancy** | U 컴포넌트, **비선형 `Cmt·(Tc+a12·Tc²·DT)·invRho`** | W 컴포넌트 (z=wall), **선형 `Cmt·(Tc−Ground_T)·invRho`** |
| **Cross-velocity 결합 M12Vn / M13Wn** | RHS 안에 직접 누적 | `M12Vn, M13Wn` 별도 변수로 명시적 누적 |
| **LHS matrix coefficient** | mACK, mAPK, mAMK (z 만) | M11MK/CK/PK + M11MI/CI/PI + M11MJ/CJ/PJ **3 방향 모두 한 번에** |
| **Wall BC** | y-wall (jum/jup) **만** | x, y, z **3 방향 모두** (`ium/iup, jum/jup, kum/kup`) |
| **IBM** | 없음 | `H_d` 마스크로 셀 단위 행렬 치환 |
| **GPU 메모리** | host only | shared memory 캐시, 3D pointer 재사용, 전치 |
| **ADI sweep 순서** | z → x → y (y=wall) | z → x → y (z=wall) ← **알고리듬은 같지만 wall 축 다름** |

---

## Part B. 통합 가능한 것 vs 통합 불가능한 것

### B.1 **구조적으로 같은 것** (= 같은 코드 그대로 공유)

| 공유 가능 | 근거 |
|---|---|
| ADI 3-stage 스켈레톤 (build → TDMA → 전치) | 두 코드 동일 |
| Convection skew-symmetric averaging 식 | 식 동일, 변수만 동일 |
| Viscous 어셈블리 (`viscous_u1..u3, u12, u13`) | 식 동일 (μ 가 상수든 변동이든) |
| Face-staggered μ harmonic mean | 식 동일 |
| Cross-velocity 결합 `M12Vn, M13Wn` | 식 동일 |
| LHS matrix coefficient 빌드 (`M11MI..PK`) | 식 동일, `invRhoc` 와 `Mu` 만 데이터 |
| TDMA call + 전치 | 라이브러리 호출 동일 |
| Wall BC contribution 식 (`ubc_down/up`) | 식 동일, BC 데이터만 다름 |
| `block_couple` (`blockLdU/V`) | 두 코드 모두 동일한 implicit 결합 보정 |
| `pseudoupdate` | 두 코드 동일 |

### B.2 **데이터 / 정책으로 분리 가능한 것**

| 분리 대상 | RBC 값 | Channel 값 | 추상화 |
|---|---|---|---|
| `Mu(i,j,k)` 필드 | T-의존 다항식 | 1.0 (또는 μ_eff = μ+ρν_t for LES) | **PropertyField** — 매 step 갱신 가능 |
| `invRho(i,j,k)` 필드 | T-의존 | 1.0 | 동일 |
| Buoyancy 항 | U 컴포넌트, 비선형 NOB | W 컴포넌트, 선형 | **SourceTerm 객체** (어느 component? 어떤 식?) |
| `presgrad1` (bulk forcing) | 0 | 사용자 정의 상수 또는 적응적 | **SourceTerm 객체** |
| Wall BC | y-wall 한 축만 | 3 축 모두 가능 | **BoundaryConditions** ([05_BC_design.md](05_BC_design.md)) |
| ADI sweep 순서 | z→x→y (y wall) | z→x→y (z wall) | **`DomainTopology::sweep_order()`** 자동 |
| IBM | 비활성 | 활성 (`H_d` 마스크) | **IbmPlugin** (matrix row 치환 hook) |
| LES (ν_t) | 비활성 | 활성 (Mu 에 누적) | **LesPlugin** (PropertyField 갱신) |

### B.3 **통합이 어려운 부분 — 정직하게**

| 어려움 | 이유 | 대처 |
|---|---|---|
| **GPU 메모리 레이아웃 (shared cache)** | MPM-STD 는 stencil 을 위해 thread block 마다 U 를 shared memory 에 캐시. PaScaL_TCS 는 CPU host array. | **C++ 에선 Backend 별 dispatch**. CPU 백엔드는 단순 배열, GPU 백엔드는 shared memory. 알고리듬 식은 동일. |
| **3 방향 행렬 한 번에 vs stage 마다** | MPM-STD 는 1 kernel call 로 AMI/CI/PI + AMJ/CJ/PJ + AMK/CK/PK 를 동시 계산 (GPU 효율). PaScaL_TCS 는 stage 마다 별도 계산 후 deallocate/reallocate. | **C++ 에선 "코어 자유함수 + Backend 별 호출 패턴"** 으로 분리. CPU 는 PaScaL_TCS 식, GPU 는 MPM-STD 식. **알고리듬 정의는 1 본**. |
| **Buoyancy 가 어느 컴포넌트** | PaScaL_TCS U, MPM-STD W. 사용자 컨벤션 (중력 방향) 차이. | **SourceTerm 이 어느 component 에 들어가는지를 데이터로** (`buoyancy.component = Component::W`). |
| **Wall BC 의 cross-velocity 항** | PaScaL_TCS U-solver 안에 V wall 값 (`VBCbt/up`) 이 직접 등장. 변동 μ + cross-derivative 가 합쳐진 결과. | **WallBC 핸들러가 변수 간 결합을 처리** — `BoundaryApplier::wall_cross_term(U, V_at_wall, mu_face)`. |

→ **결론**: 식은 같고 데이터·정책만 다르다. **단일 알고리듬 + Policy 객체** 로 통합 가능.

---

## Part C. 통합 설계 — Policy/Plugin 분리

### C.1 클래스 구조

```cpp
// ===== 핵심: 단일 MomentumEquation =====
class MomentumEquation {
public:
  MomentumEquation(Grid& g, Subdomain& sd, FieldRegistry& fr,
                   const Problem& problem,
                   TdmaRegistry& tdma, BoundaryApplier& bc,
                   PropertyPolicy& props,
                   std::vector<SourceTerm*> sources);

  void compute_coeffi();                              // PropertyPolicy 위임
  void predict(Component c, double dt);               // U, V, W 모두 같은 함수
  void block_couple_V();
  void block_couple_U();
  void pseudo_update();

private:
  // RHS 어셈블리 (PaScaL_TCS·MPM-STD 동일 식)
  void build_rhs_and_matrix(Component c, double dt, ...);

  Grid&              grid_;
  Subdomain&         sub_;
  FieldRegistry&     fields_;
  const Problem&     problem_;
  TdmaRegistry&      tdma_;
  BoundaryApplier&   bc_;
  PropertyPolicy&    props_;
  std::vector<SourceTerm*> sources_;
};
```

### C.2 PropertyPolicy — 변동 vs 상수 μ, ρ

```cpp
class PropertyPolicy {
public:
  virtual ~PropertyPolicy() = default;
  virtual void update(const FieldRegistry& fr) = 0;     // 매 step 호출 — Mu, invRho 필드 갱신
  virtual const Field& mu()      const = 0;
  virtual const Field& invRho()  const = 0;
};

// 구현 1: 상수 (Channel basic)
class ConstantProperties : public PropertyPolicy {
  Field mu_, invRho_;
public:
  ConstantProperties(Grid& g, Backend& be) {
    mu_     = Field(g, /*=*/1.0, be);
    invRho_ = Field(g, /*=*/1.0, be);
  }
  void update(const FieldRegistry&) override {}   // no-op
  const Field& mu()     const override { return mu_; }
  const Field& invRho() const override { return invRho_; }
};

// 구현 2: NOB (RBC, PaScaL_TCS)
class NobProperties : public PropertyPolicy {
  Field mu_, invRho_;
  NobCoefficients coefs_;     // a10..a15, b10..b15 등
public:
  void update(const FieldRegistry& fr) override {
    const Field& T = fr.scalar("T");
    // PaScaL_TCS 의 mpi_momentum_coeffi(T) 와 동일
    for (auto idx : T.indices()) {
      double t = T(idx) * coefs_.DeltaT;
      mu_(idx)     = polynomial5(coefs_.a, t) * polynomial5(coefs_.d, t) / coefs_.Mu0;
      invRho_(idx) = coefs_.Rho0 / polynomial5(coefs_.a, t);
    }
  }
};

// 구현 3: LES (Channel + Smagorinsky) — 나중 단계
class LesProperties : public PropertyPolicy {
  PropertyPolicy& base_;       // 합성: 기본 PropertyPolicy 위에 ν_t 누적
  Field nu_t_;
public:
  void update(const FieldRegistry& fr) override {
    base_.update(fr);
    update_smagorinsky_nu_t(fr.vector("U"), nu_t_);
    // mu_eff = mu_base + rho·nu_t
    mu_eff_ = base_.mu();   mu_eff_.add(/*rho·*/nu_t_);
  }
};
```

### C.3 SourceTerm — 추가 RHS 기여 (Buoyancy, Forcing, …)

```cpp
class SourceTerm {
public:
  virtual ~SourceTerm() = default;
  virtual Component target_component() const = 0;       // U, V, W 중 어디에 들어갈지
  virtual void add_to_rhs(Component c, const FieldRegistry& fr,
                          const PropertyPolicy& props,
                          double* RHS, /*indices...*/) const = 0;
};

// 구현 1: NOB Buoyancy (PaScaL_TCS 스타일, 비선형)
class NobBuoyancy : public SourceTerm {
  Component comp_;     // RBC: U (PaScaL_TCS) 또는 W (MPM-STD) — 사용자 지정
  double    Cmt_;
  double    a12pera11_, DeltaT_;
public:
  Component target_component() const override { return comp_; }
  void add_to_rhs(Component c, const FieldRegistry& fr,
                  const PropertyPolicy& props,
                  double* RHS, ...) const override {
    if (c != comp_) return;
    const Field& T = fr.scalar("T");
    double Tc = interpolate_to_face(T, ...);
    double invRhoc = interpolate_to_face(props.invRho(), ...);
    RHS[idx] += Cmt_ * (Tc + a12pera11_ * Tc*Tc * DeltaT_) * invRhoc;
  }
};

// 구현 2: 선형 Boussinesq Buoyancy (MPM-STD 채널-RBC 스타일)
class BoussinesqBuoyancy : public SourceTerm {
  Component comp_;
  double    Cmt_, ground_T_;
public:
  void add_to_rhs(Component c, ..., double* RHS, ...) const override {
    if (c != comp_) return;
    double Tc = ...; double invRhoc = ...;
    RHS[idx] += Cmt_ * (Tc - ground_T_) * invRhoc;
  }
};

// 구현 3: Bulk pressure gradient (forced channel)
class BulkForcing : public SourceTerm {
  Component comp_ = Component::U;     // 보통 streamwise
  double    presgrad_;                // 상수 또는 시간 적응적
public:
  Component target_component() const override { return comp_; }
  void add_to_rhs(Component c, ..., double* RHS, ...) const override {
    if (c != comp_) return;
    RHS[idx] -= presgrad_;             // ← MPM-STD line 875
  }
};
```

### C.4 핵심 — `predict` 의 구현은 한 본

```cpp
void MomentumEquation::predict(Component c, double dt) {
  // PropertyPolicy 가 Mu, invRho 제공 (NOB·상수·LES 무관)
  const Field& Mu     = props_.mu();
  const Field& invRho = props_.invRho();

  // ADI sweep 순서는 Problem 의 topology 에서 자동 도출
  const auto sweep = problem_.topology.sweep_order();

  // 첫 stage: 자기-방향(= sweep[0]) 으로 빌드
  for (auto idx : interior_indices()) {
    // (1) Convection + Viscous + Cross + Pressure (PaScaL_TCS·MPM-STD 동일 식)
    double rhs = assemble_visc_conv_pressure(c, idx, Mu, invRho, ...);

    // (2) SourceTerm 들 합산 — Buoyancy, Forcing, 미래 추가
    for (auto* s : sources_)
      s->add_to_rhs(c, fields_, props_, &rhs, idx);

    // (3) Wall BC contribution
    bc_.add_wall_rhs(c, idx, &rhs, /*using fbc */);

    // (4) LHS coefficient
    auto [Am, Ac, Ap] = build_tridiag_along(sweep[0], c, idx, Mu, invRho, ...);
    A[idx] = Am*dt; B[idx] = Ac*dt + 1; C[idx] = Ap*dt;
    D[idx] = rhs*dt;
  }

  // (5) Wall row 보정 + TDMA call
  if (problem_.topology.is_periodic(sweep[0]))
    tdma_.get(sweep[0]).solve_many_cyclic(A,B,C,D, n_sys, n_row);
  else {
    bc_.modify_tdma_row(sweep[0], fbc_, A,B,C,D, n_sys, n_row);
    tdma_.get(sweep[0]).solve_many(A,B,C,D, n_sys, n_row);
  }

  // (6) 다음 stage 2, 3 도 동일 패턴 — A,B,C 만 재구성, RHS 는 누적
  // (예: PaScaL_TCS 는 stage 마다 새로 빌드, MPM-STD 는 1 kernel 에서 미리 다 빌드 후 stage 마다 풀이)
  // → 어느 방식이든 알고리듬은 같음. 본 함수는 PaScaL_TCS 방식 (stage 별 빌드) 권장.
}
```

→ **`assemble_visc_conv_pressure` 자유함수가 PaScaL_TCS·MPM-STD 의 핵심 식 1:1 직역**. `Mu`, `invRho` 가 1 이면 (Channel) 상수 viscosity 결과, T 의존이면 (RBC) NOB 결과. **코드는 한 본**.

### C.5 RBC 셋업 (PaScaL_TCS-식 재현)

```cpp
Problem p;                                          // 자동: z=wall, periodic x,y; T BCs 자동
// (PaScaL_TCS 는 y=wall 이지만 우리 컨벤션은 z=wall RBC. wall 축은 BC 에서 자동 도출)

NobProperties props(grid, backend);
NobBuoyancy   buoy(Component::W,                   // z=wall 이므로 W 에 (Fortran MPM-STD 와 같이)
                   /*Cmt=*/..., /*a12pera11=*/..., /*DeltaT=*/...);

MomentumEquation momentum(grid, sub, fields, p, tdma, bc, props, {&buoy});
ThermalEquation  thermal(grid, sub, fields, p, tdma, bc);

Solver solver(cfg, p, mpi, backend);
solver.add_equation(&momentum);
solver.add_equation(&thermal);
solver.add_plugin<BuoyancyPropagator>(/*T→source*/);
solver.run();
```

### C.6 Channel 셋업 (MPM-STD-식 재현)

```cpp
Problem p;                                          // 동일한 z=wall topology
// ThermalEquation 생성 안 함 → T 필드 자체가 없음

ConstantProperties props(grid, backend);            // μ=1, 1/ρ=1
BulkForcing        forcing(Component::U,           // streamwise
                            /*presgrad=*/-1.0);

MomentumEquation momentum(grid, sub, fields, p, tdma, bc, props, {&forcing});

Solver solver(cfg, p, mpi, backend);
solver.add_equation(&momentum);
// thermal 없음
solver.run();
```

### C.7 Channel + LES (미래 — 같은 코드 그대로)

```cpp
Problem p;
ConstantProperties base(grid, backend);
LesProperties      props(base, /*Smagorinsky cfg*/);    // ν_t 매 step 누적
BulkForcing        forcing(Component::U, -1.0);

MomentumEquation momentum(grid, sub, fields, p, tdma, bc, props, {&forcing});
```

### C.8 Channel + IBM (미래 — Plugin 추가)

```cpp
Problem p;
ConstantProperties props(grid, backend);
BulkForcing        forcing(Component::U, -1.0);
IbmMaskField       Hmask(/* geometry */);

MomentumEquation momentum(grid, sub, fields, p, tdma, bc, props, {&forcing});

Solver solver(cfg, p, mpi, backend);
solver.add_equation(&momentum);
solver.add_plugin<IbmPlugin>(Hmask);                 // matrix row 치환 hook
solver.run();
```

→ `IbmPlugin` 이 `MomentumEquation` 의 행렬 빌드 직후 hook 으로 끼어들어 고체 셀의 `Ac=1, Am=Ap=0, RHS=0` 치환 ([core_momentum.f90:939–951](../../MPM-STD_main/MPM-STD/src/core/core_momentum.f90) 와 동일 로직).

---

## Part D. "통합 불가" 라고 결론 내야 할 부분이 있는가?

정직하게: **거의 없다**. 두 코드의 차이는:

1. **데이터** (`Mu`, `invRho` 값) → PropertyPolicy
2. **추가 항** (Buoyancy, Forcing) → SourceTerm
3. **wall 축** (y vs z) → Problem.topology → sweep_order
4. **BC 활성 면 수** (1 wall vs 3 wall) → BoundaryConditions ([05_BC_design.md](05_BC_design.md))
5. **GPU 메모리 패턴** → Backend dispatch
6. **IBM 마스크** → IbmPlugin
7. **3 방향 행렬 동시 빌드 vs stage 별** → Backend 차이 (CPU/GPU)

핵심 식 (convection skew-symmetric, viscous 5 종, cross-velocity, pressure with 1/ρ, wall BC contribution, LHS tridiagonal coefficient) 은 **두 코드가 같다**.

### D.1 한 가지 주의 — Buoyancy 의 비선형성

PaScaL_TCS: `Cmt·(Tc + a12pera11·Tc²·DeltaT)·invRhoc` — **2 차 항** 포함
MPM-STD : `Cmt·(Tc − Ground_T)·invRhoc` — **선형**

→ SourceTerm 의 자식 클래스 두 개 (NOB vs Linear) 로 분리. **MomentumEquation 본체는 무관**.

### D.2 한 가지 주의 — 컴포넌트 위치

| 코드 | wall 축 | buoyancy component | 이유 |
|---|---|---|---|
| PaScaL_TCS | y | **U** (?) | 코드 컨벤션 (gravity → x?) 또는 검증용 |
| MPM-STD Fortran | z | **W** | gravity → -z (자연스러움) |

→ `SourceTerm::target_component()` 가 데이터로 결정. **알고리듬 무관**.

---

## Part E. 결론

### E.1 통합은 가능하고 그것이 옳은 선택이다

- **알고리듬 식 (ADI, viscous, convection, cross, pressure, BC contribution, LHS) 은 1 본**.
- **차이는 모두 데이터 또는 정책**:
  - PropertyPolicy (constant / NOB / LES)
  - SourceTerm (NobBuoyancy / Boussinesq / BulkForcing / 미래)
  - Problem (BC + topology — sweep order 자동 도출)
  - Plugin (IBM / 미래 wall function 등)
  - Backend (CPU 단순 / GPU shared memory)

### E.2 통합의 이득

1. **버그가 한 곳에**: convection skew-symmetric 식이 잘못되면 한 곳만 고치면 됨. 두 클래스에 복제되어 있으면 한 쪽만 고쳐서 silent divergence.
2. **검증의 단일성**: PaScaL_TCS golden output 비교 1 회로 Channel/RBC/LES/IBM 모두 같은 식이 돌고 있는지 확인.
3. **확장의 단순성**: 새 PropertyPolicy / SourceTerm / Plugin 추가 = 자식 클래스 1 개. 본체 손대지 않음.
4. **미래 GPU 백엔드**: Backend 추상화 안에 알고리듬 1 본 + CPU/GPU 디스패치. PaScaL_TCS↔MPM-STD 의 host/device 차이가 자연스럽게 한 코드로 수렴.

### E.3 통합의 비용

1. **PropertyPolicy + SourceTerm 인터페이스 비용** — 가상함수 호출 ≤ 1 회/step (성능 영향 거의 없음, hot path 는 인라인 어셈블리 자유함수).
2. **약간 더 많은 추상 계층** — 단, [06_design_critique_and_revision.md](06_design_critique_and_revision.md) 의 검토와 일관됨.

### E.4 권고

[06_design_critique_and_revision.md](06_design_critique_and_revision.md) 의 "단일 `MomentumEquation`" 주장을 **유지하되**, 본 보고서의 Policy 분리 (`PropertyPolicy` + `SourceTerm` + `Plugin`) 를 명시적으로 추가. 이는 단일 클래스가 if-else 로 두 케이스를 분기하는 것이 *아니라* **단일 알고리듬 + 외부 주입 정책** 방식.

→ 코드 1 본 + 데이터·정책 다중 = **OOP 의 정석 (Strategy + Plugin)**. PyFR 의 BaseSystem↔EulerSystem↔NavierStokesSystem 계층과 동일한 사상이지만, 본 프로젝트의 구조격자 특성상 *상속 대신 구성(composition)* 으로 표현.
