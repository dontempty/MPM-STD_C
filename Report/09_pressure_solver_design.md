# PressureSolver — BC-aware FFT/DCT 알고리듬 상세 설계

> 대상: C++ MPM-STD 의 `PressureEquation` (incremental pressure Poisson + projection)
> 목적: BC 종류에 따라 적절한 스펙트럼 변환 (FFT / DCT / DST) 을 자동 선택하고, 비-주기 wall 축에서 TDMA 로 풀이하는 통합 알고리듬 설계
> 참조 코드: [PaScaL_TCS/src/module_solve_pressure.f90](../../PaScaL_TCS/src/module_solve_pressure.f90), [MPM-STD/src/core/core_pressure.f90](../../MPM-STD_main/MPM-STD/src/core/core_pressure.f90)
> 컨벤션: **z = wall-normal (RBC 기본)**, x, y = periodic (Problem 의 기본값)

---

## 1. 문제 정의

분리 (fractional step) 단계에서 풀어야 할 압력 보정식:

```
∇²(δP) = (1/Δt) · ∇·u*                          (constant ρ)
∇·( (1/ρ) ∇(δP) ) = (1/Δt) · ∇·u*               (variable ρ — NOB)
```

투영 단계:

```
u^{n+1} = u* − Δt · (1/ρ) ∇(δP)
P^{n+1} = P^n + δP
```

여기서 `u*` 는 운동량 predictor 의 중간속도. 우변은 `mpi_pressure_RHS` 가 빌드 ([module_solve_pressure.f90:140](../../PaScaL_TCS/src/module_solve_pressure.f90), [core_pressure.f90:361](../../MPM-STD_main/MPM-STD/src/core/core_pressure.f90)).

### 1.1 2 차 중심 차분 이산화 — 분리 가능 (separable) 형식

cell-centered MAC 격자에서:

```
(δP_{i+1,j,k} − 2δP_{i,j,k} + δP_{i−1,j,k}) / dx₁²
+ (δP_{i,j+1,k} − 2δP_{i,j,k} + δP_{i,j−1,k}) / dx₂²
+ (δP_{i,j,k+1} − 2δP_{i,j,k} + δP_{i,j,k−1}) / dx₃²
= RHS_{i,j,k}
```

(uniform grid 가정. 비-uniform 은 wall 축에서 TDMA 가 처리, periodic 축은 uniform 권장.)

→ **분리 가능 (separable) 연산자**: 각 축의 2 차 차분이 *독립*. 적절한 변환으로 *고유값 분해* 가능.

---

## 2. 스펙트럼 변환 — BC 종류에 따른 자동 선택

### 2.1 변환의 핵심 아이디어

분리 가능 연산자 `L = L_x + L_y + L_z` 의 고유벡터는 각 1D 고유벡터의 tensor product. 1D 2 차 차분 `L_d` 의 고유벡터는 **BC 종류** 에 따라 다름:

| BC 종류 | 1D 고유벡터 | 대응 변환 |
|---|---|---|
| **Periodic** | 복소수 지수 `exp(2πikx/L)` | **R2C FFT** (실수→복소수, n→n/2+1) |
| **Neumann** `∂P/∂n=0` (cell-center) | 코사인 `cos(π(k)(2j−1)/(2N))` | **DCT-II** (Discrete Cosine Transform type II) |
| **Dirichlet** `P=0` (cell-center) | 사인 `sin(π(k+1)(2j−1)/(2N))` | **DST-II** (Discrete Sine Transform type II) |

→ **wall 축은 보통 변환하지 않고 TDMA** (변환하려면 격자가 uniform 해야 하는데 wall 축은 거의 항상 stretched).

### 2.2 자동 선택 규칙 — `Problem` 으로부터

C++ 의 `FftPlanner` 는 `Problem` 객체를 받아 다음 절차로 결정:

```cpp
class FftPlanner {
public:
  FftPlanner(const Problem& problem, const Grid& g, Backend& be) {
    wall_axis_ = problem.topology.wall_axis().value();   // 1 차에 1 개 가정
    for (Direction d : {X, Y, Z}) {
      if (d == wall_axis_) {
        kind_[d] = SpectralKind::None;                    // TDMA 만 사용
      } else if (problem.topology.is_periodic(d)) {
        kind_[d] = SpectralKind::FftR2c;
      } else {
        const FaceBc& fb = problem.P.face(d, Side::Minus);
        switch (fb.kind) {
          case BcKind::Neumann:   kind_[d] = SpectralKind::DctII; break;
          case BcKind::Dirichlet: kind_[d] = SpectralKind::DstII; break;
          default: throw std::runtime_error("unsupported pressure BC on non-wall axis");
        }
      }
    }
    build_eigenvalues_();
    build_fftw_plans_();
  }

private:
  Direction                            wall_axis_;
  std::array<SpectralKind, 3>          kind_;
  std::array<std::vector<double>, 3>   eigenvalues_;     // 축별 λ_k 사전계산
  // FFTW plans...
};
```

### 2.3 자주 쓰이는 BC 조합 (1차 지원)

| 케이스 | x | y | z (wall) | 변환 |
|---|---|---|---|---|
| **RBC (기본)** | Periodic | Periodic | Neumann | FFT(x) + FFT(y) + TDMA(z) |
| **Channel forced** | Periodic | Periodic | Neumann | 동일 |
| **Channel with side walls (drawer)** | Periodic | Neumann | Neumann | FFT(x) + DCT(y) + TDMA(z) |
| **PaScaL_TCS 호환 (y=wall)** | Periodic | Neumann | Periodic | FFT(x) + FFT(z) + TDMA(y) |
| **모든 면 wall (cavity)** | Neumann | Neumann | Neumann | DCT(x) + DCT(y) + TDMA(z) |

→ `kind_[d]` 를 `Problem` 에서 자동 도출하므로, **사용자가 변환 종류를 지정할 필요 없음**.

---

## 3. 고유값 (wave number) 사전계산

각 축의 1D 고유값 `λ_k` 를 시뮬레이션 시작 시 한 번 계산.

### 3.1 Periodic (R2C FFT, N 셀)

2 차 중심 차분의 modified wave number:

```
λ_k = (2/dx²) · (1 − cos(2πk/N))
    = (4/dx²) · sin²(πk/N),       k = 0, 1, ..., N/2
```

PaScaL_TCS 구현 ([module_solve_pressure.f90:114, 120](../../PaScaL_TCS/src/module_solve_pressure.f90)):
```fortran
dzk(k) = 2.0d0 * (1.0d0 - cos(2.0d0*PI*dble(km)*ddz/L3)) / (ddz*ddz)
dxk1(i)= 2.0d0 * (1.0d0 - cos(2.0d0*PI*dble(im)*ddx/L1)) / (ddx*ddx)
```

MPM-STD 동일 (`BCtype=='P'`, [core_pressure.f90:326, 333, 342](../../MPM-STD_main/MPM-STD/src/core/core_pressure.f90)):
```fortran
dxk2(i) = 2.0 * (1.0 - dcos(2.0*real(im)*PI/real(n1m))) / (dx1*dx1)
```

### 3.2 Neumann (DCT-II, N 셀, cell-center)

cell-centered 2 차 차분 + Neumann BC 의 고유값:

```
λ_k = (4/dx²) · sin²(πk/(2N)),    k = 0, 1, ..., N−1
```

`k=0` 일 때 `λ_0 = 0` → **null space** (압력 상수 시프트 = 자유도). 평균 제거 필요 (§7).

MPM-STD 구현 (`BCtype=='N'`, [core_pressure.f90:309, 321](../../MPM-STD_main/MPM-STD/src/core/core_pressure.f90)):
```fortran
dxk2(i) = 4.0 * (dsin(0.5*real(im)*PI/real(n1m)))**2 / (dx1*dx1)
```

### 3.3 Dirichlet (DST-II, N 셀, cell-center)

```
λ_k = (4/dx²) · sin²(π(k+1)/(2N)),   k = 0, 1, ..., N−1
```

→ 가장 낮은 모드도 `λ_0 = (4/dx²)·sin²(π/(2N)) ≠ 0` → singular 아님 (null space 없음).

(MPM-STD·PaScaL_TCS 코드에 직접 구현된 곳은 없음 — 1 차에 DST 지원 추가 시 위 공식 사용.)

### 3.4 헬퍼 함수 — C++

```cpp
// include/mpmstd/pressure/eigenvalues.hpp
namespace mpmstd::pressure {

inline std::vector<double> eigvals_periodic(int N, double dx) {
  std::vector<double> lam(N/2 + 1);
  for (int k = 0; k <= N/2; ++k)
    lam[k] = 4.0 / (dx*dx) * std::pow(std::sin(M_PI*k/N), 2);
  return lam;
}

inline std::vector<double> eigvals_neumann(int N, double dx) {
  std::vector<double> lam(N);
  for (int k = 0; k < N; ++k)
    lam[k] = 4.0 / (dx*dx) * std::pow(std::sin(0.5*M_PI*k/N), 2);
  return lam;
}

inline std::vector<double> eigvals_dirichlet(int N, double dx) {
  std::vector<double> lam(N);
  for (int k = 0; k < N; ++k)
    lam[k] = 4.0 / (dx*dx) * std::pow(std::sin(0.5*M_PI*(k+1)/N), 2);
  return lam;
}

}  // namespace mpmstd::pressure
```

---

## 4. 알고리듬 전체 흐름

### 4.1 6 단계 의사코드 (wall = Z 가정)

```
[1] RHS 빌드:
    PRHS(i,j,k) = (1/Δt) · ∇·u*_{i,j,k}     + NOB 보정 (variable ρ 시)

[2] Forward transform (x 축):
    if (kind_x == FftR2c)   FFT_x(PRHS) → PRHS_hat   (complex, n1/2+1 × n2 × n3)
    if (kind_x == DctII )   DCT_x(PRHS) → PRHS_hat   (real,    n1     × n2 × n3)
    if (kind_x == DstII )   DST_x(PRHS) → PRHS_hat   (real,    n1     × n2 × n3)

[3] Forward transform (y 축):
    if (kind_y == FftR2c)   FFT_y(PRHS_hat)
    if (kind_y == DctII )   DCT_y(PRHS_hat)
    if (kind_y == DstII )   DST_y(PRHS_hat)

[4] Wall 축 TDMA — 각 (kx, ky) 모드 독립
    for each (k_x, k_y):
      build tridiag system in z:
        sub_z[k]  · δP_hat(k_x,k_y,k−1)
        + diag_z · δP_hat(k_x,k_y,k  )       diag_z = α_z(k) − λ_x(k_x) − λ_y(k_y)
        + sup_z[k]· δP_hat(k_x,k_y,k+1)
        = PRHS_hat(k_x,k_y,k)
      apply wall BC (z=0, z=N3) → matrix row 보정
      PaScaL_TDMA.solve_many(...)

[5] Inverse transform (y 축, x 축 순서로)
    inverse along y
    inverse along x
    normalization (FFT N 배, DCT 2N 배 등)

[6] Mean removal (모든 압력 BC 가 Periodic / Neumann 이면 singular)
    δP ← δP − ⟨δP⟩

→ Projection:
    u^{n+1} = u* − Δt · (1/ρ) ∇δP
    P^{n+1} = P^n + δP
    dP_prev ← δP                    (다음 step 외삽용)
```

### 4.2 의자(transpose) 의 위치 — pencil 분해와 결부

3D MPI Cartesian decomposition `np1 × np2 × np3` 하에서, 변환은 *해당 축의 전체 데이터* 가 필요. → 전치 (`MPI_Alltoallw` + derived datatype).

```
초기 분할 (C-pencil, 각 축 부분 분할):
  local size = (n1msub, n2msub, n3msub)

전치 #1 (C → I): x 방향 데이터 모음 → I-pencil
  local size = (n1m, n2msub_I, n3msub_I)
  FFT/DCT in x

전치 #2 (I → K): z 방향 데이터 모음 → K-pencil
  local size = (n1m_Kx, n2msub_K, n3m)
  FFT/DCT in z

전치 #3 (K → ?): wall 축 z 인 경우 TDMA 는 PaScaL_TDMA 가 z 분할 위에서 직접 수행 가능
  (또는 다시 C-pencil 로 복귀 후 z 방향 PaScaL_TDMA)

역변환은 역순으로 전치
```

PaScaL_TCS 는 [module_solve_pressure.f90:558–586](../../PaScaL_TCS/src/module_solve_pressure.f90) 에 C→I, I→K 두 전치를 명시. MPM-STD 도 동일 패턴.

### 4.3 우리 컨벤션 (z=wall) 의 사이즈 변화 — 예시

```
n1m × n2m × n3m = 256 × 256 × 128
np1 × np2 × np3 = 4 × 4 × 2

C-pencil:    each rank holds (64, 64, 64)
I-pencil:    each rank holds (256, 32, 32)   ← x 전체
K-pencil:    each rank holds (32, 32, 128)   ← z 전체 (wall 축, TDMA 용)
```

→ PaScaL_TDMA 의 `_cycle` (periodic) 와 `_solve` (non-periodic) 가 마지막 K-pencil 에서 호출.

---

## 5. C++ 클래스 설계

### 5.1 헤더 골격

```cpp
// include/mpmstd/pressure/pressure_equation.hpp
namespace mpmstd {

enum class SpectralKind { None, FftR2c, DctII, DstII };

class FftPlanner {
public:
  FftPlanner(const Problem& problem, const Grid& g,
             const Subdomain& sub, Backend& be);

  SpectralKind  kind(Direction d) const { return kind_[int(d)]; }
  Direction     wall_axis() const { return wall_axis_; }
  const std::vector<double>& eigvals(Direction d) const { return eigvals_[int(d)]; }

  // 변환 실행
  void forward (Direction d, double* data) const;
  void backward(Direction d, double* data) const;

private:
  Direction                            wall_axis_;
  std::array<SpectralKind, 3>          kind_;
  std::array<std::vector<double>, 3>   eigvals_;
  // FFTW plans (직접 / 또는 r2r FFTW_REDFT10 for DCT-II / FFTW_RODFT10 for DST-II)
  std::array<fftw_plan, 3>             fwd_plan_, bwd_plan_;
};

class TransposePlan {
public:
  TransposePlan(const Subdomain& sub, TransposeKind kind);
  void execute(const double* in, double* out) const;
private:
  // MPI_Alltoallw + DDT
};

class PressureEquation {
public:
  PressureEquation(const Grid& g, const Subdomain& sub, FieldRegistry& fr,
                   const Problem& problem, TdmaRegistry& tdma,
                   FftPlanner& fft, BoundaryApplier& bc);

  void compute_rhs (const FieldRegistry& fr, double dt);     // PRHS
  void solve       (FieldRegistry& fr);                       // δP
  void project     (FieldRegistry& fr, double dt);            // u, P 갱신

private:
  void mean_remove_if_singular(ScalarField& dP) const;

  const Grid&            grid_;
  const Subdomain&       sub_;
  FieldRegistry&         fields_;
  const Problem&         problem_;
  TdmaRegistry&          tdma_;
  FftPlanner&            fft_;
  BoundaryApplier&       bc_;

  ScalarField            PRHS_;
  ScalarField            dP_prev_;                            // 외삽용 (dPhat = 2·dP − dP_prev)

  // 작업 버퍼 (전치 후 데이터 보관)
  std::vector<double>    buf_C_, buf_I_, buf_K_;
  TransposePlan          c_to_i_, i_to_k_, k_to_i_, i_to_c_;
};

}  // namespace mpmstd
```

### 5.2 `solve()` 의 구현 골격

```cpp
void PressureEquation::solve(FieldRegistry& fr) {
  ScalarField& dP = fr.scalar("dP");

  // ---- Forward path ----
  // 1. C → I 전치 (x 축 모음)
  c_to_i_.execute(PRHS_.host_ptr(), buf_I_.data());

  // 2. Forward FFT/DCT/DST along x
  if (fft_.kind(Direction::X) != SpectralKind::None)
    fft_.forward(Direction::X, buf_I_.data());

  // 3. I → K 전치 (y 축 모음, 또는 직접 K)
  i_to_k_.execute(buf_I_.data(), buf_K_.data());

  // 4. Forward FFT/DCT/DST along y
  if (fft_.kind(Direction::Y) != SpectralKind::None)
    fft_.forward(Direction::Y, buf_K_.data());

  // ---- Wall-axis TDMA: each (kx, ky) mode ----
  for (auto idx : modes_xy_local()) {
    auto [Am, Ac, Ap, D] = build_tridiag_z_mode(idx,
                                                 fft_.eigvals(Direction::X),
                                                 fft_.eigvals(Direction::Y),
                                                 grid_, problem_);
    if (problem_.topology.is_periodic(fft_.wall_axis()))
      tdma_.get(fft_.wall_axis()).solve_many_cyclic(Am, Ac, Ap, D, ...);
    else
      tdma_.get(fft_.wall_axis()).solve_many       (Am, Ac, Ap, D, ...);
  }

  // ---- Backward path ----
  if (fft_.kind(Direction::Y) != SpectralKind::None)
    fft_.backward(Direction::Y, buf_K_.data());
  k_to_i_.execute(buf_K_.data(), buf_I_.data());

  if (fft_.kind(Direction::X) != SpectralKind::None)
    fft_.backward(Direction::X, buf_I_.data());
  i_to_c_.execute(buf_I_.data(), dP.host_ptr());

  // ---- 평균 제거 (singular Neumann case) ----
  mean_remove_if_singular(dP);

  // ---- 외삽 (NOB) ----
  // 다음 step 의 projection 에 dPhat = 2·dP − dP_prev 사용
  // dP_prev 는 다음 step 시작 직전에 갱신
}
```

---

## 6. wall 축 TDMA 의 상세 — 1D 시스템 구조

### 6.1 ADI 가 아니라 modal TDMA

운동량 ADI 와는 *다른 의미* 의 TDMA. 운동량은 3 stage 의 *분할* 이지만, 여기는 *단일* 1D 시스템 — 단지 spectral mode 마다 독립.

각 모드 `(k_x, k_y)` 에 대해 wall 축 (z) 의 시스템:

```
  α_z(k+1) · δP_{kx,ky,k+1}
  + (β_z(k) − λ_x(k_x) − λ_y(k_y)) · δP_{kx,ky,k}
  + γ_z(k−1) · δP_{kx,ky,k−1}
  = PRHS_hat(k_x, k_y, k)
```

여기서:
- `α_z, β_z, γ_z` = z 축 2 차 차분 계수 (비-uniform 격자 지원, `dz_c, dz_f` 로 계산)
- `λ_x, λ_y` = §3 의 사전계산 고유값

### 6.2 wall BC matrix row 보정

z 축이 Dirichlet wall 이면 `j=1, j=N` 행을 `δP_wall = 0` 강제. Neumann 이면 ghost cell + matrix row 조정. [05_BC_design.md](05_BC_design.md) 의 `BoundaryApplier::modify_tdma_row` 가 처리.

### 6.3 PaScaL_TDMA 호출 패턴

```cpp
// k 모드 batched call
int n_sys = n_modes_xy_local;      // 로컬 (k_x, k_y) 개수
int n_row = n3m;                    // wall 축 길이 (= 전체 z 분할)

if (is_periodic_wall_axis)
  tdma_z.solve_many_cyclic(Am, Ac, Ap, D, n_sys, n_row);
else
  tdma_z.solve_many       (Am, Ac, Ap, D, n_sys, n_row);
```

PaScaL_TDMA 는 z 축을 따라 분산된 데이터에서 동작 (K-pencil). 실수·복소수 모두 지원 — Periodic 후의 complex 모드는 real + imag 두 번 풀이 ([module_solve_pressure.f90:638-639](../../PaScaL_TCS/src/module_solve_pressure.f90)).

---

## 7. Singular Neumann 케이스 — 평균 제거

### 7.1 문제

모든 압력 BC 가 Periodic 또는 Neumann 이면 Poisson 연산자가 singular (null space = 상수). 한 모드 `(k_x, k_y, k_z) = (0,0,0)` 에서 `λ = 0` → 0 으로 나눔 발생.

### 7.2 대처

방법 A — **DC 모드 제외**:
```cpp
if (kx == 0 && ky == 0 && kz_wall_singular) {
  δP_hat(0, 0, k) = 0;     // 상수 시프트 제거
  continue;                 // TDMA solve 안 함
}
```

방법 B — **물리 공간에서 평균 제거**:
```cpp
double mean = compute_mean(δP);
δP -= mean;
```

PaScaL_TCS 는 방법 B 사용 ([module_solve_pressure.f90:702–714](../../PaScaL_TCS/src/module_solve_pressure.f90)). 권장: **방법 B** (구현 단순, 분명).

### 7.3 singular 판정 — `Problem` 에서 자동

```cpp
bool PressureEquation::is_singular() const {
  for (Direction d : {X, Y, Z}) {
    if (d == fft_.wall_axis()) continue;          // wall 축은 TDMA 라 별개 (Dirichlet 이면 non-singular)
    if (fft_.kind(d) == SpectralKind::DstII) return false;     // Dirichlet → non-singular
  }
  // wall 축 BC 도 확인
  const FaceBc& fb = problem_.P.face(fft_.wall_axis(), Side::Minus);
  if (fb.kind == BcKind::Dirichlet) return false;
  return true;                                     // 모든 면 Periodic/Neumann → singular
}
```

→ RBC 기본 (모두 Periodic + wall Neumann) = singular → 평균 제거 매 step.

---

## 8. NOB 의 `dPhat` 외삽 — 사용 위치

variable density (NOB) 케이스에서 projection 의 정확도를 높이기 위해 PaScaL_TCS 가 사용 ([module_solve_pressure.f90:821](../../PaScaL_TCS/src/module_solve_pressure.f90)):

```
dPhat = 2·δP^n − δP^{n−1}
```

이를 운동량 RHS 의 압력 그래디언트 그리고 projection 의 보정 항에 사용:

```cpp
// projection step
void PressureEquation::project(FieldRegistry& fr, double dt) {
  ScalarField& dP   = fr.scalar("dP");
  ScalarField  dPhat = 2.0 * dP - dP_prev_;          // 외삽

  // u^{n+1} = u* − Δt · (1/ρ) · ∇δP
  for (...) {
    U.x()(i,j,k) -= dt * (1.0 / rho(...)) * stencil::dpdx_at_face_x(dP, grid_, i, j, k);
    // NOB 보정 항: 1/ρ 곱이 추가되는 부분에서 dPhat 사용
    // (NobBuoyancy 와 짝)
  }

  // P^{n+1} = P^n + δP
  fr.scalar("P").add(dP);

  // 다음 step 용 보관
  dP_prev_ = dP;
}
```

`ConstantProperties` 케이스 (Channel): `1/ρ = 1` 이므로 외삽 효과 없음 — 그래도 무해. NOB 활성 시만 의미.

---

## 9. PaScaL_TCS vs MPM-STD Fortran vs 우리 설계 — 비교

| 항목 | PaScaL_TCS | MPM-STD Fortran | 우리 C++ |
|---|---|---|---|
| 지원 BC 조합 | y=wall, x·z=periodic **1 종** | (NN, NP, PN, PP) **4 종** (x,y 만) | **Problem 에서 자동**, 모든 조합 |
| 변환 | R2C FFT (x,z) | R2C FFT + DCT (x,y) | R2C / DCT-II / DST-II 자동 선택 |
| Wall 축 | y (하드코딩) | z (하드코딩) | **`Problem.topology.wall_axis()` 자동** |
| Singular 처리 | 평균 제거 | 평균 제거 | 평균 제거 + `is_singular()` 자동 판정 |
| dPhat 외삽 | 있음 | 있음 | 있음 (PropertyPolicy 가 NOB 일 때만 의미) |
| 전치 | C↔I↔K, 2 차례 | 동일 | 동일 (`TransposePlan` 클래스) |
| 변환 라이브러리 | FFTW3 | cuFFT (D2Z + DCT 전·후처리) | FFTW3 (1차), cuFFT (미래) |
| DCT 구현 | 없음 | **R2C + pre/post trick** | FFTW3 `FFTW_REDFT10` 직접 또는 R2C trick |

### 9.1 DCT 구현의 두 가지 길

**Option A — FFTW3 의 r2r 직접**:
```cpp
fftw_plan plan = fftw_plan_r2r_1d(N, in, out, FFTW_REDFT10, FFTW_MEASURE);
// FFTW_REDFT10 = DCT-II for cell-centered Neumann
```
→ 가장 단순. CPU 만.

**Option B — R2C + pre/post 처리** (MPM-STD Fortran 패턴):
```
DCT-II 길이 N = R2C 길이 2N (대칭 확장) + 모드 추출
```
→ GPU (cuFFT) 에서 DCT 가 직접 지원 안 될 때 사용. 1차에는 불필요.

**우리 1차: Option A**. 미래 GPU 백엔드 추가 시 Option B 검토.

---

## 10. 검증 전략

### 10.1 단위 테스트 — 변환 자체

```cpp
TEST(FftPlanner, R2cInversionIdentity) {
  // 무작위 데이터 → forward → backward → 동일 데이터 (정규화 후)
}

TEST(FftPlanner, DctIIInversionIdentity) {
  // DCT-II → DCT-III (= inverse) → identity
}

TEST(FftPlanner, EigenvaluesMatchAnalytic) {
  // cos(πk x/L) 입력 → 변환 후 해당 k 모드만 nonzero
  // 분리 가능 Laplacian 의 고유값과 일치 확인
}
```

### 10.2 단위 테스트 — Poisson 자체

```cpp
TEST(PressureEquation, AnalyticPoisson) {
  // 알려진 우변 RHS = sin(πx/L)·sin(πz/H) 등
  // 알려진 해 δP_analytic 와 L∞ 비교
  // 격자 미세화 후 EOC = 2 확인
}

TEST(PressureEquation, NeumannSingularMeanZero) {
  // 모든 면 Neumann → 해의 평균 = 0 확인
}

TEST(PressureEquation, DivergenceAnnihilation) {
  // 임의의 u* → solve → project → max|∇·u| < 1e-12
}
```

### 10.3 회귀 — PaScaL_TCS golden

```
Ra=100, Pr=1, 512×128×256, 10 step → δP 와 PaScaL_TCS golden L∞ < 1e-10
```

---

## 11. 구현 순서 (M4 마일스톤 세부)

[08_design_revision_v2.md](08_design_revision_v2.md) Part 7 의 M4 를 다음 7 단계로 세부화:

| 단계 | 작업 | DoD |
|---|---|---|
| **M4.1** | `eigenvalues.hpp` 자유함수 + 단위 테스트 | EOC = 2 확인 |
| **M4.2** | `FftPlanner` 의 변환 dispatch (FFTW3 1D, 축별) | inversion identity 통과 |
| **M4.3** | `TransposePlan` (C↔I, I↔K) + DDT | 단위 테스트 (역전치 = identity) |
| **M4.4** | `PressureEquation::compute_rhs` | div(u*) 계산 검증 |
| **M4.5** | `PressureEquation::solve` (위 4 조각 결합) | 해석해 EOC = 2 |
| **M4.6** | `mean_remove_if_singular` + `is_singular` | Neumann all-face 케이스 평균 = 0 |
| **M4.7** | `project` + `dPhat` 외삽 | divergence annihilation, PaScaL_TCS golden L∞ < 1e-10 |

---

## 12. 결론

PressureSolver 의 핵심 가치:

1. **BC 종류가 변환 종류를 결정** — `Problem` 객체 한 곳에서 도출. PaScaL_TCS 의 y-wall, MPM-STD 의 z-wall, 미래 cavity (모든 면 wall) 모두 같은 라이브러리 코드.
2. **고유값 사전계산** — 매 step 재계산하지 않음. `FftPlanner` 가 보관.
3. **전치는 두 번** (C↔I↔K) — pencil decomposition 표준 패턴.
4. **wall 축은 항상 TDMA** — 변환 없이 (stretched 격자 지원).
5. **Singular Neumann 자동 처리** — 평균 제거 매 step.
6. **NOB 외삽 (dPhat)** — variable density 정확도 보존.
7. **DCT 는 FFTW3 r2r 직접 사용** (1차 CPU) → 미래 GPU 백엔드 추가 시 R2C trick 으로 전환.

이 7 가지가 [04_PaScaL_TCS_analysis.md](04_PaScaL_TCS_analysis.md) §4.3 의 PaScaL_TCS 핵심 알고리듬과 [03_MPM-STD_Fortran_analysis.md](03_MPM-STD_Fortran_analysis.md) §6.4 의 BC-aware FFT 패턴을 통합한 결과. **모든 BC 조합이 같은 코드로 처리됨**.
