# 수정계획서 03 — main 정리: Fields 명명 통일 · 물성/부력 초기화 내재화 · Restart/통계 객체 분리

- 작성일: 2026-07-24
- 대상: [apps/rbc/main_gpu_nob.cu](../../apps/rbc/main_gpu_nob.cu) (프로덕션 main, 719줄) 우선, 같은 패턴을 쓰는 dhvc/main_gpu.cu와 CPU main들에 순차 적용
- 상위 계획과의 관계: [01 계획서](01_deabstraction_user_friendliness_plan.md) **3단계(앱 main 통합)의 선행 정지작업**. restart·통계 시설을 `src/post/`로 옮겨 두면 rbc 7→1 통합 main과 dhvc가 그대로 재사용하고, CPU main들도 같은 시설의 CPU 경로를 써서 기능 격차(CPU에 restart 없음 등)가 함께 해소된다.
- 원칙: **계산 로직·파일 포맷 불변**(restart .bin/stats.txt/monitor.csv/stats_t*.dat 전부 byte 호환 유지). 변경은 이동·개명·초기화 캡슐화뿐.

---

## 0. 현황 — main_gpu_nob.cu 719줄의 구성 분해

| 구간 | 줄수 | 내용 |
|---|---|---|
| 통계 CUDA 커널 + 헬퍼 (익명 namespace) | ~344 | `k_accum_*`/`k_flow_diag`/`k_div_max` 커널 5개, `GpuRbcAccum` 버퍼 구조체, accumulate/print/write 헬퍼 4개 (:51-344) |
| config 읽기 | ~50 | :354-400 (이미 한 블록으로 재정렬됨) |
| 조립(topology→grid→BC→domain→fields→물성→부력) | ~75 | :406-431 |
| IC / restart 로드 | ~25 | :433-456 |
| GPU 셋업 + H2D | ~25 | :458-479 |
| 시간 루프 — 물리 스텝 [1]~[8] | ~100 | :517-618 |
| 시간 루프 — 통계/모니터/파일/체크포인트/타이밍 [9]~[10] | ~90 | :620-688 |
| 마무리(최종 통계·restart 기록) | ~30 | :690-718 |

**물리 스텝은 전체의 약 1/7뿐이고, 나머지가 통계·I/O·셋업 코드다.** 요구사항 5건은 전부 "물리 아닌 부분을 main 밖으로"에 해당한다.

---

## 1. Fields 명명 통일 + 메모리 위치 접두 규약

### 문제
- `CpuField`/`GpuField`는 `core/field_cpu.hpp`/`field_gpu.hpp`에 있는데, 그 컨테이너 `CpuFields`/`GpuFields`(`FieldStore`)는 **`core/variables.hpp`** 에 정의되어 있다 — 파일명이 타입 계열과 어긋나 정의 위치를 못 찾게 한다("namespace=위치" 규약의 파일명 판 위반).
- main에서 `core::CpuFields fields;`(:414)와 `core::GpuFields gpu_fields;`(:463)가 공존 — `fields`만 보면 host인지 device인지 알 수 없다. `props`(CPU, :422) vs `gpu_props`(:473)도 같은 문제이고, 게다가 축약이다.

### 변경
1. **파일 개명**: `core/variables.hpp` → **`core/fields.hpp`** (field_cpu / field_gpu / fields 한 계열로 통일). 포함 지점 15곳 기계적 갱신. 클래스명 `FieldStore`와 별칭 `CpuFields`/`GpuFields`는 유지(이미 서술적이고, 치환 규모 최소화).
2. **선언 접두 규약 제정**: *CPU/GPU 저장이 한 번역단위에 공존하면 변수명에 메모리 위치 접두 필수* — 앱 전반에 통일 적용:

| 현재 | 변경 | 위치 |
|---|---|---|
| `fields` (CpuFields) | `cpu_fields` | 모든 main (9개 파일) |
| `gpu_fields` | 유지 (이미 규약 준수) | |
| `props` | `cpu_properties` | main_gpu_nob.cu:422 등 |
| `gpu_props` | `gpu_properties` | :473 |
| `pmodel` | `property_model` | :423 (축약 금지 규칙) |
| `buoy` | `buoyancy_params` | :424 |
| `rmeta` | `restart_state` | :434 |
| `mon_fp` | (통계 객체로 흡수 — §5) | :499 |

CPU 전용 main(`main_cpu_nob.cpp` 등)도 `cpu_fields`로 통일한다 — 파일 간 diff/복붙 시 혼동 방지가 목적이므로 예외를 두지 않는다.

---

## 2. PropertyModel 선언 단축

### 문제 (:423)
```cpp
const physics::properties::PropertyModel pmodel =
    physics::properties::PropertyModel::glycerol(static_cast<real_t>(DeltaT));
```
타입명과 정적 팩토리의 이중 자격 표기 때문에 한 줄이 두 줄로 늘어난다.

### 변경
`properties.hpp`에 자유 함수 한 줄 추가 (namespace=위치 규약 그대로, 새 추상화 아님):
```cpp
inline PropertyModel glycerol(real_t DeltaT) { return PropertyModel::glycerol(DeltaT); }
```
main:
```cpp
const auto property_model = physics::properties::glycerol(static_cast<real_t>(DeltaT));
```
향후 다른 유체 모델도 같은 패턴(`physics::properties::water(...)`)으로 추가하면 된다.

---

## 3. BuoyancyParams 초기화 내재화

### 문제 (:424-427)
```cpp
physics::buoyancy::BuoyancyParams buoy;
buoy.gravity_axis = gravity_axis; buoy.coeff = 1; buoy.T_ref = 0;
buoy.a12pera11 = (std::fabs(pmodel.a[2]) < real_t{1e-14}) ? real_t{0} : pmodel.a[2] / pmodel.a[1];
buoy.delta_T   = static_cast<real_t>(DeltaT);
```
NOB 부력 파라미터가 물성 모델에서 유도되는 관계식(a12/a11, ΔT)이 main에 노출되어 있고 케이스마다 복붙된다. `PropertyModel`은 이미 `DeltaT` 멤버를 보유하므로(properties.hpp:22) main이 ΔT를 두 번 넘길 이유도 없다.

### 변경
`buoyancy.hpp`에 정적 팩토리 추가 — 구현식은 위 4줄을 **그대로 이동**(a[1]≈0 가드 포함):
```cpp
// NOB 전용: 물성 모델에서 유도 (a12pera11=a[2]/a[1], delta_T=model.DeltaT; coeff=1, T_ref=0)
static BuoyancyParams from_property_model(const PropertyModel& model, int gravity_axis);
```
main:
```cpp
const auto buoyancy_params =
    physics::buoyancy::BuoyancyParams::from_property_model(property_model, gravity_axis);
```
OB 경로(채널·DHVC)는 `coeff`가 케이스별 유도값이므로 기존 수동 설정을 유지한다(팩토리가 NOB 전용임을 주석으로 명시).
※ `buoyancy.hpp` → `properties.hpp` 포함이 새로 생김 — physics 모듈 내부의 단방향 의존이라 문제없음.

---

## 4. Restart 객체 — `src/post/`로 이동

### 문제
- restart 읽기/쓰기가 **앱 헤더**(rbc_common.hpp:117-199 `RestartMeta`/`write_restart`/`read_restart`)에 있고, ckpt_a/b 교대·D2H 스테이징·최종 기록이 main에 흩어져 있다(:434-456, :652-667, :709-717).
- 이 코드에는 재발성 함정 회피 지식이 주석으로 박혀 있다 — NFS 데드락 회피용 gathered write, P.bin 존재 판정의 root-판정+Bcast(집단호출 불일치 데드락, Report/34). 앱 파일에 있으면 새 케이스가 복붙하다 빠뜨리는 유형의 지식이므로 src로 옮겨 한 곳에서 강제해야 한다.

### 변경 — `post::Restart` (신규 `src/post/restart.hpp` + `cpu/restart_cpu.cpp`)
```cpp
namespace mpmstd::post {
struct RestartState {           // 필드 외 페이로드: 스텝/시간 + 케이스별 통계 스칼라
  int step = 0; double time = 0.0;
  std::vector<std::pair<std::string,double>> extras;   // 기록 순서 보존 → stats.txt byte 동일
};
class Restart {
public:
  Restart(std::string dir_in, std::string dir_out, int nrestart_write);
  bool warm() const;                                   // dir_in 지정 여부
  RestartState load(core::CpuFields&, const boundary::BoundaryCondition&,
                    const parallel::mpi::Subdomain&, MPI_Comm);      // U/V/W/T(+P 하위호환) + stats.txt
  void save(const core::CpuFields&, const parallel::mpi::Subdomain&,
            const RestartState&, MPI_Comm) const;                    // dir_out에 최종 기록
  void checkpoint_if_due(int step, double time, const core::CpuFields&,
                         const parallel::mpi::Subdomain&, const RestartState&,
                         MPI_Comm);                                  // ckpt_a/b 교대 내장
};
}
```
- 구현은 rbc_common의 함수를 **그대로 이동**: gathered write(NFS 안전), P.bin root-exists+Bcast, stats.txt 17자리 정밀도·키 순서 유지 → **기존 체크포인트와 완전 호환**.
- 통계 페이로드는 `extras`(순서 있는 key-value)로 일반화 — RBC는 `{tc_sum, tc_n, wt_sum}`을 등록하고, 다른 케이스는 자기 키를 등록한다. 케이스 고유 구조체가 post 계층에 새지 않는다.
- GPU 스테이징: main의 `to_host` 5줄 반복(:659-663, :711-715)은 `core/fields.hpp` 헬퍼로:
```cpp
void copy_all_to_host(const GpuFields&, CpuFields&);   // 등록된 Var 전체 D2H
```
  `Restart` 자체는 CPU 전용으로 유지한다(팀 레이아웃 규칙: I/O는 host 계층).

---

## 5. 통계 객체 — `src/post/`로 이동

### 문제
main_gpu_nob.cu의 절반(:51-344)이 통계 기반시설이다: CUDA 커널 5개 + `GpuRbcAccum` 버퍼 + accumulate/print/write 헬퍼 4개. 시간 루프에도 [9a]~[9c] 호출, `t_next_out` 상태, monitor.csv `FILE*`, clip 카운터 보고가 흩어져 있다. rbc_common.hpp에는 같은 일의 CPU 판(`RbcStats`/accumulate/print_step/print_final)이 별도로 존재한다.

### 변경 — `post::RbcStatistics` (신규 `src/post/statistics_rbc.hpp` + `gpu/statistics_rbc_gpu.cu`)
```cpp
namespace mpmstd::post {
struct RbcStatisticsConfig {
  double t_stat_start, t_stat_out; int nmonitor;
  std::string dir_stats;                       // monitor.csv + stats_t*.dat 위치
  double Ra, DeltaT, alpha_T;
};
class RbcStatistics {
public:
  RbcStatistics(const parallel::mpi::Subdomain&, const core::GpuGridArrays&,
                const RbcStatisticsConfig&, MPI_Comm, bool is_root);
  void accumulate_if_due(double time, const core::GpuFields&);                  // [9a]
  void monitor_if_due(int step, double time, real_t dt, const core::GpuFields&); // [9b] printf+csv+clip 보고
  void write_profile_if_due(double time, int step);                             // [9c] stats_t<t>.dat (t_next_out 내장)
  void finalize(int step);                                                      // stats_final.dat + print_final
  // restart 연동
  std::vector<std::pair<std::string,double>> export_extras() const;             // {tc_sum, tc_n, wt_sum}
  void import_extras(const RestartState&, bool stats_reset);
  unsigned long long* clip_counter();          // k_clip_T_gpu가 쓰는 카운터(보고는 monitor_if_due가)
};
}
```
- 커널·버퍼·헬퍼 전부 **코드 이동만**(리덕션 수식·D2H 최적화(≤48B/스텝)·파일 헤더/컬럼 문자열 불변).
- monitor.csv 열기/닫기, `t_next_out` 갱신 등 상태 전부 객체 내부로 — main에서는 대표 호출 4개만 남는다.
- `k_clip_T_gpu` 자체는 물리 스텝([8])의 일부이므로 main에 남기고, 카운터 소유/보고만 통계 객체가 맡는다(클리핑은 통계가 아니라 물리 가드라는 구분 유지).
- rbc_common.hpp의 CPU 판 통합은 **3단계(main 통합)에서** 처리 — 이번에는 GPU 프로덕션 경로만 옮겨 diff를 최소화한다.

---

## 6. 목표 상태 — main_gpu_nob.cu 스케치 (719줄 → 약 280줄)

```cpp
int main(int argc, char** argv) {
  // [입력] config 블록 (기존 그대로, ~50줄)
  // [조립]
  ... topo / sub / grid / bc / domain ...
  core::CpuFields cpu_fields;   /* U,V,W,P,T 등록 */
  const auto property_model  = physics::properties::glycerol(DeltaT);
  const auto buoyancy_params = physics::buoyancy::BuoyancyParams::from_property_model(property_model, gravity_axis);
  post::Restart restart(restart_in, restart_out, nrestart_write);
  auto restart_state = restart.warm() ? restart.load(cpu_fields, bc, sub, comm)
                                      : post::RestartState{};   // cold면 기존 IC 블록 실행
  // [GPU] bind, gpu_grid, gpu_fields/gpu_properties 등록, H2D
  post::RbcStatistics statistics(sub, gpu_grid_arrays, stat_config, comm, mpi.is_root());
  statistics.import_extras(restart_state, stats_reset);

  while (time < t_end && step < n_steps) {
    // [1]~[8] 물리 스텝 — 기존 그대로 (CFL→물성→에너지→물성→운동량→W벽→압력→클립)
    statistics.accumulate_if_due(time, gpu_fields);            // [9a]
    statistics.monitor_if_due(step, time, dt, gpu_fields);     // [9b]
    statistics.write_profile_if_due(time, step);               // [9c]
    if (checkpoint_due) {                                      // [9d]
      core::copy_all_to_host(gpu_fields, cpu_fields);
      restart.checkpoint_if_due(step, time, cpu_fields, sub,
          make_state(step, time, statistics.export_extras()), comm);
    }
  }
  statistics.finalize(step);
  core::copy_all_to_host(gpu_fields, cpu_fields);
  restart.save(cpu_fields, sub, make_state(step, time, statistics.export_extras()), comm);
}
```
물리 스텝 [1]~[8]과 단계 주석은 그대로 남는다 — "시간 루프는 명시적으로 유지"(01 계획서 §3 원칙)와 부합하면서, 루프 안의 비물리 코드가 대표 호출 4줄로 줄어든다.

---

## 7. 이행 순서 (커밋 단위)

| 커밋 | 내용 | 규모 | 검증 |
|---|---|---|---|
| **F1** | `variables.hpp`→`fields.hpp` 개명(포함 15곳) + §1 변수 rename(앱 9파일) | 기계적 | CPU: validate 필드덤프 8/8 byte-동일 · GPU: clean build + 스모크 |
| **F2** | `glycerol()` 자유 함수 + `BuoyancyParams::from_property_model` (§2·§3) | +10줄, main −6줄 | 상동 (값 경로 동일 → byte-동일 기대) |
| **F3** | `post::Restart` 신설, rbc_common restart 코드 이동, `copy_all_to_host` 헬퍼 (§4) | 이동 ~110줄 | **기존 체크포인트 호환**: `restart_reorder_cold_dt40`을 새 바이너리로 load→즉시 save→구 파일과 U/V/W/T/P.bin **byte-diff 0** + stats.txt diff 0 |
| **F4** | `post::RbcStatistics` 신설, 커널/헬퍼 344줄 이동 (§5) | 이동 ~350줄, main 절반 감소 | monitor.csv·stats_t*.dat 헤더/포맷 문자열 diff 없음 + 재시작 이어달리기 스모크에서 Θ_c/Nu 6자리 일치(N-시리즈 검증 프로토콜 재사용) |
| **F5** | dhvc/main_gpu.cu에 F1~F4 패턴 적용 | 소규모 | dhvc GPU 스모크 |

- 검증 환경: GPU 노드에서 clean `make BACKEND=cuda apps`; 입출력은 공유 NFS 경로(`/shared/...`); CPU 빌드는 Intel oneAPI 소싱 금지. F1의 CPU측 rename은 validate byte-게이트가 커버하고, `.cu` 변경분은 매 커밋 GPU 빌드+스모크가 게이트.
- **F3의 byte-diff 왕복 검증이 핵심 관문이다** — restart 포맷이 1바이트라도 달라지면 기존 체크포인트 자산(수백만 스텝 러닝)이 무효가 되므로, "load→save 왕복 byte 동일"을 통과해야만 머지한다.

## 8. 범위 제외 (이번에 하지 않는 것)

- rbc_common.hpp의 CPU 통계/진단(div_max, flow_diag, print_* 등)과 CPU main들의 시설 교체 — 3단계 main 통합에서 함께 처리(지금 하면 7개 main을 두 번 고치게 된다).
- `k_zero_w_wall_gpu`·`k_clip_T_gpu`의 모듈 이동 — 물리 스텝 소속이라 main 잔류가 맞고, 3단계 통합 main 설계 때 재검토.
- config 키 통일·문서화 — 01 계획서 4단계 몫.
