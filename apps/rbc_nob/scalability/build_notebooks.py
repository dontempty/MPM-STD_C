"""세 노트북(main / momentum / scalar)을 같은 양식으로 생성·실행:
   §A 최적화 전: 이벤트별 막대(np 나란히) + 이벤트별 scaling 곡선
   §B 최적화 후(최신 바이너리): 같은 그림
   §C 전 vs 후: 이벤트별 감소량 막대 (같은 기종·문제 크기·np)
   저장 없음 (SAVE_FIG = False 고정).  사용: python3 build_ba_nb.py [main|momentum|scalar|all]"""
import base64, io, json, os, sys, contextlib
PLOT_DIR = "/scratch/r908a06/KDH/MPM-STD_C/apps/rbc_nob/plot"

SPECS = {
  "main": dict(
    nb="main_scalability.ipynb", title="rbc_nob 전체 스텝 프로파일 — 최적화 전 / 후 / 전·후 비교",
    what="`apps/rbc_nob/main.cpp` 시간 루프의 호출 단위 구간(plan.md §1 표). 분산 TDMA 의 MPI 시간 3개(`tdma_comm_*_z`, 풀이 안에 포함)는 참고용으로 같이 그린다.",
    before={"h200": "h200", "v100": "v100", "a100": "a100"},
    after={"h200h": "h200", "v100h": "v100", "v100r1": "v100", "v100s": "v100", "v100p": "v100", "v100pt": "v100", "v100pf": "v100", "v100pr": "v100"},
    before_note="원본 바이너리(최적화 전). H200 gpu55 4장·V100 gpu04 4장·A100 gpu36 1장 (np ≤ 4).",
    after_note="최신 바이너리 = momentum build_kernel_z 타일 전치 + 순환 Thomas 레지스터화 + scalar build_z 타일 전치 + halo 자기 이웃 디바이스 복사 + momentum_rhs 역수·융합 + 조립·커플링·scalar 스텐실 역수 + 패스 제거(scatter_z 타일 전치·밴드 핑퐁·증분 융합) + pressure 전치 커널 타일화(`v100pt`; H200 은 `h200h` 까지 — H200 잡 취소). V100 8장 (np ≤ 8).",
    events=["dt_cfl", "dt_viscous", "properties_1", "scalar_solve", "sync_T", "momentum_rhs", "momentum_solve", "sync_UVW_1",
            "properties_2", "pressure_rhs", "pressure_solve", "pressure_correct", "sync_UVW_2", "predictor", "clip", "monitor",
            "tdma_comm_scalar_z", "tdma_comm_momentum_z", "tdma_comm_pressure_z"],
    prefix="", total="step_total", require=None,
    breakdown=("I. pressure_solve 단계별 분해 (최신 바이너리 + 단계 타이머, `v100q`)",
               "`pressure::Solver::timer` 로 잰 14단계 (팩·FFT·재배열·밴드 팩·z TDMA·언팩·산포·평균 제거·halo). np 1 에서 reorder_x_to_y·pack/scatter_lines·reorder_y_to_x 가 전치 복사라 크고, np 8 에서는 tdma_z(Alltoallv)가 전부.",
               {"v100q": "v100", "v100pt": "v100", "v100pf": "v100", "v100pr": "v100"}, "pressure."),
    extra=[("E. 통신(halo 자기 전송) 수정 전 vs 후 — np 8 포함",
            "A/B 의 '전' 은 원본 바이너리라 np ≤ 4 뿐이어서 halo 수정 효과(np 8 에서 큼)가 안 보인다. 여기서는 **halo 수정 직전 바이너리**(다른 최적화는 모두 적용, 꼬리표 `h200z`·`v100z`)를 '전' 으로 두고 같은 세 그림을 그린다. 차이는 전부 halo 교환(sync_*, predictor, momentum 내부 halo)에서 나야 한다.",
            {"h200z": "h200", "v100z": "v100"}),
           ("F. momentum_rhs 최적화(역수·융합) 직전 vs 최종 — V100",
            "전 = halo 까지 적용된 바이너리(`v100h`), 후 = momentum_rhs 최적화까지 적용된 최종(`v100r1`). 차이는 momentum_rhs 에서만 나야 한다.",
            {"v100h": "v100"}),
           ("G. 조립·커플링·scalar 스텐실 역수 직전 vs 최종 — V100",
            "전 = momentum_rhs 까지 적용된 바이너리(`v100r1`), 후 = sweep_band·couple_*·axis_band 의 나눗셈을 격자 역수 곱으로 바꾼 최종(`v100s`). 차이는 momentum_solve(build_*·couple·block_couple)와 scalar_solve(build_*)에서 나야 한다.",
            {"v100r1": "v100"}, [("momentum_solve", ["momentum_solve"]), ("scalar_solve", ["scalar_solve"]), ("momentum_rhs", ["momentum_rhs"])]),
           ("H. 패스 제거(scatter_z 타일 전치·밴드 핑퐁·증분 융합) 직전 vs 최종 — V100",
            "전 = `v100s`, 후 = `v100p`. 차이는 momentum_solve(scatter_*·add_increment)와 scalar_solve(scatter_*·add_increment)에서만 나야 한다.",
            {"v100s": "v100"}, [("momentum_solve", ["momentum_solve"]), ("scalar_solve", ["scalar_solve"]), ("pressure_solve", ["pressure_solve"])]),
           ("J. pressure 전치 커널 타일화 직전 vs 최종 — V100",
            "전 = `v100q`(단계 타이머만), 후 = `v100pt`(pack_lines·reorder_x_to_y·reorder_y_to_x·scatter_lines 를 32×32 타일 전치로). 차이는 pressure_solve 에서만 나야 한다.",
            {"v100q": "v100"}, [("pressure_solve", ["pressure_solve"]), ("reorder x↔y", ["pressure.reorder_x_to_y", "pressure.reorder_y_to_x"]), ("pack+scatter_lines", ["pressure.pack_lines", "pressure.scatter_lines"]), ("fft ×4", ["pressure.fft_x_fwd", "pressure.fft_y_fwd", "pressure.fft_y_bwd", "pressure.fft_x_bwd"])]),
           ("K. pressure 우변·보정 융합 직전 vs 최종 — V100",
            "전 = `v100pt`, 후 = `v100pr`(fill·∇·u*·ExtraTerm 를 한 커널로 + 속도 보정·P 누적 융합 + 격자 역수). 융합만(`v100pf`)은 −0.4 ms 뿐이었고 역수가 핵심. 차이는 pressure_rhs·pressure_correct 에서만 나야 한다.",
            {"v100pt": "v100"}, [("pressure_rhs", ["pressure_rhs"]), ("pressure_correct", ["pressure_correct"]), ("pressure_solve", ["pressure_solve"])])],
  ),
  "momentum": dict(
    nb="momentum_scalability.ipynb", title="momentum_solve 이벤트별 프로파일 — 최적화 전 / 후 / 전·후 비교",
    what="`momentum::solve_variable_property` 가 부르는 호출 (`momentum.<dU|dV|dW>.<호출>` 을 세 성분 합으로, `momentum.post.*` 는 그대로). 상위 구간 `momentum_solve` 도 첫 막대로.",
    before={"h200m": "h200", "v100m": "v100"},
    after={"h200h": "h200", "v100h": "v100", "v100r1": "v100", "v100s": "v100", "v100p": "v100", "v100pt": "v100", "v100pf": "v100", "v100pr": "v100"},
    before_note="momentum 분해 측정 시점의 바이너리(최적화 전). H200 gpu55 4장·V100 gpu04 4장.",
    after_note="최신 바이너리(위 네 가지 최적화 + momentum_rhs 역수·융합 + 조립·커플링 스텐실 역수 + 패스 제거 `v100p`). H200 gpu55, V100 8장.",
    events=["momentum_solve", "build_x", "tdma_x", "scatter_x", "build_y", "tdma_y", "scatter_y", "build_z", "tdma_z", "scatter_z",
            "halo", "wall_ghost", "couple_rhs", "block_couple_v", "halo_dV", "wall_ghost_dV", "block_couple_u", "add_increment"],
    prefix="momentum", total="momentum_solve", require="momentum.",
    extra=[("E. 조립·커플링 스텐실 역수 직전 vs 최종 — V100",
            "전 = `v100r1`(momentum_rhs 까지), 후 = `v100s`(sweep_band·own_face_blend·couple_* 의 나눗셈을 역수 곱으로). 차이는 build_x/y/z·couple_rhs·block_couple_* 에서만 나야 한다.",
            {"v100r1": "v100"}, [("build_x+y+z (3성분 합)", ['momentum.dU.build_x', 'momentum.dU.build_y', 'momentum.dU.build_z', 'momentum.dV.build_x', 'momentum.dV.build_y', 'momentum.dV.build_z', 'momentum.dW.build_x', 'momentum.dW.build_y', 'momentum.dW.build_z']), ("couple_rhs+block_couple", ['momentum.dV.couple_rhs', 'momentum.dW.couple_rhs', 'momentum.post.block_couple_v', 'momentum.post.block_couple_u']), ("tdma_x+y+z (3성분 합)", ['momentum.dU.tdma_x', 'momentum.dU.tdma_y', 'momentum.dU.tdma_z', 'momentum.dV.tdma_x', 'momentum.dV.tdma_y', 'momentum.dV.tdma_z', 'momentum.dW.tdma_x', 'momentum.dW.tdma_y', 'momentum.dW.tdma_z'])]),
           ("F. 패스 제거 직전 vs 최종 — V100",
            "전 = `v100s`, 후 = `v100p`(scatter_z 타일 전치, scatter_x/y 제거, W += 융합). 차이는 scatter_*·add_increment 에서만 나야 한다.",
            {"v100s": "v100"}, [("scatter_x+y+z (3성분 합)", ['momentum.dU.scatter_x', 'momentum.dU.scatter_y', 'momentum.dU.scatter_z', 'momentum.dV.scatter_x', 'momentum.dV.scatter_y', 'momentum.dV.scatter_z', 'momentum.dW.scatter_x', 'momentum.dW.scatter_y', 'momentum.dW.scatter_z']), ("add_increment", ["momentum.post.add_increment"]), ("build_x+y+z (3성분 합)", ['momentum.dU.build_x', 'momentum.dU.build_y', 'momentum.dU.build_z', 'momentum.dV.build_x', 'momentum.dV.build_y', 'momentum.dV.build_z', 'momentum.dW.build_x', 'momentum.dW.build_y', 'momentum.dW.build_z'])])],
  ),
  "scalar": dict(
    nb="scalar_scalability.ipynb", title="scalar_solve(에너지 방정식) 이벤트별 프로파일 — 최적화 전 / 후 / 전·후 비교",
    what="`scalar::solve_variable_property` 가 부르는 호출 (`scalar.<build|tdma|scatter>_<x|y|z>`, `scalar.add_increment`). 상위 구간 `scalar_solve` 도 첫 막대로.",
    before={"h200k": "h200", "v100e": "v100"},
    after={"h200h": "h200", "v100h": "v100", "v100r1": "v100", "v100s": "v100", "v100p": "v100", "v100pt": "v100", "v100pf": "v100", "v100pr": "v100"},
    before_note="scalar 분해 측정 시점의 바이너리(scalar build_z 수정 전, momentum 최적화는 적용됨). H200 gpu55 4장·V100 gpu04 8장.",
    after_note="최신 바이너리(scalar build_z 타일 전치 + halo 자기 이웃 복사 + axis_band 스텐실 역수 + 패스 제거 `v100p`). H200 gpu55, V100 8장.",
    events=["scalar_solve", "build_x", "tdma_x", "scatter_x", "build_y", "tdma_y", "scatter_y", "build_z", "tdma_z", "scatter_z", "add_increment"],
    prefix="scalar", total="scalar_solve", require="scalar.",
    extra=[("E. scalar 스텐실 역수 직전 vs 최종 — V100",
            "전 = `v100r1`, 후 = `v100s`(axis_band 의 `/dx`·`/dmx` 를 역수 곱으로). 차이는 build_x/y/z 에서만 나야 한다.",
            {"v100r1": "v100"}, [("build_x", ["scalar.build_x"]), ("build_y", ["scalar.build_y"]), ("build_z", ["scalar.build_z"]), ("tdma_x+y+z", ["scalar.tdma_x", "scalar.tdma_y", "scalar.tdma_z"])]),
           ("F. 패스 제거 직전 vs 최종 — V100",
            "전 = `v100s`, 후 = `v100p`(scatter_z 타일 전치 + T += 융합, scatter_x/y 제거). 차이는 scatter_*·add_increment 에서만 나야 한다.",
            {"v100s": "v100"}, [("scatter_x+y+z", ["scalar.scatter_x", "scalar.scatter_y", "scalar.scatter_z"]), ("add_increment", ["scalar.add_increment"]), ("build_x+y+z", ["scalar.build_x", "scalar.build_y", "scalar.build_z"])])],
  ),
  "present": dict(
    nb="present.ipynb", title="rbc_nob 전체 스텝 — 제일 처음 결과 vs 최신 결과",
    what="`apps/rbc_nob/main.cpp` 시간 루프의 호출 단위 구간(plan.md §1 표). 두 바이너리만 비교한다: **처음** = 최적화 전 원본, **최신** = 현재 본 코드. 최신 바이너리로 잰 기종은 V100 뿐이라 V100 만 싣는다 (H200 은 10-17 중간 판까지만 있어 제외).",
    before={"v100": "v100"},
    after={"v100pr": "v100"},
    before_note="원본 바이너리(최적화 전). V100 gpu04 4장 (np ≤ 4).",
    after_note="최신 바이너리(모든 최적화 적용: build_z 타일 전치 · 순환 Thomas 레지스터화 · scalar build_z · halo 자기 이웃 복사 · momentum_rhs 역수·융합 · 조립·커플링·scalar 스텐실 역수 · 패스 제거 · pressure 전치 타일화). V100 8장 gpu02 (np ≤ 8).",
    events=["dt_cfl", "dt_viscous", "properties_1", "scalar_solve", "sync_T", "momentum_rhs", "momentum_solve", "sync_UVW_1",
            "properties_2", "pressure_rhs", "pressure_solve", "pressure_correct", "sync_UVW_2", "predictor", "clip", "monitor"],
    prefix="", total="step_total", require=None, present=True,
  ),
}

def make_cells(S):
    cells = []
    md = lambda s: cells.append(("markdown", s))
    code = lambda s: cells.append(("code", s))
    md(f"""# {S['title']}

{S['what']}

데이터 `apps/rbc_nob/scalability/out/scal_<mode>_<n>_np<N>/phase_timing_<tag>_<job>_r<rep>.csv` (200 스텝 중 워밍업 50 제외, 반복 중 `step_total` 중앙값 런, ms/step 랭크 max).
기종·노드·GPU 수는 꼬리표의 환경 파일(nvidia-smi 기록)에서 읽어 첫 셀 출력에 적고, 모든 제목·범례가 그 문자열과 문제 크기(strong/weak, n³, np)를 쓴다. **그림 저장 없음.**

| | 꼬리표 | 비고 |
|---|---|---|
| **최적화 전** | {', '.join(f'`{t}`' for t in S['before'])} | {S['before_note']} |
| **최적화 후** | {', '.join(f'`{t}`' for t in S['after'])} | {S['after_note']} |

strong: 전역 격자 고정(256³·512³), npz = np, $E = T(np_0)/(T(np)\\,np/np_0)$. weak: 랭크당 격자 고정(256³; H200 은 512³ 도), 전역 nz = n·np, $E = T(np_0)/T(np)$.""")

    code(f'''import csv, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, PercentFormatter
plt.rcParams.update({{"figure.dpi": 110, "font.size": 9, "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False}})
OUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "scalability", "out"))
SAVE_FIG = False                                   # 항상 끔

HW_COLOR = {{"h200": "#2a78d6", "v100": "#1baf7a", "a100": "#eb6834"}}      # 기종 = 색 (검증된 팔레트 슬롯)
NP_COLOR = {{1: "#bcd3f0", 2: "#7fa9e3", 4: "#2a78d6", 8: "#164a8a"}}         # np = 파란색 명도
BEFORE_TAGS = {S['before']!r}
AFTER_TAGS  = {S['after']!r}
EVENTS = {S['events']!r}
PREFIX = {S['prefix']!r}        # "" = main 구간, "momentum"/"scalar" = 풀이 안 호출
TOTAL = {S['total']!r}
REQUIRE = {S['require']!r}      # 이 접두사의 구간이 있는 런만 (세부 계측이 있는 바이너리)

def event_time(ph, e):
    if PREFIX == "":
        return ph.get(e, 0.0)
    if e == TOTAL:
        return ph.get(TOTAL, 0.0)
    if PREFIX == "momentum":
        if e in ("block_couple_v", "halo_dV", "wall_ghost_dV", "block_couple_u", "add_increment"):
            return ph.get(f"momentum.post.{{e}}", 0.0)
        return sum(ph.get(f"momentum.{{c}}.{{e}}", 0.0) for c in ("dU", "dV", "dW"))
    return ph.get(f"scalar.{{e}}", 0.0)

def load(tags):
    """{{(hw, mode, n, np): 대표 런}} — 같은 조합이 여러 꼬리표에 있으면 tags 에서 뒤에 오는 꼬리표 우선"""
    order = list(tags); runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(OUT_DIR, "scal_*", "phase_timing_*.csv"))):
        m = re.search(r"scal_(strong|weak)_(\\d+)_np(\\d+)/phase_timing_([a-z0-9]+)_(\\d+)_r(\\d+)\\.csv$", path)
        if not m or m.group(4) not in tags:
            continue
        ph = {{}}
        with open(path) as fh:
            for row in csv.reader(l for l in fh if not l.startswith("#")):
                if row[0] != "phase":
                    ph[row[0]] = float(row[3])
        if REQUIRE and not any(k.startswith(REQUIRE) for k in ph):
            continue
        ph["_job"] = m.group(5)
        runs[(tags[m.group(4)], m.group(1), int(m.group(2)), int(m.group(3)))].append((order.index(m.group(4)), ph))
    out, used = {{}}, {{}}
    for k, lst in runs.items():
        best = max(pr for pr, _ in lst); sel = [ph for pr, ph in lst if pr == best]
        out[k] = sorted(sel, key=lambda r: r["step_total"])[len(sel) // 2]; used[k] = order[best]
    return out, used

def describe_tag(tag, jobs=None):
    hosts, names, ngpu = [], [], 0
    for f in sorted(glob.glob(os.path.join(OUT_DIR, f"environment_{{tag}}_*.txt"))):
        if jobs is not None and re.search(r"_(\d+)\.txt$", f).group(1) not in jobs:
            continue
        txt = open(f).read()
        h = re.search(r"host=(\\S+)", txt)
        if h and h.group(1) not in hosts: hosts.append(h.group(1))
        gl = [l.split(",")[0].strip() for l in txt.splitlines() if l.startswith(("NVIDIA", "Tesla"))]
        if gl:
            ngpu = max(ngpu, len(gl))
            if gl[0] not in names: names.append(gl[0])
    return f"{{'/'.join(names) or tag}} x{{ngpu}} @ {{'/'.join(hosts) or '?'}}"

BEFORE, BEFORE_TAG = load(BEFORE_TAGS)
AFTER, AFTER_TAG = load(AFTER_TAGS)
DESC = {{}}
def describe(label, cases, used):
    """실제로 쓴 런(꼬리표·잡)만으로 기종 문자열"""
    for hw in sorted({{k[0] for k in cases}}):
        tags = sorted({{used[k] for k in cases if k[0] == hw}})
        jobs = {{cases[k]["_job"] for k in cases if k[0] == hw}}
        DESC[(label, hw)] = " / ".join(f"{{describe_tag(t, jobs)}} [{{t}}]" for t in tags)
describe("before", BEFORE, BEFORE_TAG); describe("after", AFTER, AFTER_TAG)
def hw_desc(label, hw):
    return DESC.get((label, hw), hw)

for label, cases, used in (("before", BEFORE, BEFORE_TAG), ("after", AFTER, AFTER_TAG)):
    print(f"[{{label}}] {{len(cases)}} cases")
    for hw in sorted({{k[0] for k in cases}}):
        print(f"  {{hw}} = {{hw_desc(label, hw)}}")
    for k in sorted(cases):
        print(f"    {{k[0]:5s}} {{k[1]:6s}} {{k[2]}}^3 np{{k[3]:<2d}}  step_total {{cases[k]['step_total']:7.1f}}  {{TOTAL}} {{event_time(cases[k], TOTAL):7.2f}} ms/step  [{{used[k]}}]")''')

    code(r'''def plot_event_bars(cases, label, mode, hw, n):
    keys = sorted(k for k in cases if k[:3] == (hw, mode, n))
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(12.5, 4.0))
    nps = [k[3] for k in keys]; xs = np.arange(len(EVENTS)); w = 0.8 / len(nps)
    base = np.array([event_time(cases[keys[0]], e) for e in EVENTS]); vmax = base.max()
    for j, k in enumerate(keys):
        vals = np.array([event_time(cases[k], e) for e in EVENTS]); vmax = max(vmax, vals.max())
        pos = xs - 0.4 + w * (j + 0.5)
        ax.bar(pos, vals, w * 0.92, color=NP_COLOR.get(k[3], "#164a8a"), label=f"np {k[3]}  (step_total {cases[k]['step_total']:.1f} ms)")
        if j > 0:
            for x, v, b in zip(pos, vals, base):
                if b > 0.02 * base.max():
                    ax.annotate(f"{v / b:.2f}", (x, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.3, color="#52514e", rotation=90)
    ax.set_xticks(xs); ax.set_xticklabels(EVENTS, rotation=40, ha="right"); ax.set_ylabel("ms / step (rank max)")
    ax.set_title(f"[{label}] {hw_desc(label, hw)} — {mode} {n}³: events by np (number = ratio to np {nps[0]})", fontsize=10, loc="left")
    ax.set_ylim(0, vmax * 1.15)
    ax.legend(frameon=False, ncol=len(nps), loc="upper center", bbox_to_anchor=(0.5, -0.55), fontsize=7.5)
    plt.tight_layout()

STYLE = {"before": dict(marker="o", ls="--", mfc="white", ms=10, mew=1.6, z=2), "after": dict(marker="s", ls="-", mfc=None, ms=5, mew=1.2, z=3)}   # 전 = 큰 빈 원(겹쳐도 보임), 후 = 작은 채운 사각형

def plot_event_scaling(versions, mode, n):
    """versions = [(cases, label), ...] — 같은 패널에 겹쳐 그린다. before = ○ 파선(빈 마커), after = ■ 실선(채운 마커). 점선 = 각 선의 이상선."""
    hws = sorted({k[0] for cases, _ in versions for k in cases if k[1:3] == (mode, n)})
    if not hws:
        return
    nps_all = sorted({k[3] for cases, _ in versions for k in cases if k[1:3] == (mode, n)})
    ncol = 5; nrow = int(np.ceil(len(EVENTS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13.5, 2.6 * nrow))
    for ax, e in zip(axes.flat, EVENTS):
        for cases, label in versions:
            st = STYLE[label]
            for hw in hws:
                keys = sorted(k for k in cases if k[:3] == (hw, mode, n))
                x = np.array([k[3] for k in keys], float); y = np.array([event_time(cases[k], e) for k in keys])
                if len(x) < 2:
                    continue
                kw = dict(color=HW_COLOR[hw], lw=1.6, ms=st["ms"], mfc=st["mfc"] or HW_COLOR[hw], mew=st["mew"], marker=st["marker"], ls=st["ls"],
                          zorder=st["z"], label=f"{hw_desc(label, hw)} — {label}")
                if y[0] <= 0:
                    if y.max() > 0:
                        ax.plot(x[y > 0], y[y > 0], **kw)
                    continue
                ax.plot(x, y, **kw)
                ideal = y[0] * x[0] / x if mode == "strong" else np.full_like(y, y[0])
                ax.plot(x, ideal, ":", color="#9a9a96", lw=0.9)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(nps_all)); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([str(p) for p in nps_all]); ax.set_xlim(min(nps_all) / 1.4, max(nps_all) * 1.4)
        ax.set_title(e, fontsize=9, loc="left"); ax.tick_params(labelsize=7.5)
    for ax in axes.flat[len(EVENTS):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("np")
    for ax in axes[:, 0]:
        ax.set_ylabel("ms / step")
    handles, labels = [], []
    for ax in axes.flat:
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in labels: handles.append(h_); labels.append(l_)
    fig.legend(handles, labels, frameon=False, loc="lower right", ncol=min(len(labels), 2), fontsize=7.5)
    tag = " vs ".join(l for _, l in versions)
    fig.suptitle(f"[{tag}] {mode} {n}³ — events vs np  (large ○ dashed = before, small ■ solid = after, dotted = ideal)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

def plot_bars_section(cases, label):
    for hw in ("h200", "v100", "a100"):
        for mode in ("strong", "weak"):
            for n in (256, 512):
                plot_event_bars(cases, label, mode, hw, n)
    plt.show()

def plot_scaling_section(versions):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            plot_event_scaling(versions, mode, n)
    plt.show()
print("plot helpers ready")''')

    md("## A. 최적화 전 — 이벤트별 막대(np 나란히): 어디가 병목인가")
    code('plot_bars_section(BEFORE, "before")')
    md("## B. 최적화 후(최신 바이너리) — 이벤트별 막대")
    code('plot_bars_section(AFTER, "after")')
    md("""## C. 이벤트별 scaling 곡선 — 최적화 전(○ 파선, 빈 마커)과 후(■ 실선, 채운 마커)를 같은 패널에

색 = 기종, 점선 = 각 선의 이상선(strong $T_{np_0}\,np_0/np$, weak 평평). 전·후가 겹치면 그 이벤트는 바뀌지 않은 것, 벌어지면 그만큼 줄어든 것.""")
    code('plot_scaling_section([(BEFORE, "before"), (AFTER, "after")])')
    md("""## D. 최적화 전 vs 후 — 같은 기종·문제 크기·np 에서 이벤트별 시간과 감소량

왼쪽: 전(회색)·후(파랑) ms/step. 오른쪽: 감소량 = 전 − 후 (ms/step, 양수가 개선), 막대 위 숫자는 감소율 %. 두 바이너리를 같은 노드에서 돈 조합만.""")
    code(r'''def plot_before_after(mode, hw, n, np_):
    kb, ka = (hw, mode, n, np_), (hw, mode, n, np_)
    if kb not in BEFORE or ka not in AFTER:
        return
    b, a = BEFORE[kb], AFTER[ka]
    vb = np.array([event_time(b, e) for e in EVENTS]); va = np.array([event_time(a, e) for e in EVENTS])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    xs = np.arange(len(EVENTS)); w = 0.38
    ax1.bar(xs - w / 2, vb, w, color="#9a9a96", label=f"before [{BEFORE_TAG[kb]}]  step_total {b['step_total']:.1f} ms")
    ax1.bar(xs + w / 2, va, w, color="#2a78d6", label=f"after [{AFTER_TAG[ka]}]  step_total {a['step_total']:.1f} ms")
    ax1.set_xticks(xs); ax1.set_xticklabels(EVENTS, rotation=40, ha="right"); ax1.set_ylabel("ms / step (rank max)")
    ax1.set_title(f"{hw_desc('after', hw)} — {mode} {n}³, np {np_}: before vs after", fontsize=10, loc="left")
    ax1.set_ylim(0, max(vb.max(), va.max()) * 1.12); ax1.legend(frameon=False, fontsize=7.5)
    d = vb - va
    colors = ["#2a78d6" if v >= 0 else "#e34948" for v in d]
    ax2.bar(xs, d, 0.62, color=colors)
    for x, v, bv in zip(xs, d, vb):
        if bv > 0.02 * vb.max():
            ax2.annotate(f"{100 * v / bv:+.0f}%", (x, v), xytext=(0, 2 if v >= 0 else -9), textcoords="offset points", ha="center", fontsize=6.5, color="#52514e", rotation=90)
    ax2.axhline(0, color="#52514e", lw=0.8)
    ax2.set_xticks(xs); ax2.set_xticklabels(EVENTS, rotation=40, ha="right"); ax2.set_ylabel("reduction = before − after (ms / step)")
    ax2.set_title(f"reduction per event (step_total {b['step_total']:.1f} → {a['step_total']:.1f} ms, {100*(1-a['step_total']/b['step_total']):+.0f}%)", fontsize=10, loc="left")
    fig.tight_layout()

for hw in ("h200", "v100", "a100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            for np_ in (1, 2, 4, 8):
                plot_before_after(mode, hw, n, np_)
plt.show()

# 요약표
print(f"{'hw':5s} {'mode':6s} {'n':4s} {'np':3s} | {'step before':>11s} {'after':>8s} {'gain':>6s} | {TOTAL:>15s} {'after':>8s} {'gain':>6s} | 가장 많이 줄어든 이벤트")
for hw in ("h200", "v100", "a100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            for np_ in (1, 2, 4, 8):
                k = (hw, mode, n, np_)
                if k not in BEFORE or k not in AFTER:
                    continue
                b, a = BEFORE[k], AFTER[k]
                d = sorted(((event_time(b, e) - event_time(a, e), e) for e in EVENTS if e != TOTAL), reverse=True)[:3]
                tb, ta = event_time(b, TOTAL), event_time(a, TOTAL)
                print(f"{hw:5s} {mode:6s} {n:<4d} {np_:<3d} | {b['step_total']:11.1f} {a['step_total']:8.1f} {100*(1-a['step_total']/b['step_total']):5.0f}% | "
                      f"{tb:15.2f} {ta:8.2f} {100*(1-ta/tb) if tb>0 else 0:5.0f}% | " + ", ".join(f"{e} −{v:.2f}" for v, e in d))''')
    if S.get("breakdown"):
        title, note, tags, pfx = S["breakdown"]
        md(f"## {title}\n\n{note}")
        code(f"""BD, BD_TAG = load({tags!r}); describe("breakdown", BD, BD_TAG)
PFX = {pfx!r}
for hw in ("h200", "v100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            keys = sorted(k for k in BD if k[:3] == (hw, mode, n))
            if not keys:
                continue
            names = sorted({{e for k in keys for e in BD[k] if e.startswith(PFX)}}, key=lambda e: -BD[keys[0]].get(e, 0))
            fig, ax = plt.subplots(figsize=(12.5, 4.0)); xs = np.arange(len(names)); w = 0.8 / len(keys); vmax = 0
            for j, k in enumerate(keys):
                vals = np.array([BD[k].get(e, 0.0) for e in names]); vmax = max(vmax, vals.max())
                tot = BD[k].get(PFX.rstrip('.') + '_solve', 0.0)
                ax.bar(xs - 0.4 + w * (j + 0.5), vals, w * 0.92, color=NP_COLOR.get(k[3], "#164a8a"), label=f"np {{k[3]}}  ({{PFX.rstrip('.')}}_solve {{tot:.2f}} ms, step {{BD[k]['step_total']:.1f}})")
            ax.set_xticks(xs); ax.set_xticklabels([e[len(PFX):] for e in names], rotation=40, ha="right"); ax.set_ylabel("ms / step (rank max)")
            ax.set_title(f"[breakdown] {{hw_desc('breakdown', hw)}} — {{mode}} {{n}}³: {{PFX.rstrip('.')}}_solve stages by np", fontsize=10, loc="left")
            ax.set_ylim(0, vmax * 1.15); ax.legend(frameon=False, ncol=len(keys), loc="upper center", bbox_to_anchor=(0.5, -0.5), fontsize=7.5); plt.tight_layout()
plt.show()
print(f"{{'hw':5s}} {{'mode':6s}} {{'n':4s}} {{'np':3s}} | {{PFX.rstrip('.')}}_solve | 상위 5단계 (ms)")
for k in sorted(BD):
    names = sorted((e for e in BD[k] if e.startswith(PFX)), key=lambda e: -BD[k][e])[:5]
    print(f"{{k[0]:5s}} {{k[1]:6s}} {{k[2]:<4d}} {{k[3]:<3d}} | {{BD[k].get(PFX.rstrip('.') + '_solve', 0):8.2f}} | " + ", ".join(f"{{e[len(PFX):]}} {{BD[k][e]:.2f}}" for e in names))""")
    DEFAULT_COLS = [("halo 계열 합(sync_T+sync_UVW×2+predictor)", ["sync_T", "sync_UVW_1", "sync_UVW_2", "predictor"]), ("momentum_rhs", ["momentum_rhs"])]
    for title, note, before_tags, *rest in S.get("extra", []):
        cols = rest[0] if rest else DEFAULT_COLS   # 요약표 열: (라벨, CSV 이벤트 키 목록 — 합산)
        md(f"## {title}\n\n{note}")
        code(f"""BEFORE2, BEFORE2_TAG = load({before_tags!r})
describe("before2", BEFORE2, BEFORE2_TAG)
STYLE["before2"] = STYLE["before"]
_B, _BT = BEFORE, BEFORE_TAG
BEFORE, BEFORE_TAG = BEFORE2, BEFORE2_TAG            # 아래 그림 함수들은 BEFORE 를 읽는다 — 잠시 바꿔 끼운다
for hw in ("h200", "v100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            plot_event_bars(BEFORE, "before2", mode, hw, n)
plt.show()
plot_scaling_section([(BEFORE, "before2"), (AFTER, "after")])
for hw in ("h200", "v100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            for np_ in (4, 8):
                plot_before_after(mode, hw, n, np_)
plt.show()
COLS = {cols!r}
print(f"{{'hw':5s}} {{'mode':6s}} {{'n':4s}} {{'np':3s}} | {{TOTAL + ' before':>22s}} {{'after':>8s}} {{'gain':>6s}} | " + " | ".join(f"{{lab}} 전 → 후" for lab, _ in COLS))
for hw in ("h200", "v100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            for np_ in (1, 2, 4, 8):
                k = (hw, mode, n, np_)
                if k not in BEFORE or k not in AFTER:
                    continue
                b, a = BEFORE[k], AFTER[k]
                cells_ = " | ".join(f"{{sum(b.get(e, 0) for e in keys):7.2f}} → {{sum(a.get(e, 0) for e in keys):7.2f}} ms" for _, keys in COLS)
                print(f"{{hw:5s}} {{mode:6s}} {{n:<4d}} {{np_:<3d}} | {{b[TOTAL]:22.1f}} {{a[TOTAL]:8.1f}} {{100*(1-a[TOTAL]/b[TOTAL]):5.0f}}% | {{cells_}}")
BEFORE, BEFORE_TAG = _B, _BT""")
    return cells

def make_present_cells(S):
    """처음 vs 최신 두 바이너리만: ① step_total scaling·효율, ② 조합별 이벤트 막대(처음·최신 나란히 + 감소량), ③ 요약표"""
    cells = make_cells(S)[:3]          # 머리말·데이터 적재·그림 도우미는 공용
    md = lambda s: cells.append(("markdown", s))
    code = lambda s: cells.append(("code", s))
    md("""## 1. step_total scaling — 처음(큰 빈 ○, 파선) vs 최신(작은 채운 ■, 실선)

왼쪽 ms/step, 오른쪽 효율. 색 = 기종, 점선 = 처음 선의 이상선(strong $T_{np_0}\\,np_0/np$, weak 평평). 최신은 np 8 까지 있다(V100).""")
    code(PRESENT_CELL_1)
    md("""## 2. 이벤트별 시간 — 처음(회색) vs 최신(파랑), 같은 기종·문제 크기·np

오른쪽은 감소량 = 처음 − 최신 (ms/step), 막대 위 숫자는 감소율 %. 두 바이너리가 모두 있는 조합만(np ≤ 4).""")
    code(PRESENT_CELL_2)
    md("## 3. 요약표 — 처음 → 최신")
    code(PRESENT_CELL_3)
    return cells

PRESENT_CELL_1 = r"""def plot_total(mode, n):
    hws = sorted({k[0] for cases in (BEFORE, AFTER) for k in cases if k[1:3] == (mode, n)})
    if not hws:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.9))
    nps_all = sorted({k[3] for cases in (BEFORE, AFTER) for k in cases if k[1:3] == (mode, n)})
    for hw in hws:
        for cases, label, name in ((BEFORE, "before", "first"), (AFTER, "after", "latest")):
            ks = sorted(k for k in cases if k[:3] == (hw, mode, n))
            if not ks:
                continue
            nps = np.array([k[3] for k in ks]); t = np.array([cases[k]["step_total"] for k in ks])
            st = STYLE[label]; c = HW_COLOR[hw]
            ax1.plot(nps, t, color=c, marker=st["marker"], ls=st["ls"], mfc=st["mfc"] or c, ms=st["ms"], mew=st["mew"], zorder=st["z"],
                     label=f"{name}: {hw_desc(label, hw)}")
            ideal = t[0] * nps[0] / nps if mode == "strong" else np.full_like(t, t[0])
            if label == "before":
                ax1.plot(nps, ideal, color=c, ls=":", lw=0.9, alpha=0.6)
            eff = t[0] / (t * nps / nps[0]) if mode == "strong" else t[0] / t
            ax2.plot(nps, 100 * eff, color=c, marker=st["marker"], ls=st["ls"], mfc=st["mfc"] or c, ms=st["ms"], mew=st["mew"], zorder=st["z"], label=f"{name} [{hw}]")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2); ax.xaxis.set_major_locator(FixedLocator(nps_all)); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([str(v) for v in nps_all]); ax.set_xlabel("np (GPUs, z-slab)")
    ax1.set_yscale("log" if mode == "strong" else "linear"); ax1.set_ylabel("step_total (ms / step)")
    ax1.set_title(f"{mode} {n}³ — step_total vs np: first (○ dashed) vs latest (■ solid), dotted = ideal", fontsize=10, loc="left")
    ax2.set_ylabel("efficiency"); ax2.yaxis.set_major_formatter(PercentFormatter()); ax2.set_ylim(0, 110); ax2.axhline(100, color="#52514e", lw=0.6, ls=":")
    ax2.set_title(f"{mode} {n}³ — efficiency ({'T(np0)/(T(np) np/np0)' if mode == 'strong' else 'T(np0)/T(np)'})", fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=7.2, loc="best"); ax2.legend(frameon=False, fontsize=7.2, loc="lower left")
    fig.tight_layout()

for mode in ("strong", "weak"):
    for n in (256, 512):
        plot_total(mode, n)
plt.show()"""

PRESENT_CELL_2 = r"""def plot_first_latest(mode, hw, n, np_):
    k = (hw, mode, n, np_)
    if k not in BEFORE or k not in AFTER:
        return
    b, a = BEFORE[k], AFTER[k]
    vb = np.array([event_time(b, e) for e in EVENTS]); va = np.array([event_time(a, e) for e in EVENTS])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    xs = np.arange(len(EVENTS)); w = 0.38
    ax1.bar(xs - w / 2, vb, w, color="#9a9a96", label=f"first [{BEFORE_TAG[k]}]  step_total {b['step_total']:.1f} ms")
    ax1.bar(xs + w / 2, va, w, color="#2a78d6", label=f"latest [{AFTER_TAG[k]}]  step_total {a['step_total']:.1f} ms")
    ax1.set_xticks(xs); ax1.set_xticklabels(EVENTS, rotation=40, ha="right"); ax1.set_ylabel("ms / step (rank max)")
    ax1.set_title(f"{hw_desc('after', hw)} — {mode} {n}³, np {np_}: first vs latest", fontsize=10, loc="left")
    ax1.set_ylim(0, max(vb.max(), va.max()) * 1.12); ax1.legend(frameon=False, fontsize=7.5)
    d = vb - va; colors = ["#2a78d6" if v >= 0 else "#e34948" for v in d]
    ax2.bar(xs, d, 0.62, color=colors)
    for x, v, bv in zip(xs, d, vb):
        if bv > 0.02 * vb.max():
            ax2.annotate(f"{100 * v / bv:+.0f}%", (x, v), xytext=(0, 2 if v >= 0 else -9), textcoords="offset points", ha="center", fontsize=6.5, color="#52514e", rotation=90)
    ax2.axhline(0, color="#52514e", lw=0.8)
    ax2.set_xticks(xs); ax2.set_xticklabels(EVENTS, rotation=40, ha="right"); ax2.set_ylabel("reduction = first − latest (ms / step)")
    ax2.set_title(f"reduction per event (step_total {b['step_total']:.1f} → {a['step_total']:.1f} ms, {100*(1-a['step_total']/b['step_total']):+.0f}%)", fontsize=10, loc="left")
    fig.tight_layout()

for hw in ("h200", "v100", "a100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            for np_ in (1, 2, 4, 8):
                plot_first_latest(mode, hw, n, np_)
plt.show()"""

PRESENT_CELL_3 = r"""print(f"{'hw':5s} {'mode':6s} {'n':4s} {'np':3s} | {'first':>8s} {'latest':>8s} {'gain':>6s} {'speedup':>8s} | 가장 많이 줄어든 이벤트 (ms)")
for hw in ("h200", "v100", "a100"):
    for mode in ("strong", "weak"):
        for n in (256, 512):
            for np_ in (1, 2, 4, 8):
                k = (hw, mode, n, np_)
                if k not in BEFORE or k not in AFTER:
                    continue
                b, a = BEFORE[k], AFTER[k]
                d = sorted(((event_time(b, e) - event_time(a, e), e) for e in EVENTS if e != TOTAL), reverse=True)[:3]
                print(f"{hw:5s} {mode:6s} {n:<4d} {np_:<3d} | {b['step_total']:8.1f} {a['step_total']:8.1f} {100*(1-a['step_total']/b['step_total']):5.0f}% {b['step_total']/a['step_total']:7.2f}x | " + ", ".join(f"{e} −{v:.1f}" for v, e in d))
print()
print("최신만 있는 조합(np 8 등):")
for k in sorted(AFTER):
    if k not in BEFORE:
        print(f"  {k[0]:5s} {k[1]:6s} {k[2]}^3 np{k[3]:<2d}  step_total {AFTER[k]['step_total']:7.1f} ms  [{AFTER_TAG[k]}]")"""

def run(S):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = (make_present_cells if S.get("present") else make_cells)(S); ns = {"__name__": "__main__"}
    os.chdir(PLOT_DIR); out_cells = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell["execution_count"] = None; outputs = []; buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(src, "<cell>", "exec"), ns)
            if buf.getvalue():
                outputs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO(); fig.savefig(b, format="png", bbox_inches="tight")
                outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode()}})
            plt.close("all"); cell["outputs"] = outputs
        out_cells.append(cell)
    nb = {"cells": out_cells, "metadata": {"kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(PLOT_DIR, S["nb"]); json.dump(nb, open(path, "w"), ensure_ascii=False, indent=1)
    print("wrote", path, "cells", len(out_cells), "figures", sum(1 for c in out_cells for o in c.get("outputs", []) if o["output_type"] == "display_data"))

which = sys.argv[1] if len(sys.argv) > 1 else "all"
for name, S in SPECS.items():
    if which in ("all", name):
        run(S)
