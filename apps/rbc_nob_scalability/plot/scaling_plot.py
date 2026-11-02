"""plot/strong.ipynb · plot/weak.ipynb 공용 — 결과 읽기와 그림 (TDMA_KISTI6/Scalability/plot/scaling_plot.py 양식).

숫자는 summarize.py 와 같은 계산이다:
    results/<run>/<study>/<study>_np<N>_n<노드>x<rpn>_r<k>_j<job>.csv
    → (np, 배치) 마다 반복 중 step_total 대표 런 (홀수 = 중앙값, 2 회 = 빠른 쪽)
    비용 = 구간별 ms/step 의 랭크 max (가장 늦은 랭크가 스텝을 정한다)
같은 np 를 여러 배치로 돌렸으면 노드를 가장 많이 쓴 배치를 쓴다
(np 2·4 는 노드당 1 랭크로 펼쳐 노드 안 통신을 피했다 — Build_KISTI6.md §3).
그림 안 글자는 영어 (로그인 노드 matplotlib 에 한글 글꼴이 없다). 그림은 plot/figures/ 에 png 로도 저장한다.
"""
import math
import pathlib
import statistics
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator

APP = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
import summarize  # noqa: E402  (NAME, read_csv, top_level)

FIGDIR = APP / "plot" / "figures"

# 참조 팔레트 (dataviz references/palette.md, light) — 고정 순서로만 쓴다.
#   선: study 마다 1·2·3 번 (세 칸은 모든 쌍이 검증됨).  누적 막대: 구간 묶음에 1–6 번을 인접 순서로, 나머지는 회색.
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SURFACE, INK, INK2, GRID, GUIDE, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a96", "#c3c2b7"

# 누적 막대의 구간 묶음 (아래 → 위). 나머지 top-level 구간은 전부 "other".
GROUPS = (
    ("momentum_solve", ("momentum_solve",)),
    ("scalar_solve", ("scalar_solve",)),
    ("pressure_solve", ("pressure_solve",)),
    ("momentum_rhs", ("momentum_rhs",)),
    ("pressure_rhs", ("pressure_rhs",)),
    ("halo sync", ("sync_T", "sync_UVW_1", "sync_UVW_2")),
)
# main 루프 이벤트 (호출 순서 = plan.md §1 표). 이벤트별 scalability 그림이 쓴다.
EVENTS = ("dt_cfl", "dt_viscous", "properties_1", "scalar_solve", "sync_T", "momentum_rhs", "momentum_solve",
          "sync_UVW_1", "properties_2", "pressure_rhs", "pressure_solve", "pressure_correct", "sync_UVW_2",
          "predictor", "clip", "monitor")
SOLVE_MPI = {"scalar_solve": "tdma_comm_scalar_", "momentum_solve": "tdma_comm_momentum_",
             "pressure_solve": "tdma_comm_pressure_"}   # 풀이 구간 안의 분산 TDMA MPI 몫
HALO = GROUPS[-1][1]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


# ---------------------------------------------------------------------------
#  데이터
# ---------------------------------------------------------------------------
def _roots(results):
    roots = [results] if isinstance(results, (str, pathlib.Path)) else list(results)
    return [pathlib.Path(r) if pathlib.Path(r).is_absolute() else APP / r for r in roots]


def _runs(results, study):
    """{(np, nodes, rpn): [(path, header, phases)]} — 결과 폴더 하나 또는 여러 개(리스트)를 합친다."""
    runs = {}
    for root in _roots(results):
        for path in sorted((root / study).glob("*.csv")):
            m = summarize.NAME.match(path.name)
            if not m or m.group(1) != study:
                continue
            header, phases = summarize.read_csv(path)
            if "step_total" not in phases:
                continue
            nodes = int(m.group(6)) if m.group(6) else 0
            rpn = int(m.group(7)) if m.group(7) else 0
            runs.setdefault((int(m.group(5)), nodes, rpn), []).append((path, header, phases))
    return runs


def _row(np_, nodes, rpn, lst):
    lst.sort(key=lambda r: r[2]["step_total"][2])
    path, header, phases = lst[(len(lst) - 1) // 2]
    totals = [r[2]["step_total"][2] for r in lst]
    ph = {k: v[2] for k, v in phases.items()}
    return {
        "np": np_, "nodes": nodes, "rpn": rpn,
        "place": "%dx%d" % (nodes, rpn) if nodes else "?",
        "decomposition": header.get("decomposition", "?"),
        "runs": len(lst),
        "spread": (max(totals) - min(totals)) / statistics.median(totals) * 100 if len(lst) > 1 else 0.0,
        "file": path.name, "phases": ph, "total": ph["step_total"],
        "tdma_mpi": sum(v for k, v in ph.items() if k.startswith("tdma_comm_")),
        "halo": sum(ph.get(k, 0.0) for k in HALO),
    }


def collect(results, study):
    """{np: row} — row 는 대표 런의 구간 표 (phases = {이름: ms/step 랭크 max}) + 배치·반복 정보.
       results = 결과 폴더 하나 또는 여러 개 (리스트). 같은 np 의 배치가 여럿이면 노드를 가장 많이 쓴 것."""
    data = {}
    for (np_, nodes, rpn), lst in _runs(results, study).items():
        row = _row(np_, nodes, rpn, lst)
        cur = data.get(np_)
        if cur is None or nodes > cur["nodes"]:
            data[np_] = row
    if not data:
        raise FileNotFoundError("no runs for %s under %s" % (study, [str(r) for r in _roots(results)]))
    return dict(sorted(data.items()))


def placement_table(results, studies, np_):
    """같은 np 를 배치를 바꿔 잰 비교 — 모든 배치를 보인다 (collect 는 노드를 가장 많이 쓴 것만 쓴다)."""
    head = "%-24s %4s %6s %8s %10s %10s %8s %4s" % ("study", "np", "place", "decomp", "ms/step", "TDMA MPI", "halo", "runs")
    print(head + "\n" + "-" * len(head))
    for study in studies:
        for (n, nodes, rpn), lst in sorted(_runs(results, study).items()):
            if n != np_:
                continue
            r = _row(n, nodes, rpn, lst)
            print("%-24s %4d %6s %8s %10.3f %10.3f %8.3f %4d" % (
                study, n, r["place"], r["decomposition"], r["total"], r["tdma_mpi"], r["halo"], r["runs"]))
    print("\nplace = 노드 x 노드당 랭크.  노드당 1 랭크면 모든 통신자가 노드 사이에 걸친다.")


def metrics(data, mode):
    """기준 np0 = 가장 작은 np.
       strong : S = T0 / T                E = S · np0 / np
       weak   : E = T0 / T                S = E · np / np0   (scaled speedup)"""
    np0 = min(data)
    t0 = data[np0]["total"]
    for np_, r in data.items():
        ratio = t0 / r["total"]
        if mode == "strong":
            r["speedup"], r["eff"] = ratio, ratio * np0 / np_
        else:
            r["speedup"], r["eff"] = ratio * np_ / np0, ratio
        r["np0"] = np0
    return data


def label_of(study):
    mode, n, decomp = study.split("_")
    return "%s³ %s %s" % (n, "global" if mode == "strong" else "per GPU", decomp)


def table(datasets):
    head = "%-24s %4s %6s %8s %10s %8s %6s %10s %8s %4s %7s" % (
        "study", "np", "place", "decomp", "ms/step", "speedup", "eff", "TDMA MPI", "halo", "runs", "spread%")
    print(head + "\n" + "-" * len(head))
    for study, d in datasets.items():
        for np_, r in d.items():
            print("%-24s %4d %6s %8s %10.3f %8.2f %6.3f %10.3f %8.3f %4d %7.1f" % (
                study, np_, r["place"], r["decomposition"], r["total"], r["speedup"], r["eff"],
                r["tdma_mpi"], r["halo"], r["runs"], r["spread"]))
    print("\nms/step = step_total (rank max).  TDMA MPI = tdma_comm_* (분산 TDMA 의 Alltoallv, 풀이 구간 안의 몫),"
          "\nhalo = sync_T + sync_UVW_1 + sync_UVW_2.  place = 노드 x 노드당 랭크.")


# ---------------------------------------------------------------------------
#  그림
# ---------------------------------------------------------------------------
def _save(fig, fname):
    if fname:
        FIGDIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGDIR / ("%s.png" % fname), bbox_inches="tight")


def _places(datasets):
    """{np: [배치, …]} — 같은 np 를 study 마다 다른 배치로 쟀으면 둘 다 (study 순서)."""
    out = {}
    for d in datasets.values():
        for n, r in d.items():
            if r["place"] not in out.setdefault(n, []):
                out[n].append(r["place"])
    return out


def _tick_labels(nps, places):
    return ["%d\n(%s)" % (n, " / ".join(p.replace("x", "×") for p in places.get(n, ["?"]))) for n in nps]


def _name(study, labels, short=False):
    """선 이름: labels 에 있으면 그것, 없으면 label_of (끝 라벨은 짧게 — 격자 크기)."""
    if labels and study in labels:
        return labels[study]
    return label_of(study).split()[0] if short else label_of(study)


def _np_axis(ax, nps, places, right_room=3.2):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(nps))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(_tick_labels(nps, places))
    ax.set_xlabel("GPUs  np   (nodes × ranks per node)")
    ax.set_xlim(nps[0] / 1.35, nps[-1] * right_room)     # 오른쪽은 끝 라벨 자리


def _log_y(ax, base=10):   # 1-2-5 (base 10) 또는 1-2-4-8 (base 2) 눈금을 그냥 숫자로 (6x10^1 대신 60)
    ax.set_yscale("log", base=base)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)) if base == 10 else LogLocator(base=2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _direct_labels(ax, ends, logy, gap=0.065):
    """선 끝에 이름 + 값 (글자는 잉크색, 색은 끝 마커가 맡는다). 가까우면 살짝 띄운다."""
    y0, y1 = ax.get_ylim()
    if logy:
        frac = lambda y: (math.log10(y) - math.log10(y0)) / (math.log10(y1) - math.log10(y0))
    else:
        frac = lambda y: (y - y0) / (y1 - y0)
    items = sorted((frac(y), x, txt) for x, y, txt in ends)
    for i in range(1, len(items)):
        if items[i][0] - items[i - 1][0] < gap:
            items[i] = (items[i - 1][0] + gap, items[i][1], items[i][2])
    over = items[-1][0] - 0.95 if items else 0.0
    if over > 0:
        items = [(fy - over, x, txt) for fy, x, txt in items]
    for fy, x, txt in items:
        ax.annotate(txt, xy=(x, fy), xycoords=("data", "axes fraction"), xytext=(9, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9, color=INK)


def line_chart(datasets, key, *, title, ylabel, logy, ideal=None, ideal_label=None,
               ylim=None, fmt="%.3g", legend_loc="best", log_base=10, labels=None, fname=None):
    """한 지표 vs np: study 마다 선 하나 (dict 순서대로 1·2·3 번 색), 회색 = 이상적.
       ideal(x, d) 는 그 study 의 데이터 d 로 이상선을 준다 (study 마다 다르면 선도 여러 개, 라벨은 한 번).
       labels = {study: 선 이름} (없으면 label_of)."""
    if len(datasets) > len(SLOTS[:3]):
        raise ValueError("line_chart: study 는 3 개까지 (색 구분이 검증된 칸 수)")
    fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=130)
    nps = sorted({n for d in datasets.values() for n in d})
    places = _places(datasets)

    if ideal is not None:
        xs = [nps[0] * 2 ** (i / 8) for i in range(int(8 * math.log2(nps[-1] / nps[0])) + 1)] if len(nps) > 1 else nps
        for i, d in enumerate(datasets.values()):
            ax.plot(xs, [ideal(x, d) for x in xs], color=GUIDE, lw=1.2, zorder=1,
                    label=ideal_label if i == 0 else None)

    ends = []
    for color, (study, d) in zip(SLOTS, datasets.items()):
        xs = list(d)
        ys = [d[n][key] for n in xs]
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", zorder=3, marker="o", ms=7,
                mfc=color, mec=SURFACE, mew=1.6, label=_name(study, labels))
        ends.append((xs[-1], ys[-1], fmt % ys[-1] if len(datasets) == 1
                     else "%s  %s" % (_name(study, labels, short=True), fmt % ys[-1])))

    if logy:
        _log_y(ax, log_base)
    if ylim:
        ax.set_ylim(*ylim)
    _np_axis(ax, nps, places, right_room=2.2 if len(datasets) == 1 else 3.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    # 범례: study 가 둘 이상이면 항상, 하나면 이상선 설명용으로만
    if len(datasets) > 1 or ideal_label:
        ax.legend(loc=legend_loc, fontsize=9)
    _direct_labels(ax, ends, logy)
    fig.tight_layout()
    _save(fig, fname)
    return fig, ax


def phase_groups(row):
    """누적 막대용 [(묶음, ms/step)] — GROUPS 순서 + other (나머지 top-level 구간 합)."""
    ph = row["phases"]
    out, used = [], set()
    for name, members in GROUPS:
        out.append((name, sum(ph.get(m, 0.0) for m in members)))
        used.update(members)
    other = sum(v for k, v in ph.items() if summarize.top_level(k) and k not in used)
    out.append(("other", other))
    return out


def phase_chart(data, *, title, fname=None, bar_px=24):
    """np 마다 한 스텝의 구간 분해 (누적 막대, ms/step 랭크 max). 막대 위 숫자 = step_total.
       other = dt_cfl, dt_viscous, properties_1·2, pressure_correct, predictor, clip, monitor."""
    nps = list(data)
    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=130)
    xs = list(range(len(nps)))
    colors = list(SLOTS) + [NEUTRAL]
    names = [g for g, _ in phase_groups(data[nps[0]])]
    bottoms = [0.0] * len(nps)
    segs = []   # (막대 패치들) — 폭은 그린 뒤 픽셀로 맞춘다
    for gi, name in enumerate(names):
        vals = [dict(phase_groups(data[n]))[name] for n in nps]
        bars = ax.bar(xs, vals, width=0.3, bottom=bottoms, color=colors[gi],
                      edgecolor=SURFACE, linewidth=1.1, zorder=3)   # 1.1 pt ≈ 2 px 표면 틈 @130 dpi
        segs.append(bars)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    top = max(bottoms)
    for x, n in zip(xs, nps):
        ax.annotate("%.1f" % data[n]["total"], xy=(x, data[n]["total"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(nps) - 0.4)
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks(xs)
    ax.set_xticklabels(_tick_labels(nps, {n: [data[n]["place"]] for n in nps}))
    ax.set_xlabel("GPUs  np   (nodes × ranks per node)")
    ax.set_ylabel("time per step [ms]  (rank max)")
    ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    handles = [Patch(color=c, label=n) for c, n in zip(colors, names)][::-1]   # 범례도 위 → 아래
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    # 막대 두께를 bar_px 로 고정 (칸을 채우지 않는다)
    fig.canvas.draw()
    px_per_unit = ax.get_window_extent().width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    w = bar_px / px_per_unit
    for bars in segs:
        for x, p in zip(xs, bars):
            p.set_width(w)
            p.set_x(x - w / 2)
    _save(fig, fname)
    return fig, ax


def comm_chart(datasets, *, title, labels=None, fname=None):
    """통신 두 가지를 나란히 (small multiples, 각자 y 축): 분산 TDMA 의 MPI 시간 | halo 교환 구간."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=130)
    nps = sorted({n for d in datasets.values() for n in d})
    places = _places(datasets)
    for ax, key, sub in zip(axes, ("tdma_mpi", "halo"),
                            ("distributed TDMA, MPI Alltoallv (tdma_comm_*)",
                             "halo exchange (sync_T + sync_UVW_1 + sync_UVW_2)")):
        ends = []
        for color, (study, d) in zip(SLOTS, datasets.items()):
            xs = list(d)
            ys = [d[n][key] for n in xs]
            ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", zorder=3, marker="o", ms=7,
                    mfc=color, mec=SURFACE, mew=1.6, label=_name(study, labels))
            name = "" if len(datasets) == 1 else _name(study, labels, short=True) + "  "
            ends.append((xs[-1], ys[-1], "%s%.2f ms" % (name, ys[-1])))
        ax.set_ylim(0, max(max(d[n][key] for n in d) for d in datasets.values()) * 1.25 or 1)
        _np_axis(ax, nps, places, right_room=2.4 if len(datasets) == 1 else 3.6)
        ax.set_ylabel("time per step [ms]")
        ax.set_title(sub, loc="left", fontsize=10)
        _direct_labels(ax, ends, False)
    if len(datasets) > 1:
        axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, fname)
    return fig, axes


# ---------------------------------------------------------------------------
#  이벤트별 scalability — 어느 이벤트가 확장을 해치는가 (plan.md §5 의 손실 순위를 그림으로)
# ---------------------------------------------------------------------------
def event_loss(data, mode):
    """[(이벤트, loss@np_max, 효율@np_max, T(np0), T(np_max))] — loss 내림차순.
       strong: loss = T(np) − T(np0)·np0/np,  E = T(np0)·np0 / (np·T(np))
       weak  : loss = T(np) − T(np0),          E = T(np0) / T(np)"""
    nps = list(data)
    np0, top = nps[0], nps[-1]
    out = []
    for e in EVENTS:
        t0 = data[np0]["phases"].get(e, 0.0)
        t = data[top]["phases"].get(e, 0.0)
        ideal = t0 * np0 / top if mode == "strong" else t0
        out.append((e, t - ideal, ideal / t if t > 0 else float("nan"), t0, t))
    return sorted(out, key=lambda r: r[1], reverse=True)


def loss_chart(data, mode, *, title, fname=None):
    """np 최대에서 이벤트별 손실 (ms/step) — 막대가 길수록 확장을 해치는 이벤트. 막대 끝 = 손실 · 효율."""
    rows = event_loss(data, mode)
    top = list(data)[-1]
    total_loss = data[top]["total"] - (data[list(data)[0]]["total"] * list(data)[0] / top if mode == "strong"
                                       else data[list(data)[0]]["total"])
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(rows) + 1.3), dpi=130)
    ys = list(range(len(rows)))[::-1]
    vals = [r[1] for r in rows]
    ax.barh(ys, vals, height=0.62, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.1, zorder=3)
    span = max(abs(v) for v in vals) or 1.0
    for y, (e, loss, eff, t0, t) in zip(ys, rows):
        ax.annotate("%.2f ms   E %.2f" % (loss, eff), xy=(max(loss, 0), y), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(min(0, min(vals)) - 0.02 * span, span * 1.45)
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("loss at np = %d  [ms/step]   (%s)" % (
        top, "T(np) − T(%d)·%d/np" % (list(data)[0], list(data)[0]) if mode == "strong"
        else "T(np) − T(%d)" % list(data)[0]))
    ax.set_title(title, loc="left")
    ax.annotate("step total loss %.2f ms/step" % total_loss, xy=(1, 0), xycoords="axes fraction",
                xytext=(0, -34), textcoords="offset points", ha="right", va="top", fontsize=9, color=INK2)
    fig.tight_layout()
    _save(fig, fname)
    return fig, ax


def event_grid(data, mode, *, title, cols=4, fname=None):
    """이벤트마다 작은 그림 하나 (small multiples): 측정 (선) vs 이상 (회색), x = np.
       패널 순서 = np 최대 손실 내림차순 (왼쪽 위가 가장 문제). strong 은 log–log (이상 = 1/np),
       weak 는 y 를 0 부터 (이상 = 평평). 패널 글 = 손실·효율 @np 최대, 풀이 구간은 그 안의 TDMA MPI 도."""
    rows = event_loss(data, mode)
    nps = list(data)
    places = {n: [data[n]["place"]] for n in nps}
    nrow = math.ceil(len(rows) / cols)
    fig, axes = plt.subplots(nrow, cols, figsize=(3.0 * cols, 2.45 * nrow), dpi=110, squeeze=False)
    xs_i = [nps[0] * 2 ** (i / 8) for i in range(int(8 * math.log2(nps[-1] / nps[0])) + 1)]
    for ax, (e, loss, eff, t0, t) in zip(axes.flat, rows):
        ys = [data[n]["phases"].get(e, 0.0) for n in nps]
        ideal = [t0 * nps[0] / x for x in xs_i] if mode == "strong" else [t0] * len(xs_i)
        ax.plot(xs_i, ideal, color=GUIDE, lw=1.1, zorder=1)
        ax.plot(nps, ys, color=SLOTS[0], lw=2, zorder=3, marker="o", ms=5.5, mfc=SLOTS[0], mec=SURFACE, mew=1.2)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(nps))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["%d" % n for n in nps], fontsize=8)
        ax.set_xlim(nps[0] / 1.3, nps[-1] * 1.3)
        if mode == "strong":
            _log_y(ax)   # 1-2-5 눈금 — 범위가 한 자릿수 안이어도 숫자가 둘 이상 찍힌다
            lo = min(min(ys), min(ideal)) / 1.6
            ax.set_ylim(lo, max(max(ys), max(ideal)) * 1.6)
        else:
            ax.set_ylim(0, max(max(ys), t0) * 1.35)
        ax.tick_params(labelsize=8)
        ax.set_title(e, loc="left", fontsize=9.5)
        note = "loss %.2f ms   E %.2f" % (loss, eff)
        if e in SOLVE_MPI:
            mpi = sum(v for k, v in data[nps[-1]]["phases"].items() if k.startswith(SOLVE_MPI[e]))
            note += "\nof which TDMA MPI %.2f" % mpi
        ax.annotate(note, xy=(0.03, 0.04) if mode == "strong" else (0.03, 0.96), xycoords="axes fraction",
                    ha="left", va="bottom" if mode == "strong" else "top", fontsize=7.5, color=INK2)
    for ax in list(axes.flat)[len(rows):]:
        ax.set_visible(False)
    fig.suptitle("%s   (blue = measured, gray = ideal; ms/step, rank max; x = np, %s)" % (
        title, ", ".join("%d: %s" % (n, places[n][0].replace("x", "×")) for n in nps)),
        x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    _save(fig, fname)
    return fig, axes
