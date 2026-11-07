#!/usr/bin/env python3
"""같은 문제를 푼 MPM-STD(평면묶음·펜슬)·FFTU·CaNS·PaScaL 비교 그림과 표 (plan.md §10).

    /scratch/t525a06/KDH/.venv-plot/bin/python plot_compare.py [problem ...]     (기본 weak256 strong1024)

입력  results/compare/<problem>/np<N>/j<job>/  (np 마다 가장 최근 잡)
출력  plot/compare_<problem>_total.png    풀이 시간과 확장 효율
      plot/compare_<problem>_events.png   MPM-STD·FFTU 의 7 단계 event 별 시간
      plot/compare_<problem>_stack.png    MPM-STD·FFTU 의 시간 구성 (전치·z 풀이·FFT·기타·후처리)
      plot/compare_<problem>.csv          위 그림의 모든 값 + RMS

총시간은 풀이별 rank-max 의 평균을 반복 사이 중앙값으로 잡는다 (세 코드 같은 정의). event·구성은 구간의 랭크 평균이다.
CaNS·PaScaL 은 총시간과 RMS 만 쓴다 (프로파일을 돌리지 않는다, 사용자 결정 2026-11-06).
event 는 MPM-STD·FFTU 의 20 회 풀이, 구간 경계마다 동기화한 값이다.
펜슬은 q = 1 에서 슬랩과 같은 경로라 효율 기준을 평면묶음·슬랩의 첫 점으로 잡는다 (자기 첫 점이면 혼자 좋아 보인다).
"""
import csv
import math
import pathlib
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
METHODS = ("mpm_plane", "mpm_pencil", "fftu", "cans", "pascal")
LABEL = {"mpm_plane": "MPM-STD plane (slab at q = 1)", "mpm_pencil": "MPM-STD pencil", "fftu": "FFTU",
         "cans": "CaNS", "pascal": "PaScaL_POISSON_FFT"}
SHORT = {"mpm_plane": "plane", "mpm_pencil": "pencil", "fftu": "FFTU", "cans": "CaNS", "pascal": "PaScaL"}
# 색은 코드 계열 (3 색까지만 전체 쌍 구분 검증을 통과한다): MPM-STD 파랑, FFTU 주황, cuDecomp 계열 초록. 계열 안은 선 모양·마커로.
COLOR = {"mpm_plane": "#2a78d6", "mpm_pencil": "#2a78d6", "fftu": "#eb6834", "cans": "#1baf7a", "pascal": "#1baf7a"}
MARKER = {"mpm_plane": "s", "mpm_pencil": "D", "fftu": "o", "cans": "^", "pascal": "v"}
STYLE = {"mpm_plane": "-", "mpm_pencil": "--", "fftu": "-", "cans": "-", "pascal": "--"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})

EVENTS = (("layout_in", "layout in (block to x lines)"), ("fftx_fwd", "forward x FFT"),
          ("xy_ffty_fwd", "x-y transpose + forward y FFT"), ("z_solve", "z solve"),
          ("ffty_inv_yx", "inverse y FFT + y-x transpose"), ("fftx_inv_out", "inverse x FFT + layout out"),
          ("post", "mean removal + halo"))
# 시간 구성 (누적 순서). transpose·post 는 통신이 든 항목이라 빗금으로 구분한다.
STACK = (("fft", "FFT", "#2a78d6", False), ("other", "local copy / reorder", "#eb6834", False),
         ("z_solve", "z solve (includes z communication)", "#1baf7a", False),
         ("transpose", "x-y transposes (pack + MPI)", "#eda100", True), ("post", "mean removal + halo", "#e87ba4", True))

STEADY = re.compile(r"\[(?:CANS-)?PERF\]\s+Steady-state average\s+(\S+)\s+(\S+)\s+(\S+)")
RMS = re.compile(r"RMS error against analytic solution:\s+(\S+)")


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


def read_csv_rows(path):
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.reader(lines))


def phase_average(path):
    """phase_timer CSV → {구간: 랭크 평균 ms/solve}. 구간 합이 풀이 시간과 맞는 값은 평균뿐이다 (rank-max 의 합은 다른 랭크를 더한다)."""
    return {row[0]: float(row[2]) for row in read_csv_rows(path)[1:]}


def median_dicts(dicts):
    keys = sorted(set().union(*[d.keys() for d in dicts])) if dicts else []
    return {k: statistics.median(d[k] for d in dicts if k in d) for k in keys}


def mpm_events(method, p):
    """phase_timer 값 → (7 event, 시간 구성). 경로마다 구간 이름이 다르다."""
    g = lambda *names: sum(p.get(n, 0.0) for n in names)
    post = g("pressure.mean_remove", "pressure.halo")
    if method == "mpm_plane":
        if "pressure.pack_lines" in p:   # q = 1: 슬랩 경로, 전치 없음
            return ({"layout_in": g("pressure.pack_lines"), "fftx_fwd": g("pressure.fft_x_fwd"),
                     "xy_ffty_fwd": g("pressure.reorder_x_to_y", "pressure.fft_y_fwd"),
                     "z_solve": g("pressure.pack_bands", "pressure.build_z", "pressure.tdma_z"),
                     "ffty_inv_yx": g("pressure.fft_y_bwd", "pressure.reorder_y_to_x"),
                     "fftx_inv_out": g("pressure.fft_x_bwd", "pressure.scatter_lines"), "post": post},
                    {"fft": g("pressure.fft_x_fwd", "pressure.fft_y_fwd", "pressure.fft_y_bwd", "pressure.fft_x_bwd"),
                     "other": g("pressure.pack_lines", "pressure.scatter_lines", "pressure.reorder_x_to_y", "pressure.reorder_y_to_x"),
                     "z_solve": g("pressure.pack_bands", "pressure.build_z", "pressure.tdma_z"), "transpose": 0.0, "post": post})
        return ({"layout_in": g("pressure.plane.block_to_planes"), "fftx_fwd": g("pressure.fft_x_fwd"),
                 "xy_ffty_fwd": g("pressure.reorder_x_to_y", "pressure.fft_y_fwd"),
                 "z_solve": g("pressure.build_z", "pressure.tdma_z"),
                 "ffty_inv_yx": g("pressure.fft_y_bwd", "pressure.reorder_y_to_x"),
                 "fftx_inv_out": g("pressure.fft_x_bwd", "pressure.plane.planes_to_block"), "post": post},
                {"fft": g("pressure.fft_x_fwd", "pressure.fft_y_fwd", "pressure.fft_y_bwd", "pressure.fft_x_bwd"),
                 "other": g("pressure.reorder_x_to_y", "pressure.reorder_y_to_x"),
                 "z_solve": g("pressure.build_z", "pressure.tdma_z"),
                 "transpose": g("pressure.plane.block_to_planes", "pressure.plane.planes_to_block"), "post": post})
    if method == "mpm_pencil":
        return ({"layout_in": g("pressure.pencil.block_to_lines"), "fftx_fwd": g("pressure.pencil.fft_x_fwd"),
                 "xy_ffty_fwd": g("pressure.pencil.x_to_y", "pressure.pencil.fft_y_fwd"),
                 "z_solve": g("pressure.pencil.build_z", "pressure.pencil.solve_z"),
                 "ffty_inv_yx": g("pressure.pencil.fft_y_bwd", "pressure.pencil.y_to_x"),
                 "fftx_inv_out": g("pressure.pencil.fft_x_bwd", "pressure.pencil.lines_to_block"), "post": post},
                {"fft": g("pressure.pencil.fft_x_fwd", "pressure.pencil.fft_y_fwd", "pressure.pencil.fft_y_bwd", "pressure.pencil.fft_x_bwd"),
                 "other": 0.0, "z_solve": g("pressure.pencil.build_z", "pressure.pencil.solve_z"),
                 "transpose": g("pressure.pencil.block_to_lines", "pressure.pencil.lines_to_block", "pressure.pencil.x_to_y", "pressure.pencil.y_to_x"),
                 "post": post})
    post = g("fftu.mean_remove", "fftu.halo")
    transpose = g("fftu.block_to_cyclic", "fftu.cyclic_to_block", "fftu.twiddle_pack_fwd", "fftu.a2a_fwd", "fftu.unpack_fwd",
                  "fftu.pack_bwd", "fftu.a2a_bwd", "fftu.unpack_twiddle_bwd")
    return ({"layout_in": g("fftu.block_to_cyclic"), "fftx_fwd": g("fftu.fft_x_fwd"),
             "xy_ffty_fwd": g("fftu.reorder_x_to_y", "fftu.fft_y_fwd", "fftu.twiddle_pack_fwd", "fftu.a2a_fwd", "fftu.unpack_fwd", "fftu.fft_small_fwd"),
             "z_solve": g("fftu.build_z", "fftu.tdma_z"),
             "ffty_inv_yx": g("fftu.fft_small_bwd", "fftu.pack_bwd", "fftu.a2a_bwd", "fftu.unpack_twiddle_bwd", "fftu.fft_y_bwd", "fftu.reorder_y_to_x"),
             "fftx_inv_out": g("fftu.fft_x_bwd", "fftu.cyclic_to_block"), "post": post},
            {"fft": g("fftu.fft_x_fwd", "fftu.fft_y_fwd", "fftu.fft_small_fwd", "fftu.fft_small_bwd", "fftu.fft_y_bwd", "fftu.fft_x_bwd"),
             "other": g("fftu.reorder_x_to_y", "fftu.reorder_y_to_x"), "z_solve": g("fftu.build_z", "fftu.tdma_z"),
             "transpose": transpose, "post": post})


def load_np(folder):
    """np 폴더의 가장 최근 잡 → {method: {total, rms, events, stack, reps}}"""
    jobs = sorted((p for p in folder.glob("j*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))
    if not jobs:
        return None
    base = jobs[-1]
    out = {"job": base.name[1:], "methods": {}}
    # MPM-STD·FFTU — PASS 런만
    totals, rms, phases = {}, {}, {}
    placement = ""
    for log in sorted((base / "mpm").glob("r*.log")):
        text = log.read_text(errors="replace")
        if "PASS FFT_test" not in text:
            continue
        m = re.search(r"# placement: (.*)", text)
        placement = m.group(1) if m else placement
        times = read_csv_rows(log.with_name(log.stem + "_times.csv"))
        for row in times[1:]:
            name = "mpm_plane" if row[0] in ("mpm_plane", "mpm_slab") else row[0]
            totals.setdefault(name, []).append(float(row[1]))
            rms.setdefault(name, []).append(float(row[5]))
            ph = log.with_name(f"{log.stem}_{row[0]}_phases.csv")
            if ph.exists():
                phases.setdefault(name, []).append(phase_average(ph))
    for name, values in totals.items():
        events, stack = mpm_events(name, median_dicts(phases.get(name, [])))
        out["methods"][name] = {"total": statistics.median(values), "rms": statistics.median(rms[name]), "events": events,
                                "stack": stack, "reps": len(values)}
    out["placement"] = placement
    # CaNS·PaScaL — 정상 로그만
    for solver in ("cans", "pascal"):
        values, rms_values = [], []
        for log in sorted((base / solver / "perf").glob("r*/solver.log")):
            text = log.read_text(errors="replace")
            m, e = STEADY.findall(text), RMS.findall(text)
            if m and e:
                values.append(1e3 * number(m[-1][2]))
                rms_values.append(number(e[-1]))
        if not values:
            continue
        out["methods"][solver] = {"total": statistics.median(values), "rms": statistics.median(rms_values), "events": {},
                                  "stack": {}, "reps": len(values)}
    return out


def np_axis(ax, nps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(nps)
    ax.set_xticklabels([str(n) for n in nps])
    ax.minorticks_off()
    ax.set_xlim(nps[0] / 1.35, nps[-1] * 2.3)
    ax.tick_params(axis="x", length=0)


def series(data, nps, method, value, need=None):
    """need = "events"·"stack" 이면 그 분해가 있는 점만 (CaNS·PaScaL 은 총시간뿐이다)."""
    xs = [n for n in nps if method in data[n]["methods"] and (need is None or data[n]["methods"][method][need])]
    return xs, [value(data[n]["methods"][method]) for n in xs]


def end_labels(ax, labels, log=False):
    """선 끝 값 라벨 — 가까우면 위아래로 벌린다."""
    labels = sorted(labels)
    placed = []
    lo, hi = ax.get_ylim()
    span = (math.log10(hi) - math.log10(lo)) if log else (hi - lo)
    pos = lambda y: (math.log10(y) - math.log10(lo)) / span if log else (y - lo) / span
    for y, x, text in labels:
        frac = pos(y) if y > 0 else 0
        if placed and frac - placed[-1] < 0.075:
            frac = placed[-1] + 0.075
        placed.append(frac)
        ax.annotate(text, xy=(x, frac), xycoords=("data", "axes fraction"), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)


def legend_handles(methods=METHODS):
    return [Line2D([], [], color=COLOR[m], ls=STYLE[m], marker=MARKER[m], lw=2, ms=6.5, mec=SURFACE, mew=1.4, label=LABEL[m]) for m in methods]


def total_figure(data, nps, problem, out):
    weak = problem.startswith("weak")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=130)
    labels = []
    for m in METHODS:
        xs, ys = series(data, nps, m, lambda v: v["total"])
        if not xs:
            continue
        left.plot(xs, ys, color=COLOR[m], ls=STYLE[m], lw=2, marker=MARKER[m], ms=6.5, mec=SURFACE, mew=1.4, zorder=3)
        labels.append((ys[-1], xs[-1], f"{SHORT[m]} {ys[-1]:.3g}"))
        base_x, base_y = xs[0], ys[0]
        if m == "mpm_pencil":   # 펜슬은 q > 1 부터 있다 — 같은 경로로 도는 평면묶음·슬랩의 첫 점이 기준
            plane_x, plane_y = series(data, nps, "mpm_plane", lambda v: v["total"])
            base_x, base_y = plane_x[0], plane_y[0]
        eff = [100 * base_y / y if weak else 100 * base_y * base_x / (y * x) for x, y in zip(xs, ys)]
        right.plot(xs, eff, color=COLOR[m], ls=STYLE[m], lw=2, marker=MARKER[m], ms=6.5, mec=SURFACE, mew=1.4, zorder=3)
    if not weak:
        left.set_yscale("log")
        left.set_ylim(min(l[0] for l in labels) / 1.6, max(max(series(data, nps, m, lambda v: v["total"])[1] or [0]) for m in METHODS) * 1.5)
    else:
        left.set_ylim(0, max(max(series(data, nps, m, lambda v: v["total"])[1] or [0]) for m in METHODS) * 1.2)
    end_labels(left, labels, log=not weak)
    np_axis(left, nps)
    left.set_xlabel("GPUs")
    left.set_ylabel("ms / solve (rank-max)")
    left.set_title("time per solve", loc="left")
    right.axhline(100, color=GRID, lw=1.2, zorder=1)
    right.set_ylim(0, 115)
    np_axis(right, nps)
    right.set_xlabel("GPUs")
    right.set_ylabel(("weak efficiency  T(first) / T" if weak else "strong efficiency  T(first)·np(first) / (T·np)") + "  [%]")
    right.set_title("scaling efficiency against the first point", loc="left")
    right.legend(handles=legend_handles(), fontsize=8, loc="upper right")
    fig.suptitle(f"{problem}: same problem, same process grid, same allocation", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def events_figure(data, nps, problem, out):
    panels = [("__total__", "total solve")] + list(EVENTS)
    methods = [m for m in METHODS if any(m in data[n]["methods"] and data[n]["methods"][m]["events"] for n in nps)]
    ncols = 4
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.85 * nrows + 0.9), dpi=130, squeeze=False)
    flat = list(axes.flat)
    for i, ax in enumerate(flat):
        if i >= len(panels):
            ax.axis("off")
            continue
        key, title = panels[i]
        top = 0.0
        labels = []
        for m in methods:
            xs, ys = series(data, nps, m, (lambda v: v["total"]) if key == "__total__" else (lambda v, k=key: v["events"][k]), need="events")
            if not xs:
                continue
            ax.plot(xs, ys, color=COLOR[m], ls=STYLE[m], lw=1.8, marker=MARKER[m], ms=5.5, mec=SURFACE, mew=1.2, zorder=3)
            top = max(top, max(ys))
            labels.append((ys[-1], xs[-1], f"{ys[-1]:.3g}"))
        ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        end_labels(ax, labels)
        np_axis(ax, nps)
        ax.set_title(title, loc="left", fontsize=9, pad=5)
        if i % ncols == 0:
            ax.set_ylabel("ms / solve")
        if i + ncols >= len(panels):
            ax.set_xlabel("GPUs")
    fig.legend(handles=legend_handles(methods), loc="upper right", ncol=3, fontsize=8.5, bbox_to_anchor=(0.995, 0.985))
    fig.suptitle(f"{problem}: time per solve by event (MPM-STD and FFTU)", x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.005, "Each panel has its own y-range. Events: rank average, phase pass with a device sync at every boundary. "
             "Total panel: rank-max, pass without phase timers.", fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stack_figure(data, nps, problem, out):
    methods = [m for m in METHODS if any(m in data[n]["methods"] and data[n]["methods"][m]["stack"] for n in nps)]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.6), dpi=130, sharey=True, squeeze=False)
    top = 0.0
    for ax, m in zip(axes[0], methods):
        xs = [n for n in nps if m in data[n]["methods"] and data[n]["methods"][m]["stack"]]
        positions = [nps.index(n) for n in xs]
        bottom = [0.0] * len(xs)
        for key, _, color, comm in STACK:
            ys = [data[n]["methods"][m]["stack"][key] for n in xs]
            ax.bar(positions, ys, 0.68, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, hatch="///" if comm else None)
            bottom = [b + y for b, y in zip(bottom, ys)]
        for p, b in zip(positions, bottom):
            ax.text(p, b, f"{b:.3g}", ha="center", va="bottom", fontsize=7.5, color=INK)
        top = max(top, max(bottom))
        ax.set_xticks(range(len(nps)))
        ax.set_xticklabels([str(n) for n in nps])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("GPUs")
        ax.set_title(SHORT[m], loc="left")
    axes[0][0].set_ylim(0, top * 1.12)
    axes[0][0].set_ylabel("ms / solve (rank average)")
    handles = [Patch(facecolor=color, edgecolor=SURFACE, hatch="///" if comm else None, label=title) for _, title, color, comm in STACK]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{problem}: where MPM-STD and FFTU spend their time", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_table(data, nps, problem, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "np", "job", "method", "repeats", "item", "group", "ms_per_solve_rank_max"])
        for n in nps:
            d = data[n]
            for m, v in d["methods"].items():
                common = [problem, n, d["job"], m, v["reps"]]
                w.writerow(common + ["total", "total", round(v["total"], 4)])
                for key, value in v["events"].items():
                    w.writerow(common + [key, "event", round(value, 4)])
                for key, value in v["stack"].items():
                    w.writerow(common + [key, "stack", round(value, 4)])
                w.writerow(common + ["rms", "accuracy", f"{v['rms']:.10e}"])


def print_table(data, nps, problem):
    print(f"\n{problem}: ms/solve (rank-max), np {nps}")
    print(f"  {'method':<12}" + "".join(f"{n:>10}" for n in nps))
    for m in METHODS:
        row = "".join(f"{data[n]['methods'][m]['total']:>10.2f}" if m in data[n]["methods"] else f"{'-':>10}" for n in nps)
        print(f"  {m:<12}{row}")
    for n in nps:
        have = data[n]["methods"]
        if "cans" not in have or "pascal" not in have:
            print(f"  np {n}: missing CaNS or PaScaL timing")
    print("  RMS against analytic solution, worst relative spread between methods per np:")
    for n in nps:
        values = [v["rms"] for v in data[n]["methods"].values()]
        spread = (max(values) - min(values)) / min(values) if values else math.nan
        print(f"    np {n:<4} rms {values[0]:.6e}  spread {spread:.2e}  jobs {data[n]['job']}  reps {[data[n]['methods'][m]['reps'] for m in data[n]['methods']]}")
    print("  placement:", {n: data[n]["placement"] for n in nps})


def main():
    problems = sys.argv[1:] or ["weak256", "strong1024"]
    plot = HERE / "plot"
    plot.mkdir(exist_ok=True)
    made = 0
    for problem in problems:
        root = HERE / "results" / "compare" / problem
        data = {}
        for folder in root.glob("np*"):
            d = load_np(folder)
            if d and d["methods"]:
                data[int(folder.name[2:])] = d
        nps = sorted(data)
        if not nps:
            print(f"{problem}: no results")
            continue
        print_table(data, nps, problem)
        write_table(data, nps, problem, plot / f"compare_{problem}.csv")
        if len(nps) >= 2:
            total_figure(data, nps, problem, plot / f"compare_{problem}_total.png")
            events_figure(data, nps, problem, plot / f"compare_{problem}_events.png")
            stack_figure(data, nps, problem, plot / f"compare_{problem}_stack.png")
            made += 1
    print(f"\nfigures and tables -> {plot}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
