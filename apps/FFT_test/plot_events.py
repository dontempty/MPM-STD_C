#!/usr/bin/env python3
"""FFT_test 측정 결과 → event 별 시간·계산/통신 비교 그림과 표.

    /scratch/t525a06/KDH/.venv-plot/bin/python plot_events.py [study]      (기본 weak_512_v2)

입력  results/<study>/<study>_np<N>_r<k>_j<job>_{total,phases}.csv
      np 마다 가장 최근 잡의, 로그가 PASS 인 런만 쓴다 (멈춘 런·실패한 런·이전 측정이 중앙값에 섞이지 않게).
출력  plot/<study>_events_time.png      event 마다 패널 하나: 기존·새 솔버 ms/solve vs GPU
      plot/<study>_compute_comm.png     계산·통신 누적 막대 (기존·새 나란히) + weak 효율
      plot/<study>_event_breakdown.png  GPU 수별 event 누적 막대 (어느 event 가 시간을 쓰는가)
      plot/<study>_events.csv           위 그림의 모든 값 (rank min·avg·max, 분류 포함)

값: 런마다 구간의 랭크 평균을 잡고 반복 사이 중앙값을 쓴다. 전치·TDMA 를 MPI 와 나머지로 나누는 뺄셈이
같은 랭크 안에서 성립하는 값은 평균뿐이기 때문이다 (rank-max 끼리 빼면 다른 랭크의 값을 뺀다).
총시간 패널과 효율만 rank-max 이고, 동기화를 넣지 않은 총시간 패스의 값이다.
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
TAG = re.compile(r"_np(?P<np>\d+)_r(?P<rep>\d+)_j(?P<job>\w+)_(?P<kind>total|phases)\.csv$")

SOLVERS = ("old", "new")
LABEL = {"old": "existing (x-y FFT + PaScaL z)", "new": "FFTU (x r2c + cyclic y FFT + PaScaL z)"}
COLOR = {"old": "#2a78d6", "new": "#eb6834"}
MARKER = {"old": "s", "new": "o"}
KIND_COLOR = {"compute": "#1baf7a", "comm": "#6250d6"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def phase(rows, name):
    """구간 하나의 (min, avg, max) ms/solve. 없으면 0."""
    return rows.get(name, (0.0, 0.0, 0.0))


def add(*values):
    return tuple(sum(v[i] for v in values) for i in range(3))


def minus(total, part):
    """total − part 를 랭크 평균으로만 정의한다 (min·max 는 다른 랭크끼리라 의미가 없어 nan)."""
    return (math.nan, max(0.0, total[1] - part[1]), math.nan)


def events_of(solver, rows):
    """[(event key, title, class, (min, avg, max))] — 없는 event 는 0 으로 둔다 (두 솔버 패널을 맞춘다)."""
    p = lambda *names: add(*[phase(rows, n) for n in names])
    if solver == "old":
        transpose_total = p("pressure.plane.block_to_planes", "pressure.plane.planes_to_block")
        transpose_comm = p("old.comm.transpose")
        tdma_total, tdma_comm = p("pressure.tdma_z"), p("old.comm.tdma_z")
        return [
            ("fft", "FFT (all transforms)", "compute",
             p("pressure.fft_x_fwd", "pressure.fft_y_fwd", "pressure.fft_y_bwd", "pressure.fft_x_bwd")),
            ("pack", "pack / unpack / reorder / twiddle", "compute",
             add(p("pressure.pack_lines", "pressure.scatter_lines", "pressure.reorder_x_to_y",
                   "pressure.reorder_y_to_x", "pressure.pack_bands"), minus(transpose_total, transpose_comm))),
            ("build_z", "z band assembly", "compute", p("pressure.build_z")),
            ("tdma_local", "z TDMA local solve", "compute", minus(tdma_total, tdma_comm)),
            ("redistribution", "block redistribution MPI\n(existing: plane transpose)", "comm", transpose_comm),
            ("fftu_a2a", "FFTU all-to-all (2)", "comm", (0.0, 0.0, 0.0)),
            ("tdma_comm", "z TDMA MPI_Alltoallv", "comm", tdma_comm),
            ("mean", "mean removal (Allreduce)", "comm", p("pressure.mean_remove")),
            ("halo", "halo exchange + wall ghosts", "comm", p("pressure.halo")),
        ]
    redist_total = p("fftu.block_to_cyclic", "fftu.cyclic_to_block")
    redist_comm = p("fftu.block_to_cyclic_mpi", "fftu.cyclic_to_block_mpi")
    tdma_total, tdma_comm = p("fftu.tdma_z"), p("fftu.comm.tdma_z")
    return [
        ("fft", "FFT (all transforms)", "compute",
         p("fftu.fft_x_fwd", "fftu.fft_y_fwd", "fftu.fft_y_bwd", "fftu.fft_x_bwd", "fftu.fft_small_fwd", "fftu.fft_small_bwd",
           "fftu.fft_local_fwd", "fftu.fft_local_bwd")),   # 뒤의 둘은 v1 결과용
        ("pack", "pack / unpack / reorder / twiddle", "compute",
         add(p("fftu.reorder_x_to_y", "fftu.reorder_y_to_x", "fftu.twiddle_pack_fwd", "fftu.unpack_fwd", "fftu.pack_bwd",
               "fftu.unpack_twiddle_bwd"),
             minus(redist_total, redist_comm))),
        ("build_z", "z band assembly", "compute", p("fftu.build_z")),
        ("tdma_local", "z TDMA local solve", "compute", minus(tdma_total, tdma_comm)),
        ("redistribution", "block redistribution MPI\n(existing: plane transpose)", "comm", redist_comm),
        ("fftu_a2a", "FFTU all-to-all (2)", "comm", p("fftu.a2a_fwd", "fftu.a2a_bwd")),
        ("tdma_comm", "z TDMA MPI_Alltoallv", "comm", tdma_comm),
        ("mean", "mean removal (Allreduce)", "comm", p("fftu.mean_remove")),
        ("halo", "halo exchange + wall ghosts", "comm", p("fftu.halo")),
    ]


def read_csv(path):
    rows = {}
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    for row in list(csv.reader(lines))[1:]:
        rows[row[0]] = (float(row[1]), float(row[2]), float(row[3]))
    return rows


def header(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "=" in line and " " not in line[2:].split("=")[0]:
                k, v = line[2:].strip().split("=", 1)
                out[k] = v.split()[0]
    return out


def med(values):
    good = [v for v in values if not math.isnan(v)]
    return statistics.median(good) if good else math.nan


def passed(csv_path, kind):
    log = csv_path.with_name(csv_path.name[: -len(f"_{kind}.csv")] + ".log")
    try:
        return "PASS FFT_test" in log.read_text(errors="replace")
    except OSError:
        return False


def load(study):
    """{np: {"total": {solver: rank-max ms}, "events": {solver: [...]}, "reps": n, "header": {...}}}"""
    found = {}
    for path in (HERE / "results" / study).glob("*.csv"):
        m = TAG.search(path.name)
        if m and passed(path, m["kind"]):
            found.setdefault(int(m["np"]), {}).setdefault(int(m["job"]), {}).setdefault(m["kind"], []).append(path)
    data = {}
    for np, jobs in found.items():
        job = max(jobs)                 # 이 np 의 가장 최근 잡만
        kinds = jobs[job]
        if "total" not in kinds or "phases" not in kinds:
            continue
        total_rows = [read_csv(p) for p in kinds["total"]]
        phase_rows = [read_csv(p) for p in kinds["phases"]]
        if not all("old.comm.tdma_z" in r for r in phase_rows):
            print(f"np {np}: skipped (older run without comm-time columns)")
            continue
        entry = {"total": {}, "events": {}, "reps": len(total_rows), "job": job, "header": header(kinds["total"][0])}
        for s in SOLVERS:
            entry["total"][s] = med([r[f"{s}.solve_total"][2] for r in total_rows])
            per_run = [events_of(s, r) for r in phase_rows]
            merged = []
            for i, (key, title, kind, _) in enumerate(per_run[0]):
                merged.append((key, title, kind, tuple(med([run[i][3][c] for run in per_run]) for c in range(3))))
            entry["events"][s] = merged
        data[np] = entry
    return data


def np_axis(ax, nps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(nps)
    ax.set_xticklabels([str(n) for n in nps])
    ax.minorticks_off()
    ax.set_xlim(nps[0] / 1.35, nps[-1] * 1.9)
    ax.tick_params(axis="x", length=0)


def events_time_figure(data, nps, study, out):
    first = data[nps[0]]["events"]["old"]
    keys = [(e[0], e[1], e[2]) for e in first]
    panels = [("__total__", "total solve (rank-max, no extra sync)", None)] + keys
    ncols = 4
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.85 * nrows + 1.0), dpi=130, squeeze=False)
    flat = list(axes.flat)
    for i, ax in enumerate(flat):
        if i >= len(panels):
            ax.axis("off")
            continue
        key, title, kind = panels[i]
        top = 0.0
        labels = []
        for s in SOLVERS:
            if key == "__total__":
                ys = [data[n]["total"][s] for n in nps]
            else:
                ys = [next(e[3][1] for e in data[n]["events"][s] if e[0] == key) for n in nps]
            if max(ys) <= 0:
                continue   # 그 솔버에 없는 event
            ax.plot(nps, ys, color=COLOR[s], lw=2, marker=MARKER[s], ms=6.5, mec=SURFACE, mew=1.6, zorder=3)
            top = max(top, max(ys))
            labels.append((ys[-1], f"{ys[-1]:.3g}", s))
        ax.set_ylim(0, top * 1.22 if top > 0 else 1)
        labels.sort()
        for j, (y, text, s) in enumerate(labels):
            offset = (j - (len(labels) - 1) / 2) * 11 if len(labels) > 1 and abs(labels[0][0] - labels[-1][0]) < 0.12 * top else 0
            ax.annotate(text, xy=(nps[-1], y), xytext=(7, offset), textcoords="offset points", va="center", fontsize=8, color=INK)
        np_axis(ax, nps)
        tag = "" if kind is None else ("[compute] " if kind == "compute" else "[comm] ")
        ax.set_title(tag + title, loc="left", fontsize=8.8, pad=5)
        if i % ncols == 0:
            ax.set_ylabel("ms / solve")
        if i + ncols >= len(panels):
            ax.set_xlabel("GPUs")
    handles = [Line2D([], [], color=COLOR[s], marker=MARKER[s], lw=2, ms=6.5, mec=SURFACE, mew=1.6, label=LABEL[s]) for s in SOLVERS]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=9, bbox_to_anchor=(0.995, 0.985))
    fig.suptitle(f"{study}: time per solve by event", x=0.01, y=0.985, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.005, "Each panel has its own y-range. Events are rank-average (phase pass, device sync at every boundary), median of repeats; "
             "the total panel is rank-max from the pass without phase timers.", fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_comm_figure(data, nps, study, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.5, 4.6), dpi=130, gridspec_kw={"width_ratios": [3, 1]})
    x = list(range(len(nps)))
    width = 0.36
    totals = {}
    for offset, s in ((-0.2, "old"), (0.2, "new")):
        compute = [sum(e[3][1] for e in data[n]["events"][s] if e[2] == "compute") for n in nps]
        comm = [sum(e[3][1] for e in data[n]["events"][s] if e[2] == "comm") for n in nps]
        positions = [p + offset for p in x]
        left.bar(positions, compute, width, color=KIND_COLOR["compute"], edgecolor=SURFACE, linewidth=2, zorder=3)
        left.bar(positions, comm, width, bottom=compute, color=KIND_COLOR["comm"], edgecolor=SURFACE, linewidth=2, zorder=3)
        totals[s] = [a + b for a, b in zip(compute, comm)]
        for p, a, b in zip(positions, compute, comm):
            left.text(p, a + b, f"{a + b:.0f}\n{100 * b / (a + b):.0f}%" if a + b > 0 else "", ha="center", va="bottom", fontsize=7.5, color=INK)
            left.text(p, -0.012 * max(1, max(totals[s])), "existing" if s == "old" else "FFTU", ha="center", va="top", fontsize=7, color=INK2)
    left.set_xticks(x)
    left.set_xticklabels([f"\n{n}" for n in nps])
    left.tick_params(axis="x", length=0)
    left.grid(axis="x", visible=False)
    left.set_ylim(0, max(max(v) for v in totals.values()) * 1.2)
    left.set_xlabel("GPUs")
    left.set_ylabel("ms / solve")
    left.set_title("compute vs communication (rank-average; labels: ms/solve and communication share)", loc="left")
    left.legend(handles=[Patch(color=KIND_COLOR["comm"], label="communication"), Patch(color=KIND_COLOR["compute"], label="compute")],
                loc="upper left", fontsize=9)

    efficiency = {s: [100 * data[nps[0]]["total"][s] / data[n]["total"][s] for n in nps] for s in SOLVERS}
    for s in SOLVERS:
        ys = efficiency[s]
        right.plot(nps, ys, color=COLOR[s], lw=2, marker=MARKER[s], ms=6.5, mec=SURFACE, mew=1.6, zorder=3,
                   label="existing" if s == "old" else "FFTU")
        higher = ys[-1] >= max(efficiency[o][-1] for o in SOLVERS)   # 끝값이 가까우면 위아래로 벌린다
        right.annotate(f"{ys[-1]:.0f}%", xy=(nps[-1], ys[-1]), xytext=(7, 5 if higher else -5), textcoords="offset points",
                       va="center", fontsize=8.5, color=INK)
    right.axhline(100, color=GRID, lw=1.2, zorder=1)
    right.set_ylim(0, 115)
    np_axis(right, nps)
    right.set_xlabel("GPUs")
    right.set_ylabel(f"weak efficiency  T({nps[0]}) / T  [%]")
    right.set_title("weak efficiency (total, rank-max)", loc="left")
    right.legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"{study}: compute and communication time", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.text(0.01, -0.005, "Communication = MPI time of redistribution, FFTU all-to-all and z TDMA, plus mean removal and the whole halo step (its pack kernels included). "
             "MPI phases include the device copy of the self block.", fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 누적 막대 전용: 팔레트 8 칸을 고정 순서로. event 는 엔티티라 두 솔버에서 같은 색이다.
# mean 과 halo 는 작아서 한 칸으로 합친다 (9 번째 색을 만들지 않는다). 패널·표에서는 따로 보인다.
# 통신은 빗금으로 한 번 더 구분한다 — 색만으로 계산·통신을 가리지 않게.
STACK = (
    ("fft", ("fft",), "#2a78d6", "compute", "FFT (all transforms)"),
    ("pack", ("pack",), "#eb6834", "compute", "pack / unpack / reorder / twiddle"),
    ("build_z", ("build_z",), "#1baf7a", "compute", "z band assembly"),
    ("tdma_local", ("tdma_local",), "#eda100", "compute", "z TDMA local solve"),
    ("redistribution", ("redistribution",), "#e87ba4", "comm", "block redistribution MPI (existing: plane transpose)"),
    ("fftu_a2a", ("fftu_a2a",), "#008300", "comm", "FFTU all-to-all (2)"),
    ("tdma_comm", ("tdma_comm",), "#6250d6", "comm", "z TDMA MPI_Alltoallv"),
    ("mean_halo", ("mean", "halo"), "#e34948", "comm", "mean removal + halo"),
)


def breakdown_figure(data, nps, study, out):
    fig, axes = plt.subplots(1, len(SOLVERS), figsize=(12.6, 4.8), dpi=130, sharey=True)
    x = list(range(len(nps)))
    top = 0.0
    for ax, s in zip(axes, SOLVERS):
        bottom = [0.0] * len(nps)
        for key, members, color, kind, _ in STACK:
            ys = [sum(e[3][1] for e in data[n]["events"][s] if e[0] in members) for n in nps]
            if max(ys) <= 0:
                continue
            ax.bar(x, ys, 0.62, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   hatch="///" if kind == "comm" else None)
            bottom = [b + y for b, y in zip(bottom, ys)]
        for p, b in zip(x, bottom):
            ax.text(p, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        top = max(top, max(bottom))
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in nps])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("GPUs")
        ax.set_title(LABEL[s], loc="left")
    axes[0].set_ylim(0, top * 1.1)
    axes[0].set_ylabel("ms / solve")
    handles = [Patch(facecolor=color, edgecolor=SURFACE, hatch="///" if kind == "comm" else None, label=f"[{kind}] {title}")
               for _, _, color, kind, title in STACK]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle(f"{study}: where each solve spends its time", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_table(data, nps, study, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "np", "process", "q", "old_path", "job", "repeats", "solver", "event", "class", "ms_per_solve", "statistic"])
        for n in nps:
            d = data[n]
            h = d["header"]
            q = h.get("q", f"{h.get('q1')}x{h.get('q2')}")
            common = [study, n, f"{h.get('npx')}x{h.get('npy')}x{h.get('npz')}", q, h.get("old_path"), d["job"], d["reps"]]
            for s in SOLVERS:
                w.writerow(common + [s, "total", "total", round(d["total"][s], 4), "rank_max_no_phase_timers"])
                for key, _, kind, values in d["events"][s]:
                    w.writerow(common + [s, key, kind, round(values[1], 4), "rank_average_phase_pass"])


def print_table(data, nps):
    head = f"  {'event':<16}{'class':<9}" + "".join(f"{n:>10}" for n in nps)
    for s in SOLVERS:
        print(f"\n{LABEL[s]} (ms/solve, rank-average; total = rank-max)")
        print(head)
        print(f"  {'total':<16}{'total':<9}" + "".join(f"{data[n]['total'][s]:>10.2f}" for n in nps))
        for i, (key, _, kind, _) in enumerate(data[nps[0]]["events"][s]):
            print(f"  {key:<16}{kind:<9}" + "".join(f"{data[n]['events'][s][i][3][1]:>10.2f}" for n in nps))
        for cls in ("compute", "comm"):
            print(f"  {'sum ' + cls:<16}{cls:<9}" + "".join(f"{sum(e[3][1] for e in data[n]['events'][s] if e[2] == cls):>10.2f}" for n in nps))


def main():
    study = sys.argv[1] if len(sys.argv) > 1 else "weak_512_v2"
    data = load(study)
    nps = sorted(data)
    if not nps:
        print(f"{study}: no complete runs")
        return 1
    print(f"{study}: np {nps}, PASS repeats {[data[n]['reps'] for n in nps]}, jobs {[data[n]['job'] for n in nps]}")
    print_table(data, nps)
    plot = HERE / "plot"
    plot.mkdir(exist_ok=True)
    write_table(data, nps, study, plot / f"{study}_events.csv")
    events_time_figure(data, nps, study, plot / f"{study}_events_time.png")
    compute_comm_figure(data, nps, study, plot / f"{study}_compute_comm.png")
    breakdown_figure(data, nps, study, plot / f"{study}_event_breakdown.png")
    print(f"\nfigures and table -> {plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
