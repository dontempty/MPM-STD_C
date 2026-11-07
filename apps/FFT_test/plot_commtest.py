#!/usr/bin/env python3
"""통신 방식 시험 결과 → 표와 그림 (plan.md §11).

    /scratch/t525a06/KDH/.venv-plot/bin/python plot_commtest.py

입력  results/commtest/<problem>/np<N>/j<job>/<mode>_r<k>*  (np 마다 가장 최근 잡의 PASS 런)
출력  plot/commtest.png, plot/commtest.csv

통신 시간은 FFTU 의 구간 (재분배 MPI 2 회 + FFTU all-to-all 2 회) 랭크 평균이다. 재분배 MPI 2 회는 MPM-STD 전치와 같은 양을 보낸다.
MPM-STD 평면묶음·펜슬은 항상 MPI_Alltoallv 라 같은 실행의 대조군이다: 방식 사이에 이 값이 변하면 방식이 아니라 시간대 변동이다.
"""
import csv
import pathlib
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
MODES = ("alltoallv", "p2p", "pairwise")
COLOR = {"alltoallv": "#2a78d6", "p2p": "#eb6834", "pairwise": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def rows(path):
    with open(path) as f:
        return list(csv.reader(line for line in f if not line.startswith("#")))[1:]


def load():
    out = {}
    for folder in sorted((HERE / "results" / "commtest").glob("*/np*")):
        jobs = sorted((p for p in folder.glob("j*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))
        if not jobs:
            continue
        problem, np = folder.parent.name, int(folder.name[2:])
        for log in sorted(jobs[-1].glob("*_r*.log")):
            mode = re.match(r"(\w+?)_r\d+$", log.stem).group(1)
            if "PASS FFT_test" not in log.read_text(errors="replace"):
                continue
            times = {r[0]: float(r[1]) for r in rows(log.with_name(log.stem + "_times.csv"))}
            phases = {r[0]: float(r[2]) for r in rows(log.with_name(log.stem + "_fftu_phases.csv"))}
            entry = out.setdefault((problem, np), {"job": jobs[-1].name[1:]}).setdefault(mode, [])
            entry.append({"fftu": times["fftu"], "plane": times.get("mpm_plane", float("nan")), "pencil": times.get("mpm_pencil", float("nan")),
                          "redistribution": phases.get("fftu.block_to_cyclic_mpi", 0) + phases.get("fftu.cyclic_to_block_mpi", 0),
                          "a2a": phases.get("fftu.a2a_fwd", 0) + phases.get("fftu.a2a_bwd", 0)})
    return out


def median(items, key):
    return statistics.median(i[key] for i in items)


def main():
    data = load()
    if not data:
        print("no results")
        return 1
    with open(HERE / "plot" / "commtest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "np", "job", "mode", "repeats", "fftu_total_ms", "redistribution_mpi_ms", "fftu_a2a_ms", "comm_total_ms", "control_plane_ms", "control_pencil_ms"])
        print(f"{'problem':<11}{'np':>4} {'mode':<11}{'reps':>4}{'FFTU':>9}{'redist':>9}{'a2a':>8}{'comm':>8}{'vs a2av':>9}{'plane*':>9}{'pencil*':>9}")
        for (problem, np), d in sorted(data.items()):
            base = None
            for mode in MODES:
                if mode not in d:
                    continue
                items = d[mode]
                red, a2a = median(items, "redistribution"), median(items, "a2a")
                comm = red + a2a
                base = base or comm
                values = (median(items, "fftu"), red, a2a, comm, median(items, "plane"), median(items, "pencil"))
                print(f"{problem:<11}{np:>4} {mode:<11}{len(items):>4}{values[0]:>9.2f}{red:>9.2f}{a2a:>8.2f}{comm:>8.2f}{100 * (comm / base - 1):>8.1f}%{values[4]:>9.2f}{values[5]:>9.2f}")
                w.writerow([problem, np, d["job"], mode, len(items)] + [round(v, 4) for v in values])
    print("* MPM-STD paths always use MPI_Alltoallv: the control for time-of-day drift.")

    problems = sorted({p for p, _ in data})
    fig, axes = plt.subplots(1, len(problems), figsize=(5.6 * len(problems), 4.2), dpi=130, squeeze=False)
    for ax, problem in zip(axes[0], problems):
        nps = sorted(n for p, n in data if p == problem)
        width = 0.26
        top = 0.0
        for i, mode in enumerate(MODES):
            xs, ys = [], []
            for j, n in enumerate(nps):
                if mode in data[(problem, n)]:
                    items = data[(problem, n)][mode]
                    xs.append(j + (i - 1) * width)
                    ys.append(median(items, "redistribution") + median(items, "a2a"))
            ax.bar(xs, ys, width * 0.92, color=COLOR[mode], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=mode)
            for x, y in zip(xs, ys):
                ax.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8, color=INK)
            top = max([top] + ys)
        ax.set_xticks(range(len(nps)))
        ax.set_xticklabels([str(n) for n in nps])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, top * 1.15)
        ax.set_xlabel("GPUs")
        ax.set_title(problem, loc="left")
    axes[0][0].set_ylabel("FFTU MPI time, ms / solve (rank average)")
    axes[0][-1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Same data pattern, three ways to send it: redistribution + FFTU all-to-all", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "plot" / "commtest.png", dpi=200, bbox_inches="tight")
    print(f"figure and table -> {HERE / 'plot'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
