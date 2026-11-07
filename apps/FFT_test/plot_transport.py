#!/usr/bin/env python3
"""MPM-STD 펜슬 경로의 전치 통신 방식 비교 → 그림과 표 (plan.md §11).

    /scratch/t525a06/KDH/.venv-plot/bin/python plot_transport.py [problem ...]     (기본 weak256 strong1024)

입력  results/transport/<problem>/np<N>/j<job>/  (np 마다 가장 최근 잡의 PASS 런)
출력  plot/transport_<problem>_total.png   풀이 시간 (MPI_Alltoallv, pairwise, NCCL, PaScaL) 과 전치 MPI 시간
      plot/transport_<problem>_stack.png   방식별 시간 구성 (계산과 통신을 나눈다)
      plot/transport_<problem>.csv         위 그림의 모든 값 + RMS

총시간은 풀이별 rank-max 의 평균, 반복 중앙값 (PaScaL 과 같은 정의). 구성은 단계 패스의 랭크 평균이다 (구간 합이 맞는 값).
단계 패스는 총시간과 따로 돈다. 구성 그림의 검은 눈금이 타이머 없는 패스의 랭크 평균이고, 막대 끝과 벌어지면 그 런의 구성은 그만큼 부풀어 있다.
PaScaL 은 총시간만 있다 (프로파일을 돌리지 않는다). PaScaL 의 전치는 cuDecomp mpi_p2p_pl 이다.
NCCL 런은 aws-ofi-nccl 모듈의 FI_CXI_* 설정이 같은 런의 MPI (z TDMA, halo) 에도 걸린다.
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
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
MODES = ("alltoallv", "pairwise", "nccl")
LABEL = {"alltoallv": "MPM-STD pencil, MPI_Alltoallv", "pairwise": "MPM-STD pencil, pairwise", "nccl": "MPM-STD pencil, NCCL",
         "pascal": "PaScaL (cuDecomp mpi_p2p_pl)"}
SHORT = {"alltoallv": "Alltoallv", "pairwise": "pairwise", "nccl": "NCCL", "pascal": "PaScaL"}
# 전체 쌍 구분 검증을 통과하는 3 색 + 기준선은 무채색 점선
COLOR = {"alltoallv": "#2a78d6", "pairwise": "#eb6834", "nccl": "#1baf7a", "pascal": "#52514e"}
MARKER = {"alltoallv": "s", "pairwise": "o", "nccl": "^", "pascal": "v"}
STYLE = {"alltoallv": "-", "pairwise": "-", "nccl": "-", "pascal": "--"}
STACK = (("fft", "FFT", "#2a78d6", False), ("pack", "pack / unpack", "#eb6834", False), ("z_local", "z local solve", "#1baf7a", False),
         ("transpose_mpi", "x-y transpose communication", "#eda100", True), ("z_mpi", "z TDMA MPI", "#e87ba4", True),
         ("post", "mean removal + halo", "#008300", True))
PHASE_GAP = 1.10   # 단계 패스가 타이머 없는 패스보다 이만큼 길면 그림에 표시한다
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})

STEADY = re.compile(r"\[PERF\]\s+Steady-state average\s+(\S+)\s+(\S+)\s+(\S+)")
RMS = re.compile(r"RMS error against analytic solution:\s+(\S+)")


def number(text):
    return float(text.replace("D", "E").replace("d", "e"))


def rows(path):
    with open(path) as f:
        return list(csv.reader(line for line in f if not line.startswith("#")))[1:]


def stack_of(p):
    """펜슬 경로 구간 (랭크 평균 ms/solve) → 시간 구성"""
    g = lambda *names: sum(p.get(n, 0.0) for n in names)
    transpose_total = g("pressure.pencil.block_to_lines", "pressure.pencil.lines_to_block", "pressure.pencil.x_to_y", "pressure.pencil.y_to_x")
    transpose_mpi = g("comm.transpose")
    z_total = g("pressure.pencil.build_z", "pressure.pencil.solve_z")
    z_mpi = g("comm.tdma_z")
    return {"fft": g("pressure.pencil.fft_x_fwd", "pressure.pencil.fft_y_fwd", "pressure.pencil.fft_y_bwd", "pressure.pencil.fft_x_bwd"),
            "pack": max(0.0, transpose_total - transpose_mpi), "z_local": max(0.0, z_total - z_mpi),
            "transpose_mpi": transpose_mpi, "z_mpi": z_mpi, "post": g("pressure.mean_remove", "pressure.halo")}


def load_np(folder):
    jobs = sorted((p for p in folder.glob("j*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))
    if not jobs:
        return None
    base = jobs[-1]
    out = {"job": base.name[1:], "methods": {}, "failed": {}}
    for mode in MODES:
        totals, averages, phase_totals, rms, stacks = [], [], [], [], []
        logs = sorted(base.glob(f"{mode}_r*.log"))
        for log in logs:
            if log.name.endswith(".no_cxi.log") or "PASS FFT_test" not in log.read_text(errors="replace"):
                continue
            times = {r[0]: r for r in rows(log.with_name(log.stem + "_times.csv"))}
            totals.append(float(times["mpm_pencil"][1]))
            averages.append(float(times["mpm_pencil"][2]))
            rms.append(float(times["mpm_pencil"][5]))
            phases = {r[0]: float(r[2]) for r in rows(log.with_name(log.stem + "_mpm_pencil_phases.csv"))}
            phase_totals.append(phases["solve_total"])
            stacks.append(stack_of(phases))
        out["failed"][mode] = len(logs) - len(totals)
        if totals:
            out["methods"][mode] = {"total": statistics.median(totals), "average": statistics.median(averages),
                                    "phase_total": statistics.median(phase_totals), "rms": statistics.median(rms), "reps": len(totals),
                                    "stack": {k: statistics.median(s[k] for s in stacks) for k in stacks[0]}}
    values, rms = [], []
    for log in sorted(base.glob("pascal/r*/solver.log")):
        text = log.read_text(errors="replace")
        m, e = STEADY.findall(text), RMS.findall(text)
        if m and e:
            values.append(1e3 * number(m[-1][2]))
            rms.append(number(e[-1]))
    if values:
        out["methods"]["pascal"] = {"total": statistics.median(values), "rms": statistics.median(rms), "reps": len(values), "stack": {}}
    return out


def np_axis(ax, nps):
    ax.set_xscale("log", base=2)
    ax.set_xticks(nps)
    ax.set_xticklabels([str(n) for n in nps])
    ax.minorticks_off()
    ax.set_xlim(nps[0] / 1.3, nps[-1] * 2.3)
    ax.tick_params(axis="x", length=0)


def y_axis(ax, top, bottom, log):
    if not log:
        ax.set_ylim(0, top * 1.2)
        return
    lo, hi = bottom / 1.35, top * 1.35
    ax.set_yscale("log", base=2)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1.25, 2.5, 5, 10, 20, 40, 80, 160) if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def axis_fraction(ax, y):
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        return (math.log(y) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (y - lo) / (hi - lo)


def end_labels(ax, labels):
    labels = sorted(labels)
    placed = []
    for y, x, text in labels:
        frac = axis_fraction(ax, y)
        if placed and frac - placed[-1] < 0.07:
            frac = placed[-1] + 0.07
        placed.append(frac)
        ax.annotate(text, xy=(x, frac), xycoords=("data", "axes fraction"), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def total_figure(data, nps, problem, out):
    log = problem.startswith("strong")   # 강확장은 풀이 시간이 몇 배씩 줄어 선형 축이면 큰 np 의 차이가 안 보인다
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.6), dpi=130)
    for ax, members, get, title in ((left, MODES + ("pascal",), lambda v: v["total"], "time per solve (rank-max)"),
                                    (right, MODES, lambda v: v["stack"]["transpose_mpi"], "x-y transpose communication only (phase pass, rank average)")):
        labels = []
        top, bottom = 0.0, math.inf
        for m in members:
            xs = [n for n in nps if m in data[n]["methods"]]
            ys = [get(data[n]["methods"][m]) for n in xs]
            if not xs:
                continue
            ax.plot(xs, ys, color=COLOR[m], ls=STYLE[m], lw=2, marker=MARKER[m], ms=6.5, mec=SURFACE, mew=1.4, zorder=3)
            labels.append((ys[-1], xs[-1], f"{SHORT[m]} {ys[-1]:.3g}"))
            top = max(top, max(ys))
            bottom = min(bottom, min(ys))
        y_axis(ax, top, bottom, log)
        end_labels(ax, labels)
        np_axis(ax, nps)
        if ax is right:
            for m in MODES:
                for n in nps:
                    v = data[n]["methods"].get(m)
                    if v and v["phase_total"] > PHASE_GAP * v["average"]:
                        ax.annotate(f"upper bound: phase pass\nran {100 * (v['phase_total'] / v['average'] - 1):.0f}% long", xy=(n, get(v)),
                                    xytext=(8, 2), textcoords="offset points", fontsize=8, color=INK2, va="bottom")
        ax.set_xlabel("GPUs")
        ax.set_ylabel("ms / solve, log scale" if log else "ms / solve")
        ax.set_title(title, loc="left")
    fig.legend(handles=[Line2D([], [], color=COLOR[m], ls=STYLE[m], marker=MARKER[m], lw=2, ms=6.5, mec=SURFACE, mew=1.4, label=LABEL[m]) for m in MODES + ("pascal",)],
               loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{problem}: communication method in the MPM-STD pencil path", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stack_figure(data, nps, problem, out):
    fig, axes = plt.subplots(1, len(MODES), figsize=(4.2 * len(MODES), 4.6), dpi=130, sharey=True)
    top = 0.0
    for ax, m in zip(axes, MODES):
        xs = [n for n in nps if m in data[n]["methods"]]
        positions = [nps.index(n) for n in xs]
        bottom = [0.0] * len(xs)
        for key, _, color, comm in STACK:
            ys = [data[n]["methods"][m]["stack"][key] for n in xs]
            ax.bar(positions, ys, 0.68, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, hatch="///" if comm else None)
            bottom = [b + y for b, y in zip(bottom, ys)]
        timed = [data[n]["methods"][m]["average"] for n in xs]
        ax.hlines(timed, [p - 0.42 for p in positions], [p + 0.42 for p in positions], color=INK, lw=1.8, zorder=5,
                  path_effects=[path_effects.withStroke(linewidth=3.6, foreground=SURFACE)])
        for p, b, t in zip(positions, bottom, timed):
            ax.text(p, max(b, t), f"{b:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
            if b > PHASE_GAP * t:
                ax.annotate(f"timed {t:.3g}", xy=(p + 0.42, t), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
        top = max([top] + bottom + timed)
        ax.set_xticks(range(len(nps)))
        ax.set_xticklabels([str(n) for n in nps])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("GPUs")
        ax.set_title(SHORT[m], loc="left")
    axes[0].set_ylim(0, top * 1.12)
    axes[0].set_ylabel("ms / solve (rank average)")
    handles = [Patch(facecolor=c, edgecolor=SURFACE, hatch="///" if comm else None, label=t) for _, t, c, comm in STACK]
    handles.append(Line2D([], [], color=INK, lw=1.8, label="solve time, timed pass without event timers"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(f"{problem}: compute (solid) and communication (hatched) by method", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    problems = sys.argv[1:] or ["weak256", "strong1024"]
    (HERE / "plot").mkdir(exist_ok=True)
    made = 0
    for problem in problems:
        data = {}
        for folder in (HERE / "results" / "transport" / problem).glob("np*"):
            d = load_np(folder)
            if d and d["methods"]:
                data[int(folder.name[2:])] = d
        nps = sorted(data)
        if not nps:
            print(f"{problem}: no results")
            continue
        print(f"\n{problem}: ms/solve (rank-max, median of PASS runs)")
        print(f"  {'GPUs':>4}" + "".join(f"{SHORT[m]:>11}" for m in MODES + ("pascal",)) + f"{'x-y MPI A2Av':>14}{'pairwise':>10}{'NCCL':>8}   reps, failed, rms")
        with open(HERE / "plot" / f"transport_{problem}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["problem", "np", "job", "method", "repeats", "item", "ms_per_solve"])
            for n in nps:
                d = data[n]
                cell = lambda m, get: f"{get(d['methods'][m]):.2f}" if m in d["methods"] else "-"
                totals = "".join(f"{cell(m, lambda v: v['total']):>11}" for m in MODES + ("pascal",))
                mpi = [cell(m, lambda v: v["stack"]["transpose_mpi"]) for m in MODES]
                rms = {SHORT[m]: f"{v['rms']:.6e}" for m, v in d["methods"].items()}
                print(f"  {n:>4}{totals}{mpi[0]:>14}{mpi[1]:>10}{mpi[2]:>8}   {[d['methods'][m]['reps'] for m in d['methods']]} {d['failed']} {sorted(set(rms.values()))}")
                for m, v in d["methods"].items():
                    common = [problem, n, d["job"], m, v["reps"]]
                    w.writerow(common + ["total_rank_max", round(v["total"], 4)])
                    if "average" in v:
                        w.writerow(common + ["total_rank_average", round(v["average"], 4)])
                        w.writerow(common + ["phase_pass_solve_total", round(v["phase_total"], 4)])
                    for key, value in v["stack"].items():
                        w.writerow(common + [key, round(value, 4)])
                    w.writerow(common + ["rms", f"{v['rms']:.10e}"])
        if len(nps) >= 2:
            total_figure(data, nps, problem, HERE / "plot" / f"transport_{problem}_total.png")
            stack_figure(data, nps, problem, HERE / "plot" / f"transport_{problem}_stack.png")
            made += 1
    print(f"\nfigures and tables -> {HERE / 'plot'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
