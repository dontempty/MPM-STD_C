#!/usr/bin/env python3
"""cube 분할에서 MPM-STD 압력 경로의 통신 방식 비교 → 그림과 표 (plan.md §12).

    /scratch/t525a06/KDH/.venv-plot/bin/python plot_cube.py

입력  results/cube/weak512/np<N>/j<job>/  (np 마다 가장 최근 잡의 PASS 런)
출력  plot/cube_weak512_modes.png   경로별 패널, 선 = 통신 방식 (Alltoallv 와 pairwise 비교)
      plot/cube_weak512_paths.png   방식별 패널, 선 = 경로 (xy 그룹 평면묶음과 펜슬 비교)
      plot/cube_weak512_stack.png   행 = 경로, 열 = 방식. 계산과 통신을 나눈 시간 구성
      plot/cube_weak512.csv         위 그림의 모든 값 + RMS

총시간은 풀이별 rank-max 의 평균, 반복 중앙값. 몇몇 풀이가 튀는 잡음이 평균을 흔들어 (np 64 에서 7 ms) 가장 빠른 풀이도 함께 그린다.
구성은 단계 패스의 랭크 평균이다. 구성 그림의 검은 눈금이 타이머 없는 패스의 랭크 평균이고,
막대 끝과 벌어지면 그 런의 구성은 그만큼 부풀어 있다. 스트레칭 z 라 해석 RMS 가 없어 같은 np 안의 RMS 가 서로 같은지로 대조한다.
"""
import csv
import pathlib
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_transport import INK, STACK, SURFACE, end_labels, np_axis, rows

HERE = pathlib.Path(__file__).resolve().parent
MODES = ("alltoallv", "pairwise", "pairwise_z")
PATHS = ("mpm_plane", "mpm_pencil")
MODE_LABEL = {"alltoallv": "MPI_Alltoallv", "pairwise": "pairwise: x-y transposes", "pairwise_z": "pairwise: x-y transposes + z TDMA"}
MODE_SHORT = {"alltoallv": "Alltoallv", "pairwise": "pairwise x-y", "pairwise_z": "pairwise x-y+z"}
PATH_LABEL = {"mpm_plane": "plane bundle (xy groups)", "mpm_pencil": "pencil"}
PATH_SHORT = {"mpm_plane": "plane", "mpm_pencil": "pencil"}
MODE_COLOR = {"alltoallv": "#2a78d6", "pairwise": "#eb6834", "pairwise_z": "#1baf7a"}
MODE_MARKER = {"alltoallv": "s", "pairwise": "o", "pairwise_z": "^"}
PATH_COLOR = {"mpm_plane": "#6250d6", "mpm_pencil": "#eda100"}
PATH_MARKER = {"mpm_plane": "s", "mpm_pencil": "D"}
RMS_TOLERANCE = 1e-6   # 같은 np 안에서 경로·방식 사이 RMS 의 상대 차 상한
TIMED_GAP = 0.05       # 구성 막대와 타이머 없는 패스가 이만큼 벌어지면 그 값을 적는다


def stack_of(p, path):
    """구간 (랭크 평균 ms/solve) → 시간 구성. 경로마다 구간 이름이 다르다."""
    g = lambda *names: sum(p.get(n, 0.0) for n in names)
    transpose_mpi, z_mpi = g("comm.transpose"), g("comm.tdma_z")
    post = g("pressure.mean_remove", "pressure.halo")
    if path == "mpm_plane":
        fft = g("pressure.fft_x_fwd", "pressure.fft_y_fwd", "pressure.fft_y_bwd", "pressure.fft_x_bwd")
        transpose_total = g("pressure.plane.block_to_planes", "pressure.plane.planes_to_block", "pressure.reorder_x_to_y", "pressure.reorder_y_to_x")
        z_total = g("pressure.build_z", "pressure.tdma_z")
    else:
        fft = g("pressure.pencil.fft_x_fwd", "pressure.pencil.fft_y_fwd", "pressure.pencil.fft_y_bwd", "pressure.pencil.fft_x_bwd")
        transpose_total = g("pressure.pencil.block_to_lines", "pressure.pencil.lines_to_block", "pressure.pencil.x_to_y", "pressure.pencil.y_to_x")
        z_total = g("pressure.pencil.build_z", "pressure.pencil.solve_z")
    return {"fft": fft, "pack": max(0.0, transpose_total - transpose_mpi), "z_local": max(0.0, z_total - z_mpi),
            "transpose_mpi": transpose_mpi, "z_mpi": z_mpi, "post": post}


def load_np(folder):
    jobs = sorted((p for p in folder.glob("j*") if p.name[1:].isdigit()), key=lambda p: int(p.name[1:]))
    if not jobs:
        return None
    base = jobs[-1]
    out = {"job": base.name[1:], "cells": {}, "failed": {}, "rms": []}
    for mode in MODES:
        logs = sorted(base.glob(f"{mode}_r*.log"))
        passed = [log for log in logs if "PASS FFT_test" in log.read_text(errors="replace")]
        out["failed"][mode] = len(logs) - len(passed)
        for path in PATHS:
            totals, averages, bests, worsts, phase_totals, stacks = [], [], [], [], [], []
            for log in passed:
                times = {r[0]: r for r in rows(log.with_name(log.stem + "_times.csv"))}
                if path not in times:
                    continue
                totals.append(float(times[path][1]))
                averages.append(float(times[path][2]))
                bests.append(float(times[path][3]))
                worsts.append(float(times[path][4]))
                out["rms"].append(float(times[path][5]))
                phases = {r[0]: float(r[2]) for r in rows(log.with_name(f"{log.stem}_{path}_phases.csv"))}
                phase_totals.append(phases["solve_total"])
                stacks.append(stack_of(phases, path))
            if totals:
                out["cells"][path, mode] = {"total": statistics.median(totals), "average": statistics.median(averages),
                                            "best": statistics.median(bests), "worst": max(worsts),
                                            "phase_total": statistics.median(phase_totals), "reps": len(totals),
                                            "stack": {k: statistics.median(s[k] for s in stacks) for k in stacks[0]}}
    return out


def line_panel(ax, data, nps, series, top, metric="total"):
    """series: (라벨, 색, 표식, 선 종류, 셀 키)"""
    labels = []
    for label, color, marker, style, key in series:
        xs = [n for n in nps if key in data[n]["cells"]]
        ys = [data[n]["cells"][key][metric] for n in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=color, ls=style, lw=2, marker=marker, ms=6.5, mec=SURFACE, mew=1.4, zorder=3)
        labels.append((ys[-1], xs[-1], f"{label} {ys[-1]:.3g}"))
    ax.set_ylim(0, top * 1.12)
    end_labels(ax, labels)
    np_axis(ax, nps)
    ax.set_xlabel("GPUs")


def modes_figure(data, nps, top, out):
    fig, grid = plt.subplots(2, 2, figsize=(11.4, 7.8), dpi=130, sharey=True, sharex=True)
    for row, metric, note in zip(grid, ("total", "best"), ("mean of 20 solves", "fastest of 20 solves")):
        for ax, path in zip(row, PATHS):
            line_panel(ax, data, nps, [(MODE_SHORT[m], MODE_COLOR[m], MODE_MARKER[m], "-", (path, m)) for m in MODES], top, metric)
            ax.set_title(f"{PATH_LABEL[path]}: {note}", loc="left")
        row[0].set_ylabel("ms / solve (rank-max)")
    for ax in grid[0]:
        ax.set_xlabel("")
    fig.legend(handles=[Line2D([], [], color=MODE_COLOR[m], marker=MODE_MARKER[m], lw=2, ms=6.5, mec=SURFACE, mew=1.4, label=MODE_LABEL[m]) for m in MODES],
               loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Cube decomposition, 512³ per GPU: communication method in each pressure path", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def paths_figure(data, nps, top, out):
    fig, axes = plt.subplots(1, len(MODES), figsize=(4.4 * len(MODES), 4.6), dpi=130, sharey=True)
    for ax, mode in zip(axes, MODES):
        line_panel(ax, data, nps, [(PATH_SHORT[p], PATH_COLOR[p], PATH_MARKER[p], "-", (p, mode)) for p in PATHS], top)
        ax.set_title(MODE_LABEL[mode], loc="left")
    axes[0].set_ylabel("ms / solve (rank-max)")
    fig.legend(handles=[Line2D([], [], color=PATH_COLOR[p], marker=PATH_MARKER[p], lw=2, ms=6.5, mec=SURFACE, mew=1.4, label=PATH_LABEL[p]) for p in PATHS],
               loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Cube decomposition, 512³ per GPU: xy-group plane bundle against pencil", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stack_figure(data, nps, out):
    fig, grid = plt.subplots(len(PATHS), len(MODES), figsize=(4.2 * len(MODES), 3.5 * len(PATHS)), dpi=130, sharey=True, sharex=True, squeeze=False)
    top = 0.0
    for row, path in zip(grid, PATHS):
        for ax, mode in zip(row, MODES):
            key = path, mode
            xs = [n for n in nps if key in data[n]["cells"]]
            positions = [nps.index(n) for n in xs]
            bottom = [0.0] * len(xs)
            for name, _, color, comm in STACK:
                ys = [data[n]["cells"][key]["stack"][name] for n in xs]
                ax.bar(positions, ys, 0.68, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, hatch="///" if comm else None)
                bottom = [b + y for b, y in zip(bottom, ys)]
            timed = [data[n]["cells"][key]["average"] for n in xs]
            ax.hlines(timed, [p - 0.42 for p in positions], [p + 0.42 for p in positions], color=INK, lw=1.8, zorder=5,
                      path_effects=[path_effects.withStroke(linewidth=3.6, foreground=SURFACE)])
            for p, b, t in zip(positions, bottom, timed):
                ax.text(p, max(b, t), f"{b:.3g}", ha="center", va="bottom", fontsize=8, color=INK)
                if abs(b - t) > TIMED_GAP * t:
                    ax.annotate(f"timed {t:.3g}", xy=(p - 0.42, t), xytext=(-3, 0), textcoords="offset points", ha="right", va="center", fontsize=8, color=INK)
            top = max([top] + bottom + timed)
            ax.set_xticks(range(len(nps)))
            ax.set_xticklabels([str(n) for n in nps])
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{PATH_SHORT[path]}, {MODE_SHORT[mode]}", loc="left")
    grid[0][0].set_ylim(0, top * 1.12)
    for row in grid:
        row[0].set_ylabel("ms / solve (rank average)")
    for ax in grid[-1]:
        ax.set_xlabel("GPUs")
    handles = [Patch(facecolor=c, edgecolor=SURFACE, hatch="///" if comm else None, label=t) for _, t, c, comm in STACK]
    handles.append(Line2D([], [], color=INK, lw=1.8, label="solve time, timed pass without event timers"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Cube decomposition, 512³ per GPU: compute (solid) and communication (hatched)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    data = {}
    for folder in (HERE / "results" / "cube" / "weak512").glob("np*"):
        d = load_np(folder)
        if d and d["cells"]:
            data[int(folder.name[2:])] = d
    nps = sorted(data)
    if not nps:
        print("no results")
        return 1
    (HERE / "plot").mkdir(exist_ok=True)
    print("cube weak 512^3/GPU: ms/solve (rank-max, median of PASS runs); transpose MPI, z MPI (phase pass, rank average)")
    head = "".join(f"{PATH_SHORT[p] + ' ' + MODE_SHORT[m]:>24}" for p in PATHS for m in MODES)
    print(f"  {'GPUs':>4}{head}")
    bad = 0
    with open(HERE / "plot" / "cube_weak512.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["np", "job", "path", "mode", "repeats", "item", "ms_per_solve"])
        for n in nps:
            d = data[n]
            line = ""
            for p in PATHS:
                for m in MODES:
                    c = d["cells"].get((p, m))
                    if c is None:
                        line += f"{'-':>24}"
                        continue
                    line += f"{c['total']:>10.2f} ({c['stack']['transpose_mpi']:5.1f},{c['stack']['z_mpi']:5.1f})"
                    common = [n, d["job"], p, m, c["reps"]]
                    w.writerow(common + ["total_rank_max", round(c["total"], 4)])
                    w.writerow(common + ["total_rank_average", round(c["average"], 4)])
                    w.writerow(common + ["fastest_solve_rank_max", round(c["best"], 4)])
                    w.writerow(common + ["slowest_solve_rank_max", round(c["worst"], 4)])
                    w.writerow(common + ["phase_pass_solve_total", round(c["phase_total"], 4)])
                    for key, value in c["stack"].items():
                        w.writerow(common + [key, round(value, 4)])
            spread = (max(d["rms"]) - min(d["rms"])) / min(d["rms"])
            verdict = "ok" if spread < RMS_TOLERANCE else "RMS MISMATCH"
            bad += verdict != "ok"
            print(f"  {n:>4}{line}   rms {min(d['rms']):.6e} spread {spread:.1e} {verdict}  failed {d['failed']}  j{d['job']}")
            w.writerow([n, d["job"], "all", "all", len(d["rms"]), "rms_min", f"{min(d['rms']):.10e}"])
            w.writerow([n, d["job"], "all", "all", len(d["rms"]), "rms_max", f"{max(d['rms']):.10e}"])
    if len(nps) >= 2:
        top = max(c["total"] for d in data.values() for c in d["cells"].values())
        modes_figure(data, nps, top, HERE / "plot" / "cube_weak512_modes.png")
        paths_figure(data, nps, top, HERE / "plot" / "cube_weak512_paths.png")
        stack_figure(data, nps, HERE / "plot" / "cube_weak512_stack.png")
    print(f"figures and tables -> {HERE / 'plot'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
