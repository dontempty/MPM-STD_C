#!/usr/bin/env python3
"""job_scal.sh 결과 집계 — docs/Scalability/plan.md §5.

  python3 summarize.py [results/] [--study strong_256_slab] [--phases]

  results/<study>/<study>_np<N>_n<노드>x<rpn>_r<k>_j<job>.csv 를 (study, np, 배치) 로 묶어 반복 중
  step_total 중앙값 런을 대표로 쓴다 (plan.md §4.2). 출력:
    ① 효율표 — strong: speedup = T(np0)/T(np), eff = speedup/(np/np0);  weak: eff = T(np0)/T(np).
       comm% = 분산 TDMA MPI 시간(tdma_comm_*) / step_total, untimed% = step_total − 구간 합 (plan.md §2.1: ≤ 2%).
    ② 손실 순위 @최대 np — strong: T_p(np) − T_p(np0)·np0/np, weak: T_p(np) − T_p(np0)  (ms/step).
    ③ strong 결과 불변 — 마지막 모니터 줄(θ_c, Tmin, Tmax, Nu_b, Nu_t, |U|)의 np0 대비 최대 상대차 (plan.md §4.3).
    --phases: 조합마다 구간 표 전체 (ms/step 랭크 max, share, 불균형).
  비용 지표는 전부 랭크 max (가장 늦은 랭크가 스텝을 정한다). 대표 런 전부를 results/summary_phases.csv 로도 쓴다.
"""
import argparse
import csv
import glob
import os
import re
import statistics

NAME = re.compile(r"^((strong|weak)_(\d+)_(slab|pencil))_np(\d+)(?:_n(\d+)x(\d+))?_r(\d+)_j(\w+)\.csv$")


def read_csv(path):
    header, phases = {}, {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                for kv in line[1:].split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        header[k] = v
                continue
            row = line.strip().split(",")
            if not row or row[0] in ("", "phase"):
                continue
            phases[row[0]] = tuple(float(x) for x in row[1:6])  # min avg max share imbalance
    return header, phases


def last_monitor(log_path):
    """마지막 모니터 줄의 (θ_c, Tmin, Tmax, Nu_b, Nu_t, u_bot, u_top) — 없으면 None"""
    last = None
    try:
        for line in open(log_path):
            if re.match(r"^\s+\d+\s", line):
                last = line.split()
    except OSError:
        return None
    if last is None:
        return None
    f = [float(x) for x in last]
    return (f[4], f[6], f[7], f[8], f[9], f[11], f[12])


def top_level(name):
    return name != "step_total" and "." not in name and not name.startswith("tdma_comm_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--study")
    ap.add_argument("--phases", action="store_true")
    a = ap.parse_args()

    runs = {}  # study -> {(np, 배치 "노드x노드당랭크"): [(path, header, phases)]}
    meta = {}
    for path in sorted(glob.glob(os.path.join(a.results, "*", "*.csv"))):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        study, mode, n, decomp, np_ = m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        place = f"{m.group(6)}x{m.group(7)}" if m.group(6) else "?"   # 배치 칸이 없는 옛 이름
        if a.study and study != a.study:
            continue
        header, phases = read_csv(path)
        if "step_total" not in phases:
            continue
        runs.setdefault(study, {}).setdefault((np_, place), []).append((path, header, phases))
        meta[study] = (mode, n, decomp)
    if not runs:
        print(f"결과 없음: {a.results}/*/<study>_np<N>_n<노드>x<rpn>_r<k>_j<job>.csv")
        return

    out_rows = []
    for study in sorted(runs):
        mode, n, decomp = meta[study]
        rep = {}
        for key, lst in sorted(runs[study].items()):
            lst = sorted(lst, key=lambda r: r[2]["step_total"][2])
            totals = [r[2]["step_total"][2] for r in lst]
            spread = (max(totals) - min(totals)) / statistics.median(totals) * 100 if len(totals) > 1 else 0.0
            rep[key] = (lst[(len(lst) - 1) // 2], len(lst), spread)  # 홀수 = 중앙값, 2 회 = 빠른 쪽
        keys = sorted(rep)   # (np, 배치) — 같은 np 의 다른 배치는 다른 행
        np0 = keys[0][0]
        T0 = rep[keys[0]][0][2]["step_total"][2]
        print(f"\n## {study} — 기준 np {np0}  ({mode}, {'전역' if mode == 'strong' else '랭크당'} {n}^3, {decomp})")
        print(f"   {'np':>4s} {'place':>6s} {'decomp':>8s} {'ms/step':>9s} {'speedup':>8s} {'eff%':>6s} {'comm%':>6s} "
              f"{'untimed%':>8s} {'runs':>4s} {'spread%':>7s}")
        for key in keys:
            np_, place = key
            (path, header, ph), nrun, spread = rep[key]
            T = ph["step_total"][2]
            speedup = T0 / T
            eff = speedup / (np_ / np0) if mode == "strong" else speedup
            comm = sum(v[2] for k, v in ph.items() if k.startswith("tdma_comm_")) / T
            untimed = (T - sum(v[2] for k, v in ph.items() if top_level(k))) / T
            print(f"   {np_:4d} {place:>6s} {header.get('decomposition', '?'):>8s} {T:9.3f} {speedup:8.2f} {100 * eff:6.1f} "
                  f"{100 * comm:6.1f} {100 * untimed:8.2f} {nrun:4d} {spread:7.1f}")
            for name, v in ph.items():
                out_rows.append([study, mode, n, decomp, np_, place, header.get("decomposition", ""), name] + list(v))

        base = rep[keys[0]][0][2]
        if len(keys) > 1:
            top, top_place = keys[-1]
            ph = rep[keys[-1]][0][2]
            print(f"   손실 순위 @np {top} ({top_place}) (ms/step; strong: T_p − T_p(np0)·np0/np, weak: T_p − T_p(np0)):")
            rows = []
            for name, v in ph.items():
                if not top_level(name) and not name.startswith("tdma_comm_"):
                    continue
                Tp0 = base.get(name, (0, 0, 0))[2]
                loss = v[2] - (Tp0 * np0 / top if mode == "strong" else Tp0)
                rows.append((loss, name, v[2], 100 * v[3], 100 * v[4]))
            for loss, name, Tp, share, imb in sorted(rows, reverse=True)[:10]:
                print(f"     {name:24s} loss {loss:8.3f}  now {Tp:8.3f}  share {share:5.1f}%  imbal {imb:5.1f}%")

        if mode == "strong" and len(keys) > 1:
            ref = last_monitor(rep[keys[0]][0][0][:-4] + ".log")
            worst = 0.0
            for key in keys[1:]:
                cur = last_monitor(rep[key][0][0][:-4] + ".log")
                if ref is None or cur is None:
                    worst = float("nan")
                    break
                for x, y in zip(ref, cur):
                    worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-300))
            print(f"   결과 불변 (마지막 모니터 줄, np0 대비 최대 상대차, 출력 자릿수 한계 ~1e-5): {worst:.2e}")

        if a.phases:
            for key in keys:
                (path, header, ph), _, _ = rep[key]
                print(f"\n   -- np {key[0]} [{key[1]}] ({header.get('decomposition', '?')}, {header.get('tdma', '?')}, "
                      f"{header.get('steps', '?')} steps)  {os.path.basename(path)}")
                print(f"   {'phase':30s} {'max ms':>9s} {'share%':>7s} {'imbal%':>7s}")
                for name, v in ph.items():
                    print(f"   {name:30s} {v[2]:9.3f} {100 * v[3]:7.2f} {100 * v[4]:7.2f}")

    out = os.path.join(a.results, "summary_phases.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "mode", "n", "decomp", "np", "place", "decomposition", "phase",
                    "min_ms", "avg_ms", "max_ms", "share", "imbalance"])
        w.writerows(out_rows)
    print(f"\n대표 런 구간 표 → {out}")


if __name__ == "__main__":
    main()
