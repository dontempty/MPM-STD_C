#!/usr/bin/env python3
"""phase_timing_*.csv 집계 — docs/Scalability/plan.md §5.
   python3 summarize.py [out/]  → 조합별 반복 중앙값(step_total 기준 런)의 구간 ms/step 표, strong/weak 효율, 손실 순위.
"""
import csv, glob, os, re, statistics, sys
root = sys.argv[1] if len(sys.argv) > 1 else "out"
tag_filter = sys.argv[2] if len(sys.argv) > 2 else None   # 예: h200m — 그 하드웨어 꼬리표만
runs = {}  # (hw, mode, n, np) -> list of dict(phase -> (min,avg,max,share,imb))
for path in sorted(glob.glob(os.path.join(root, "scal_*", "phase_timing_*.csv"))):
    m = re.search(r"scal_(strong|weak)_(\d+)_np(\d+)/phase_timing_([a-z0-9]+)_", path)
    if not m: continue
    if tag_filter and m.group(4) != tag_filter: continue
    key = (m.group(4), m.group(1), int(m.group(2)), int(m.group(3)))  # (hw, mode, n, np)
    phases = {}
    with open(path) as f:
        for row in csv.reader(l for l in f if not l.startswith("#")):
            if row[0] == "phase": continue
            phases[row[0]] = tuple(float(x) for x in row[1:])
    runs.setdefault(key, []).append((path, phases))

def median_run(lst):
    lst = sorted(lst, key=lambda r: r[1]["step_total"][2])
    return lst[len(lst) // 2]

by = {}
for key, lst in sorted(runs.items()):
    path, ph = median_run(lst)
    by[key] = ph
    totals = [r[1]["step_total"][2] for r in lst]
    spread = (max(totals) - min(totals)) / statistics.median(totals) * 100 if len(totals) > 1 else 0
    print(f"\n== [{key[0]}] {key[1]} {key[2]}^3 np={key[3]}  (runs {len(lst)}, step_total max {ph['step_total'][2]:.2f} ms/step, spread {spread:.1f}%)  {os.path.basename(path)}")
    print(f"   {'phase':24s} {'max ms':>9s} {'share%':>7s} {'imbal%':>7s}")
    for name, v in ph.items():
        print(f"   {name:36s} {v[2]:9.3f} {100*v[3]:7.2f} {100*v[4]:7.2f}")

# 효율 + 손실 순위
for hw in sorted({k[0] for k in by}):
  for mode in ("strong", "weak"):
    for n in sorted({k[2] for k in by if k[0] == hw and k[1] == mode}):
        keys = sorted(k for k in by if k[0] == hw and k[1] == mode and k[2] == n)
        base = by[keys[0]]; np0 = keys[0][3]
        print(f"\n## [{hw}] {mode} {n}^3 — 기준 np {np0}: step_total {base['step_total'][2]:.2f} ms/step")
        print(f"   {'np':>4s} {'ms/step':>9s} {'speedup':>8s} {'eff%':>6s}")
        for k in keys:
            T = by[k]["step_total"][2]; T0 = base["step_total"][2]
            if mode == "strong":
                sp = T0 / T; eff = sp / (k[3] / np0)
            else:
                sp = T0 / T; eff = sp
            print(f"   {k[3]:4d} {T:9.2f} {sp:8.2f} {100*eff:6.1f}")
        top = keys[-1]
        print(f"   손실 순위 @np {top[3]} (loss = T_p(np) − T_p(np0)/(np/np0) [strong] / T_p(np) − T_p(np0) [weak], ms/step):")
        rows = []
        parents = {n.split(".")[0] for n in by[top] if "." in n}   # 하위 구간이 있는 부모는 순위에서 뺀다 (중복)
        if "momentum" in parents: parents.add("momentum_solve")      # momentum.* 의 부모 구간 이름은 momentum_solve
        for name in by[top]:
            if name == "step_total" or name not in base or name in parents: continue
            Tp = by[top][name][2]; Tp0 = base[name][2]
            loss = Tp - (Tp0 / (top[3] / np0) if mode == "strong" else Tp0)
            rows.append((loss, name, Tp, 100 * by[top][name][3]))
        for loss, name, Tp, share in sorted(rows, reverse=True)[:12]:
            print(f"     {name:36s} loss {loss:8.3f}  now {Tp:8.3f}  share {share:5.1f}%")
