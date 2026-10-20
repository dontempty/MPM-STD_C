#!/usr/bin/env python3
"""채널 펜슬 검증: runs/<case>/checkpoints/*.bin 의 마지막 체크포인트를 기준 케이스와 비교.
사용: compare_checkpoints.py <runs_dir> <reference_case> <case...>
체크포인트 형식 v1 (post/checkpoint.hpp): magic 8 | int32 ver | int32 sizeof | int64 step | double t |
int32 ngx ngy ngz nfields nscalars | nscalars×(name16+double) | nfields×(name16 + double[ngx·ngy·ngz])."""
import glob, os, struct, sys
import numpy as np

def read(path):
    with open(path, "rb") as f:
        f.read(8); ver, rs = struct.unpack("ii", f.read(8)); step, t = struct.unpack("qd", f.read(16))
        ngx, ngy, ngz, nf, ns = struct.unpack("iiiii", f.read(20))
        scal = {}
        for _ in range(ns):
            n = f.read(16).split(b"\0")[0].decode(); scal[n] = struct.unpack("d", f.read(8))[0]
        fields = {}
        for _ in range(nf):
            n = f.read(16).split(b"\0")[0].decode()
            fields[n] = np.frombuffer(f.read(8 * ngx * ngy * ngz), dtype=np.float64).reshape(ngx, ngy, ngz)
    return dict(step=step, t=t, scal=scal, fields=fields)

runs, ref_case, cases = sys.argv[1], sys.argv[2], sys.argv[3:]
def last(case):
    p = sorted(glob.glob(os.path.join(runs, case, "checkpoints", "*.bin")))
    return read(p[-1]) if p else None
ref = last(ref_case)
if ref is None: sys.exit(f"no checkpoint for reference {ref_case}")
print(f"reference {ref_case}: step {ref['step']} t {ref['t']:.10f} mean_dPdx {ref['scal'].get('mean_dPdx', float('nan')):.12e}")
print(f"{'case':>8} {'step':>5} {'dt':>9} {'max|dU|':>10} {'max|dV|':>10} {'max|dW|':>10} {'max|dP|':>10} {'|U|max':>8} {'|P|max':>8}  verdict")
ok_all = True
for c in cases:
    r = last(c)
    if r is None: print(f"{c:>8}  (no checkpoint)"); ok_all = False; continue
    d = {k: np.abs(r["fields"][k] - ref["fields"][k]).max() for k in ("U", "V", "W", "P")}
    big = max(d.values())
    ok = r["step"] == ref["step"] and big < 1e-9
    ok_all &= ok
    print(f"{c:>8} {r['step']:5d} {r['t'] - ref['t']:+9.1e} {d['U']:10.2e} {d['V']:10.2e} {d['W']:10.2e} {d['P']:10.2e} "
          f"{np.abs(r['fields']['U']).max():8.4f} {np.abs(r['fields']['P']).max():8.4f}  {'PASS' if ok else 'FAIL'}")
print("ALL PASS (max |diff| < 1e-9 vs reference)" if ok_all else "SOME CASES FAIL")
