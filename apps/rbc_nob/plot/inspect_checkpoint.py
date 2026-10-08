#!/usr/bin/env python3
"""rbc_nob 체크포인트에서 벽 인접 평면의 |U| |V| |W| 통계·핫스팟 기둥·연속식으로 역산한 윗벽 면 W 를 출력한다
(2026-10-09 윗벽 rms 스파이크 진단; docs/Design/20261009_rbc_nob_dt_criterion.md §6).
사용: python3 inspect_checkpoint.py <checkpoint.bin> [dz_wall=0.00153]
체크포인트 형식 v1 (post/checkpoint.hpp): magic 8 | int32 version | int32 sizeof(real) | int64 step | double t |
int32 ngx ngy ngz | int32 nfields | int32 nscalars | nscalars×(name16+double) | nfields×(name16 + real[ngx·ngy·ngz]) — (i·ngy+j)·ngz+k.
MAC: U 는 x 면(i−½), V 는 y 면, W 는 z 면(셀 k 아래 면; 윗벽 면은 ghost 슬롯이라 저장 안 됨)."""
import struct
import sys

import numpy as np

path = sys.argv[1]
dz_wall = float(sys.argv[2]) if len(sys.argv) > 2 else 0.00153

with open(path, 'rb') as f:
    f.read(8)
    version, real_size = struct.unpack('ii', f.read(8))
    step, t = struct.unpack('qd', f.read(16))
    ngx, ngy, ngz, nf, ns = struct.unpack('iiiii', f.read(20))
    scalars = {}
    for _ in range(ns):
        name = f.read(16).split(b'\0')[0].decode()
        scalars[name] = struct.unpack('d', f.read(8))[0]
    off = f.tell()
n = ngx * ngy * ngz
mm = np.memmap(path, dtype=np.uint8, mode='r')
fields = {}
for q in range(nf):
    name = bytes(mm[off:off + 16]).split(b'\0')[0].decode()
    off += 16
    fields[name] = np.ndarray((ngx, ngy, ngz), dtype=np.float64, buffer=mm, offset=off)
    off += n * 8
print(f"{path}: step {step} t {t:.2f} grid {ngx}x{ngy}x{ngz} fields {list(fields)} scalars {scalars}")
U, V, W, T = fields['U'], fields['V'], fields['W'], fields['T']
dx = 6.0 / ngx  # 수평 격자 (6×6 박스 전제; 다른 박스면 인자화)

for name, F in (('U', U), ('V', V), ('W', W)):
    for label, k in (('bottom k=0', 0), ('bottom k=2', 2), ('top k=nz-1', ngz - 1), ('top k=nz-3', ngz - 3)):
        plane = F[:, :, k]
        a = np.abs(plane)
        rms = np.sqrt((plane ** 2).mean())
        i, j = np.unravel_index(a.argmax(), a.shape)
        print(f"  {name} {label:12s} rms {rms:9.3e} max {a.max():9.3e} at (i={i},j={j})  "
              f"cells>10rms: {(a > 10 * rms).sum():5d}  cells>0.05: {(a > 0.05).sum():5d}  median {np.median(a):9.3e}")

# ── 윗벽 핫스팟 기둥 ──
a = np.abs(U[:, :, ngz - 1])
i, j = np.unravel_index(a.argmax(), a.shape)
print(f"\n U top-plane hotspot at (i={i}, j={j}); signed values k = nz-8..nz-1:")
for name, G in (('U', U), ('V', V), ('W', W), ('T', T)):
    print(f"   {name}: ", np.array2string(G[i, j, -8:], precision=4, floatmode='fixed'))
print(f"   same column |U| bottom k=0..7: ", np.array2string(np.abs(U[i, j, :8]), precision=4, floatmode='fixed'))
ii = [(i + d) % ngx for d in range(-2, 3)]
jj = [(j + d) % ngy for d in range(-2, 3)]
print(" |U| top plane 5x5 neighbourhood (periodic):")
print(np.array2string(np.abs(U[np.ix_(ii, jj, [ngz - 1])])[:, :, 0], precision=3))
mask = a > 0.05
if mask.any():
    I, J = np.nonzero(mask)
    print(f" cells |U|>0.05 in top plane: {mask.sum()}, i range {I.min()}..{I.max()}, j range {J.min()}..{J.max()}")
k_max = np.abs(U).reshape(-1, ngz).max(axis=0)
print(" max|U| per k (top 8):", np.array2string(k_max[-8:], precision=3), " bulk max (8 cells off walls):", f"{k_max[8:-8].max():.3e}")


# ── 연속식으로 윗벽 면 W 역산: (U[i+1]−U[i])/dx + (V[j+1]−V[j])/dy + (W[k+1]−W[k])/dz = 0 ──
def horizontal_divergence(k):
    return ((np.roll(U[:, :, k], -1, 0) - U[:, :, k]) / dx
            + (np.roll(V[:, :, k], -1, 1) - V[:, :, k]) / dx)


d0 = horizontal_divergence(0) + (W[:, :, 1] - W[:, :, 0]) / dz_wall
print(f"\n bottom cell k=0 divergence with stored faces: max {np.abs(d0).max():.2e}  (checks dz_wall/indexing)")
w_top = W[:, :, -1] - dz_wall * horizontal_divergence(ngz - 1)
w_last = W[:, :, -1]
print(f" top cell: implied wall-face W rms {np.sqrt((w_top ** 2).mean()):.3e}, max {np.abs(w_top).max():.3e}"
      f"  (0 이면 벽면 투과 없음);  W[nz-1] rms {np.sqrt((w_last ** 2).mean()):.3e}")
print(f" first interior face rms: bottom W[1] {np.sqrt((W[:, :, 1] ** 2).mean()):.3e}  top W[nz-1] {np.sqrt((w_last ** 2).mean()):.3e}"
      f"  (ratio {np.sqrt((w_last ** 2).mean()) / np.sqrt((W[:, :, 1] ** 2).mean()):.0f})")
