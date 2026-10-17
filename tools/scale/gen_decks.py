#!/usr/bin/env python3
"""앱별 확장성 측정 덱 생성: apps/<app>/scale/decks/input_{strong,weak}_256_np{N}.ini
   strong: 256³ 고정, weak: 256×256×(256·np). npz = np, 200 스텝(측정은 MPMSTD_PHASE_START=50 부터), 통계·체크포인트 끔.
   사용: python3 tools/scale/gen_decks.py [app ...]   (기본: dhvc channel boundary_layer thermal_boundary_layer)
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = {  # 앱: (기본 덱, 추가로 바꿀 키 {섹션: {키: 값}})
    'dhvc': ('input_dhvc_1e5.ini', {'output': {'stats_start_time': '1.0e9', 'stats_interval': '0', 'spectra_interval': '0', 'conditional_interval': '0'}}),
    'channel': ('input_channel_detcheck.ini', {'output': {'stats_start_step': '1000000', 'stats_write_interval': '1000000', 'budget_interval': '0'}}),
    'boundary_layer': ('input_bl_blasius_coarse.ini', {'domain': {'ly': '20.0'}, 'output': {'stats_interval': '1.0e9'}}),
    'thermal_boundary_layer': ('input_tbl_laminar_check.ini', {'domain': {'ly': '20.0'}, 'output': {'stats_interval': '1.0e9'}}),
}
COMMON = {'process': {'npx': '1', 'npy': '1'}, 'time': {'max_step': '200', 'end_time': '1.0e9'},
          'output': {'monitor_interval': '50', 'checkpoint_interval': '0', 'stats_dir': '"scale/out_tmp/"'}}

def set_keys(text, updates):
    lines = text.split('\n'); out = []; sec = None; pending = {}
    def flush(sec):
        for k, v in pending.get(sec, {}).items():
            out.append(f'{k} = {v}')
        pending.pop(sec, None)
    todo = {s: dict(kv) for s, kv in updates.items()}
    for l in lines:
        m = re.match(r'^\[(\w+)\]', l)
        if m:
            if sec in todo: out.extend(f'{k} = {v}' for k, v in todo.pop(sec).items())
            sec = m.group(1); out.append(l); continue
        km = re.match(r'^(\w+)\s*=', l)
        if km and sec in todo and km.group(1) in todo[sec]:
            out.append(f'{km.group(1)} = {todo[sec].pop(km.group(1))}'); continue
        out.append(l)
    if sec in todo: out.extend(f'{k} = {v}' for k, v in todo.pop(sec).items())
    for s, kv in todo.items():   # 없는 섹션은 새로
        if kv: out.append(f'[{s}]'); out.extend(f'{k} = {v}' for k, v in kv.items())
    return '\n'.join(out)

def main(apps):
    for app in apps:
        base, extra = BASE[app]
        text = open(os.path.join(ROOT, 'apps', app, base)).read()
        d = os.path.join(ROOT, 'apps', app, 'scale', 'decks'); os.makedirs(d, exist_ok=True)
        for mode in ('strong', 'weak'):
            for np_ in (1, 2, 4, 8):
                nz = 256 if mode == 'strong' else 256 * np_
                upd = {'mesh': {'nx': '256', 'ny': '256', 'nz': str(nz)}, 'process': {'npz': str(np_)}}
                for s, kv in COMMON.items(): upd.setdefault(s, {}).update(kv)
                if not re.search(r'^end_time\s*=', text, re.M): upd['time'].pop('end_time', None)   # channel 은 max_step 만 받는다
                for s, kv in extra.items(): upd.setdefault(s, {}).update(kv)
                upd['process'].update(COMMON['process'])
                hdr = f'# 확장성 측정 덱 (tools/scale/gen_decks.py 생성) — {app} {mode} 256³{"/rank" if mode=="weak" else ""} np {np_}; 기반 {base}\n'
                open(os.path.join(d, f'input_{mode}_256_np{np_}.ini'), 'w').write(hdr + set_keys(text, upd))
        print(app, 'decks written to', d)

if __name__ == '__main__':
    main(sys.argv[1:] or list(BASE))
