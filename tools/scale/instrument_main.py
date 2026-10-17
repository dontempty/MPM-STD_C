#!/usr/bin/env python3
"""apps/<app>/main.cpp → apps/<app>/main_scale.cpp : 시간 측정용 쌍둥이 main.
   - 전체를 #ifdef MPMSTD_SCALE_MAIN 으로 감싼다 (일반 빌드에서는 빈 TU; make <app>_scale 가 켠다).
   - 시간 루프의 호출 단위 문장을 PhaseTimer 구간(timer.begin/end)으로 감싼다. 수치는 불변(동기화만 추가).
   - 환경변수: MPMSTD_PHASE_START(측정 시작 스텝, 기본 50), MPMSTD_PHASE_CSV(CSV 경로, 기본 phase_timing.csv).
   사용: python3 tools/scale/instrument_main.py apps/dhvc/main.cpp [--no-solver-timers] [-o apps/dhvc/main_scale.cpp]
"""
import re, sys, argparse

SECTIONS = [  # (구간 이름, 문장 시작을 찾는 부분 문자열들)
    ("dt_cfl", ["compute_cfl_dt("]),
    ("inflow_outflow", ["update_outflow_convective(", "make_inflow_plane(", "boundary::set_inflow("]),
    ("scalar_solve", ["scalar::solve_constant_property(", "scalar::solve_variable_property("]),
    ("sync_T", ["sync_temperature(", "sync_scalar("]),
    ("les_viscosity", ["calculate_eddy_viscosity_wale(", "calculate_eddy_viscosity_constant_vreman(", "synchronize_viscosity_ratio("]),
    ("momentum_rhs", ["build_constant_property_rhs(", "build_variable_viscosity_rhs(", "build_variable_property_rhs(",
                      "rhs_U.fill(0)", "rhs_V.fill(0)", "rhs_W.fill(0)", "add_advection_diffusion_", "add_buoyancy_x_constant_property(",
                      "add_buoyancy_z_", "add_gradient_constant_property(", "add_gradient_variable_property(", "add_mean_pressure_gradient("]),
    ("trip", ["apply_trip("]),
    ("momentum_solve", ["momentum::solve_variable_property(", "momentum::solve_constant_property("]),
    ("sync_UVW", ["sync_velocity("]),
    ("sync_P", ["sync_pressure("]),
    ("bulk_correct", ["correct_bulk_velocity("]),
    ("outflow_flux", ["correct_outflow_flux("]),
    ("pressure_rhs", ["rhs_P.fill(0)", "pressure::add_divergence("]),
    ("pressure_solve", ["pressure::solve("]),
    ("pressure_correct", ["correct_velocity_constant_property(", "correct_velocity_variable_property(", "pressure::accumulate("]),
    ("statistics", ["statistics.accumulate(", "stats.sample(", "->sample("]),
    ("monitor", ["max_divergence("]),
]
DECL_RE = re.compile(r'^\s*(const\s+)?(real_t|auto|double|int|long|bool)\b')
STR_RE = re.compile(r'"(\\.|[^"\\])*"|\'(\\.|[^\'\\])\'')

def code_of(line):
    """문자열·문자 리터럴과 // 주석을 뺀 코드 부분 (괄호·중괄호 세기용)"""
    return STR_RE.sub('""', line).split('//')[0]

def split_comment(line):
    """(코드, 꼬리 주석) — 문자열 안의 // 는 주석이 아니다"""
    stripped = STR_RE.sub(lambda m: 'X' * len(m.group(0)), line)
    k = stripped.find('//')
    return (line, '') if k < 0 else (line[:k].rstrip(), '  ' + line[k:])

def find_statement_end(lines, i):
    """lines[i] 에서 시작하는 문장이 끝나는 줄 index (괄호 깊이 0 에서 ';')"""
    depth = 0
    for j in range(i, len(lines)):
        code = code_of(lines[j])
        for ch in code:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            elif ch == ';' and depth == 0:
                return j
    raise RuntimeError(f"statement end not found from line {i}: {lines[i]}")

def loop_range(lines):
    for i, l in enumerate(lines):
        if re.match(r'^    (while \(t < |for \(long step)', l):
            depth = 0
            for j in range(i, len(lines)):
                code = code_of(lines[j])
                depth += code.count('{') - code.count('}')
                if depth == 0 and j > i:
                    return i, j
    raise RuntimeError("time loop not found")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('-o', '--out'); ap.add_argument('--no-solver-timers', action='store_true')
    a = ap.parse_args(); out = a.out or a.src.replace('main.cpp', 'main_scale.cpp')
    lines = open(a.src).read().split('\n')
    has_scalar = any('scalar_solver(' in l for l in lines)
    lo, hi = loop_range(lines)
    # ① 구간 감싸기 (루프 본문, 뒤에서 앞으로 — 인덱스 보존)
    used = set()
    i = hi - 1
    while i > lo:
        l = lines[i]; code = code_of(l)
        name = next((n for n, keys in SECTIONS if any(k in code for k in keys)), None)
        if name and not code.strip().startswith(('timer.', '//')) and ('(' in code or '.fill(0)' in code):
            # 이 줄이 문장의 시작인가: 바로 앞 줄이 문장 중간(;{} 로 안 끝남)이면 건너뜀
            prev = code_of(lines[i-1]).rstrip()
            if prev and not prev.endswith((';', '{', '}', ')')) and prev.strip() != 'else':
                i -= 1; continue
            if prev.endswith(')') and not re.match(r'^\s*(if|else|for|while)\b', lines[i-1]):
                i -= 1; continue
            j = find_statement_end(lines, i)
            indent = re.match(r'^\s*', l).group(0)
            body, tail = split_comment(lines[j])
            if DECL_RE.match(code):
                lines[j] = body + f' timer.end("{name}");' + tail
                lines.insert(i, f'{indent}timer.begin("{name}");')
            else:
                lines[j] = body + f' timer.end("{name}"); }}' + tail
                lines.insert(i, f'{indent}{{ timer.begin("{name}");')
            used.add(name)
        i -= 1
    lo, hi = loop_range(lines)
    # ② 루프 머리·꼬리
    solver_timers = '' if a.no_solver_timers else '''    if (timing_start > 0) { momentum_solver.timer = &timer; pressure_solver.timer = &timer; }
''' + ('    if (timing_start > 0) scalar_solver.timer = &timer;\n' if has_scalar else '')
    head = f'''    // [scale] 구간 타이머 — tools/scale/instrument_main.py 가 main.cpp 에서 생성 (수치 불변, 동기화만 추가)
    mpmstd::common::PhaseTimer timer;
    const long timing_start = std::getenv("MPMSTD_PHASE_START") ? std::atol(std::getenv("MPMSTD_PHASE_START")) : 50;
{solver_timers}    double tdma_base_momentum = 0, tdma_base_pressure = 0{', tdma_base_scalar = 0' if has_scalar else ''};
'''
    lines.insert(lo, head.rstrip('\n'))
    lo, hi = loop_range(lines)
    comm = not a.no_solver_timers   # 분산 TDMA 의 comm_seconds 훅은 최적화 뒤 코드에만 있다
    comm_base = ('        tdma_base_momentum = momentum_solver.plan_z.comm_seconds;\n        tdma_base_pressure = pressure_solver.plan_z.comm_seconds;'
                 + (chr(10)+'        tdma_base_scalar = scalar_solver.plan_z.comm_seconds;' if has_scalar else '')) if comm else ''
    activate = f'''      if (timing_start > 0 && !timer.active() && step >= timing_start) {{
        timer.activate();
{comm_base}
      }}
      timer.begin("step_total");'''
    lines.insert(lo + 1, activate)
    lo, hi = loop_range(lines)
    lines.insert(hi, '      timer.end("step_total"); timer.count_step();')
    hi += 1
    comm_ext = ('      timer.set_external("tdma_comm_momentum_z", momentum_solver.plan_z.comm_seconds - tdma_base_momentum);\n      timer.set_external("tdma_comm_pressure_z", pressure_solver.plan_z.comm_seconds - tdma_base_pressure);'
                + (chr(10)+'      timer.set_external("tdma_comm_scalar_z", scalar_solver.plan_z.comm_seconds - tdma_base_scalar);' if has_scalar else '')) if comm else ''
    report = f'''    if (timing_start > 0) {{
{comm_ext}
      int world_ranks = 1; MPI_Comm_size(MPI_COMM_WORLD, &world_ranks);
      const char* csv = std::getenv("MPMSTD_PHASE_CSV");
      timer.report(MPI_COMM_WORLD, csv ? csv : "phase_timing.csv",
                   {{{{"nx", std::to_string(s.mesh.nx)}}, {{"ny", std::to_string(s.mesh.ny)}}, {{"nz", std::to_string(s.mesh.nz)}},
                    {{"npz", std::to_string(s.process.npz)}}, {{"ranks", std::to_string(world_ranks)}},
                    {{"phase_timing_start_step", std::to_string(timing_start)}}}});
    }}'''
    lines.insert(hi + 1, report)
    # ③ include + 가드
    k = max(i for i, l in enumerate(lines) if l.startswith('#include'))
    lines.insert(k + 1, '#include <cstdlib>\n#include "common/phase_timer.hpp"')
    text = '\n'.join(lines)
    banner = f'''// ══ 시간 측정용 쌍둥이 main — tools/scale/instrument_main.py 가 main.cpp 에서 자동 생성 (손으로 고치지 말 것).
//    make <app>_scale 가 -DMPMSTD_SCALE_MAIN 으로 컴파일한다 (일반 빌드에서는 빈 TU). 구간: {", ".join(sorted(used))}.
#ifdef MPMSTD_SCALE_MAIN
'''
    open(out, 'w').write(banner + text.rstrip('\n') + '\n#endif  // MPMSTD_SCALE_MAIN\n')
    print(f"{out}: sections {sorted(used)}")

if __name__ == '__main__':
    main()
