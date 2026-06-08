# tests/integration

Single-equation / small end-to-end tests for the refactored `mpmstd` lib
(rev.2). Each `*.cpp` is one binary linking `libmpmstd` (built by
`mpmstd/Makefile`'s `tests` target). Populated during P1+ as modules are ported
(e.g. thermal-only manufactured solution, momentum 1-step vs frozen, pressure
Poisson divergence). Old `test/` (singular) stays as the baseline until parity.
