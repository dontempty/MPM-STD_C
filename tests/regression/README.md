# tests/regression

Golden regression suite for the refactored `mpmstd` lib (rev.2 §10): validated
Re_tau=180 channel statistics, lid-driven cavity vs Ghia et al. (P3b), and the
final Fig 7 (OB-DHVC) / Fig 9 (NOB-RBC) targets. CPU stages compare against the
frozen `src/` baseline; the GPU stage compares by statistical agreement (the
pointwise 1e-10 criterion was dropped, rev.2 M4). Populated P1+.
