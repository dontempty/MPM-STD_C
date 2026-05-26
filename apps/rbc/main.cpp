// =============================================================================
//  apps/rbc/main.cpp  —  Rayleigh-Bénard convection, SETUP-ONLY (no time loop)
//
//  This file is intentionally written top-to-bottom in plain procedural style
//  so you can read it as a "tour" of the MPM-STD library.  Each numbered
//  section corresponds to one module in src/.  The time loop is deferred
//  until M5 (it will sit at the end of section 12).
//
//  Run with:
//      mpirun -np 1 build/cpu/bin/rbc apps/rbc/input.toml
//
//  Layer cake of dependencies (top-to-bottom in this file matches bottom-to-
//  top in the dependency graph):
//
//      common  ──► parallel/mpi ──► parallel/backend
//                          │
//                          └──► grid
//                                │
//                                └──► field ──► boundary ──► linear_solver
// =============================================================================

#include "boundary/main.hpp"
#include "common/main.hpp"
#include "config/main.hpp"
#include "field/main.hpp"
#include "grid/main.hpp"
#include "linear_solver/tdma/main.hpp"
#include "parallel/main.hpp"
#include "post/main.hpp"

#include <array>
#include <iostream>
#include <string>

using namespace mpmstd;


int main(int argc, char** argv) {

  // -------------------------------------------------------------------------
  //  (1)  MPI startup.
  //
  //  MpiContext is RAII — the ctor calls MPI_Init_thread and the dtor calls
  //  MPI_Finalize.  It also splits MPI_COMM_WORLD by node-shared memory to
  //  get a node-local communicator; on the CUDA build that's used to bind
  //  one MPI rank to one GPU.  On CPU build it's harmless.
  // -------------------------------------------------------------------------
  parallel::mpi::MpiContext mpi(&argc, &argv);


  // -------------------------------------------------------------------------
  //  (2)  Logger + config file.
  //
  //  Logger::init binds the world rank.  Only rank 0 actually writes; all
  //  other ranks ignore log calls silently.
  //
  //  Config::load parses an INI-style key/value file.  Missing keys throw,
  //  unless you use the get_or<T>(...) variant with a default.
  // -------------------------------------------------------------------------
  config::Logger::init(mpi.world_rank(), config::LogLevel::Info);

  const std::string input_path = (argc >= 2) ? argv[1] : "apps/rbc/input.toml";
  auto cfg = config::Config::load(input_path);
  config::Logger::info("loaded config from '%s'", cfg.source_path().c_str());


  // -------------------------------------------------------------------------
  //  (3)  Cartesian MPI topology.
  //
  //  Reads (np1, np2, np3) from [mpi].  The product must equal the world
  //  size; otherwise MpiTopology's ctor throws.
  //
  //  periodic[]:  x and y are periodic (typical RBC), z is non-periodic
  //               because it holds the two thermal walls.
  //
  //  Internally MpiTopology builds:
  //    * the 3D Cartesian comm,
  //    * three 1-D axis sub-comms (used by TDMA and halo exchange),
  //    * neighbor ranks via MPI_Cart_shift (MPI_PROC_NULL at non-periodic
  //      global faces).
  // -------------------------------------------------------------------------
  std::array<int, 3> dims = {
    cfg.get<int>("mpi", "np1"),
    cfg.get<int>("mpi", "np2"),
    cfg.get<int>("mpi", "np3"),
  };
  std::array<bool, 3> periodic = { true, true, false };
  parallel::mpi::MpiTopology topo(mpi, dims, periodic);


  // -------------------------------------------------------------------------
  //  (4)  Subdomain decomposition.
  //
  //  n_global = interior cell counts in each axis.  Subdomain uses para_range
  //  (matching PaScaL_TDMA_C) to split them across the axis sub-comms; its
  //  state includes:
  //
  //      n_interior(d) : local interior cell count along d
  //      n_total(d)    : n_interior(d) + 2 * kHaloWidth   (halos included)
  //      global_offset(d) : index of the first local interior cell in global
  //
  //  Subdomain also builds the MPI derived datatypes used by exchange_halo()
  //  and the post::restart_io routines.
  // -------------------------------------------------------------------------
  std::array<int, 3> n_global = {
    cfg.get<int>("mesh", "n1m"),
    cfg.get<int>("mesh", "n2m"),
    cfg.get<int>("mesh", "n3m"),
  };
  parallel::mpi::Subdomain sub(topo, n_global);


  // -------------------------------------------------------------------------
  //  (5)  Grid (per-axis coordinates + metrics).
  //
  //  For each axis we pass:
  //      n_global, length, stretch kind, gamma (tanh parameter)
  //
  //  Grid::build_axis_ generates the global face coordinates, then slices
  //  out the local-rank portion (interior + halos) into:
  //      xf[d] : face coordinates  (size n_total(d) + 1)
  //      xc[d] : cell-center coords (size n_total(d))
  //      dx[d] : cell widths       (size n_total(d))
  //      dmx[d]: face-to-face dist (size n_total(d))
  //
  //  Stencil helpers in field/stencil/ take dx, dmx and field arrays.
  // -------------------------------------------------------------------------
  std::array<grid::AxisConfig, 3> axes;
  for (int a = 0; a < 3; ++a) {
    const std::string gamma_key = "gamma"   + std::to_string(a + 1);
    const std::string uni_key   = "uniform" + std::to_string(a + 1);
    const std::string len_key   = std::string("L") + char('x' + a);

    const int    is_uniform = cfg.get_or<int>("uniform_mesh",  uni_key,   1);
    const double gamma      = cfg.get_or<double>("mesh_stretch", gamma_key, 0.0);
    const double L          = cfg.get<double>("domain", len_key);

    axes[a] = grid::AxisConfig{
      /*.n_global=*/ n_global[a],
      /*.length  =*/ L,
      /*.stretch =*/ is_uniform ? grid::StretchKind::Uniform
                                : grid::StretchKind::Tanh,
      /*.gamma   =*/ gamma,
    };
  }
  grid::Grid g(sub, axes);


  // -------------------------------------------------------------------------
  //  (6)  Problem  =  Topology + per-field BC table.
  //
  //  load_problem_from_config() reads:
  //      [topology]      x/y/z = "periodic" | "non_periodic"
  //      [bc.<a>.<s>]    Dirichlet values for U, V, W, T
  //                       (P is always Neumann 0, Periodic faces auto.)
  //
  //  Missing Dirichlet entries default to 0 — see problem_loader.hpp.
  //  validate() throws if face periodicity disagrees with axis topology.
  // -------------------------------------------------------------------------
  boundary::Problem problem = boundary::load_problem_from_config(cfg);
  problem.validate();


  // -------------------------------------------------------------------------
  //  (7)  Backend  =  memory + stream abstraction.
  //
  //  make_default_backend() returns a CpuBackend on the CPU build and (later)
  //  a CudaBackend on BACKEND=cuda.  Past memory ownership the backend is
  //  intentionally thin — kernel dispatch happens by which .cpp/.cu the
  //  build system selects, not by virtual dispatch.
  // -------------------------------------------------------------------------
  auto backend = parallel::make_default_backend();


  // -------------------------------------------------------------------------
  //  (8)  FieldRegistry  =  named storage for all 3-D arrays.
  //
  //  Following the PaScaL_TCS convention, every velocity component (U, V, W)
  //  is its own ScalarField. The staggered MAC interpretation —
  //  U on the x-face, V on the y-face, W on the z-face — is encoded by the
  //  stencil helpers that consume the arrays, not by the storage type.
  //  All arrays share the same (n_total) shape (see Report/07).
  //
  //  dU, dV, dW are the per-step velocity increments used by the momentum
  //  predictor; we pre-allocate them once so the future time loop can reuse
  //  the buffers every step.
  // -------------------------------------------------------------------------
  field::FieldRegistry fields(sub, *backend);
  auto& U  = fields.add_scalar("U");   // x-velocity on FaceX
  auto& V  = fields.add_scalar("V");   // y-velocity on FaceY
  auto& W  = fields.add_scalar("W");   // z-velocity on FaceZ
  auto& dU = fields.add_scalar("dU");
  auto& dV = fields.add_scalar("dV");
  auto& dW = fields.add_scalar("dW");
  auto& P  = fields.add_scalar("P");
  auto& dP = fields.add_scalar("dP");
  auto& T  = fields.add_scalar("T");


  // -------------------------------------------------------------------------
  //  (9)  BoundaryApplier.
  //
  //  Owns a reference to `problem` and does two things:
  //
  //    apply_ghost(field, FieldBoundary, t)
  //        Fills ghost cells on global boundary faces only (Periodic faces
  //        are no-op — halo exchange handles them via MPI_Cart wrap-around).
  //
  //    modify_tdma_row(direction, fbc, A, B, C, D, n_sys, n_row)
  //        Will be implemented in M3 when the momentum solver starts
  //        building tridiagonal systems; for now it throws if called.
  // -------------------------------------------------------------------------
  boundary::BoundaryApplier bc(problem);


  // -------------------------------------------------------------------------
  //  (10) TdmaRegistry  =  one TDMA solver per axis.
  //
  //  Wraps PaScaL_TDMA_C.  Each axis gets its own plan, lazily cached by
  //  n_sys, so the same registry serves every ADI stage across the whole
  //  time loop.  We don't call solve_*() in this file yet — the equation
  //  modules will (M2 thermal, M3 momentum, M4 pressure).
  // -------------------------------------------------------------------------
  auto tdma = linear_solver::tdma::TdmaRegistry::make_default(topo);


  // -------------------------------------------------------------------------
  //  (11) Initial conditions.
  //
  //  For RBC at rest:
  //      U, V, W   = 0
  //      P, dP     = 0
  //      T         = linear conductive profile  T_hot + (T_cold-T_hot)*z/Lz
  //
  //  In a real run we'd add a small velocity perturbation to break symmetry
  //  (otherwise the conductive state is a fixed point of the equations).
  //  Here we keep it simple — the time loop in M5 will optionally add a
  //  perturbation.
  // -------------------------------------------------------------------------
  U.fill_host(0.0);   V.fill_host(0.0);   W.fill_host(0.0);
  dU.fill_host(0.0);  dV.fill_host(0.0);  dW.fill_host(0.0);
  P.fill_host(0.0);
  dP.fill_host(0.0);

  // Linear conductive temperature profile between the two z-wall Dirichlet
  // values that were loaded into `problem`.  Reading from the FaceBc objects
  // makes `problem` the single source of truth for wall temperatures.
  {
    const double Lz      = cfg.get<double>("domain", "Lz");
    const double T_minus = problem.T.face(Direction::Z, Side::Minus).value(0, 0, 0, 0);
    const double T_plus  = problem.T.face(Direction::Z, Side::Plus ).value(0, 0, 0, 0);
    const auto&  zc      = g.xc(Direction::Z);   // cell-center z coordinates

    const int n1 = sub.n_total()[0];
    const int n2 = sub.n_total()[1];
    const int n3 = sub.n_total()[2];
    for (int i = 0; i < n1; ++i)
      for (int j = 0; j < n2; ++j)
        for (int k = 0; k < n3; ++k) {
          const double frac = zc[k] / Lz;       // 0 at lower wall, 1 at upper
          T.host_at(i, j, k) = T_minus + (T_plus - T_minus) * frac;
        }
  }


  // -------------------------------------------------------------------------
  //  (12) Halo exchange + boundary fill  (twice the same pattern: first MPI,
  //       then global walls).
  //
  //  Order matters:
  //    [a]  exchange_halo()   fills interior rank-rank interface halos AND
  //                            Periodic wrap-around halos.
  //    [b]  apply_ghost(...)  fills global boundary halos (Dirichlet/Neumann).
  //
  //  After this pair of calls, every field has well-defined values in every
  //  cell — including all halos — and is ready for the first stencil
  //  evaluation in the time loop.
  // -------------------------------------------------------------------------
  // --- [a] velocity (3 separate scalars: U, V, W) ---
  U.exchange_halo();  bc.apply_ghost(U, problem.U);
  V.exchange_halo();  bc.apply_ghost(V, problem.V);
  W.exchange_halo();  bc.apply_ghost(W, problem.W);

  // --- [b] pressure ---
  P.exchange_halo();  bc.apply_ghost(P, problem.P);

  // --- [c] temperature ---
  T.exchange_halo();  bc.apply_ghost(T, problem.T);


  // -------------------------------------------------------------------------
  //  Setup summary (root rank only).
  // -------------------------------------------------------------------------
  if (mpi.is_root()) {
    auto wax = problem.topology.wall_axis();
    auto so  = problem.topology.sweep_order();

    std::cout << "\n=== MPM-STD RBC setup complete ===\n"
              << "  ranks      : " << mpi.world_size() << "\n"
              << "  proc grid  : " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n"
              << "  mesh (n_m) : " << n_global[0] << " x " << n_global[1]
                                   << " x " << n_global[2] << "\n"
              << "  domain L   : " << cfg.get<double>("domain", "Lx") << " x "
                                   << cfg.get<double>("domain", "Ly") << " x "
                                   << cfg.get<double>("domain", "Lz") << "\n"
              << "  Ra / Pr    : " << cfg.get<double>("physics", "Ra")
                                   << " / " << cfg.get<double>("physics", "Pr") << "\n"
              << "  T walls    : "
              << problem.T.face(Direction::Z, Side::Minus).value(0,0,0,0) << " (-z) , "
              << problem.T.face(Direction::Z, Side::Plus ).value(0,0,0,0) << " (+z)\n"
              << "  U walls    : "
              << problem.U.face(Direction::Z, Side::Minus).value(0,0,0,0) << " (-z) , "
              << problem.U.face(Direction::Z, Side::Plus ).value(0,0,0,0) << " (+z)\n"
              << "  U.kind ±z  : "
              << boundary::bc_kind_name(problem.U.face(Direction::Z, Side::Minus).kind) << " / "
              << boundary::bc_kind_name(problem.U.face(Direction::Z, Side::Plus ).kind) << "\n"
              << "  P.kind ±z  : "
              << boundary::bc_kind_name(problem.P.face(Direction::Z, Side::Minus).kind) << " / "
              << boundary::bc_kind_name(problem.P.face(Direction::Z, Side::Plus ).kind) << "\n"
              << "  scalars    :";
    for (auto& n : fields.scalar_names()) std::cout << " " << n;
    std::cout << "\n  wall axis  : "
              << (wax ? direction_name(wax.value()) : "(ambiguous)") << "\n"
              << "  sweep ord  : "
              << direction_name(so[0]) << " -> "
              << direction_name(so[1]) << " -> "
              << direction_name(so[2]) << "\n";

    // Print a 1-D temperature slice through the centre to confirm the
    // conductive profile is correct.
    std::cout << "\n  T(x_mid, y_mid, z):";
    const int n3 = sub.n_total()[2];
    const int im = sub.n_total()[0] / 2;
    const int jm = sub.n_total()[1] / 2;
    for (int k = kHaloWidth; k < n3 - kHaloWidth; ++k)
      std::cout << "  " << T.host_at(im, jm, k);
    std::cout << "\n";

    std::cout << "\n[ next:  time loop  (added in milestone M5) ]\n\n";
  }


  // -------------------------------------------------------------------------
  //  (12') TIME LOOP would start here in M5.
  //
  //      while (time < t_end) {
  //          1. thermal solver  (M2)        T^{n+1}
  //          2. momentum solver (M3)        dU, then U* = U + dU
  //          3. pressure solver (M4)        dP, then U^{n+1} = U* - dt grad dP
  //          4. plugins / stats / io
  //          5. dt = CFL controller
  //      }
  //
  //  None of those exist yet — this file ends here.
  // -------------------------------------------------------------------------

  return 0;
}
