#pragma once

#include "core/field_cpu.hpp"
#include "core/mpi_topology.hpp"        // Subdomain, CartComm1D (via parallel::mpi)
#include "core/variables.hpp"           // Var, CpuFields
#include "core/boundary.hpp"            // BoundaryCondition (= boundary::Problem)
#include "common/direction.hpp"
#include "boundary/field_boundary.hpp"  // FieldBoundary, FaceBc, BcKind, GhostPolicy (host-single)

namespace mpmstd::core {

using FieldBoundary = boundary::FieldBoundary;

// rev.2 P1: BC operations ported to FREE FUNCTIONS on CpuField / raw bands —
// fully decoupled from the old ScalarField-based BoundaryApplier (and thus from
// the dropped virtual Backend). Logic is identical to the validated original.

// Fill the ghost layer of `phi` on GLOBAL boundary faces only (faces where this
// rank's axis sub-comm neighbour is MPI_PROC_NULL). Interior interfaces are
// handled by exchange_halo_cpu. Periodic = no-op (halo wrap); Dirichlet
// (ZeroGhost / Antisymmetric) and Neumann implemented.
void apply_ghost_cpu(CpuField& phi, const FieldBoundary& fbc,
                     const Subdomain& sub, real_t t = real_t{0});

// Amend a cell-centered ADI tridiagonal system (row-major [n_row x n_sys]) so
// the wall BC on `wall_axis` is enforced, ONLY on ranks owning the global wall.
// Dirichlet/Neumann fold: B += A, A = 0 (lower) / B += C, C = 0 (upper).
void modify_tdma_row_cpu(Direction wall_axis, const FieldBoundary& fbc,
                         const parallel::mpi::CartComm1D& axis_comm,
                         real_t* A, real_t* B, real_t* C, real_t* D,
                         int n_sys, int n_row);

// Make a field consistent across rank interfaces AND domain boundaries in one
// step = exchange_halo_cpu (neighbour halos) + apply_ghost_cpu (global-boundary
// ghost). This is the "sync after a solve" the recipe does around every field;
// folding the pair keeps the main loop readable and prevents a forgotten ghost.
void sync_field_cpu(CpuField& field, const FieldBoundary& fbc, const Subdomain& sub,
                    real_t t = real_t{0});

// The per-field FieldBoundary for a variable (bc.U for Var::U, bc.V for V, …).
const FieldBoundary& face_bc_for(const BoundaryCondition& bc, Var v);

// Container convenience: sync Fields[v] (halo + that variable's BC ghost) in one.
void sync_field_cpu(CpuFields& fields, Var v, const BoundaryCondition& bc, const Subdomain& sub);

} // namespace mpmstd::core
