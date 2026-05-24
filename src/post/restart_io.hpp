#pragma once

#include "field/scalar_field.hpp"

#include <mpi.h>
#include <string>

namespace mpmstd::post {

// Parallel binary I/O for ScalarField.
//
// File layout
// -----------
// A single binary file holds the **global interior** of the field, in
// row-major order (axis 0 slowest), shape (n1m, n2m, n3m). Halo cells are not
// written. The file format is just a raw block of `real_t` values — no header.
//
// Each rank writes its subarray with a collective MPI-IO call using a
// subarray datatype derived from the Subdomain.
//
// Use cases (M0)
// --------------
// - Restart checkpoint: write_scalar() at the end of a run, read_scalar() at
//   the start of the next run.
// - Unit test: write_scalar(f1, path); read_scalar(f2, path); compare f1==f2.

void write_scalar(const field::ScalarField& field, const std::string& path);

void read_scalar(field::ScalarField& field, const std::string& path);

} // namespace mpmstd::post
