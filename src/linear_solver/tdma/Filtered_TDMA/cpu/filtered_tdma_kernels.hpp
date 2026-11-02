#ifndef FILTERED_TDMA_KERNELS_HPP
#define FILTERED_TDMA_KERNELS_HPP

// Shared CPU "kernels" (algorithm steps) for the Filtered TDMA solves.
//
// These are the reusable building blocks called by both the production solves
// in filtered_tdma.cpp and the per-phase timed solves in
// filtered_tdma_profile.cpp. Each is `static inline`, so every translation unit
// that includes this header gets its own internal-linkage copy — no link
// conflicts with the pascal_tdma library.
//
// Every step takes the matrix/RHS as raw base pointers (A, B, C, D) and derives
// the row offsets it needs internally — mirroring the GPU kernels, which take
// d_A/d_B/d_C/d_D. The base pointers are marked `__restrict`: A, B, C and D are
// four distinct arrays, so this is true, and it lets the vectorizer keep the
// four columns apart.
//
// Rows are addressed by an EXPLICIT offset (`j*n_sys`) recomputed each row, and
// never by pointers walked across the row loop.  That is a correctness
// requirement, not a style choice.  With walked pointers (Ajm/Cjm/Djm advanced
// in the loop increment) nvc++ 25.11 miscompiles all three sweeps at -O2 and
// above: the row-(j-1) values a row read are the ones from BEFORE the previous
// iteration's stores, so the forward sweep propagates the un-normalised
// coefficients and every multi-rank solve comes out wrong by O(|A|) — the
// interface coupling, i.e. the whole point of the distributed solve.
//
// It was masked for a long time: n_sys == 1 (one lane) and nprocs == 1 (local
// solve, these kernels unused) both come out right, `-O0`/`-O1`/`-Mnovect` come
// out right, and g++ at -O3 comes out right.  Only nvc++ -O2+ with n_sys > 1
// and nprocs > 1 — i.e. exactly the production CPU configuration — is affected.
// `__restrict` alone does NOT cure it; the offsets do.  A regression here shows
// up as heat_cpu's `option = order` error jumping to O(0.1) as soon as any
// direction is decomposed, while the GPU path stays correct.
//
// The MPI neighbour exchange is NOT here: it needs the solver's MPI plan
// (datatypes, ranks, recv buffers) and lives as FilteredTDMA::exchange_interfaces_().

#include <cstddef>
#include <cmath>
#include <algorithm>

namespace ftdma_kernel {

// ---------------------------------------------------------------------------
//  Step 1: Forward elimination (normalize rows 0,1 then eliminate downward)
// ---------------------------------------------------------------------------

// Full update for j=2..J, then skip the A update for j=J+1..n_row-1:
// beyond the cutoff the A coefficients no longer influence the answer.
[[gnu::always_inline]] static inline
void fwd_elim(double* __restrict A, double* __restrict B,
                 double* __restrict C, double* __restrict D,
                 int n_sys, int n_row, int J) {
    const std::size_t ns = (std::size_t)n_sys;
    #pragma omp simd
    for (int i = 0; i < n_sys; ++i) {
        double r0 = 1.0 / B[i];
        A[i]      *= r0;  C[i]      *= r0;  D[i]      *= r0;
        double r1 = 1.0 / B[ns + i];
        A[ns + i] *= r1;  C[ns + i] *= r1;  D[ns + i] *= r1;
    }
    // Phase 1: j=2..J — full update (D, C, A)
    for (int j = 2; j <= J; ++j) {
        const std::size_t o = (std::size_t)j * ns, p = o - ns;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (B[o+i] - A[o+i] * C[p+i]);
            double dn  =  inv * (D[o+i] - A[o+i] * D[p+i]);
            double cn  =  inv * C[o+i];
            double an  = -inv * A[o+i] * A[p+i];
            D[o+i] = dn;  C[o+i] = cn;  A[o+i] = an;
        }
    }
    // Phase 2: j=max(J+1,2)..n_row-1 — skip A update.
    // Rows 0 and 1 are normalised outside the loop, so the A-carrying pass
    // covers [2, J] and this one must resume at max(J+1, 2).  A bare `J + 1`
    // restarts at row 1 when J = 0 and eliminates it a second time.
    for (int j = (J + 1 > 2 ? J + 1 : 2); j < n_row; ++j) {
        const std::size_t o = (std::size_t)j * ns, p = o - ns;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) {
            double inv = 1.0 / (B[o+i] - A[o+i] * C[p+i]);
            double dn  = inv * (D[o+i] - A[o+i] * D[p+i]);
            double cn  = inv * C[o+i];
            D[o+i] = dn;  C[o+i] = cn;
        }
    }
}

// ---------------------------------------------------------------------------
//  Step 2: Backward substitution
// ---------------------------------------------------------------------------

// Three phases: D over all rows, A only below J, C only above lo. Splitting
// the sweep is what lets the A and C passes stop at the cutoff.
[[gnu::always_inline]] static inline
void bwd_subst(double* __restrict A, double* __restrict C,
                  double* __restrict D, int n_sys, int n_row,
                  int J, int lo) {
    const std::size_t ns = (std::size_t)n_sys;
    // D over all rows
    for (int j = n_row - 3; j >= 1; --j) {
        const std::size_t o = (std::size_t)j * ns, q = o + ns;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) D[o+i] -= C[o+i] * D[q+i];
    }
    // A only below J — reads the still-unmodified C, so it must precede the C pass
    if (J >= 2) {
        for (int j = J - 1; j >= 1; --j) {
            const std::size_t o = (std::size_t)j * ns, q = o + ns;
            #pragma omp simd
            for (int i = 0; i < n_sys; ++i) A[o+i] -= C[o+i] * A[q+i];
        }
    }
    // C only above lo
    for (int j = n_row - 3; j >= lo + 1; --j) {
        const std::size_t o = (std::size_t)j * ns, q = o + ns;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) C[o+i] *= -C[q+i];
    }
}

// ---------------------------------------------------------------------------
//  Step 3: Pack reduced system — decouple row 0 from x_{N-1}
// ---------------------------------------------------------------------------

[[gnu::always_inline]] static inline
void pack_reduced(double* __restrict A, double* __restrict C,
                  double* __restrict D, int n_sys) {
    const std::size_t ns = (std::size_t)n_sys;
    #pragma omp simd
    for (int i = 0; i < n_sys; ++i) {
        double r = 1.0 / (1.0 - A[ns+i] * C[i]);
        D[i] = r * (D[i] - C[i] * D[ns+i]);
        A[i] = r * A[i];
    }
}

// ---------------------------------------------------------------------------
//  Step 5: Solve both interface 2x2 blocks locally (post-exchange)
//    left  : D[0]   = (D0 - A0*D_left)    / (1 - C_left*A0)
//    right : D[N-1] = (DN - CN*D0_right)  / (1 - A_right*CN)
// ---------------------------------------------------------------------------

[[gnu::always_inline]] static inline
void solve_interfaces(double* __restrict A, double* __restrict C,
                                    double* __restrict D,
                                    const double* __restrict C_left,
                                    const double* __restrict D_left,
                                    const double* __restrict A_right,
                                    const double* __restrict D0_right,
                                    bool has_left, bool has_right,
                                    int n_sys, int n_row) {
    if (has_left) {
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i)
            D[i] = (D[i] - A[i] * D_left[i]) / (1.0 - C_left[i] * A[i]);
    }
    if (has_right) {
        double* CN = C + (std::size_t)(n_row - 1) * n_sys;
        double* DN = D + (std::size_t)(n_row - 1) * n_sys;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i)
            DN[i] = (DN[i] - CN[i] * D0_right[i]) / (1.0 - A_right[i] * CN[i]);
    }
}

// ---------------------------------------------------------------------------
//  Step 6: Final local solve (boundary correction with J cutoff)
//    left  rows 1..left_end    : D[j] -= A[j]*D0
//    right rows right_beg..N-2 : D[j] -= C[j]*DN
//  Callers pass the exact loop bounds (they differ between v1 and v2).
// ---------------------------------------------------------------------------

[[gnu::always_inline]] static inline
void final_solve(double* __restrict A, double* __restrict C,
                 double* __restrict D, int n_sys, int n_row,
                 int left_end, int right_beg) {
    const std::size_t ns = (std::size_t)n_sys;
    const std::size_t oN = (std::size_t)(n_row - 1) * ns;   // row 0 is offset 0
    for (int j = 1; j <= left_end; ++j) {
        const std::size_t o = (std::size_t)j * ns;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) D[o+i] -= A[o+i] * D[i];
    }
    for (int j = right_beg; j < n_row - 1; ++j) {
        const std::size_t o = (std::size_t)j * ns;
        #pragma omp simd
        for (int i = 0; i < n_sys; ++i) D[o+i] -= C[o+i] * D[oN+i];
    }
}

} // namespace ftdma_kernel

#endif // FILTERED_TDMA_KERNELS_HPP
