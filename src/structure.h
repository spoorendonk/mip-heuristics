#pragma once

#include "util/HighsInt.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

struct ProblemView;

// What one row of the model is, by the shape of its coefficients and sides
// (#206).  One O(nnz) pass over the view's row-wise matrix; nothing is
// propagated or substituted, so a row is classified as written.
//
// "Uniform binary" below means every entry is a binary column (the view's
// `binary` mask) with the same positive coefficient `a`, up to
// `kExactKTol` relative to `max(1, a)` — cuOpt's `apply_exact_k_start`
// test.  A side `s` then reads as the count `s / a`, which must be an
// integer within `kCountTol`.
enum class RowKind : uint8_t {
    kGeneral,
    // Uniform binary, `= 1` (a set-partitioning row).
    kPartitioning,
    // Uniform binary, `<= 1` with no binding lower side.
    kPacking,
    // Uniform binary, `>= 1` with no binding upper side.
    kCovering,
    // Uniform binary, `= k`, `<= k` or `>= k` for an integer `k >= 2`.
    kCardinality,
    // Two entries of opposite sign and at least one finite side, not an
    // equality: `a x - b y <= c` and its mirror.
    kPrecedence,
    // An equality whose coefficients are all `+1` or `-1`, with both signs
    // present: a flow-conservation row of a network.
    kFlowConservation,
};
inline constexpr size_t kNumRowKinds = 7;

// An exact-k row as cuOpt's `apply_exact_k_start` collects it: a uniform
// binary equality row of width 2 to `kExactKMaxWidth`, `k` its count side,
// `0 <= k <= width`.  Partitioning rows are the `k = 1` ones.
struct ExactKRow {
    HighsInt row = 0;
    HighsInt k = 0;
};

struct RowStructure {
    std::array<HighsInt, kNumRowKinds> count{};
    // Uniform binary rows with a binding lower side: the covering and
    // partitioning rows, and the `= k` / `>= k` cardinality ones.  The
    // covering start only acts on these from the bound start; a `<= k`
    // row is satisfied there already.
    HighsInt lower_bound_rows = 0;
    // In row order.
    std::vector<ExactKRow> exact_k;

    [[nodiscard]] HighsInt count_of(RowKind kind) const { return count[static_cast<size_t>(kind)]; }
};

// cuOpt's `exact_k_tol` (its `fj_cpu` tuning), the agreement both the
// equality test and the uniform-coefficient test use.
inline constexpr double kExactKTol = 1e-6;
// cuOpt's distance from an integer that a count side `s / a` may have
// (`apply_exact_k_start`'s `1e-4`).
inline constexpr double kCountTol = 1e-4;
// cuOpt's `exact_k_max_width`: wider equality rows are not exact-k rows.
inline constexpr HighsInt kExactKMaxWidth = 20000;

// Classify every row of `problem`.  Reads the view only; safe on any
// thread, and run once on the dispatching thread by `fj::run`.
RowStructure detect_structure(const ProblemView& problem);
