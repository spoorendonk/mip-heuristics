#include "structure.h"

#include "heuristic_context.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace {

// A side `s` of a uniform row with coefficient `a`, as the integer count
// `s / a`, or nothing when it is not one.
std::optional<HighsInt> count_side(double side, double coefficient) {
    const double count = side / coefficient;
    const double rounded = std::round(count);
    // Out of HighsInt's range, the cast below is undefined; such a count
    // exceeds every row's width anyway.
    if (std::abs(count - rounded) > kCountTol ||
        std::abs(rounded) > static_cast<double>(std::numeric_limits<HighsInt>::max())) {
        return std::nullopt;
    }
    return static_cast<HighsInt>(rounded);
}

// The kind of a uniform binary row of width `width` with coefficient `a`
// and sides `[lower, upper]`.  A side at or beyond what the row can reach
// (`lower <= 0`, `upper >= width`) binds nothing and is read as absent.
RowKind uniform_binary_kind(double lower, double upper, double a, HighsInt width) {
    const bool has_lower = std::isfinite(lower) && lower > kExactKTol * std::max(1.0, a);
    const bool has_upper = std::isfinite(upper) && upper < (width * a) - (kExactKTol * a);
    const bool equality =
        std::isfinite(lower) && std::isfinite(upper) && std::abs(lower - upper) <= kExactKTol;
    if (equality) {
        const auto k = count_side(lower, a);
        if (!k || *k < 1) {
            return RowKind::kGeneral;
        }
        return *k == 1 ? RowKind::kPartitioning : RowKind::kCardinality;
    }
    if (has_lower == has_upper) {
        // A ranged row, or one that binds nothing.
        return RowKind::kGeneral;
    }
    const auto k = count_side(has_lower ? lower : upper, a);
    if (!k || *k < 1) {
        return RowKind::kGeneral;
    }
    if (*k == 1) {
        return has_lower ? RowKind::kCovering : RowKind::kPacking;
    }
    return RowKind::kCardinality;
}

// The per-entry properties the row kinds need, gathered in one pass.
struct RowEntries {
    double a = 0;  // the first coefficient
    bool uniform_binary = false;
    bool unit = true;
    bool positive = false;
    bool negative = false;
};

RowEntries scan_row(const ProblemView& problem, HighsInt begin, HighsInt end) {
    const auto& index = *problem.ar_index;
    const auto& value = *problem.ar_value;
    RowEntries e;
    e.a = value[begin];
    const double agreement = kExactKTol * std::max(1.0, std::abs(e.a));
    e.uniform_binary = e.a > 0;
    for (HighsInt p = begin; p < end; ++p) {
        const double c = value[p];
        e.uniform_binary =
            e.uniform_binary && problem.binary[index[p]] != 0 && std::abs(c - e.a) <= agreement;
        e.unit = e.unit && std::abs(std::abs(c) - 1.0) <= kExactKTol;
        e.positive = e.positive || c > 0;
        e.negative = e.negative || c < 0;
    }
    return e;
}

}  // namespace

RowStructure detect_structure(const ProblemView& problem) {
    RowStructure out;
    const auto& start = *problem.ar_start;
    for (HighsInt row = 0; row < problem.nrow; ++row) {
        const HighsInt width = start[row + 1] - start[row];
        const double lower = (*problem.row_lower)[row];
        const double upper = (*problem.row_upper)[row];
        const bool has_lower = std::isfinite(lower);
        const bool has_upper = std::isfinite(upper);
        const bool equality = has_lower && has_upper && std::abs(lower - upper) <= kExactKTol;

        // A singleton or empty row is a bound, not structure.
        if (width < 2) {
            ++out.count[static_cast<size_t>(RowKind::kGeneral)];
            continue;
        }
        RowKind kind = RowKind::kGeneral;
        const RowEntries e = scan_row(problem, start[row], start[row + 1]);
        if (e.uniform_binary) {
            kind = uniform_binary_kind(lower, upper, e.a, width);
            // `uniform_binary_kind`'s binding lower side.
            const bool binds_lower = has_lower && lower > kExactKTol * std::max(1.0, e.a);
            out.lower_bound_rows += kind != RowKind::kGeneral && binds_lower ? 1 : 0;
            // cuOpt's exact-k row: an equality whose count side lies in
            // `[0, width]`, `k = 0` included.
            const auto k =
                equality && width <= kExactKMaxWidth ? count_side(lower, e.a) : std::nullopt;
            if (k && *k >= 0 && *k <= width) {
                out.exact_k.push_back({row, *k});
            }
        } else if (width == 2 && e.positive && e.negative && !equality &&
                   (has_lower || has_upper)) {
            kind = RowKind::kPrecedence;
        } else if (equality && e.unit && e.positive && e.negative) {
            kind = RowKind::kFlowConservation;
        }
        ++out.count[static_cast<size_t>(kind)];
    }
    return out;
}
