#include "structure.h"

#include "heuristic_context.h"

#include <algorithm>
#include <cmath>
#include <optional>

namespace {

// A side `s` of a uniform row with coefficient `a`, as the integer count
// `s / a`, or nothing when it is not one.
std::optional<HighsInt> count_side(double side, double coefficient) {
    const double count = side / coefficient;
    const double rounded = std::round(count);
    if (std::abs(count - rounded) > kCountTol) {
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

}  // namespace

RowStructure detect_structure(const ProblemView& problem) {
    RowStructure out;
    const auto& start = *problem.ar_start;
    const auto& index = *problem.ar_index;
    const auto& value = *problem.ar_value;
    for (HighsInt row = 0; row < problem.nrow; ++row) {
        const HighsInt begin = start[row];
        const HighsInt end = start[row + 1];
        const HighsInt width = end - begin;
        const double lower = (*problem.row_lower)[row];
        const double upper = (*problem.row_upper)[row];
        const bool has_lower = std::isfinite(lower);
        const bool has_upper = std::isfinite(upper);
        const bool equality = has_lower && has_upper && std::abs(lower - upper) <= kExactKTol;

        RowKind kind = RowKind::kGeneral;
        if (width >= 2) {
            // One pass for every per-entry property the kinds need.
            const double a = value[begin];
            const double agreement = kExactKTol * std::max(1.0, std::abs(a));
            bool uniform_binary = a > 0;
            bool unit = true;
            bool positive = false;
            bool negative = false;
            for (HighsInt p = begin; p < end; ++p) {
                const double c = value[p];
                uniform_binary =
                    uniform_binary && problem.binary[index[p]] != 0 && std::abs(c - a) <= agreement;
                unit = unit && std::abs(std::abs(c) - 1.0) <= kExactKTol;
                positive = positive || c > 0;
                negative = negative || c < 0;
            }
            if (uniform_binary) {
                kind = uniform_binary_kind(lower, upper, a, width);
                // cuOpt's exact-k row: an equality whose count side lies in
                // `[0, width]`, `k = 0` included.
                if (equality && width <= kExactKMaxWidth) {
                    const auto k = count_side(lower, a);
                    if (k && *k >= 0 && *k <= width) {
                        out.exact_k.push_back({row, *k});
                    }
                }
            } else if (width == 2 && positive && negative && !equality &&
                       (has_lower || has_upper)) {
                kind = RowKind::kPrecedence;
            } else if (equality && unit && positive && negative) {
                kind = RowKind::kFlowConservation;
            }
        }
        ++out.count[static_cast<size_t>(kind)];
    }
    return out;
}
