#pragma once

#include "heuristic_context.h"
#include "sat_solver.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

// A feasibility search for pure-binary models (#236): each row becomes CNF and
// a CDCL solver looks for a point, ignoring the objective.
namespace sat {

// More clauses than this and a model is refused: what matters is the encoding's
// memory on a model the search would not reach anyway.  Measured on #212's
// probe at this value, where 4 of 12 models were solved and 7 went over.
constexpr size_t kDefaultMaxClauses = 5'000'000;

enum class SatSearchStatus {
    // `point` holds integer values satisfying every row within `feastol`.
    kFeasible,
    // No point exists, as far as the scaled rows say; see `integer_scale`'s
    // tolerance in the source before treating this as a proof about the model.
    kInfeasible,
    kStopped,
    kRefused,
};

enum class SatRefusal {
    kNone,
    // A column that is neither fixed, an integer one at an integral value, nor
    // an integer with bounds exactly [0, 1].
    kNotPureBinary,
    // A row with a coefficient no power of ten up to `kMaxScale` makes integral,
    // or whose scaled weights are too large to add up exactly.
    kNotIntegral,
    // Over `SatSearchConfig::max_clauses`.
    kTooLarge,
    // The solver's point failed the check against the original rows, which
    // only numerics in scaling can cause.  The point is not returned.
    kCheckFailed,
};

struct SatSearchConfig {
    std::uint64_t seed = 0;
    SatLimits limits;
    size_t max_clauses = kDefaultMaxClauses;
};

struct SatSearchResult {
    SatSearchStatus status = SatSearchStatus::kStopped;
    SatRefusal refusal = SatRefusal::kNone;
    size_t clauses = 0;
    size_t aux_vars = 0;
    size_t conflicts = 0;
    // Row coefficients read, diagram nodes built and level entries shifted, plus
    // the solver's effort.
    size_t effort = 0;
};

// `stop` is polled on this thread, at least every `kEncodePollWork` units while
// encoding and as `SatSolver` documents while solving; it may be empty.
// `config.limits.max_effort` covers encoding and solving together; `max_conflicts`
// is the solver's.  `point` is sized `problem.ncol` and written only on `kFeasible`.
SatSearchResult sat_search(const ProblemView& problem, const SatSearchConfig& config,
                           const std::function<bool()>& stop, std::vector<double>& point);

}  // namespace sat
