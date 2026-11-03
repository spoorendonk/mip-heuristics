#include "sat_search.h"

#include "heuristic_common.h"
#include "heuristic_context.h"
#include "sat_solver.h"
#include "util/HighsInt.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace sat {

namespace {

// The largest power of ten tried to make a row's coefficients integral.
constexpr std::int64_t kMaxScale = 1'000'000;
// How close to an integer a scaled coefficient must be.
constexpr double kIntegralTol = 1e-9;
// A row's scaled weights must add up below this, exact in a double and with
// room in an `int64_t` for the diagrams' open interval ends.
constexpr std::int64_t kMaxWeightSum = std::int64_t{1} << 52;
constexpr std::int64_t kUnbounded = std::int64_t{1} << 62;

// A diagram's terminals, which are not literals.
constexpr Lit kTrueNode = -1;
constexpr Lit kFalseNode = -2;

enum class EncodeStatus { kEncoded, kInfeasible, kRefused, kStopped };

struct Item {
    Lit lit;
    std::int64_t weight;
};

// Encodes `sum weight * lit <= bound` over positive weights as a reduced ordered
// BDD (Een and Sorensson 2006), which one construction covers at-most-one,
// cardinality and weighted rows with: a node is the suffix of the items from
// some position with some capacity left, and capacities that behave alike share
// a node, kept as intervals per position.  Heavier items go first, which is what
// keeps the diagram small.  Each node `y` means "this suffix fits": `y` implies
// its low child, and `y` and the item imply its high child.  The high child
// implies the low one, so these two clauses alone propagate completely (Abio et
// al. 2012, monotonic BDDs).
class RowEncoder {
public:
    RowEncoder(SatSolver& solver, size_t max_clauses)
        : solver_(solver), max_clauses_(max_clauses) {}

    EncodeStatus encode_leq(std::span<Item> items, std::int64_t bound);

    [[nodiscard]] size_t clauses() const { return clauses_; }
    [[nodiscard]] size_t aux_vars() const { return aux_vars_; }
    [[nodiscard]] size_t work() const { return work_; }

private:
    struct Result {
        Lit node;
        std::int64_t lo;
        std::int64_t hi;
    };
    struct Frame {
        size_t pos;
        std::int64_t capacity;
        int stage;
        Result low;
    };

    bool add(std::span<const Lit> clause);
    bool build_diagram(std::span<const Item> items, std::int64_t bound, Result& root);
    bool lookup(size_t pos, std::int64_t capacity, Result& out) const;
    bool make_node(const Frame& frame, const Item& item, const Result& low, const Result& high,
                   Result& out);

    SatSolver& solver_;
    size_t max_clauses_;
    size_t clauses_ = 0;
    size_t aux_vars_ = 0;
    size_t work_ = 0;
    std::vector<std::vector<Result>> levels_;
    std::vector<std::int64_t> suffix_;
    std::vector<Frame> stack_;
    std::vector<Lit> clause_;
};

bool RowEncoder::add(std::span<const Lit> clause) {
    if (clauses_ >= max_clauses_) {
        return false;
    }
    ++clauses_;
    solver_.add_clause(clause);
    return true;
}

EncodeStatus RowEncoder::encode_leq(std::span<Item> items, std::int64_t bound) {
    if (bound < 0) {
        solver_.add_clause({});
        return EncodeStatus::kInfeasible;
    }
    // An item heavier than the whole capacity cannot be true.
    size_t kept = 0;
    std::int64_t total = 0;
    std::int64_t divisor = 0;
    std::int64_t lightest = kUnbounded;
    for (const Item& item : items) {
        if (item.weight > bound) {
            const Lit unit = item.lit ^ 1;
            if (!add({&unit, 1})) {
                return EncodeStatus::kRefused;
            }
            continue;
        }
        items[kept++] = item;
        total += item.weight;
        divisor = std::gcd(divisor, item.weight);
        lightest = std::min(lightest, item.weight);
    }
    if (total <= bound) {
        return EncodeStatus::kEncoded;
    }
    items = items.first(kept);
    // True unless every item is: one clause, whatever the weights.
    if (total - lightest <= bound) {
        clause_.clear();
        for (const Item& item : items) {
            clause_.push_back(item.lit ^ 1);
        }
        return add(clause_) ? EncodeStatus::kEncoded : EncodeStatus::kRefused;
    }
    for (Item& item : items) {
        item.weight /= divisor;
    }
    std::ranges::sort(items, [](const Item& a, const Item& b) {
        return a.weight != b.weight ? a.weight > b.weight : a.lit < b.lit;
    });
    Result root{};
    if (!build_diagram(items, bound / divisor, root)) {
        return EncodeStatus::kRefused;
    }
    return add({&root.node, 1}) ? EncodeStatus::kEncoded : EncodeStatus::kRefused;
}

bool RowEncoder::lookup(size_t pos, std::int64_t capacity, Result& out) const {
    if (capacity < 0) {
        out = {kFalseNode, -kUnbounded, -1};
        return true;
    }
    if (capacity >= suffix_[pos]) {
        out = {kTrueNode, suffix_[pos], kUnbounded};
        return true;
    }
    const auto& level = levels_[pos];
    const auto it = std::ranges::upper_bound(level, capacity, {}, &Result::lo);
    if (it != level.begin() && std::prev(it)->hi >= capacity) {
        out = *std::prev(it);
        return true;
    }
    return false;
}

bool RowEncoder::make_node(const Frame& frame, const Item& item, const Result& low,
                           const Result& high, Result& out) {
    out.lo = std::max(low.lo, high.lo + item.weight);
    out.hi = std::min(low.hi, high.hi + item.weight);
    if (low.node == high.node) {
        out.node = low.node;
    } else {
        // The low child keeps this node's capacity, so it cannot be false; the
        // two children differ, so this node is not the true terminal.
        if (clauses_ + 2 > max_clauses_) {
            return false;
        }
        const Lit node = make_lit(solver_.new_var());
        ++aux_vars_;
        if (low.node != kTrueNode) {
            const std::array<Lit, 2> implies_low{node ^ 1, low.node};
            add(implies_low);
        }
        if (high.node == kFalseNode) {
            const std::array<Lit, 2> excludes_item{node ^ 1, item.lit ^ 1};
            add(excludes_item);
        } else {
            const std::array<Lit, 3> implies_high{node ^ 1, item.lit ^ 1, high.node};
            add(implies_high);
        }
        out.node = node;
    }
    auto& level = levels_[frame.pos];
    const auto at = std::ranges::upper_bound(level, out.lo, {}, &Result::lo);
    level.insert(at, out);
    return true;
}

// Iterative because a row can be as long as the model is wide, which a worker
// thread's stack would not survive as recursion.
bool RowEncoder::build_diagram(std::span<const Item> items, std::int64_t bound, Result& root) {
    const size_t n = items.size();
    suffix_.assign(n + 1, 0);
    for (size_t k = n; k > 0; --k) {
        suffix_[k - 1] = suffix_[k] + items[k - 1].weight;
    }
    if (levels_.size() < n + 1) {
        levels_.resize(n + 1);
    }
    for (size_t k = 0; k <= n; ++k) {
        levels_[k].clear();
    }
    work_ += n;

    Result ret{};
    stack_.clear();
    stack_.push_back({0, bound, 0, {}});
    while (!stack_.empty()) {
        const size_t top = stack_.size() - 1;
        Frame& frame = stack_[top];
        const size_t pos = frame.pos;
        const std::int64_t capacity = frame.capacity;
        if (frame.stage == 0) {
            if (lookup(pos, capacity, ret)) {
                stack_.pop_back();
                continue;
            }
            frame.stage = 1;
            stack_.push_back({pos + 1, capacity, 0, {}});
        } else if (frame.stage == 1) {
            frame.low = ret;
            frame.stage = 2;
            stack_.push_back({pos + 1, capacity - items[pos].weight, 0, {}});
        } else {
            ++work_;
            const Result high = ret;
            if (!make_node(frame, items[pos], frame.low, high, ret)) {
                return false;
            }
            stack_.pop_back();
        }
    }
    root = ret;
    return true;
}

bool is_fixed(double lower, double upper) {
    return lower == upper;
}

bool is_pure_binary(const ProblemView& p) {
    for (HighsInt j = 0; j < p.ncol; ++j) {
        const double lower = (*p.col_lower)[j];
        const double upper = (*p.col_upper)[j];
        if (is_fixed(lower, upper)) {
            continue;
        }
        if ((*p.integrality)[j] == HighsVarType::kContinuous || lower < 0.0 || upper > 1.0) {
            return false;
        }
    }
    return true;
}

// The smallest power of ten making every coefficient integral, or 0 if none up
// to `kMaxScale` does.  `kIntegralTol` is why an `kInfeasible` is about the
// scaled row: a coefficient within it of an integer is taken to be that integer.
double integer_scale(std::span<const double> values) {
    for (std::int64_t scale = 1; scale <= kMaxScale; scale *= 10) {
        const auto factor = static_cast<double>(scale);
        const bool integral = std::ranges::all_of(values, [factor](double a) {
            const double v = a * factor;
            return std::abs(v - std::round(v)) <= kIntegralTol;
        });
        if (integral) {
            return factor;
        }
    }
    return 0.0;
}

class ProblemEncoder {
public:
    ProblemEncoder(const ProblemView& p, SatSolver& solver, size_t max_clauses)
        : p_(p), rows_(solver, max_clauses) {}

    EncodeStatus encode(const std::function<bool()>& stop, SatRefusal& refusal);

    [[nodiscard]] const RowEncoder& rows() const { return rows_; }

private:
    // `sum sign * a[k] x[k] <= rhs` for row `i`.
    EncodeStatus encode_side(HighsInt i, double sign, double rhs, SatRefusal& refusal);

    const ProblemView& p_;
    RowEncoder rows_;
    std::vector<Item> items_;
    std::vector<double> values_;
};

EncodeStatus ProblemEncoder::encode_side(HighsInt i, double sign, double rhs, SatRefusal& refusal) {
    items_.clear();
    values_.clear();
    double residual = rhs;
    for (HighsInt k = (*p_.ar_start)[i]; k < (*p_.ar_start)[i + 1]; ++k) {
        const HighsInt j = (*p_.ar_index)[k];
        const double a = sign * (*p_.ar_value)[k];
        const double lower = (*p_.col_lower)[j];
        if (is_fixed(lower, (*p_.col_upper)[j])) {
            residual -= a * lower;
            continue;
        }
        values_.push_back(a);
        items_.push_back({make_lit(static_cast<std::int32_t>(j)), 0});
    }
    const double scale = integer_scale(values_);
    if (scale == 0.0) {
        refusal = SatRefusal::kNotIntegral;
        return EncodeStatus::kRefused;
    }
    // Half the feasibility tolerance forgives rounding noise in the right-hand
    // side, and leaves the check against the original rows room to pass.
    auto bound = std::floor((residual + (0.5 * p_.feastol)) * scale);
    double weight_sum = 0.0;
    for (size_t k = 0; k < items_.size(); ++k) {
        const double w = std::round(values_[k] * scale);
        weight_sum += std::abs(w);
        // A negative weight on `x` is a positive one on its negation.
        if (w < 0.0) {
            items_[k].lit ^= 1;
            bound -= w;
        }
        items_[k].weight = static_cast<std::int64_t>(std::abs(w));
    }
    if (weight_sum >= static_cast<double>(kMaxWeightSum)) {
        refusal = SatRefusal::kNotIntegral;
        return EncodeStatus::kRefused;
    }
    // A bound past the sum of weights says the same as the sum, and any negative
    // one the same as -1, so the cast below cannot overflow on a huge right-hand
    // side.
    bound = std::clamp(bound, -1.0, weight_sum);
    const EncodeStatus status = rows_.encode_leq(items_, static_cast<std::int64_t>(bound));
    if (status == EncodeStatus::kRefused) {
        refusal = SatRefusal::kTooLarge;
    }
    return status;
}

EncodeStatus ProblemEncoder::encode(const std::function<bool()>& stop, SatRefusal& refusal) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    for (HighsInt i = 0; i < p_.nrow; ++i) {
        if (stop && stop()) {
            return EncodeStatus::kStopped;
        }
        const double upper = (*p_.row_upper)[i];
        const double lower = (*p_.row_lower)[i];
        if (upper < kInf) {
            const EncodeStatus status = encode_side(i, 1.0, upper, refusal);
            if (status != EncodeStatus::kEncoded) {
                return status;
            }
        }
        if (lower > -kInf) {
            const EncodeStatus status = encode_side(i, -1.0, -lower, refusal);
            if (status != EncodeStatus::kEncoded) {
                return status;
            }
        }
    }
    return EncodeStatus::kEncoded;
}

bool satisfies_rows(const ProblemView& p, const std::vector<double>& x) {
    for (HighsInt i = 0; i < p.nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = (*p.ar_start)[i]; k < (*p.ar_start)[i + 1]; ++k) {
            lhs += (*p.ar_value)[k] * x[(*p.ar_index)[k]];
        }
        if (is_row_violated(lhs, (*p.row_lower)[i], (*p.row_upper)[i], p.feastol)) {
            return false;
        }
    }
    return true;
}

}  // namespace

SatSearchResult sat_search(const ProblemView& problem, const SatSearchConfig& config,
                           const std::function<bool()>& stop, std::vector<double>& point) {
    SatSearchResult result;
    if (!is_pure_binary(problem)) {
        result.status = SatSearchStatus::kRefused;
        result.refusal = SatRefusal::kNotPureBinary;
        return result;
    }
    SatSolver solver(static_cast<std::int32_t>(problem.ncol), config.seed);
    ProblemEncoder encoder(problem, solver, config.max_clauses);
    const EncodeStatus encoded = encoder.encode(stop, result.refusal);
    result.clauses = encoder.rows().clauses();
    result.aux_vars = encoder.rows().aux_vars();
    result.effort = encoder.rows().work() + problem.nnz;
    if (encoded == EncodeStatus::kRefused) {
        result.status = SatSearchStatus::kRefused;
        return result;
    }
    if (encoded == EncodeStatus::kStopped) {
        result.status = SatSearchStatus::kStopped;
        return result;
    }
    const SatResult solved = solver.solve(config.limits, stop);
    result.conflicts = solver.conflicts();
    result.effort += solver.effort();
    if (solved == SatResult::kStopped) {
        result.status = SatSearchStatus::kStopped;
        return result;
    }
    if (solved == SatResult::kUnsatisfiable) {
        result.status = SatSearchStatus::kInfeasible;
        return result;
    }
    std::vector<double> x(static_cast<size_t>(problem.ncol));
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        const double lower = (*problem.col_lower)[j];
        const bool fixed = is_fixed(lower, (*problem.col_upper)[j]);
        const bool value = solver.model_value(static_cast<std::int32_t>(j));
        const double free_value = value ? 1.0 : 0.0;
        x[static_cast<size_t>(j)] = fixed ? lower : free_value;
    }
    if (!satisfies_rows(problem, x)) {
        result.status = SatSearchStatus::kRefused;
        result.refusal = SatRefusal::kCheckFailed;
        return result;
    }
    point = std::move(x);
    result.status = SatSearchStatus::kFeasible;
    return result;
}

}  // namespace sat
