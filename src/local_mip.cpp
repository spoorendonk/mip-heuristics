#include "local_mip.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "heuristic_common.h"
#include "lp_data/HConst.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

namespace {

// --- File-scope constants ---
constexpr double kViolTol = 5e-7;
constexpr HighsInt kRestartInterval = 200000;
constexpr HighsInt kTermCheckInterval = 1000;
constexpr HighsInt kActivityPeriod = 100000;
constexpr double kSmoothProb = 3e-4;
constexpr HighsInt kBmsConstraints = 12;
constexpr HighsInt kBmsBudget = 2250;
constexpr HighsInt kBmsSatCon = 1;
constexpr HighsInt kBmsSatBudget = 80;
constexpr HighsInt kBoolFlipBudget = 5000;
constexpr HighsInt kEasyBudget = 5;
constexpr HighsInt kTabuBase = 3;
constexpr HighsInt kTabuVar = 10;
constexpr HighsInt kFeasibleRecheckPeriod = 100;
constexpr HighsInt kFeasiblePlateau = 5000;
constexpr double kEpsZero = 1e-15;

double compute_objective(const HighsLp *model,
                         const std::vector<double> &solution) {
  double obj = model->offset_;
  for (HighsInt j = 0; j < model->num_col_; ++j) {
    obj += model->col_cost_[j] * solution[j];
  }
  return obj;
}

// --- IndexedSet: O(1) add/remove with iteration ---
struct IndexedSet {
  std::vector<HighsInt> elements;
  std::vector<HighsInt> pos; // -1 = absent

  explicit IndexedSet(HighsInt n) : pos(n, -1) { elements.reserve(n); }

  void add(HighsInt i) {
    if (pos[i] != -1) {
      return;
    }
    pos[i] = static_cast<HighsInt>(elements.size());
    elements.push_back(i);
  }

  void remove(HighsInt i) {
    HighsInt p = pos[i];
    if (p == -1) {
      return;
    }
    HighsInt last = elements.back();
    elements[p] = last;
    pos[last] = p;
    elements.pop_back();
    pos[i] = -1;
  }

  bool contains(HighsInt i) const { return pos[i] != -1; }
  bool empty() const { return elements.empty(); }
  HighsInt size() const { return static_cast<HighsInt>(elements.size()); }
  HighsInt operator[](HighsInt idx) const { return elements[idx]; }

  void clear() {
    for (HighsInt e : elements) {
      pos[e] = -1;
    }
    elements.clear();
  }

  auto begin() const { return elements.begin(); }
  auto end() const { return elements.end(); }
};

// --- ViolCache: memoize row violations within a candidate batch ---
struct ViolCache {
  std::vector<double> cache;
  std::vector<HighsInt> used;
  static constexpr double kSentinel = -1.0;

  explicit ViolCache(HighsInt n) : cache(n, kSentinel) { used.reserve(n); }

  double get_or_compute(HighsInt i, double lhs_i, double row_lo_i,
                        double row_hi_i) {
    if (cache[i] >= 0.0) {
      return cache[i];
    }
    double v = row_violation(lhs_i, row_lo_i, row_hi_i);
    cache[i] = v;
    used.push_back(i);
    return v;
  }

  void reset() {
    for (HighsInt i : used) {
      cache[i] = kSentinel;
    }
    used.clear();
  }
};

// --- Candidate structs (hoisted from worker()) ---
struct Candidate {
  HighsInt var_idx = -1;
  double new_val = 0.0;
  double score = -std::numeric_limits<double>::infinity();
  double bonus = 0.0;
};

struct BatchCand {
  HighsInt var_idx;
  double new_val;
};

struct WeightedCon {
  HighsInt ci;
  uint64_t w;
};

// Forward declaration
struct WorkerCtx;

// --- LiftCache ---
struct LiftCache {
  std::vector<double> lo, hi, score;
  std::vector<bool> dirty;
  std::vector<HighsInt> dirty_list;
  bool all_dirty = true;
  std::vector<HighsInt> positive_list;
  std::vector<bool> in_positive;
  const std::vector<HighsInt> *costed_vars = nullptr;

  explicit LiftCache(HighsInt ncol)
      : lo(ncol), hi(ncol), score(ncol), dirty(ncol, true),
        in_positive(ncol, false) {
    dirty_list.reserve(ncol);
    positive_list.reserve(ncol);
  }

  void mark_dirty(HighsInt j) {
    if (!dirty[j]) {
      dirty[j] = true;
      dirty_list.push_back(j);
    }
  }

  void mark_all_dirty() {
    all_dirty = true;
    dirty_list.clear();
    std::fill(dirty.begin(), dirty.end(), true);
    positive_list.clear();
    std::fill(in_positive.begin(), in_positive.end(), false);
  }

  void recompute_one(HighsInt j, WorkerCtx &ctx);
  void recompute_all(WorkerCtx &ctx);
};

// --- WorkerCtx: central context for the local search worker ---
struct WorkerCtx {
  // Model refs
  const HighsLp *model;
  const std::vector<HighsInt> &ARstart;
  const std::vector<HighsInt> &ARindex;
  const std::vector<double> &ARvalue;
  const std::vector<double> &col_lb;
  const std::vector<double> &col_ub;
  const std::vector<double> &col_cost;
  const std::vector<double> &row_lo;
  const std::vector<double> &row_hi;
  const std::vector<HighsVarType> &integrality;
  const CscMatrix &csc;
  const double feastol;
  const double epsilon;
  const bool minimize;
  const HighsInt ncol;
  const HighsInt nrow;
  HighsMipSolverData *mipdata;

  // Mutable state
  std::vector<double> solution;
  std::vector<double> lhs;
  std::vector<uint64_t> weight;
  uint64_t obj_weight = 1;
  double current_obj = 0.0;

  // Sub-structures
  IndexedSet violated;
  IndexedSet satisfied;
  ViolCache viol_cache;
  LiftCache lift;

  // Tabu
  std::vector<HighsInt> tabu_inc_until;
  std::vector<HighsInt> tabu_dec_until;

  // Reusable buffers
  std::vector<BatchCand> batch;
  std::vector<WeightedCon> sampled;

  // Feasibility tracking
  bool was_infeasible = true;
  HighsInt feasible_recheck_counter = 0;

  // Effort tracking (coefficient accesses)
  size_t effort = 0;

  WorkerCtx(HighsMipSolver &mipsolver, const CscMatrix &csc_)
      : model(mipsolver.model_), ARstart(mipsolver.mipdata_->ARstart_),
        ARindex(mipsolver.mipdata_->ARindex_),
        ARvalue(mipsolver.mipdata_->ARvalue_),
        col_lb(mipsolver.model_->col_lower_),
        col_ub(mipsolver.model_->col_upper_),
        col_cost(mipsolver.model_->col_cost_),
        row_lo(mipsolver.model_->row_lower_),
        row_hi(mipsolver.model_->row_upper_),
        integrality(mipsolver.model_->integrality_), csc(csc_),
        feastol(mipsolver.mipdata_->feastol),
        epsilon(mipsolver.mipdata_->epsilon),
        minimize(mipsolver.model_->sense_ == ObjSense::kMinimize),
        ncol(mipsolver.model_->num_col_), nrow(mipsolver.model_->num_row_),
        mipdata(mipsolver.mipdata_.get()), solution(ncol), lhs(nrow),
        weight(nrow, 1), violated(nrow), satisfied(nrow), viol_cache(nrow),
        lift(ncol), tabu_inc_until(ncol, 0), tabu_dec_until(ncol, 0) {
    batch.reserve(kBmsBudget);
    sampled.reserve(static_cast<size_t>(kBmsConstraints) * 3);
  }

  bool is_int(HighsInt j) const { return ::is_integer(integrality, j); }

  double clamp_and_round(HighsInt j, double val) const {
    return clamp_round(val, col_lb[j], col_ub[j], is_int(j));
  }

  double compute_violation(HighsInt i, double l) const {
    return row_violation(l, row_lo[i], row_hi[i]);
  }

  bool is_violated(HighsInt i, double l) const {
    return l > row_hi[i] + feastol || l < row_lo[i] - feastol;
  }

  bool is_equality(HighsInt i) const {
    return row_lo[i] == row_hi[i] && row_lo[i] > -kHighsInf;
  }

  bool is_tabu(HighsInt j, double delta, HighsInt step) const {
    if (delta > 0 && step < tabu_inc_until[j]) {
      return true;
    }
    if (delta < 0 && step < tabu_dec_until[j]) {
      return true;
    }
    return false;
  }

  void update_violated(HighsInt i) {
    double viol = compute_violation(i, lhs[i]);
    bool was = violated.contains(i);
    bool now = (viol > kViolTol);
    if (now && !was) {
      violated.add(i);
      satisfied.remove(i);
    } else if (!now && was) {
      violated.remove(i);
      if (!is_equality(i)) {
        satisfied.add(i);
      }
    }
  }

  void apply_move(HighsInt j, double new_val) {
    double old_val = solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < kEpsZero) {
      return;
    }
    solution[j] = new_val;
    current_obj += col_cost[j] * delta;
    effort += csc.col_start[j + 1] - csc.col_start[j];
    // Only maintain LiftCache during feasible mode; on the
    // infeasible→feasible transition, rebuild_state marks all dirty.
    bool dirty_lift = !was_infeasible && !lift.all_dirty;
    if (dirty_lift) {
      lift.mark_dirty(j);
    }
    for (HighsInt p = csc.col_start[j]; p < csc.col_start[j + 1]; ++p) {
      HighsInt i = csc.col_row[p];
      lhs[i] += csc.col_val[p] * delta;
      update_violated(i);
      if (dirty_lift) {
        for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
          lift.mark_dirty(ARindex[k]);
        }
      }
    }
  }

  void apply_move_with_tabu(HighsInt j, double new_val, HighsInt step,
                            std::mt19937 &rng) {
    double delta = new_val - solution[j];
    apply_move(j, new_val);
    HighsInt tabu_len = kTabuBase + static_cast<HighsInt>(rng() % kTabuVar);
    if (delta > 0) {
      tabu_dec_until[j] = step + tabu_len;
    } else {
      tabu_inc_until[j] = step + tabu_len;
    }
  }

  void rebuild_state() {
    was_infeasible = true;
    feasible_recheck_counter = 0;
    violated.clear();
    satisfied.clear();
    effort += ARindex.size(); // full O(nnz) LHS recomputation
    for (HighsInt i = 0; i < nrow; ++i) {
      double l = 0.0;
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        l += ARvalue[k] * solution[ARindex[k]];
      }
      lhs[i] = l;
      if (compute_violation(i, l) > kViolTol) {
        violated.add(i);
      } else if (!is_equality(i)) {
        satisfied.add(i);
      }
    }
    lift.mark_all_dirty();
    current_obj = compute_objective(model, solution);
  }

  double compute_tight_delta(HighsInt i, HighsInt j, double coeff) const {
    if (std::abs(coeff) < kEpsZero) {
      return 0.0;
    }
    double l = lhs[i];
    double gap;
    // NOLINTBEGIN(bugprone-branch-clone) — same expression form, different
    // bounds
    if (l > row_hi[i] + feastol) {
      gap = l - row_hi[i]; // upper violated
    } else if (l < row_lo[i] - feastol) {
      gap = l - row_lo[i]; // lower violated
    } else {
      // Satisfied: push toward the nearest bound
      double gap_hi = (row_hi[i] < kHighsInf) ? (l - row_hi[i]) : kHighsInf;
      double gap_lo = (row_lo[i] > -kHighsInf) ? (l - row_lo[i]) : kHighsInf;
      gap = (std::abs(gap_hi) <= std::abs(gap_lo)) ? gap_hi : gap_lo;
    }
    // NOLINTEND(bugprone-branch-clone)

    double delta = -gap / coeff;

    if (is_equality(i)) {
      if (is_int(j)) {
        delta = (coeff > 0) ? std::floor(delta) : std::ceil(delta);
      }
      double new_val = solution[j] + delta;
      if (new_val < col_lb[j] || new_val > col_ub[j]) {
        if ((gap > 0 && coeff > 0) || (gap < 0 && coeff < 0)) {
          delta = col_lb[j] - solution[j];
        } else {
          delta = col_ub[j] - solution[j];
        }
      }
    } else {
      // Paper Eq 5: integer rounding depends on coefficient sign.
      if (is_int(j)) {
        delta = (coeff > 0) ? std::floor(delta) : std::ceil(delta);
      }
      // Clamp to variable bounds (Paper Eq 5 min/max with l_j, u_j).
      double new_val = solution[j] + delta;
      if (new_val < col_lb[j]) {
        delta = col_lb[j] - solution[j];
      } else if (new_val > col_ub[j]) {
        delta = col_ub[j] - solution[j];
      }
    }
    return delta;
  }

  // Paper Section 4.1: weighting scheme for MIP.
  // Called when at a local optimum (no positive operation found).
  void update_weights(std::mt19937 &rng, bool is_feasible, bool best_feasible,
                      double best_obj) {
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    if (coin(rng) >= kSmoothProb) {
      // With probability 1 - sp: strengthen
      if (is_feasible) {
        obj_weight += 1;
      } else {
        for (auto ci : violated) {
          weight[ci] += 1;
        }
      }
    } else {
      // With probability sp: smooth (weaken)
      bool obj_better = best_feasible && (minimize ? (current_obj < best_obj)
                                                   : (current_obj > best_obj));
      if (obj_better && obj_weight > 1) {
        obj_weight -= 1;
      }
      for (auto ci : satisfied) {
        if (weight[ci] > 1) {
          weight[ci] -= 1;
        }
      }
    }
  }
};

// --- LiftCache method implementations ---

void LiftCache::recompute_one(HighsInt j, WorkerCtx &ctx) {
  double old_score = score[j];
  if (std::abs(ctx.col_cost[j]) < kEpsZero) {
    score[j] = 0.0;
    dirty[j] = false;
    if (old_score > 0.0 && in_positive[j]) {
      in_positive[j] = false;
      // lazy removal: stale entries filtered during scan
    }
    return;
  }
  // Compute lift bounds
  ctx.effort += ctx.csc.col_start[j + 1] - ctx.csc.col_start[j];
  double lo_j = ctx.col_lb[j];
  double hi_j = ctx.col_ub[j];
  for (HighsInt p = ctx.csc.col_start[j]; p < ctx.csc.col_start[j + 1]; ++p) {
    HighsInt i = ctx.csc.col_row[p];
    double coeff = ctx.csc.col_val[p];
    if (std::abs(coeff) < kEpsZero) {
      continue;
    }
    double residual = ctx.lhs[i] - coeff * ctx.solution[j];
    if (ctx.row_hi[i] < kHighsInf) {
      double bound = (ctx.row_hi[i] - residual) / coeff;
      if (coeff > 0) {
        hi_j = std::min(hi_j, bound);
      } else {
        lo_j = std::max(lo_j, bound);
      }
    }
    if (ctx.row_lo[i] > -kHighsInf) {
      double bound = (ctx.row_lo[i] - residual) / coeff;
      if (coeff > 0) {
        lo_j = std::max(lo_j, bound);
      } else {
        hi_j = std::min(hi_j, bound);
      }
    }
  }
  if (ctx.is_int(j)) {
    lo_j = std::ceil(lo_j - ctx.feastol);
    hi_j = std::floor(hi_j + ctx.feastol);
  }
  lo[j] = lo_j;
  hi[j] = hi_j;

  if (lo_j > hi_j) {
    score[j] = 0.0;
  } else {
    double target;
    if (ctx.minimize) {
      target = (ctx.col_cost[j] > 0) ? lo_j : hi_j;
    } else {
      target = (ctx.col_cost[j] > 0) ? hi_j : lo_j;
    }
    target = ctx.clamp_and_round(j, target);
    if (std::abs(target - ctx.solution[j]) < kEpsZero) {
      score[j] = 0.0;
    } else {
      double obj_delta = ctx.col_cost[j] * (target - ctx.solution[j]);
      if (!ctx.minimize) {
        obj_delta = -obj_delta;
      }
      score[j] = -obj_delta; // positive = improving
    }
  }
  // Maintain positive-lift list
  if (score[j] > 0.0) {
    if (!in_positive[j]) {
      in_positive[j] = true;
      positive_list.push_back(j);
    }
  } else {
    if (in_positive[j]) {
      in_positive[j] = false;
      // lazy removal: stale entries filtered during scan
    }
  }
  dirty[j] = false;
}

void LiftCache::recompute_all(WorkerCtx &ctx) {
  if (all_dirty) {
    // Only recompute columns with nonzero cost; zero-cost columns
    // always have score=0 and never need lift recomputation.
    if (costed_vars) {
      for (HighsInt j : *costed_vars) {
        recompute_one(j, ctx);
      }
    } else {
      for (HighsInt j = 0; j < ctx.ncol; ++j) {
        recompute_one(j, ctx);
      }
    }
    dirty_list.clear();
  } else {
    for (HighsInt j : dirty_list) {
      if (dirty[j]) {
        recompute_one(j, ctx);
      }
    }
    dirty_list.clear();
  }
  all_dirty = false;
}

// --- Candidate selection free functions ---

// Paper Definitions 5-10: two-level scoring function.
// Progress score (level 1): discrete constraint-transition scores + objective.
// Bonus score (level 2): breakthrough bonus + robustness bonus.
std::pair<double, double>
compute_candidate_scores(WorkerCtx &ctx, HighsInt j, double new_val,
                         bool best_feasible, double best_obj) {
  double old_val = ctx.solution[j];
  double delta = new_val - old_val;
  if (std::abs(delta) < kEpsZero) {
    return {-std::numeric_limits<double>::infinity(), 0.0};
  }

  ctx.effort += ctx.csc.col_start[j + 1] - ctx.csc.col_start[j];

  // Def 5: progress score for objective
  double obj_delta = ctx.col_cost[j] * delta;
  double new_obj = ctx.current_obj + obj_delta;
  double eps = ctx.epsilon;
  double progress = 0.0;
  if ((!ctx.minimize && new_obj > ctx.current_obj + eps) ||
      (ctx.minimize && new_obj < ctx.current_obj - eps)) {
    progress += static_cast<double>(ctx.obj_weight); // objective improved
  } else if ((!ctx.minimize && new_obj < ctx.current_obj - eps) ||
             (ctx.minimize && new_obj > ctx.current_obj + eps)) {
    progress -= static_cast<double>(ctx.obj_weight); // objective worsened
  }

  // Def 8: breakthrough bonus (beats best-found solution)
  double bonus = 0.0;
  if (best_feasible) {
    bool beats_best = ctx.minimize ? (new_obj < best_obj - eps)
                                   : (new_obj > best_obj + eps);
    if (beats_best) {
      bonus += static_cast<double>(ctx.obj_weight);
    }
  }

  // Defs 6-7, 9-10: constraint progress + robustness
  for (HighsInt p = ctx.csc.col_start[j]; p < ctx.csc.col_start[j + 1]; ++p) {
    HighsInt i = ctx.csc.col_row[p];
    double coeff = ctx.csc.col_val[p];
    double old_lhs = ctx.lhs[i];
    double new_lhs = old_lhs + coeff * delta;
    double old_viol =
        ctx.viol_cache.get_or_compute(i, old_lhs, ctx.row_lo[i], ctx.row_hi[i]);
    double new_viol = ctx.compute_violation(i, new_lhs);
    double w = static_cast<double>(ctx.weight[i]);

    // Def 6: constraint progress score
    bool was_viol = (old_viol > kViolTol);
    bool now_viol = (new_viol > kViolTol);
    if (was_viol && !now_viol) {
      progress += w; // violated → satisfied
    } else if (!was_viol && now_viol) {
      progress -= w; // satisfied → violated
    } else if (was_viol && now_viol) {
      if (new_viol < old_viol - kViolTol) {
        progress += w; // still violated, improved
      } else if (new_viol > old_viol + kViolTol) {
        progress -= w; // still violated, worsened
      }
    }

    // Def 9: robustness bonus — only for transitions into strictly
    // satisfied (was violated or tight, now strictly interior).
    if (!now_viol) {
      bool old_strict =
          !was_viol &&
          (ctx.row_hi[i] >= kHighsInf ||
           old_lhs < ctx.row_hi[i] - ctx.feastol) &&
          (ctx.row_lo[i] <= -kHighsInf ||
           old_lhs > ctx.row_lo[i] + ctx.feastol);
      if (!old_strict) {
        bool new_strict =
            (ctx.row_hi[i] >= kHighsInf ||
             new_lhs < ctx.row_hi[i] - ctx.feastol) &&
            (ctx.row_lo[i] <= -kHighsInf ||
             new_lhs > ctx.row_lo[i] + ctx.feastol);
        if (new_strict) {
          bonus += w;
        }
      }
    }
  }

  return {progress, bonus};
}

bool is_aspiration(const WorkerCtx &ctx, HighsInt j, double new_val,
                   double best_obj, bool best_feasible) {
  if (!best_feasible) {
    return false;
  }
  double delta = new_val - ctx.solution[j];
  double obj_delta = ctx.col_cost[j] * delta;
  double new_obj = ctx.current_obj + obj_delta;
  return ctx.minimize ? (new_obj < best_obj - ctx.epsilon)
                      : (new_obj > best_obj + ctx.epsilon);
}

double compute_breakthrough_delta(const WorkerCtx &ctx, HighsInt j,
                                  double cur_obj, double best_obj) {
  double obj_coeff = ctx.col_cost[j];
  if (std::abs(obj_coeff) < kEpsZero) {
    return 0.0;
  }

  double obj_gap = cur_obj - best_obj;
  if (!ctx.minimize) {
    obj_gap = -obj_gap;
  }

  double delta = -obj_gap / obj_coeff;

  if (ctx.is_int(j)) {
    delta = (obj_coeff > 0) ? std::floor(delta) : std::ceil(delta);
  }
  double new_val = ctx.solution[j] + delta;
  if (new_val < ctx.col_lb[j] || new_val > ctx.col_ub[j]) {
    delta = (obj_coeff > 0) ? (ctx.col_lb[j] - ctx.solution[j])
                            : (ctx.col_ub[j] - ctx.solution[j]);
  }
  return delta;
}

Candidate select_best_from_batch(WorkerCtx &ctx, std::vector<BatchCand> &batch,
                                 HighsInt step, bool aspiration,
                                 double best_obj, bool best_feasible) {
  Candidate best;
  for (const auto &c : batch) {
    double delta = c.new_val - ctx.solution[c.var_idx];
    if (std::abs(delta) < kEpsZero) {
      continue;
    }

    if (ctx.is_tabu(c.var_idx, delta, step)) {
      if (!(aspiration && is_aspiration(ctx, c.var_idx, c.new_val, best_obj,
                                        best_feasible))) {
        continue;
      }
    }

    auto [prog, bon] = compute_candidate_scores(ctx, c.var_idx, c.new_val,
                                                best_feasible, best_obj);

    if (prog > best.score + kViolTol) {
      best = {c.var_idx, c.new_val, prog, bon};
    } else if (prog > best.score - kViolTol) {
      if (bon > best.bonus) {
        best = {c.var_idx, c.new_val, prog, bon};
      }
    }
  }
  ctx.viol_cache.reset();
  return best;
}

void append_candidate(WorkerCtx &ctx, std::vector<BatchCand> &batch, HighsInt j,
                      double delta) {
  double new_val = ctx.clamp_and_round(j, ctx.solution[j] + delta);
  if (std::abs(new_val - ctx.solution[j]) < kEpsZero) {
    return;
  }
  batch.push_back({j, new_val});
}

// --- infeasible_step: candidate generation following paper's Algorithm 2 ---
//
// Phase ordering (Algorithm 2):
// 1. MTM in violated (+ BM if post-feasible)
// 2. MTM in satisfied constraints (Alg 2 lines 7-8)
// 3. Boolean flips (Alg 2 lines 9-11)
// 4. Weight update + random constraint fallback (Alg 2 lines 12-14)
// Additional (our engineering additions):
// 5. Perturbation (generalizes Boolean flip to non-binary)
// 6. Easy moves

Candidate infeasible_step(WorkerCtx &ctx, std::mt19937 &rng, HighsInt step,
                          bool best_feasible, double best_objective,
                          const std::vector<HighsInt> &costed_vars,
                          const std::vector<HighsInt> &binary_vars) {
  ctx.was_infeasible = true;

  auto &batch = ctx.batch;
  auto &sampled = ctx.sampled;

  // --- Phase 1: BMS tight moves from violated constraints ---
  HighsInt num_to_sample = std::min(kBmsConstraints * 3, ctx.violated.size());
  HighsInt num_to_keep = std::min(kBmsConstraints, ctx.violated.size());

  sampled.clear();
  if (num_to_sample == ctx.violated.size()) {
    for (auto ci : ctx.violated) {
      sampled.push_back({ci, ctx.weight[ci]});
    }
  } else {
    for (HighsInt s = 0; s < num_to_sample; ++s) {
      HighsInt idx = static_cast<HighsInt>(rng() % ctx.violated.size());
      sampled.push_back({ctx.violated[idx], ctx.weight[ctx.violated[idx]]});
    }
  }

  if (static_cast<HighsInt>(sampled.size()) > num_to_keep) {
    std::partial_sort(
        sampled.begin(), sampled.begin() + num_to_keep, sampled.end(),
        [](const WeightedCon &a, const WeightedCon &b) { return a.w > b.w; });
    sampled.resize(num_to_keep);
  }

  batch.clear();
  HighsInt budget_remaining = kBmsBudget;

  for (auto &[ci, w] : sampled) {
    (void)w;
    if (budget_remaining <= 0) {
      break;
    }
    for (HighsInt k = ctx.ARstart[ci];
         k < ctx.ARstart[ci + 1] && budget_remaining > 0; ++k) {
      HighsInt j = ctx.ARindex[k];
      --budget_remaining;
      double delta = ctx.compute_tight_delta(ci, j, ctx.ARvalue[k]);
      append_candidate(ctx, batch, j, delta);
    }
  }

  // --- Phase 1b: Breakthrough moves (only post-feasible, Alg 2 line 5-6) ---
  if (best_feasible) {
    for (HighsInt j : costed_vars) {
      double delta =
          compute_breakthrough_delta(ctx, j, ctx.current_obj, best_objective);
      append_candidate(ctx, batch, j, delta);
    }
  }

  Candidate cand =
      select_best_from_batch(ctx, batch, step, true, best_objective,
                             best_feasible);

  // If positive candidate found, done (Alg 2 lines 1-6)
  if (cand.var_idx != -1 && cand.score > kViolTol) {
    return cand;
  }

  // --- Phase 2: MTM in satisfied constraints (Alg 2 lines 7-8) ---
  if (!ctx.satisfied.empty()) {
    batch.clear();
    HighsInt num_sat_sample = std::min(kBmsSatCon, ctx.satisfied.size());
    HighsInt sat_budget = kBmsSatBudget;
    for (HighsInt s = 0; s < num_sat_sample && sat_budget > 0; ++s) {
      HighsInt ci = ctx.satisfied[rng() % ctx.satisfied.size()];
      for (HighsInt k = ctx.ARstart[ci];
           k < ctx.ARstart[ci + 1] && sat_budget > 0; ++k) {
        HighsInt j = ctx.ARindex[k];
        --sat_budget;
        double delta = ctx.compute_tight_delta(ci, j, ctx.ARvalue[k]);
        append_candidate(ctx, batch, j, delta);
      }
    }
    auto sat_cand = select_best_from_batch(ctx, batch, step, false,
                                           best_objective, best_feasible);
    if (sat_cand.var_idx != -1 && sat_cand.score > cand.score + kViolTol) {
      cand = sat_cand;
    }
  }

  if (cand.var_idx != -1 && cand.score > kViolTol) {
    return cand;
  }

  // --- Phase 3: Boolean flip (Alg 2 lines 9-11) ---
  if (!binary_vars.empty()) {
    batch.clear();
    HighsInt nbinary = static_cast<HighsInt>(binary_vars.size());
    HighsInt offset = (nbinary > 0) ? static_cast<HighsInt>(rng() % nbinary) : 0;
    for (HighsInt idx = 0; idx < nbinary && idx < kBoolFlipBudget; ++idx) {
      HighsInt j = binary_vars[(offset + idx) % nbinary];
      double new_val = (ctx.solution[j] < 0.5) ? 1.0 : 0.0;
      if (std::abs(new_val - ctx.solution[j]) < kEpsZero) {
        continue;
      }
      batch.push_back({j, new_val});
    }
    if (!batch.empty()) {
      auto flip_cand = select_best_from_batch(ctx, batch, step, true,
                                              best_objective, best_feasible);
      if (flip_cand.var_idx != -1 && flip_cand.score > cand.score + kViolTol) {
        cand = flip_cand;
      }
    }
  }

  if (cand.var_idx != -1 && cand.score > kViolTol) {
    return cand;
  }

  // --- Phase 4: Weight update + random constraint fallback (Alg 2 lines 12-14) ---
  ctx.update_weights(rng, /*is_feasible=*/false, best_feasible, best_objective);

  if (!ctx.violated.empty()) {
    batch.clear();
    HighsInt ci = ctx.violated[rng() % ctx.violated.size()];
    for (HighsInt k = ctx.ARstart[ci]; k < ctx.ARstart[ci + 1]; ++k) {
      HighsInt j = ctx.ARindex[k];
      double delta = ctx.compute_tight_delta(ci, j, ctx.ARvalue[k]);
      append_candidate(ctx, batch, j, delta);
    }
    // Breakthrough candidates already scored in Phase 1; skip re-scoring.
    auto fallback = select_best_from_batch(ctx, batch, step, false,
                                           best_objective, best_feasible);
    if (fallback.var_idx != -1 &&
        (cand.var_idx == -1 || fallback.score > cand.score + kViolTol ||
         (fallback.score > cand.score - kViolTol &&
          fallback.bonus > cand.bonus))) {
      cand = fallback;
    }
  }

  if (cand.var_idx != -1) {
    return cand;
  }

  // --- Phase 5: Perturbation (our addition, last resort) ---
  if (!ctx.violated.empty()) {
    HighsInt ci = ctx.violated[rng() % ctx.violated.size()];
    HighsInt row_len = ctx.ARstart[ci + 1] - ctx.ARstart[ci];
    if (row_len > 0) {
      HighsInt k = ctx.ARstart[ci] + static_cast<HighsInt>(rng() % row_len);
      HighsInt j = ctx.ARindex[k];
      double new_val;
      if (ctx.mipdata->domain.isBinary(j)) {
        new_val = (ctx.solution[j] < 0.5) ? 1.0 : 0.0;
      } else if (ctx.is_int(j)) {
        HighsInt dir = (rng() % 2 == 0) ? 1 : -1;
        new_val = ctx.clamp_and_round(j, ctx.solution[j] + dir);
      } else {
        double range =
            std::min(ctx.col_ub[j], ctx.col_lb[j] + 1e6) - ctx.col_lb[j];
        double perturbation = std::uniform_real_distribution<double>(
            -0.1 * range, 0.1 * range)(rng);
        new_val = ctx.clamp_and_round(j, ctx.solution[j] + perturbation);
      }
      if (std::abs(new_val - ctx.solution[j]) > kEpsZero) {
        auto [prog, bon] = compute_candidate_scores(ctx, j, new_val,
                                                    best_feasible,
                                                    best_objective);
        ctx.viol_cache.reset();
        cand = {j, new_val, prog, bon};
      }
    }
  }

  if (cand.var_idx != -1) {
    return cand;
  }

  // --- Phase 6: Easy moves (our addition) ---
  {
    batch.clear();
    HighsInt num_easy = std::min(kEasyBudget, ctx.ncol);
    for (HighsInt s = 0; s < num_easy; ++s) {
      HighsInt j = static_cast<HighsInt>(rng() % ctx.ncol);
      double target;
      if (ctx.col_lb[j] > 0) {
        target = ctx.col_lb[j];
      } else if (ctx.col_ub[j] < 0) {
        target = ctx.col_ub[j];
      } else {
        target = 0.0;
      }
      append_candidate(ctx, batch, j, target - ctx.solution[j]);
      // Try: toward lower bound
      if (ctx.col_lb[j] > -1e15 && ctx.col_lb[j] < 0) {
        append_candidate(ctx, batch, j, ctx.col_lb[j] - ctx.solution[j]);
      }
      // Try: toward upper bound
      if (ctx.col_ub[j] < 1e15 && ctx.col_ub[j] > 0) {
        append_candidate(ctx, batch, j, ctx.col_ub[j] - ctx.solution[j]);
      }
      // Try: midpoint for continuous
      if (!ctx.is_int(j) && ctx.col_lb[j] > -1e15 && ctx.col_ub[j] < 1e15) {
        append_candidate(ctx, batch, j,
                         (ctx.col_lb[j] + ctx.col_ub[j]) * 0.5 -
                             ctx.solution[j]);
      }
    }
    auto easy_cand = select_best_from_batch(ctx, batch, step, false,
                                            best_objective, best_feasible);
    if (easy_cand.var_idx != -1) {
      cand = easy_cand;
    }
  }

  return cand;
}

} // namespace

namespace local_mip {

void run(HighsMipSolver &mipsolver, size_t max_effort) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  const HighsInt nrow = model->num_row_;
  if (ncol == 0 || nrow == 0) {
    return;
  }
  if (mipdata->incumbent.empty()) {
    return;
  }

  auto csc = build_csc(ncol, nrow, mipdata->ARstart_, mipdata->ARindex_,
                       mipdata->ARvalue_);
  std::mt19937 rng(mipdata->numImprovingSols + 137);

  auto result = worker(mipsolver, csc, rng, nullptr, max_effort);
  mipdata->heuristic_effort_used += result.effort;
  if (result.found_feasible) {
    mipdata->trySolution(result.solution, kSolutionSourceLocalMIP);
  }
}

HeuristicResult worker(HighsMipSolver &mipsolver, const CscMatrix &csc,
                       std::mt19937 &rng, const double *initial_solution,
                       size_t max_effort) {
  const HighsInt ncol = mipsolver.model_->num_col_;
  const HighsInt nrow = mipsolver.model_->num_row_;

  HeuristicResult result;
  if (ncol == 0 || nrow == 0) {
    return result;
  }

  WorkerCtx ctx(mipsolver, csc);

  // Best solution tracking
  bool best_feasible = false;
  double best_objective = ctx.minimize
                              ? std::numeric_limits<double>::infinity()
                              : -std::numeric_limits<double>::infinity();
  std::vector<double> best_solution(ncol);

  // Precompute variable subsets for breakthrough moves and Boolean flips
  auto *mipdata = ctx.mipdata;
  std::vector<HighsInt> costed_vars;
  std::vector<HighsInt> binary_vars;
  for (HighsInt j = 0; j < ncol; ++j) {
    if (std::abs(ctx.col_cost[j]) >= kEpsZero) {
      costed_vars.push_back(j);
    }
    if (mipdata->domain.isBinary(j)) {
      binary_vars.push_back(j);
    }
  }
  ctx.lift.costed_vars = &costed_vars;

  // --- Initialize solution ---
  const double *src = initial_solution
                          ? initial_solution
                          : (!mipdata->incumbent.empty()
                                 ? mipdata->incumbent.data()
                                 : nullptr);
  if (src) {
    for (HighsInt j = 0; j < ncol; ++j) {
      double v = src[j];
      if (ctx.is_int(j)) {
        v = std::round(v);
      }
      ctx.solution[j] = std::max(ctx.col_lb[j], std::min(ctx.col_ub[j], v));
    }
  } else {
    for (HighsInt j = 0; j < ncol; ++j) {
      if (mipdata->domain.isBinary(j)) {
        ctx.solution[j] = 0.0;
      } else if (ctx.is_int(j)) {
        double lo = std::max(ctx.col_lb[j], -1e8);
        double hi = std::min(ctx.col_ub[j], lo + 100.0);
        ctx.solution[j] =
            std::max(ctx.col_lb[j],
                     std::min(ctx.col_ub[j], std::round((lo + hi) * 0.5)));
      } else {
        double val = 0.0;
        if (ctx.col_lb[j] > 0.0) {
          val = ctx.col_lb[j];
        } else if (ctx.col_ub[j] < 0.0) {
          val = ctx.col_ub[j];
        }
        ctx.solution[j] = std::max(ctx.col_lb[j], std::min(ctx.col_ub[j], val));
      }
    }
  }

  // Build initial LHS and violated/satisfied lists
  ctx.rebuild_state();

  HighsInt steps_since_improvement = 0;
  HighsInt restart_count = 0;
  const double time_limit = mipsolver.options_mip_->time_limit;

  // --- Main loop ---
  for (HighsInt step = 0;; ++step) {
    if (step % kTermCheckInterval == 0) {
      if (mipdata->terminatorTerminated() ||
          mipsolver.timer_.read() >= time_limit) {
        break;
      }
      if (ctx.effort >= max_effort) {
        break;
      }
    }

    bool feasible_mode = ctx.violated.empty();

    if (feasible_mode) {
      // Full O(nnz) recheck on infeasible->feasible transition or periodically;
      // otherwise trust incremental lhs[] (O(nrow) check only).
      bool need_full_recheck =
          ctx.was_infeasible ||
          (ctx.feasible_recheck_counter % kFeasibleRecheckPeriod == 0);
      ctx.was_infeasible = false;
      ++ctx.feasible_recheck_counter;

      bool truly_feasible = true;
      if (need_full_recheck) {
        ctx.effort += ctx.ARindex.size();
        for (HighsInt i = 0; i < nrow; ++i) {
          double l = 0.0;
          for (HighsInt k = ctx.ARstart[i]; k < ctx.ARstart[i + 1]; ++k) {
            l += ctx.ARvalue[k] * ctx.solution[ctx.ARindex[k]];
          }
          ctx.lhs[i] = l;
          if (ctx.is_violated(i, l)) {
            truly_feasible = false;
            ctx.violated.add(i);
            ctx.satisfied.remove(i);
          }
        }
      }
      // When !need_full_recheck, trust incremental state: apply_move's
      // update_violated() already maintains the violated set for every
      // row touched by each move, so no row can become violated without
      // being caught.  The periodic full recheck guards against FP drift.
      if (!truly_feasible) {
        continue;
      }

      // Track best solution
      double obj = ctx.current_obj;
      bool improved = false;
      if (!best_feasible) {
        improved = true;
      } else if (ctx.minimize) {
        improved = (obj < best_objective - ctx.epsilon);
      } else {
        improved = (obj > best_objective + ctx.epsilon);
      }

      if (improved) {
        // Full recheck before recording best (guard against FP drift)
        if (!need_full_recheck) {
          ctx.effort += ctx.ARindex.size();
          bool still_ok = true;
          for (HighsInt i = 0; i < nrow; ++i) {
            double l = 0.0;
            for (HighsInt k = ctx.ARstart[i]; k < ctx.ARstart[i + 1]; ++k) {
              l += ctx.ARvalue[k] * ctx.solution[ctx.ARindex[k]];
            }
            ctx.lhs[i] = l;
            if (ctx.is_violated(i, l)) {
              still_ok = false;
              break;
            }
          }
          if (!still_ok) {
            ctx.rebuild_state();
            continue;
          }
        }
        best_feasible = true;
        best_objective = obj;
        best_solution = ctx.solution;
        steps_since_improvement = 0;
      }

      // Lift move: find variable giving best feasible objective improvement
      ctx.lift.recompute_all(ctx);
      Candidate lift_best;
      lift_best.score = 0.0; // must strictly improve
      // Compact stale entries and find best lift in a single pass
      {
        HighsInt write = 0;
        for (HighsInt read = 0;
             read < static_cast<HighsInt>(ctx.lift.positive_list.size());
             ++read) {
          HighsInt j = ctx.lift.positive_list[read];
          if (!ctx.lift.in_positive[j]) {
            continue;
          }
          ctx.lift.positive_list[write++] = j;
          if (ctx.lift.score[j] <= lift_best.score) {
            continue;
          }
          double lo = ctx.lift.lo[j], hi = ctx.lift.hi[j];
          if (lo > hi) {
            continue;
          }
          double target;
          if (ctx.minimize) {
            target = (ctx.col_cost[j] > 0) ? lo : hi;
          } else {
            target = (ctx.col_cost[j] > 0) ? hi : lo;
          }
          target = ctx.clamp_and_round(j, target);
          if (std::abs(target - ctx.solution[j]) < kEpsZero) {
            continue;
          }
          lift_best = {j, target, ctx.lift.score[j], 0.0};
        }
        ctx.lift.positive_list.resize(write);
      }

      if (lift_best.var_idx != -1) {
        ctx.apply_move_with_tabu(lift_best.var_idx, lift_best.new_val,
                                 step, rng);
      } else {
        ctx.update_weights(rng, /*is_feasible=*/true, best_feasible,
                           best_objective);
      }

      // Count every feasible step toward the plateau — only genuine
      // improvements (recorded above) reset the counter.  When stuck,
      // exit so the caller sees only the effort actually consumed.
      ++steps_since_improvement;
      if (steps_since_improvement >= kFeasiblePlateau) {
        break;
      }
    } else {
      // --- Infeasible mode ---
      Candidate cand = infeasible_step(ctx, rng, step, best_feasible,
                                       best_objective, costed_vars,
                                       binary_vars);

      // Apply move
      if (cand.var_idx != -1) {
        ctx.apply_move_with_tabu(cand.var_idx, cand.new_val, step, rng);
      }

      ++steps_since_improvement;
      if (ctx.violated.empty()) {
        steps_since_improvement = 0;
      }
    }

    // Activity refresh: recompute all LHS to prevent FP drift
    if (step % kActivityPeriod == 0 && step > 0) {
      ctx.rebuild_state();
    }

    // Restart logic
    if (steps_since_improvement >= kRestartInterval) {
      steps_since_improvement = 0;
      ++restart_count;

      // Try incumbent on odd restarts
      if (best_feasible && (restart_count % 2 == 1)) {
        ctx.solution = best_solution;
      } else {
        // Random restart
        for (HighsInt j = 0; j < ncol; ++j) {
          if (mipdata->domain.isBinary(j)) {
            ctx.solution[j] = (rng() % 2 == 0) ? 0.0 : 1.0;
          } else if (ctx.is_int(j)) {
            double lo = std::max(ctx.col_lb[j], -1e8);
            double hi = std::min(ctx.col_ub[j], lo + 100.0);
            ctx.solution[j] = std::max(
                ctx.col_lb[j],
                std::min(ctx.col_ub[j],
                         std::round(std::uniform_real_distribution<double>(
                             lo, hi)(rng))));
          } else {
            double lo = ctx.col_lb[j] > -kHighsInf ? ctx.col_lb[j] : -1e6;
            double hi = ctx.col_ub[j] < kHighsInf ? ctx.col_ub[j] : lo + 1e6;
            if (hi > lo) {
              ctx.solution[j] =
                  std::uniform_real_distribution<double>(lo, hi)(rng);
            } else {
              ctx.solution[j] = lo;
            }
          }
        }
      }

      ctx.rebuild_state();
      std::fill(ctx.tabu_inc_until.begin(), ctx.tabu_inc_until.end(), 0);
      std::fill(ctx.tabu_dec_until.begin(), ctx.tabu_dec_until.end(), 0);
    }
  }

  result.effort = ctx.effort;
  if (best_feasible) {
    result.found_feasible = true;
    result.objective = best_objective;
    result.solution = std::move(best_solution);
  }

  return result;
}

} // namespace local_mip
