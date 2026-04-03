#include "fpr_strategies.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "Highs.h"
#include "heuristic_common.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

// ===================================================================
// Variable ranking
// ===================================================================

namespace {

// Bucket binary, general-integer, and continuous variables.
struct TypeBuckets {
  std::vector<HighsInt> bin;
  std::vector<HighsInt> gen_int;
  std::vector<HighsInt> cont;
};

TypeBuckets bucket_by_type(const HighsMipSolver& mipsolver) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const HighsInt ncol = model->num_col_;
  TypeBuckets b;
  for (HighsInt j = 0; j < ncol; ++j) {
    if (!is_integer(model->integrality_, j)) {
      b.cont.push_back(j);
    } else if (mipdata->domain.isBinary(j)) {
      b.bin.push_back(j);
    } else {
      b.gen_int.push_back(j);
    }
  }
  return b;
}

std::vector<HighsInt> concat_buckets(TypeBuckets& b) {
  std::vector<HighsInt> order;
  order.reserve(b.bin.size() + b.gen_int.size() + b.cont.size());
  order.insert(order.end(), b.bin.begin(), b.bin.end());
  order.insert(order.end(), b.gen_int.begin(), b.gen_int.end());
  order.insert(order.end(), b.cont.begin(), b.cont.end());
  return order;
}

// --- LR: formulation order ---
std::vector<HighsInt> rank_lr(const HighsMipSolver& mipsolver) {
  const HighsInt ncol = mipsolver.model_->num_col_;
  std::vector<HighsInt> order(ncol);
  std::iota(order.begin(), order.end(), 0);
  return order;
}

// --- type: grouped by type, formulation order within ---
std::vector<HighsInt> rank_type(const HighsMipSolver& mipsolver) {
  auto b = bucket_by_type(mipsolver);
  return concat_buckets(b);
}

// --- random: type buckets, random within ---
std::vector<HighsInt> rank_random(const HighsMipSolver& mipsolver,
                                  std::mt19937& rng) {
  auto b = bucket_by_type(mipsolver);
  std::shuffle(b.bin.begin(), b.bin.end(), rng);
  std::shuffle(b.gen_int.begin(), b.gen_int.end(), rng);
  std::shuffle(b.cont.begin(), b.cont.end(), rng);
  return concat_buckets(b);
}

// --- locks: sorted by max(uplocks, downlocks) descending within type ---
std::vector<HighsInt> rank_locks(const HighsMipSolver& mipsolver) {
  auto* mipdata = mipsolver.mipdata_.get();
  const auto& up = mipdata->uplocks;
  const auto& dn = mipdata->downlocks;

  auto b = bucket_by_type(mipsolver);
  auto cmp = [&](HighsInt a, HighsInt b_idx) {
    return std::max(up[a], dn[a]) > std::max(up[b_idx], dn[b_idx]);
  };
  // Stable sort preserves formulation order as tiebreak
  std::stable_sort(b.bin.begin(), b.bin.end(), cmp);
  std::stable_sort(b.gen_int.begin(), b.gen_int.end(), cmp);
  // continuous locks are irrelevant for fixing order, but sort for consistency
  std::stable_sort(b.cont.begin(), b.cont.end(), cmp);
  return concat_buckets(b);
}

// --- typecl: clique cover for binaries, then type ---
// Paper's greedy algorithm: process equality cliques first, add disjoint ones,
// then remaining. Within each clique, formulation order.
std::vector<HighsInt> rank_typecl(const HighsMipSolver& mipsolver) {
  auto* mipdata = mipsolver.mipdata_.get();
  auto b = bucket_by_type(mipsolver);

  // Build CliqueVar vector for all binary variables (positive literal)
  using CV = HighsCliqueTable::CliqueVar;
  std::vector<CV> clq_vars;
  clq_vars.reserve(b.bin.size());
  for (HighsInt j : b.bin) {
    clq_vars.push_back(CV(j, 1));
  }

  if (clq_vars.empty()) {
    return concat_buckets(b);
  }

  std::vector<HighsInt> partition_start;
  mipdata->cliquetable.cliquePartition(clq_vars, partition_start);

  // Rebuild binary bucket: clique order, formulation order within each clique
  b.bin.clear();
  for (size_t c = 0; c + 1 < partition_start.size(); ++c) {
    HighsInt start = partition_start[c];
    HighsInt end = partition_start[c + 1];
    // Extract columns in this clique
    std::vector<HighsInt> clique_cols;
    for (HighsInt k = start; k < end; ++k) {
      clique_cols.push_back(static_cast<HighsInt>(clq_vars[k].col));
    }
    // Sort by formulation order within clique
    std::sort(clique_cols.begin(), clique_cols.end());
    b.bin.insert(b.bin.end(), clique_cols.begin(), clique_cols.end());
  }

  return concat_buckets(b);
}

// --- cliques: clique partition + analytic-center-weighted random sort ---
// Paper's Fig. 2: within each clique, sort using weighted discrete distribution
// based on analytic center values.
std::vector<HighsInt> rank_cliques(const HighsMipSolver& mipsolver,
                                   std::mt19937& rng,
                                   const double* lp_ref) {
  auto* mipdata = mipsolver.mipdata_.get();
  const auto& col_lb = mipsolver.model_->col_lower_;
  const auto& col_ub = mipsolver.model_->col_upper_;
  auto b = bucket_by_type(mipsolver);

  using CV = HighsCliqueTable::CliqueVar;
  std::vector<CV> clq_vars;
  clq_vars.reserve(b.bin.size());
  for (HighsInt j : b.bin) {
    clq_vars.push_back(CV(j, 1));
  }

  if (clq_vars.empty() || !lp_ref) {
    return rank_typecl(mipsolver);
  }

  std::vector<HighsInt> partition_start;
  mipdata->cliquetable.cliquePartition(clq_vars, partition_start);

  // Rebuild binary bucket with weighted random sort within each clique
  b.bin.clear();
  for (size_t c = 0; c + 1 < partition_start.size(); ++c) {
    HighsInt start = partition_start[c];
    HighsInt end = partition_start[c + 1];
    // Collect vars and weights (paper Fig. 2)
    std::vector<std::pair<HighsInt, double>> vars_weights;
    for (HighsInt k = start; k < end; ++k) {
      HighsInt col = static_cast<HighsInt>(clq_vars[k].col);
      // Skip fixed variables
      if (col_ub[col] - col_lb[col] < 1e-6) continue;
      // Weight = x_ac[col] for positive literal (val=1)
      double w = clq_vars[k].val ? lp_ref[col] : 1.0 - lp_ref[col];
      w = std::max(w, 1e-10);  // avoid zero weights
      vars_weights.push_back({col, w});
    }

    if (vars_weights.empty()) continue;

    // Weighted random sort: paper uses log(Rand(0,1))/w as sort key
    // This is the Gumbel-max trick for weighted sampling without replacement
    for (auto& [col, w] : vars_weights) {
      double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
      w = std::log(u) / w;
    }
    std::sort(vars_weights.begin(), vars_weights.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (const auto& [col, _] : vars_weights) {
      b.bin.push_back(col);
    }
  }

  return concat_buckets(b);
}

// --- cliques2: dynamic clique cover using LP solution (paper Fig. 3) ---
// For each clique, pick the most positive literal w.r.t. LP solution,
// then remaining uncovered binaries.
std::vector<HighsInt> rank_cliques2(const HighsMipSolver& mipsolver,
                                    const double* lp_ref) {
  auto* mipdata = mipsolver.mipdata_.get();
  auto b = bucket_by_type(mipsolver);

  if (b.bin.empty() || !lp_ref) {
    return rank_typecl(mipsolver);
  }

  using CV = HighsCliqueTable::CliqueVar;
  std::vector<CV> clq_vars;
  clq_vars.reserve(b.bin.size());
  for (HighsInt j : b.bin) {
    clq_vars.push_back(CV(j, 1));
  }

  // Use the objective-weighted clique partition variant
  // This sorts within each clique by LP-weighted preference
  std::vector<HighsInt> partition_start;
  std::vector<double> lp_vec(lp_ref, lp_ref + mipsolver.model_->num_col_);
  mipdata->cliquetable.cliquePartition(lp_vec, clq_vars, partition_start);

  // Paper Fig. 3: for each clique, pick best variable, then rest
  b.bin.clear();
  const auto& col_lb = mipsolver.model_->col_lower_;
  const auto& col_ub = mipsolver.model_->col_upper_;

  for (size_t c = 0; c + 1 < partition_start.size(); ++c) {
    HighsInt start = partition_start[c];
    HighsInt end = partition_start[c + 1];

    // Paper Fig. 3: compute sum of LP literal values for tightness check
    double sum = 0.0;
    HighsInt best_col = -1;
    double best_val = -1.0;

    for (HighsInt k = start; k < end; ++k) {
      HighsInt col = static_cast<HighsInt>(clq_vars[k].col);
      double v = clq_vars[k].val ? lp_ref[col] : 1.0 - lp_ref[col];
      sum += v;
      if (col_ub[col] - col_lb[col] < 1e-6) continue;
      if (v > best_val) {
        best_val = v;
        best_col = col;
      }
    }

    // Paper Fig. 3, line 24: only reorder if clique is LP-tight (sum ≈ 1)
    if (best_col >= 0 && sum >= 1.0 - 1e-6) {
      b.bin.push_back(best_col);
      for (HighsInt k = start; k < end; ++k) {
        HighsInt col = static_cast<HighsInt>(clq_vars[k].col);
        if (col != best_col) {
          b.bin.push_back(col);
        }
      }
    } else {
      // Not tight — keep original clique order
      for (HighsInt k = start; k < end; ++k) {
        b.bin.push_back(static_cast<HighsInt>(clq_vars[k].col));
      }
    }
  }

  return concat_buckets(b);
}

}  // namespace

std::vector<HighsInt> compute_var_order(const HighsMipSolver& mipsolver,
                                        VarStrategy strategy,
                                        std::mt19937& rng,
                                        const double* lp_ref) {
  switch (strategy) {
    case VarStrategy::kLR:
      return rank_lr(mipsolver);
    case VarStrategy::kType:
      return rank_type(mipsolver);
    case VarStrategy::kRandom:
      return rank_random(mipsolver, rng);
    case VarStrategy::kLocks:
      return rank_locks(mipsolver);
    case VarStrategy::kTypecl:
      return rank_typecl(mipsolver);
    case VarStrategy::kCliques:
      return rank_cliques(mipsolver, rng, lp_ref);
    case VarStrategy::kCliques2:
      return rank_cliques2(mipsolver, lp_ref);
  }
  return rank_type(mipsolver);  // unreachable
}

// ===================================================================
// Value selection
// ===================================================================

namespace {

double val_up(double ub) { return ub; }

double val_random(double lb, double ub, bool is_int, std::mt19937& rng) {
  double v = std::uniform_real_distribution<double>(lb, ub)(rng);
  if (is_int) v = std::round(v);
  return std::max(lb, std::min(ub, v));
}

double val_goodobj(double lb, double ub, bool minimize, double cost) {
  if (std::abs(cost) < 1e-15) {
    return lb;  // paper Section 4.2: "always pick the lower bound"
  }
  if (minimize) {
    return (cost > 0) ? lb : ub;
  }
  return (cost > 0) ? ub : lb;
}

double val_badobj(double lb, double ub, bool minimize, double cost) {
  if (std::abs(cost) < 1e-15) {
    return lb;  // paper Section 4.2: "always pick the lower bound"
  }
  // Opposite of goodobj
  if (minimize) {
    return (cost > 0) ? ub : lb;
  }
  return (cost > 0) ? lb : ub;
}

// LP-based probabilistic rounding (paper Section 4.2):
// v_j = ceil(x^LP_j) with probability f_j = frac(x^LP_j), else floor(x^LP_j)
double val_lp_based(double lb, double ub, bool is_int, double lp_val,
                    std::mt19937& rng) {
  double clamped = std::max(lb, std::min(ub, lp_val));
  if (!is_int) return clamped;

  double f = clamped - std::floor(clamped);
  double v;
  if (f < 1e-10) {
    v = std::round(clamped);
  } else {
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    v = (u < f) ? std::ceil(clamped) : std::floor(clamped);
  }
  return std::max(lb, std::min(ub, v));
}

// Dynamic locks (paper Section 4.2, "loosedyn"):
// Compute current min/max activity for each constraint the variable appears in.
// Count how many constraints would become infeasible (or tighter) if variable
// goes up vs down. Pick direction with fewer dynamic locks.
double val_loosedyn(HighsInt j, double lb, double ub, bool is_int,
                    bool minimize, double cost,
                    const HighsMipSolver& mipsolver, const VarState* vs,
                    const CscMatrix& csc) {
  const auto* model = mipsolver.model_;
  auto* mipdata = mipsolver.mipdata_.get();
  const auto& ARstart = mipdata->ARstart_;
  const auto& ARindex = mipdata->ARindex_;
  const auto& ARvalue = mipdata->ARvalue_;
  const auto& row_lo = model->row_lower_;
  const auto& row_hi = model->row_upper_;

  // Count dynamic up-locks and down-locks for variable j
  HighsInt up_locks = 0;
  HighsInt down_locks = 0;
  const HighsInt total_rows = csc.col_start[j + 1] - csc.col_start[j];

  for (HighsInt r = 0; r < total_rows; ++r) {
    HighsInt p = csc.col_start[j] + r;
    HighsInt i = csc.col_row[p];
    double a = csc.col_val[p];

    // Compute current min/max activity for this row from current VarState
    double min_act = 0.0, max_act = 0.0;
    for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
      HighsInt jj = ARindex[k];
      double aa = ARvalue[k];
      if (vs[jj].fixed) {
        min_act += aa * vs[jj].val;
        max_act += aa * vs[jj].val;
      } else {
        if (aa > 0) {
          min_act += aa * vs[jj].lb;
          max_act += aa * vs[jj].ub;
        } else {
          min_act += aa * vs[jj].ub;
          max_act += aa * vs[jj].lb;
        }
      }
    }

    // Check if constraint has become redundant (already satisfied for any value)
    bool has_upper = row_hi[i] < kHighsInf;
    bool has_lower = row_lo[i] > -kHighsInf;
    if (has_upper && min_act > row_hi[i]) {
      // Already infeasible — skip
    } else if (has_lower && max_act < row_lo[i]) {
      // Already infeasible — skip
    } else {
      // Not redundant: count locks
      if (has_upper && a > 0) ++up_locks;
      if (has_lower && a < 0) ++up_locks;
      if (has_upper && a < 0) ++down_locks;
      if (has_lower && a > 0) ++down_locks;
    }

    // Early exit: if one direction already wins by more than remaining rows
    HighsInt remaining = total_rows - r - 1;
    if (up_locks > down_locks + remaining) return lb;
    if (down_locks > up_locks + remaining) return ub;
  }

  // Pick direction with fewer locks
  if (up_locks < down_locks) {
    return ub;
  } else if (down_locks < up_locks) {
    return lb;
  }
  // Tie: fall back to objective direction
  return val_goodobj(lb, ub, minimize, cost);
}

}  // namespace

double choose_value(HighsInt j, double lb, double ub, bool is_int,
                    bool minimize, double cost, ValStrategy strategy,
                    std::mt19937& rng, const double* lp_ref,
                    const HighsMipSolver* mipsolver, const VarState* vs,
                    const CscMatrix* csc) {
  double v;
  switch (strategy) {
    case ValStrategy::kUp:
      v = val_up(ub);
      break;
    case ValStrategy::kRandom:
      v = val_random(lb, ub, is_int, rng);
      break;
    case ValStrategy::kGoodobj:
      v = val_goodobj(lb, ub, minimize, cost);
      break;
    case ValStrategy::kBadobj:
      v = val_badobj(lb, ub, minimize, cost);
      break;
    case ValStrategy::kLoosedyn:
      if (mipsolver && vs && csc) {
        v = val_loosedyn(j, lb, ub, is_int, minimize, cost, *mipsolver, vs,
                         *csc);
      } else {
        v = val_goodobj(lb, ub, minimize, cost);
      }
      break;
    case ValStrategy::kZerocore:
    case ValStrategy::kZerolp:
    case ValStrategy::kCore:
    case ValStrategy::kLp:
      if (lp_ref) {
        v = val_lp_based(lb, ub, is_int, lp_ref[j], rng);
      } else {
        v = val_goodobj(lb, ub, minimize, cost);
      }
      break;
    default:
      v = val_goodobj(lb, ub, minimize, cost);
      break;
  }
  if (is_int) v = std::round(v);
  return std::max(lb, std::min(ub, v));
}

// ===================================================================
// LP reference solutions
// ===================================================================

namespace {

// Solve an LP relaxation of the presolved MIP model.
// use_ipm: barrier solver (analytic center); otherwise simplex (vertex).
// run_crossover: false disables crossover (for analytic center).
// use_objective: true uses model cost; false uses zero objective.
std::vector<double> solve_lp_relaxation(const HighsMipSolver& mipsolver,
                                        bool use_ipm, bool run_crossover,
                                        bool use_objective) {
  const auto* model = mipsolver.model_;
  const auto& mipdata = *mipsolver.mipdata_;
  const HighsInt ncol = model->num_col_;

  HighsLp lp;
  lp.num_col_ = ncol;
  lp.num_row_ = model->num_row_;
  lp.col_lower_ = model->col_lower_;
  lp.col_upper_ = model->col_upper_;
  lp.row_lower_ = model->row_lower_;
  lp.row_upper_ = model->row_upper_;
  lp.a_matrix_.format_ = MatrixFormat::kRowwise;
  lp.a_matrix_.num_col_ = ncol;
  lp.a_matrix_.num_row_ = model->num_row_;
  lp.a_matrix_.start_ = mipdata.ARstart_;
  lp.a_matrix_.index_ = mipdata.ARindex_;
  lp.a_matrix_.value_ = mipdata.ARvalue_;

  if (use_objective) {
    lp.col_cost_ = model->col_cost_;
    lp.sense_ = model->sense_;
    lp.offset_ = model->offset_;
  } else {
    lp.col_cost_.assign(ncol, 0.0);
  }

  Highs highs;
  highs.setOptionValue("output_flag", false);
  highs.setOptionValue("time_limit", 30.0);
  if (use_ipm) {
    highs.setOptionValue("solver", "ipm");
  }
  if (!run_crossover) {
    highs.setOptionValue("run_crossover", "off");
  }

  highs.passModel(std::move(lp));
  highs.run();

  const auto& sol = highs.getSolution();
  if (static_cast<HighsInt>(sol.col_value.size()) == ncol) {
    return sol.col_value;
  }
  return {};
}

}  // namespace

std::vector<double> compute_analytic_center(const HighsMipSolver& mipsolver,
                                            bool use_objective) {
  return solve_lp_relaxation(mipsolver, /*use_ipm=*/true,
                             /*run_crossover=*/false, use_objective);
}

std::vector<double> compute_zero_obj_vertex(const HighsMipSolver& mipsolver) {
  return solve_lp_relaxation(mipsolver, /*use_ipm=*/false,
                             /*run_crossover=*/true, /*use_objective=*/false);
}
