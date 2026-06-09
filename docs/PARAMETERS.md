# Tunable `constexpr` Parameters

This document lists every `constexpr` in the codebase that a researcher
might want to tune. Parameters are organized by heuristic/subsystem.
File paths are relative to the repository root.

---

## FPR (Fix, Propagate, and Repair)

### `repair_iterations` — RepairSearch DFS node limit

- **File**: `src/fpr_core.h` (field of `FprConfig`, line 186)
- **Default**: `50`
- **Meaning**: Maximum number of DFS nodes expanded by `repair_search`
  (paper Fig. 5). The paper quotes 200; we cap at 50 because
  RepairSearch runs two full PropEngine fixpoints per node, which
  dominates cost on tight instances (~760k coefficient accesses on
  9k-nnz LPs). 200 nodes can burn ~1.4 s regardless of the effort cap
  (see `bench/FPR_REPAIR_SEARCH_LOCKS.md`).
- **Suggested range**: 10–200. Raise on fast instances or when
  RepairSearch quality matters; lower on dense LPs where each node is
  expensive.

---

### `walksat_iterations` — WalkSAT step limit

- **File**: `src/fpr_core.h` (field of `FprConfig`, line 191)
- **Default**: `200`
- **Meaning**: Maximum number of WalkSAT repair steps (paper Fig. 4,
  loop bound). Kept at the paper's value because each step is cheap
  (O(row degree) coefficient accesses) and the RepairSearch blow-up
  rationale does not apply.
- **Suggested range**: 50–1000. Increasing helps on highly infeasible
  starting points; decreasing speeds up fast-feasible instances.

---

### `repair_noise` — WalkSAT random-walk probability

- **File**: `src/fpr_core.h` (field of `FprConfig`, line 181)
- **Default**: `0.75`
- **Meaning**: Probability of taking a random move rather than a greedy
  (minimum-damage) move in `walksat_select_move` (paper Fig. 4, line
  17). Paper default is 0.75. Greedy probability = 1 − `repair_noise`.
- **Suggested range**: 0.5–0.95. Lower values (more greedy) can work
  better on structured instances; higher values add diversification on
  hard instances.

---

### `kBox` — artificial bounding box for infinite bounds

- **File**: `src/fpr_core.cpp` (anonymous namespace, line 62)
- **Default**: `1e5`
- **Meaning**: When a variable has an unbounded side (lb = −∞ or ub =
  +∞), `finite_clamp_helper` maps the variable into a box of width
  `kBox` anchored at the finite bound (or `[−kBox, +kBox]` if both
  sides are infinite). Paper specifies `[−100000, +100000]`.
- **Suggested range**: 1e3–1e6. Smaller values keep the DFS closer to
  the feasible region on unbounded models; larger values allow more
  diverse initial points.

---

### `kInitialFprConfigs` — curated (strategy, mode) rotation

- **File**: `src/fpr.cpp` (anonymous namespace, lines 151–161)
- **Default**: 8 entries — `{BadobjclDfs, Locks2Dfs, Locks2Dive,
  LocksDfsrep, BadobjclDfsrep, RandomDiveprop, LocksRepairSearch,
  DomsizeDfs}`
- **Meaning**: Paper Section 6.3 Class 1 LP-free configs. Each
  `FprWorker` cycles through this list keyed on
  `(worker_idx + attempt_idx) % kNumInitialFprConfigs`. Adding or
  reordering entries changes which strategies are explored first and
  how they interleave across workers.
- **Note**: The full 8×5 (strategy × mode) grid is not used because
  `(kStratDomsize, kRepairSearch)` exposes a latent activity-undo gap
  in `repair_search`'s secondary backtrack (see comment in
  `fpr.cpp:select_config_for_current_attempt`). Widening the rotation
  requires that gap to be fixed first.

---

### `kMaxAttemptsPerCall` — multi-attempt fill cap per epoch

- **File**: `src/fpr.cpp` (`FprWorker::run_epoch`, line 291)
- **Default**: `32`
- **Meaning**: Maximum number of new FPR attempts started within a
  single `run_epoch` call. Guards against degenerate models where
  attempts verdict near-instantly (e.g. `infeasible-mip0`), which
  would otherwise fill the epoch budget purely with
  `fpr_attempt_begin` setup overhead.
- **Suggested range**: 8–64. Larger values let fast workers fill the
  epoch budget completely; smaller values reduce setup churn on
  degenerate models.

---

### `kNumInitialFprConfigs` — size of the curated rotation

- **File**: `src/fpr.cpp` (line 161)
- **Default**: `8` (derived from `std::size(kInitialFprConfigs)`)
- **Meaning**: Number of distinct (strategy, mode) pairs in the
  `kInitialFprConfigs` rotation. Changing the array changes this
  automatically.

---

## FPR-LP (`fpr_lp`)

### `kHardRandomizationLimit` — per-worker hard attempt restart cap

- **File**: `src/fpr_lp.cpp` (`LpFprWorker`, line 292)
- **Default**: `50`
- **Meaning**: After this many consecutive stale epochs (no improvement
  and no arm switch) the worker forces a new random seed, resetting its
  LP arm assignment. Prevents a worker from replaying the same arm
  forever on degenerate instances.
- **Suggested range**: 20–200.

---

### `kStaleEpochThreshold` — staleness trigger for randomization

- **File**: `src/fpr_lp.cpp` (`LpFprWorker`, line 295)
- **Default**: `3`
- **Meaning**: Number of consecutive stale epochs before incrementing
  the randomization counter. Lower values trigger diversification
  sooner.
- **Suggested range**: 1–10.

---

### `kNumLpArms` — total LP-dependent FPR arms

- **File**: `src/fpr_lp.cpp` (line 68)
- **Default**: `10` (`kNumClass2=4` + `kNumClass3a=2` + `kNumClass3b=4`)
- **Meaning**: Total number of LP-arm configs across Classes 2, 3a, 3b.
  Workers are assigned `w % kNumLpArms`; excess workers wrap around
  with distinct seeds.

---

## LocalMIP

### `kViolTol` — violation tolerance for constraint classification

- **File**: `src/local_mip_caches.h` (line 12)
- **Default**: `5e-7`
- **Meaning**: Threshold below which a constraint's violation is
  considered zero (used to classify rows into `violated` vs
  `satisfied`). Tighter than HiGHS's default feasibility tolerance
  to avoid misclassifying nearly-satisfied rows.
- **Suggested range**: 1e-8–1e-6. Tighten for higher accuracy;
  loosen to accept near-feasible solutions faster.

---

### `kRestartInterval` — steps between weight-based restarts

- **File**: `src/local_mip_caches.h` (line 13)
- **Default**: `200000`
- **Meaning**: Every `kRestartInterval` steps (measured in search
  iterations) the worker considers resetting its solution and weights.
  Higher values allow more exploration before resetting.
- **Suggested range**: 50000–500000.

---

### `kTermCheckInterval` — termination check period

- **File**: `src/local_mip_caches.h` (line 14)
- **Default**: `1000`
- **Meaning**: Interval (in steps) between checks for termination
  conditions (time limit, effort budget). Finer values add overhead but
  improve deadline precision.
- **Suggested range**: 100–10000.

---

### `kActivityPeriod` — weight smoothing period

- **File**: `src/local_mip_caches.h` (line 15)
- **Default**: `100000`
- **Meaning**: Interval (in steps) at which the PAWS-style weight
  smoothing is evaluated. Controls how often the weighting scheme
  adapts to constraint difficulty.
- **Suggested range**: 10000–500000.

---

### `kSmoothProb` — PAWS smoothing probability

- **File**: `src/local_mip_caches.h` (line 16)
- **Default**: `3e-4`
- **Meaning**: Probability applied each step of choosing to smooth
  (weaken) constraint weights rather than strengthen them (paper §4.1
  PAWS update). With probability `1 - kSmoothProb` the algorithm
  strengthens weights on violated constraints; with `kSmoothProb` it
  weakens weights on satisfied constraints.
- **Suggested range**: 1e-5–1e-2.

---

### `kBmsConstraints` — BMS sample size (violated constraints)

- **File**: `src/local_mip_caches.h` (line 17)
- **Default**: `12`
- **Meaning**: Number of violated constraints selected as the "best"
  (by weight) from a preliminary sample of `kBmsConstraints * 3`
  candidates, following the paper's BMS (Best-move Selection) operator.
  Determines the scope of the move search.
- **Suggested range**: 4–32.

---

### `kBmsBudget` — BMS candidate variable budget

- **File**: `src/local_mip_caches.h` (line 18)
- **Default**: `2250`
- **Meaning**: Maximum number of variable–move candidates evaluated per
  infeasible step from the BMS violated-constraint sample. Caps the
  inner candidate-generation loop.
- **Suggested range**: 500–10000. Higher values improve move quality at
  the cost of more coefficient accesses per step.

---

### `kBmsSatCon` — satisfied-constraint BMS sample count

- **File**: `src/local_mip_caches.h` (line 19)
- **Default**: `1`
- **Meaning**: Number of randomly sampled satisfied constraints used in
  Phase 2 of `infeasible_step` (paper Algorithm 2, lines 7–8). Raises
  move diversity at the cost of more coefficient access.
- **Suggested range**: 1–5.

---

### `kBmsSatBudget` — satisfied-constraint variable budget

- **File**: `src/local_mip_caches.h` (line 20)
- **Default**: `80`
- **Meaning**: Maximum variable–move candidates generated from the
  satisfied-constraint sample per step (Phase 2 cap).
- **Suggested range**: 20–500.

---

### `kBoolFlipBudget` — Boolean flip scan budget

- **File**: `src/local_mip_caches.h` (line 21)
- **Default**: `5000`
- **Meaning**: Maximum number of binary variables scanned for flipping
  in Phase 3 of `infeasible_step` (paper Algorithm 2, lines 9–11).
  When there are fewer than `kBoolFlipBudget` binary variables all are
  scanned; otherwise a random-offset window of this size is used.
- **Suggested range**: 500–20000.

---

### `kEasyBudget` — random easy-move fallback count

- **File**: `src/local_mip_caches.h` (line 22)
- **Default**: `5`
- **Meaning**: Number of randomly chosen variables tried in the Phase 6
  "easy moves" fallback (engineering extension to Algorithm 2). Provides
  a last-resort candidate when all earlier phases fail.
- **Suggested range**: 1–20.

---

### `kTabuBase` — base tabu tenure

- **File**: `src/local_mip_caches.h` (line 23)
- **Default**: `3`
- **Meaning**: Minimum number of steps a variable's move direction is
  forbidden after being flipped (tabu tenure base). Actual tenure is
  `kTabuBase + uniform(0, kTabuVar)`.
- **Suggested range**: 1–20.

---

### `kTabuVar` — tabu tenure random variation

- **File**: `src/local_mip_caches.h` (line 24)
- **Default**: `10`
- **Meaning**: Range of randomness added to the tabu tenure:
  `tabu_len = kTabuBase + rng() % kTabuVar`. Higher values make tenure
  more variable, diversifying the search.
- **Suggested range**: 0–50.

---

### `kFeasibleRecheckPeriod` — feasibility recheck interval

- **File**: `src/local_mip_caches.h` (line 25)
- **Default**: `100`
- **Meaning**: Number of feasible-mode steps between full LHS rechecks
  (`full_recheck`). Incremental updates keep the LHS cache consistent
  between rechecks; full rechecks guard against accumulated floating-
  point drift. Lower values are safer; higher values reduce overhead.
- **Suggested range**: 10–1000.

---

### `kFeasiblePlateau` — feasible-mode plateau detection threshold

- **File**: `src/local_mip_caches.h` (line 33)
- **Default**: `5000`
- **Meaning**: Number of feasible-mode steps without an improving move
  before triggering a random-walk perturbation (engineering extension to
  paper §4.1). After `kFeasiblePlateau` plateau steps the worker
  perturbs its solution and resets weights to escape the local optimum.
- **Suggested range**: 1000–20000.

---

### `kFeasibleMaxRandomWalks` — perturbation cap per worker

- **File**: `src/local_mip_caches.h` (line 34)
- **Default**: `20`
- **Meaning**: Maximum number of random-walk perturbations a single
  worker may attempt before declaring itself finished. Prevents infinite
  looping on pathological instances where perturbation cannot break the
  plateau.
- **Suggested range**: 5–100.

---

### `kEpsZero` — numerical zero threshold

- **File**: `src/local_mip_caches.h` (line 35)
- **Default**: `1e-15`
- **Meaning**: Values with absolute magnitude below `kEpsZero` are
  treated as zero throughout LocalMIP (objective coefficients, move
  deltas). Changing this can affect which variables are considered
  cost-free.

---

### `kPerturbBinaryFraction` — binary perturbation probability

- **File**: `src/local_mip_worker.h` (line 18)
- **Default**: `0.2`
- **Meaning**: Probability that any given integer variable is perturbed
  during a random-walk perturbation step. For binary variables: flip
  with this probability. For general integers: shift by a random amount
  within the domain with this probability. Applies to both LocalMIP's
  `perturb_solution` and Scylla's `pump::perturb`.
- **Suggested range**: 0.05–0.5.

---

### `kConstructionEffortFraction` — cold-start construction budget fraction

- **File**: `src/local_mip_construction.h` (line 54)
- **Default**: `0.10`
- **Meaning**: Fraction of the total worker effort budget allocated to
  the Phase B greedy construction sweep (cold-start, when neither FJ
  nor FPR has produced an incumbent). 10% leaves the bulk of the budget
  for the search loop. Phase A (zero-start) is always performed before
  this.
- **Suggested range**: 0.05–0.20. Raise if the greedy sweep reliably
  produces a tighter starting point worth the cost; lower if
  construction rarely helps on your instance set.

---

### `kMaxTightPerVar` — tight-delta candidate limit per variable

- **File**: `src/local_mip_construction.cpp` (anonymous namespace,
  line 314)
- **Default**: `4`
- **Meaning**: Maximum number of currently-violated rows from which a
  "tight delta" candidate is derived for each variable during the
  construction sweep. Caps the per-variable candidate-generation cost
  at O(col_nnz) coefficient accesses.
- **Suggested range**: 2–10.

---

## Scylla (Feasibility Pump)

### `kAlpha` — objective blending decay factor

- **File**: `src/pump_common.h` (line 14)
- **Default**: `0.9`
- **Meaning**: Per-iteration multiplicative decay applied to `α_K` in
  the modified objective `α_K * c + (1 - α_K) * rounding_term` (Mexi
  et al. 2023 Algorithm 1.1). Smaller values decay the original
  objective faster, biasing the pump toward pure integrality rounding.
- **Suggested range**: 0.8–0.99.

---

### `kEpsilonInit` — initial PDLP tolerance

- **File**: `src/pump_common.h` (line 15)
- **Default**: `0.01`
- **Meaning**: Starting tolerance for the PDLP approximate LP solver.
  Each iteration the tolerance decays by `kBeta` until it reaches
  `kEpsilonFloor`. Larger initial values allow faster but less accurate
  early solves.
- **Suggested range**: 1e-4–0.1.

---

### `kBeta` — PDLP tolerance decay factor

- **File**: `src/pump_common.h` (line 16)
- **Default**: `0.98`
- **Meaning**: Per-iteration multiplicative decay applied to the PDLP
  solve tolerance `ε`. The sequence is `ε_{K+1} = max(kBeta * ε_K,
  kEpsilonFloor)`. Closer to 1.0 means slower tightening.
- **Suggested range**: 0.9–0.999.

---

### `kEpsilonFloor` — minimum PDLP tolerance

- **File**: `src/pump_common.h` (line 17)
- **Default**: `1e-8`
- **Meaning**: Floor value for the PDLP solve tolerance. Once `ε` decays
  to this level it stays there for the remainder of the pump.
- **Suggested range**: 1e-10–1e-6.

---

### `kCycleWindow` — cycling detection history depth

- **File**: `src/pump_common.h` (line 18)
- **Default**: `3`
- **Meaning**: Number of past rounded solutions kept in the cycle-
  detection history. A new rounded solution that matches any of the
  last `kCycleWindow` solutions triggers perturbation (Mexi et al.
  Algorithm 1.1, line 13).
- **Suggested range**: 2–10. Larger values catch slower cycles but
  increase memory use and comparison cost.

---

### `kPerturbFraction` — cycling perturbation rate

- **File**: `src/pump_common.h` (line 19)
- **Default**: `0.2`
- **Meaning**: Fraction of integer variables perturbed when cycling is
  detected (Algorithm 1.1 line 14). Each integer variable is perturbed
  independently with probability `kPerturbFraction`.
  Also used by LocalMIP's perturbation (`kPerturbBinaryFraction` in
  `src/local_mip_worker.h` is the corresponding constant for LocalMIP
  and is set to the same value, 0.2).
- **Suggested range**: 0.05–0.5.

---

### `kCycleTol` — cycling detection tolerance

- **File**: `src/pump_common.h` (line 20)
- **Default**: `0.5`
- **Meaning**: Maximum allowed difference in any integer variable's
  value between the current rounded solution and a historical solution
  for the two to be considered identical. Since integer variables differ
  by at least 1.0, `0.5` is the correct binary threshold.
- **Note**: Changing this value would affect which solutions are
  considered cycles. It should remain at 0.5 for correct binary-integer
  semantics.

---

### `kMaxPdlpStalls` — PDLP zero-iteration stall limit

- **File**: `src/pump_common.h` (line 21)
- **Default**: `3`
- **Meaning**: Number of consecutive PDLP solve calls that return 0
  iterations before the ScyllaWorker declares itself finished. Guards
  against infinite stalling when the LP solver converges immediately
  (e.g., trivially feasible LP).
- **Suggested range**: 1–10.

---

### `kMaxStaleRoundsDefault` — default stale-snapshot cap per worker

- **File**: `src/scylla_worker.h` (line 32)
- **Default**: `4`
- **Meaning**: Default number of consecutive stale-snapshot rounds a
  ScyllaWorker may take before it must force a fresh blocking PDLP
  solve. A stale round rounds against a peer's cached LP solution
  without solving; too many stale rounds in a row risks stagnation on
  a degenerate snapshot. Scaled up by `compute_max_stale_rounds` for
  large LPs (see `kNnzPerExtraStaleRound`).
- **Suggested range**: 2–8 (as base; effective cap may be higher for
  large LPs).

---

### `kMaxStaleRoundsMin` — minimum stale-snapshot cap

- **File**: `src/scylla_worker.h` (line 33)
- **Default**: `2`
- **Meaning**: Floor applied by `compute_max_stale_rounds`. Even on very
  small LPs, each worker is allowed at least 2 stale rounds before
  forcing a fresh solve.

---

### `kMaxStaleRoundsMax` — maximum stale-snapshot cap

- **File**: `src/scylla_worker.h` (line 34)
- **Default**: `16`
- **Meaning**: Ceiling applied by `compute_max_stale_rounds`. On very
  large LPs (PDLP solve may take seconds), workers are allowed up to 16
  stale rounds before being forced to block.
- **Suggested range**: 8–32 for very large instances.

---

### `kNnzPerExtraStaleRound` — nnz-per-extra stale round scale factor

- **File**: `src/scylla_worker.h` (line 43)
- **Default**: `83000`
- **Meaning**: For every `kNnzPerExtraStaleRound` nnz in the LP,
  `compute_max_stale_rounds` adds 1 extra allowed stale round above the
  default. Calibrated so a 1M-nnz LP reaches approximately the
  `kMaxStaleRoundsMax` ceiling (= 4 + 1M / 83000 ≈ 16).
- **Suggested range**: Decrease to force more frequent fresh solves on
  medium LPs; increase if PDLP solves on large LPs are dominating
  wall-clock time.

---

### `kNumFprConfigs` — number of distinct FPR rounding configs for Scylla

- **File**: `src/scylla_worker.h` (line 67)
- **Default**: `4`
- **Meaning**: Number of entries in `kFprConfigs` (the per-worker static
  FPR rounding strategy assignment). Workers `0..kNumFprConfigs-1` are
  assigned deterministically; additional workers draw pseudo-randomly.

---

## Thompson Sampler / Portfolio

### `kFjAlpha`, `kFprArmAlpha`, `kLocalMipAlpha`, `kScyllaAlpha` — Beta prior α values

- **File**: `src/portfolio.cpp` (lines 71–74)
- **Default**: All `1.0`
- **Meaning**: Initial `α` parameter of the Beta prior for each arm in
  the Thompson sampler (`β` is always initialized to 1.0). The Beta
  prior `Beta(α, 1)` with `α=1` corresponds to a uniform prior over
  `[0, 1]`. Increasing `α` for an arm biases the sampler toward
  selecting that arm early (optimistic initialization).
- **Suggested range**: 0.5–5.0. Setting `α > 1` gives an arm a head
  start; `α < 1` gives a pessimistic start.

---

### `kEmaAlpha` — effort EMA smoothing factor

- **File**: `src/thompson_sampler.cpp` (`record_effort`, line 87)
- **Default**: `0.3`
- **Meaning**: Smoothing factor for the exponential moving average of
  per-arm effort in `select_effort_aware`. New observations are blended
  as `α * new + (1−α) * old`. Higher values give more weight to recent
  observations.
- **Suggested range**: 0.1–0.7.

---

### `kPoolCapacity` — solution pool size

- **File**: `src/solution_pool.h` (line 10)
- **Default**: `10`
- **Meaning**: Maximum number of distinct solutions stored in the shared
  `SolutionPool`. When full, a new solution replaces the worst entry
  (if better) or the most similar entry (if within
  `kDiversityObjTolerance` of the best and sufficiently diverse).
- **Suggested range**: 5–50. Larger pools provide more restart
  diversity but increase lock contention and crossover cost.

---

### `kDiversityObjTolerance` — diversity insertion objective tolerance

- **File**: `src/solution_pool.h` (line 14)
- **Default**: `0.10`
- **Meaning**: Maximum relative degradation in objective value that a
  diverse solution can have relative to the pool's current best and
  still be admitted (10%). A solution within 10% of the best objective
  may replace the most similar existing entry if its Hamming distance
  exceeds `kDiversityMinHammingFrac`.
- **Suggested range**: 0.0–0.5.

---

### `kDiversityMinHammingFrac` — minimum Hamming distance for diversity

- **File**: `src/solution_pool.h` (line 16)
- **Default**: `0.05`
- **Meaning**: A solution is considered structurally diverse if its
  Hamming distance (fraction of integer variables that differ) from all
  existing pool entries exceeds this threshold. Used to qualify
  solutions for the diversity-aware insertion path.
- **Suggested range**: 0.01–0.20.

---

### `kEpochsPerWorker` — epoch cadence for FPR, LocalMIP, Scylla

- **File**: `src/parallel_setup.h` (line 68)
- **Default**: `10`
- **Meaning**: Number of epochs each worker takes within its total
  budget in the deterministic epoch-gated runner. Smaller values
  synchronize workers more frequently (finer improvement broadcast
  across the epoch barrier) at the cost of more per-epoch overhead.
- **Suggested range**: 5–50.

---

### `kEpochsPerWorkerFj` — epoch cadence for FJ

- **File**: `src/parallel_setup.h` (line 75)
- **Default**: `20`
- **Meaning**: Epoch cadence for FeasibilityJump, kept separate from
  the unified `kEpochsPerWorker` because FJ's synchronization cadence
  affects pool-crossover behavior. Halving it to 10 was unvalidated as
  of the last MIPLIB benchmark; see issue #71.
- **Suggested range**: 10–40.

---

### `kBudgetCapMultiplier` — opportunistic arm budget cap multiplier

- **File**: `src/bandit_runner.h` (line 93)
- **Default**: `2.5`
- **Meaning**: In the opportunistic bandit loop, the per-arm call budget
  is capped at `kBudgetCapMultiplier * default_run_cap`. Prevents any
  single arm pull from consuming a disproportionate share of the total
  budget on a slow arm.
- **Suggested range**: 1.5–5.0.

---

## Budget Allocation Weights (mode_dispatch)

These weights tune each heuristic's share of the common effort budget
so that equal weights yield roughly equal wall-clock spend in seq/det
mode. They are calibrated against MIPLIB geomean `effort_per_ms` using
`bench/check_effort_drift.py`. Recalibrate after any change to effort
accounting.

### `kWeightFj` — FeasibilityJump budget weight

- **File**: `src/mode_dispatch.cpp` (line 101)
- **Default**: `1.54`
- **Meaning**: Proportional to FJ's geomean `effort_per_ms` (~403k
  coefficient-equivalent units/ms on the calibration set).

---

### `kWeightFpr` — FPR budget weight

- **File**: `src/mode_dispatch.cpp` (line 102)
- **Default**: `2.43`
- **Meaning**: Proportional to FPR's geomean `effort_per_ms` (~636k/ms).

---

### `kWeightLocalMip` — LocalMIP budget weight

- **File**: `src/mode_dispatch.cpp` (line 103)
- **Default**: `4.68`
- **Meaning**: Proportional to LocalMIP's geomean `effort_per_ms`
  (~1222k/ms, which includes the cold-start construction sweep as of
  issue #78). Largest weight because LocalMIP has the highest
  coefficient-access rate.

---

### `kWeightScylla` — Scylla budget weight

- **File**: `src/mode_dispatch.cpp` (line 104)
- **Default**: `1.00`
- **Meaning**: Normalized to 1.0 (slowest-per-effort heuristic, geomean
  ~261k/ms). Scylla's effort is measured in PDLP iters × nnz, a
  different unit than the other heuristics' coefficient accesses.

---

## Repair Search (`repair_search`)

### `kProgressThreshold` — no-progress trigger for best-open jump

- **File**: `src/repair_search.cpp` (anonymous namespace, line 298)
- **Default**: `10`
- **Meaning**: Number of consecutive RepairSearch DFS nodes without a
  violation improvement before the algorithm swaps to the lowest-
  violation open node (paper Fig. 5 best-first steering, line 27).
- **Suggested range**: 5–30.

---

## FJ Option Note

`mip_heuristic_run_feasibility_jump` is a **native HiGHS option**
(registered by HiGHS itself, default: `true`). The patch repurposes
it: when set to `true`, HiGHS's internal FeasibilityJump handler is
disabled and FJ runs through our presolve infrastructure (epoch-gated
or opportunistic, depending on `mip_heuristic_opportunistic`). Setting
it to `false` disables FJ entirely. It is **not** one of the custom
patch-added options — it is a pre-existing HiGHS option whose default
behavior is overridden by the patch.

The custom patch-added options are:
- `mip_heuristic_effort` — overall effort budget multiplier (default `0.30`)
- `mip_heuristic_run_fpr` — enable/disable FPR
- `mip_heuristic_run_local_mip` — enable/disable LocalMIP
- `mip_heuristic_run_scylla` — enable/disable Scylla
- `mip_heuristic_portfolio` — enable Thompson-sampling portfolio bandit
- `mip_heuristic_opportunistic` — use continuous (opportunistic) parallelism
