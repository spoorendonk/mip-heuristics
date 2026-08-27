# Tunable `constexpr` Parameters

This document lists every `constexpr` in the codebase that a researcher
might want to tune. Parameters are organized by heuristic/subsystem.
File paths are relative to the repository root.

Entries name **symbols, not line numbers**. `bench/check_docs_refs.py` runs as
the `docs_parameter_references` ctest test and fails the suite when a
documented constant or file no longer exists — and rejects a `**File**:`
reference that carries a line number, because those drifted on essentially
every refactor. Renaming a constant here means updating its entry in the same
commit.

For the runtime options a user actually sets (`mip_heuristic_suite`, the
four `mip_heuristic_<name>_effort` budgets, `mip_heuristic_effort`), see the
closing section of this file and `README.md`. For what is and is not reproducible when
you change these, see `docs/REPRODUCIBILITY.md`.

---

## FPR (Fix, Propagate, and Repair)

### `repair_iterations` — RepairSearch DFS node limit

- **File**: `src/fpr_core.h` (field of `FprConfig`)
- **Default**: `50`
- **Meaning**: Maximum number of DFS nodes expanded by `repair_search`
  (paper Fig. 5). The paper quotes 200; we cap at 50 because
  RepairSearch runs two full PropEngine fixpoints per node, which
  dominates cost on tight instances (~760k coefficient accesses on
  9k-nnz LPs). 200 nodes can burn ~1.4 s regardless of the effort cap.
- **Suggested range**: 10–200. Raise on fast instances or when
  RepairSearch quality matters; lower on dense LPs where each node is
  expensive.

---

### `walksat_iterations` — WalkSAT step limit

- **File**: `src/fpr_core.h` (field of `FprConfig`)
- **Default**: `200`
- **Meaning**: Maximum number of WalkSAT repair steps (paper Fig. 4,
  loop bound). Kept at the paper's value because each step is cheap
  (O(row degree) coefficient accesses) and the RepairSearch blow-up
  rationale does not apply.
- **Suggested range**: 50–1000. Increasing helps on highly infeasible
  starting points; decreasing speeds up fast-feasible instances.

---

### `repair_noise` — WalkSAT random-walk probability

- **File**: `src/fpr_core.h` (field of `FprConfig`)
- **Default**: `0.75`
- **Meaning**: Probability of taking a random move rather than a greedy
  (minimum-damage) move in `walksat_select_move` (paper Fig. 4, line
  17). Paper default is 0.75. Greedy probability = 1 − `repair_noise`.
- **Suggested range**: 0.5–0.95. Lower values (more greedy) can work
  better on structured instances; higher values add diversification on
  hard instances.

---

### `kBox` — artificial bounding box for infinite bounds

- **File**: `src/fpr_core.cpp` (anonymous namespace)
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

- **File**: `src/fpr.cpp` (anonymous namespace)
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

### `kMaxAttemptsPerCall` — multi-attempt fill cap per call

- **File**: `src/fpr.cpp` (`FprWorker::run_attempt`)
- **Default**: `32`
- **Meaning**: Maximum number of new FPR attempts started within a
  single `run_attempt` call. Guards against degenerate models where
  attempts verdict near-instantly (e.g. `infeasible-mip0`), which
  would otherwise fill the attempt budget purely with
  `fpr_attempt_begin` setup overhead.
- **Suggested range**: 8–64. Larger values let fast workers fill the
  attempt budget completely; smaller values reduce setup churn on
  degenerate models.

---

### `kNumInitialFprConfigs` — size of the curated rotation

- **File**: `src/fpr.cpp`
- **Default**: `8` (derived from `std::size(kInitialFprConfigs)`)
- **Meaning**: Number of distinct (strategy, mode) pairs in the
  `kInitialFprConfigs` rotation. Changing the array changes this
  automatically.

---

## FPR-LP (`fpr_lp`)

### `kHardRandomizationLimit` — per-worker hard attempt restart cap

- **File**: `src/fpr_lp.cpp` (`LpFprWorker`)
- **Default**: `50`
- **Meaning**: After this many consecutive stale attempts (no improvement
  and no arm switch) the worker forces a new random seed, resetting its
  LP arm assignment. Prevents a worker from replaying the same arm
  forever on degenerate instances.
- **Suggested range**: 20–200.

---

### `kStaleAttemptThreshold` — staleness trigger for randomization

- **File**: `src/fpr_lp.cpp` (`LpFprWorker`)
- **Default**: `3`
- **Meaning**: Number of consecutive stale attempts before incrementing
  the randomization counter. Lower values trigger diversification
  sooner.
- **Suggested range**: 1–10.

---

### `kNumLpArms` — total LP-dependent FPR arms

- **File**: `src/fpr_lp.cpp`
- **Default**: `10` (`kNumClass2=4` + `kNumClass3a=2` + `kNumClass3b=4`)
- **Meaning**: Total number of LP-arm configs across Classes 2, 3a, 3b.
  Workers are assigned `w % kNumLpArms`; excess workers wrap around
  with distinct seeds.

---

## LocalMIP

### `kViolTol` — violation tolerance for constraint classification

- **File**: `src/local_mip_caches.h`
- **Default**: `5e-7`
- **Meaning**: Threshold below which a constraint's violation is
  considered zero (used to classify rows into `violated` vs
  `satisfied`). Tighter than HiGHS's default feasibility tolerance
  to avoid misclassifying nearly-satisfied rows.
- **Suggested range**: 1e-8–1e-6. Tighten for higher accuracy;
  loosen to accept near-feasible solutions faster.

---

### `kRestartInterval` — steps between weight-based restarts

- **File**: `src/local_mip_caches.h`
- **Default**: `200000`
- **Meaning**: Every `kRestartInterval` steps (measured in search
  iterations) the worker considers resetting its solution and weights.
  Higher values allow more exploration before resetting.
- **Suggested range**: 50000–500000.

---

### `kTermCheckInterval` — termination check period

- **File**: `src/local_mip_caches.h`
- **Default**: `1000`
- **Meaning**: Interval (in steps) between checks of the solve's
  wall-clock deadline (`ExecutionContext::past_deadline()`) in
  `LocalMipWorker::run_attempt`'s search loop. Finer values add a
  `clock_gettime` per interval but bound the overshoot more tightly.
- **Suggested range**: 100–10000.
- **Note**: until #114 this constant was defined, documented here, and
  referenced nowhere — LocalMIP had no sub-attempt deadline check at all,
  and the `docs_parameter_references` gate passed because it verifies that
  a documented symbol *exists*, not that anything uses it. The cadence is
  the reason the check is affordable: a clock read per inner iteration was
  measured at ~3% of total instruction refs on small instances, which is
  why the check was originally declined; at 1-in-1000 it is not
  measurable.

---

### `kActivityPeriod` — weight smoothing period

- **File**: `src/local_mip_caches.h`
- **Default**: `100000`
- **Meaning**: Interval (in steps) at which the PAWS-style weight
  smoothing is evaluated. Controls how often the weighting scheme
  adapts to constraint difficulty.
- **Suggested range**: 10000–500000.

---

### `kSmoothProb` — PAWS smoothing probability

- **File**: `src/local_mip_caches.h`
- **Default**: `3e-4`
- **Meaning**: Probability applied each step of choosing to smooth
  (weaken) constraint weights rather than strengthen them (paper §4.1
  PAWS update). With probability `1 - kSmoothProb` the algorithm
  strengthens weights on violated constraints; with `kSmoothProb` it
  weakens weights on satisfied constraints.
- **Suggested range**: 1e-5–1e-2.

---

### `kBmsConstraints` — BMS sample size (violated constraints)

- **File**: `src/local_mip_caches.h`
- **Default**: `12`
- **Meaning**: Number of violated constraints selected as the "best"
  (by weight) from a preliminary sample of `kBmsConstraints * 3`
  candidates, following the paper's BMS (Best-move Selection) operator.
  Determines the scope of the move search.
- **Suggested range**: 4–32.

---

### `kBmsBudget` — BMS candidate variable budget

- **File**: `src/local_mip_caches.h`
- **Default**: `2250`
- **Meaning**: Maximum number of variable–move candidates evaluated per
  infeasible step from the BMS violated-constraint sample. Caps the
  inner candidate-generation loop.
- **Suggested range**: 500–10000. Higher values improve move quality at
  the cost of more coefficient accesses per step.

---

### `kBmsSatCon` — satisfied-constraint BMS sample count

- **File**: `src/local_mip_caches.h`
- **Default**: `1`
- **Meaning**: Number of randomly sampled satisfied constraints used in
  Phase 2 of `infeasible_step` (paper Algorithm 2, lines 7–8). Raises
  move diversity at the cost of more coefficient access.
- **Suggested range**: 1–5.

---

### `kBmsSatBudget` — satisfied-constraint variable budget

- **File**: `src/local_mip_caches.h`
- **Default**: `80`
- **Meaning**: Maximum variable–move candidates generated from the
  satisfied-constraint sample per step (Phase 2 cap).
- **Suggested range**: 20–500.

---

### `kBoolFlipBudget` — Boolean flip scan budget

- **File**: `src/local_mip_caches.h`
- **Default**: `5000`
- **Meaning**: Maximum number of binary variables scanned for flipping
  in Phase 3 of `infeasible_step` (paper Algorithm 2, lines 9–11).
  When there are fewer than `kBoolFlipBudget` binary variables all are
  scanned; otherwise a random-offset window of this size is used.
- **Suggested range**: 500–20000.

---

### `kEasyBudget` — random easy-move fallback count

- **File**: `src/local_mip_caches.h`
- **Default**: `5`
- **Meaning**: Number of randomly chosen variables tried in the Phase 6
  "easy moves" fallback (engineering extension to Algorithm 2). Provides
  a last-resort candidate when all earlier phases fail.
- **Suggested range**: 1–20.

---

### `kTabuBase` — base tabu tenure

- **File**: `src/local_mip_caches.h`
- **Default**: `3`
- **Meaning**: Minimum number of steps a variable's move direction is
  forbidden after being flipped (tabu tenure base). Actual tenure is
  `kTabuBase + uniform(0, kTabuVar)`.
- **Suggested range**: 1–20.

---

### `kTabuVar` — tabu tenure random variation

- **File**: `src/local_mip_caches.h`
- **Default**: `10`
- **Meaning**: Range of randomness added to the tabu tenure:
  `tabu_len = kTabuBase + rng() % kTabuVar`. Higher values make tenure
  more variable, diversifying the search.
- **Suggested range**: 0–50.

---

### `kFeasibleRecheckPeriod` — feasibility recheck interval

- **File**: `src/local_mip_caches.h`
- **Default**: `100`
- **Meaning**: Number of feasible-mode steps between full LHS rechecks
  (`full_recheck`). Incremental updates keep the LHS cache consistent
  between rechecks; full rechecks guard against accumulated floating-
  point drift. Lower values are safer; higher values reduce overhead.
- **Suggested range**: 10–1000.

---

### `kFeasiblePlateau` — feasible-mode plateau detection threshold

- **File**: `src/local_mip_caches.h`
- **Default**: `5000`
- **Meaning**: Number of feasible-mode steps without an improving move
  before triggering a random-walk perturbation (engineering extension to
  paper §4.1). After `kFeasiblePlateau` plateau steps the worker
  perturbs its solution and resets weights to escape the local optimum.
- **Suggested range**: 1000–20000.

---

### `kFeasibleMaxRandomWalks` — perturbation cap per worker

- **File**: `src/local_mip_caches.h`
- **Default**: `20`
- **Meaning**: Maximum number of random-walk perturbations a single
  worker may attempt before declaring itself finished. Prevents infinite
  looping on pathological instances where perturbation cannot break the
  plateau.
- **Suggested range**: 5–100.

---

### `kEpsZero` — numerical zero threshold

- **File**: `src/local_mip_caches.h`
- **Default**: `1e-15`
- **Meaning**: Values with absolute magnitude below `kEpsZero` are
  treated as zero throughout LocalMIP (objective coefficients, move
  deltas). Changing this can affect which variables are considered
  cost-free.

---

### `kPerturbBinaryFraction` — binary perturbation probability

- **File**: `src/local_mip_worker.h`
- **Default**: `0.2`
- **Meaning**: Probability that any given integer variable is perturbed
  during a random-walk perturbation step. For binary variables: flip
  with this probability. For general integers: shift by a random amount
  within the domain with this probability. Applies to both LocalMIP's
  `perturb_solution` and Scylla's `pump::perturb`.
- **Suggested range**: 0.05–0.5.

---

### `kConstructionEffortFraction` — cold-start construction budget fraction

- **File**: `src/local_mip_construction.h`
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

- **File**: `src/local_mip_construction.cpp` (anonymous namespace)
- **Default**: `4`
- **Meaning**: Maximum number of currently-violated rows from which a
  "tight delta" candidate is derived for each variable during the
  construction sweep. Caps the per-variable candidate-generation cost
  at O(col_nnz) coefficient accesses.
- **Suggested range**: 2–10.

---

## Scylla (Feasibility Pump)

### `kAlpha` — objective blending decay factor

- **File**: `src/pump_common.h`
- **Default**: `0.9`
- **Meaning**: Per-iteration multiplicative decay applied to `α_K` in
  the modified objective `α_K * c + (1 - α_K) * rounding_term` (Mexi
  et al. 2023 Algorithm 1.1). Smaller values decay the original
  objective faster, biasing the pump toward pure integrality rounding.
- **Suggested range**: 0.8–0.99.

---

### `kEpsilonInit` — initial PDLP tolerance

- **File**: `src/pump_common.h`
- **Default**: `0.01`
- **Meaning**: Starting tolerance for the PDLP approximate LP solver.
  Each iteration the tolerance decays by `kBeta` until it reaches
  `kEpsilonFloor`. Larger initial values allow faster but less accurate
  early solves.
- **Suggested range**: 1e-4–0.1.

---

### `kBeta` — PDLP tolerance decay factor

- **File**: `src/pump_common.h`
- **Default**: `0.98`
- **Meaning**: Per-iteration multiplicative decay applied to the PDLP
  solve tolerance `ε`. The sequence is `ε_{K+1} = max(kBeta * ε_K,
  kEpsilonFloor)`. Closer to 1.0 means slower tightening.
- **Suggested range**: 0.9–0.999.

---

### `kEpsilonFloor` — minimum PDLP tolerance

- **File**: `src/pump_common.h`
- **Default**: `1e-8`
- **Meaning**: Floor value for the PDLP solve tolerance. Once `ε` decays
  to this level it stays there for the remainder of the pump.
- **Suggested range**: 1e-10–1e-6.

---

### `kCycleWindow` — cycling detection history depth

- **File**: `src/pump_common.h`
- **Default**: `3`
- **Meaning**: Number of past rounded solutions kept in the cycle-
  detection history. A new rounded solution that matches any of the
  last `kCycleWindow` solutions triggers perturbation (Mexi et al.
  Algorithm 1.1, line 13).
- **Suggested range**: 2–10. Larger values catch slower cycles but
  increase memory use and comparison cost.

---

### `kPerturbFraction` — cycling perturbation rate

- **File**: `src/pump_common.h`
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

- **File**: `src/pump_common.h`
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

- **File**: `src/pump_common.h`
- **Default**: `3`
- **Meaning**: Number of consecutive PDLP solve calls that return 0
  iterations before the ScyllaWorker declares itself finished. Guards
  against infinite stalling when the LP solver converges immediately
  (e.g., trivially feasible LP).
- **Suggested range**: 1–10.

---

### `kMaxStaleRoundsDefault` — default stale-snapshot cap per worker

- **File**: `src/scylla_worker.h`
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

- **File**: `src/scylla_worker.h`
- **Default**: `2`
- **Meaning**: Floor applied by `compute_max_stale_rounds`. Even on very
  small LPs, each worker is allowed at least 2 stale rounds before
  forcing a fresh solve.

---

### `kMaxStaleRoundsMax` — maximum stale-snapshot cap

- **File**: `src/scylla_worker.h`
- **Default**: `16`
- **Meaning**: Ceiling applied by `compute_max_stale_rounds`. On very
  large LPs (PDLP solve may take seconds), workers are allowed up to 16
  stale rounds before being forced to block.
- **Suggested range**: 8–32 for very large instances.

---

### `kNnzPerExtraStaleRound` — nnz-per-extra stale round scale factor

- **File**: `src/scylla_worker.h`
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

- **File**: `src/scylla_worker.h`
- **Default**: `4`
- **Meaning**: Number of entries in `kFprConfigs` (the per-worker static
  FPR rounding strategy assignment). Workers `0..kNumFprConfigs-1` are
  assigned deterministically; additional workers draw pseudo-randomly.

---

## Solution Pool and Parallel Runner

### `kPoolCapacity` — solution pool size

- **File**: `src/solution_pool.h`
- **Default**: `10`
- **Meaning**: Maximum number of distinct solutions stored in the shared
  `SolutionPool`. When full, a new solution replaces the worst entry
  (if better) or the most similar entry (if within
  `kDiversityObjTolerance` of the best and sufficiently diverse).
- **Suggested range**: 5–50. Larger pools provide more restart
  diversity but increase lock contention and crossover cost.

---

### `kDiversityObjTolerance` — diversity insertion objective tolerance

- **File**: `src/solution_pool.h`
- **Default**: `0.10`
- **Meaning**: Maximum relative degradation in objective value that a
  diverse solution can have relative to the pool's current best and
  still be admitted (10%). A solution within 10% of the best objective
  may replace the most similar existing entry if its Hamming distance
  exceeds `kDiversityMinHammingFrac`.
- **Suggested range**: 0.0–0.5.

---

### `kDiversityMinHammingFrac` — minimum Hamming distance for diversity

- **File**: `src/solution_pool.h`
- **Default**: `0.05`
- **Meaning**: A solution is considered structurally diverse if its
  Hamming distance (fraction of integer variables that differ) from all
  existing pool entries exceeds this threshold. Used to qualify
  solutions for the diversity-aware insertion path.
- **Suggested range**: 0.01–0.20.

---

## Per-Heuristic Effort Budgets (mode_dispatch)

Each presolve heuristic has its own effort-budget multiplier option, read
in `run_sequential` and turned into that heuristic's budget by
`heuristic_effort_budget(nnz, value)`: `nnz << 12` effort units at the
anchor 0.05, linear in the value, so a budget scales with model size.
The options are registered by
`third_party/highs_patch/apply_patch.cmake` — which nothing compiles, so
`tests/test_smoke.cpp` pins all four defaults — and are documented for
users in the closing section of this file.

They are independent by construction: there is no shared envelope, so
raising one heuristic's budget does not lower another's.

### The defaults approximate the retired scheme — they do not reproduce it

The four defaults below are the closest *scalar* approximation to the
scheme they replaced, and no scalar can be exact. The old envelope handed
a heuristic `budget x max(1 - N/(80e), 1/4) x w/sum(w over enabled)`,
which depends on two runtime facts a constant cannot see: the worker
count `N`, and which *other* heuristics the suite enabled. These
reproduce the `suite=all` share with the worker-count term dropped, so
they are exact at neither end:

| configuration | budget vs the retired scheme |
|---|---|
| `suite=all`, N=1 | 1.04x |
| `suite=all`, N=6 | 1.33x |
| `suite=all`, N=12 | 2x |
| `suite=all`, N>=18 | 4x — the old quarter-floor capped the FJ deduction there |
| `suite=fpr` alone | 0.29x |
| `suite=local_mip` alone | 0.61x |
| `suite=scylla` alone | 0.10x |

`mip_heuristic_fj_effort` is the one exact case, at every `N` and every
suite value.

**The single-heuristic rows are the ones a per-heuristic calibration has
to know about.** The retired allocation divided by the weight sum over
the *enabled* heuristics only, so a suite naming one heuristic handed it
the entire post-FJ envelope. A sweep that runs one heuristic alone —
which is exactly how a per-heuristic budget is measured — therefore
starts at 0.29x / 0.61x / 0.10x of what that configuration used to spend,
not at parity. Those are budget ratios; charged effort tracks them for
FPR and LocalMIP but not for Scylla, which overshoots its budget by up to
a whole PDLP solve (measured 0.25x rather than 0.10x on p0548 at N=6).

The anchor is a choice, not a constraint: the deviation is smallest at
the low worker counts the test suite actually runs at (1–2 on CI,
`(hardware_concurrency()+1)/2` locally), and erring high is the cheap
direction once patience is absolute (#111) — an over-large
budget is truncated by a gate that fires, an under-large one is a hard
cap nothing recovers. Treat all four as the starting point of a
calibration, not the result of one.

`bench/run_benchmark.py --extra-options mip_heuristic_<name>_effort=<V>`
moves one heuristic's budget off its default for a run; the calibration
itself is driven by a tracked target runner rather than by config names.

### `mip_heuristic_fj_effort` — FeasibilityJump budget

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `2.84` (measured, #113; `0.0355` before the unit change, #116)
- **Meaning**: Sizes one *worker's* allowance rather than the whole
  dispatch — the only entry that does, flagged `per_worker` in `kChain`.
  At the default it is exactly `nnz << 10` steps per worker, which is
  vanilla HiGHS's hardcoded single-thread FJ limit
  (`HighsFeasibilityJump.cpp`), so each of the N parallel workers
  searches at least as deep as vanilla does on one thread and the
  dispatch total scales with the pool.
- **Granularity, and a dead zone at the bottom**: FJ's budget is only
  ever *checked* inside upstream's progress callback, which
  `feasibilityjump.hh` fires every `CALLBACK_EFFORT = 500000` effort
  units (`src/fj_worker.cpp`, `run_attempt`). Charged effort therefore
  moves in 500k steps, and this option is a **no-op whenever
  `nnz x value < 6.1`** — the whole budget is consumed before the first
  check. That is `nnz < 489` at the default and `nnz < 2035` at the low
  end of the range below, so on toy models the option looks flat: measured
  at `threads=1`, p0548 moves 500k -> 24.0M across 0.0125 -> 1.00 while
  flugpl sits at 500,009 until 1.00. Real MIPLIB instances (nnz 10^4-10^6)
  are clear of it, but a budget sweep that includes small instances will
  read as noise at the bottom of the range. The other bound is the
  patience gate (`stale_budget`, from `mip_heuristic_fj_patience`), which
  ends a worker early on models where FJ converges.
- **Suggested range**: 0.25–8 (a quarter to eight times vanilla's
  per-thread depth), subject to the dead zone above. The record's ceiling
  is `1e6`; see "A budget that cannot bind" below.

---

### `mip_heuristic_fpr_effort` — FPR budget

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `7.672` (measured, #113; `0.0959` before the unit change, #116)
- **Meaning**: Whole-dispatch budget for the presolve FPR chain, divided
  across the workers by `make_budget`. The default is `0.30 x 2.99/10.15`
  — FPR's 29.5% share of the retired shared envelope at its 0.30 default.
- **Suggested range**: 0.01–1.0. The record's ceiling is `1e6`; see
  "A budget that cannot bind" below.

---

### `mip_heuristic_local_mip_effort` — LocalMIP budget

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `29.232` (measured, #113; `0.3654` before the unit change, #116)
- **Meaning**: Whole-dispatch budget for LocalMIP. The default is
  `0.30 x 6.16/10.15`, its 60.7% share of the retired envelope — the
  largest of the three because the retired weights were proportional to
  `effort_per_ms` and LocalMIP has the highest coefficient-access rate.
  Its effort counter includes the cold-start construction sweep.
- **Suggested range**: 0.01–1.0. The record's ceiling is `1e6`; see
  "A budget that cannot bind" below.

---

### `mip_heuristic_scylla_effort` — Scylla budget

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `1.136` (measured, #113; `0.0142` before the unit change, #116)
- **Meaning**: Whole-dispatch budget for Scylla. The default is
  `0.30 x 1.00/10.15`, its 9.9% share of the retired envelope. Scylla's
  effort is measured in PDLP iters x nnz, a different unit from the other
  heuristics' coefficient accesses, and it saturates: PDLP stalls and
  stale rounds usually bound a Scylla dispatch before the budget does.
- **Suggested range**: 0.01–1.0. The record's ceiling is `1e6`; see
  "A budget that cannot bind" below.

---

### Where these defaults come from, and how to re-derive them

The four effort defaults above are **inherited**, not measured: FJ's is
vanilla HiGHS's hardcoded `nnz << 10` per-worker limit, and the other three
are `0.30 x w/Sw` for weights proportional to a geomean `effort_per_ms`
measured on a different instance set. #113's calibration probe measured them
for the first time, and `bench/derive_from_probe.sh` re-derives the whole
vector from a probe results tree in one command.

The rule, so a number in this file can always be traced to a measurement:

* **effort** = the p50 of the *yield knee* over dispatches that finished
  improving. Every `[HeurSol]` line stamps an accepted solution with the
  charged effort at which it arrived, so one clock-bound dispatch is a whole
  cumulative-yield curve; the knee is the budget that reached its last
  incumbent improvement. Barren dispatches have no knee and are excluded —
  they are a cost question, not a budget question — and a dispatch still
  improving when the cap fired is right-censored, its true knee being larger
  than what it was seen to spend.
* **patience** = `min(p95 of the inter-improvement gaps, 0.25 x effort)`.
  The p95 is a retention claim: at most 5 % of the improvements that would
  ever arrive are cut off. The clamp is a cost claim: a barren dispatch
  spends exactly the patience, on 30-46 % of dispatches depending on the
  heuristic. Both an *improvement* here and the improvement the shipped
  gate resets on are "moved the best objective known", which is the whole
  point of #116 — a patience measured on improvements and spent against a
  gate that reset on pool acceptances would be a number in one unit
  charged in another.

Measured over the 233-instance PLATO list, presolve-only, 30 s, 16 workers,
the p50 knee lands within a factor of two of every shipped value and within
8 % for FPR — so the inherited numbers were roughly right, and now there is
evidence. The directional result is **double LocalMIP, halve Scylla, hold
FPR**; FJ's knee is not identified at that cap (8 of 220 dispatches finished
improving, 115 were still improving) and its quality argues against giving it
more.

Both are valid only at the worker count they were measured at. FJ's budget is
per worker while the other three are per dispatch, so changing the count
**reallocates** budget between heuristics rather than rescaling it.

---

### One axis, a floor and a ceiling

The two options bound the *same* quantity — the effort one heuristic spends
in one dispatch — from below and from above, and since #116 they are
**multiples of the same base**, `nnz << 10`, vanilla HiGHS's hardcoded
single-thread FeasibilityJump limit and the one figure in this arithmetic
upstream itself picked:

```
ceiling = effort   x (nnz << 10)    the most this heuristic may spend
floor   = patience x (nnz << 10)    what it spends before giving up unimproved
```

A dispatch that never improves the incumbent spends exactly the floor and
stops; one that keeps improving spends up to the ceiling. So `effort = 1.0`
is exactly one vanilla FJ budget, and the constraint between the two —

```
0 < patience <= effort / 4          (0 means no gate at all)
```

— is legible without a conversion. **The upper half of that constraint is
enforced, not merely advised**: `patience_threshold` clamps to
`effort / kPatienceCeilingDivisor`, a quarter, because a patience at or
above the ceiling fires exactly at exhaustion and is then indistinguishable
from having no gate at all. That is not a corner case — the p95 #113
measured exceeds the ceiling on three of the four heuristics, FJ's by 4,400x
— so without the clamp an honestly derived value would silently mean "never
give up". A quarter is the shape FJ has always shipped (`nnz << 8` against
`nnz << 10`), and where all four shipped defaults already sat. Before that the effort option multiplied
`nnz << 12` scaled by `value / 0.05` while the threshold was already absolute
per nonzero, so comparing the pair meant knowing that `(1 << 12) / 0.05` is
81,920. Both constants were historical — the 4096 existed only so FJ's
default came out at `nnz << 10` anyway, and the 0.05 was upstream's own
`mip_heuristic_effort` default used as an anchor.

Two consequences worth knowing. `mip_heuristic_effort` — upstream's *own* B&B
knob, which `fpr_lp` caps itself against — is on the old scale and goes
through `vanilla_effort_budget`, which restores the anchor so its meaning is
unchanged. And every shipped patience default is visibly `0.25 x` its
effort, because the clamp bound all four: the measured waits contributed
nothing to them, which is a statement about how slowly these heuristics
improve rather than about the calibration.

---

### A budget that cannot bind

All four effort records are bounded at **`1e6`**, not at `1.0`. Nothing
ships or tunes above `1.0` — the suggested ranges above are the real
operating range — but one measurement needs a value outside it.

`WorkerBudgetState` retires a worker in exactly two places: `exhausted()`
(its total budget) and `stale()` (its staleness gate). Set the gate to `0`
and hand the heuristic a budget it cannot reach, and neither fires, so the
solver's wall clock becomes the *single* stopping rule — the same one for
all four heuristics on every instance. That is what #113's calibration
probe runs at, and it is the difference between measuring a heuristic and
measuring the setting being derived from it: at any binding budget the
trace stops where the budget stops, and each heuristic's budget binds at a
different model size (at effort `1.0`, FJ exhausts its budget after 7.8 s
on a 41k-nonzero model while Scylla is already clock-bound there).

At `1e6` the budget is `1.0e9` effort units per matrix nonzero —
unreachable inside any per-run cap the campaign uses, and far enough below
`size_t` overflow on the largest MIPLIB model that
`heuristic_effort_budget` never has to saturate (it does anyway; a
`double -> size_t` conversion out of range is undefined, and the same
reasoning that gives `saturating_mul` its guard applies one level up).
"Cannot bind" is checked rather than assumed:
`bench/analyze_presolve_probe.py` reports charged effort against the
budget, so a run that did hit it is visible.

---

## Patience (issue #111; options since #106; named and clamped by #116)

Every presolve heuristic stops early when improvement-free effort crosses
a threshold, at two levels: a runner-level gate over the whole dispatch
(`ContinuousLoopState::effort_since_improvement` against
`HeuristicBudget::stale`) and a worker-level gate in `WorkerBudgetState`.

The four options below are that threshold. It is a **patience** — the
improvement-free effort a heuristic tolerates before giving up, and hence
a floor on what a dispatch spends — expressed as a multiple of
`nnz << 10`, the same unit as the effort option beside it. It is absolute
and instance-scaled, never a fraction of the heuristic's own budget: a
fraction cannot bound over-budgeting, because doubling the budget doubles
the tolerance, so the gate never fires relatively sooner and charged
effort tracks the effort option one-for-one. That is what makes the four
independent budgets of #110 composable — a heuristic that stops finding
things exits instead of spending an allowance that was tuned in isolation.
(They were spelled `mip_heuristic_<name>_stall` until #116, which renamed
them with no alias.)

A gate has two operands, and the other is the **improvement signal** that
resets the counter. It has been wrong twice. Each worker used to supply
its own — "I beat my own best" — which restarts at infinity on every
rebuild, so a rebuilt worker cleared the dispatch's staleness by
rediscovering a solution the pool already held; fixing the threshold alone
left FPR at 19.98x over a 20x sweep on `flugpl`, where it spent forty
ceilings' worth of effort for one accepted solution. #111 pointed both
gates at `IncumbentSink::offer`'s verdict instead, and called pool
acceptance the project's definition of production. **#116 reverses that
for the gate**, on the probe's evidence: the pool keeps a top-K, so a
heuristic that merely beats its own worst entry resets staleness forever.
Over 233 instances, presolve-only, 30 s, 16 workers, FPR earns ~3.3 M pool
acceptances against 590 incumbent improvements, LocalMIP ~3.3 M against
24,598, Scylla 367,801 against 374, FJ 1,557 against 297. Five orders of
magnitude on FPR — so a patience calibrated on improvements (the only
thing it can honestly be calibrated on) cannot be spent against a gate
that resets on acceptances. Both gates now read
`IncumbentSink::OfferResult::improved_incumbent`, decided inside
`SolutionPool`'s own lock against the best objective the pool has ever
accepted — a monotone watermark, not its front entry, which the diversity
path can evict — while
`accepted()`, `[Heur] found` and `[HeurSol] accepted` keep reporting the
acceptance for the tooling that consumes them. `offer` is `[[nodiscard]]`
and returns both facts, so neither can be dropped or silently substituted
for the other. Each local search's own bookkeeping — FJ's
`effortSinceLastImprovement`, LocalMIP's `best_objective_` /
`steps_since_improvement_` — is untouched: those are the search's notion,
not the dispatch's.

**Zero means no gate at all.** `patience_threshold` returns an unbounded
threshold when the multiplier is `0`, before the clamp below — not a
threshold of zero, which would retire every worker before it did any
work. That semantic is load-bearing rather than defensive: searching the
patience axis needs a point where the gate provably never fires, or "what
does this gate cost?" has no zero to measure against. `0` is the bottom
of every one of the four ranges, and the top is `1e6`.

**The clamp.** `patience_threshold` clamps to a quarter of the allowance,
so a live gate always fires strictly *before* budget exhaustion. Clamping
to the allowance itself — which is what it did until #116 — made a
too-large patience fire exactly at exhaustion, which is behaviourally
identical to no gate at all and reported nowhere. Since the measured p95
exceeds the ceiling on three of the four heuristics, that was the common
case for an honest value rather than a corner. Two consequences for a
search: any `patience >= effort / 4` gives the same gate, the loosest one
that can still fire; and a search that wants "no gate" must ask for `0`
explicitly, because the top of the range no longer means it.

`patience_threshold(nnz, per_base, budget)` in `src/heuristic_common.h`
applies one, clamped to `budget / kPatienceCeilingDivisor`, with the
product taken through `saturating_mul` — both factors are user-supplied,
and a wrapped product would hand the gate a *small* threshold, silently
reducing the heuristic to nothing at the top of the option's range. The runner-level
gate is sized in `run_sequential` (multiplied by the worker count for the
one option whose scope is per-worker, FJ's); the worker-level gate is
`HeuristicBudget::worker_stale`, that value divided by the worker count.
Scylla is the documented exception and takes the dispatch-level value,
because its per-worker counter is charged the PDLP cost already divided
by the worker count.

### These options do not mean the same thing at every worker count

- **File**: `src/mode_dispatch.cpp` (`kChain`, `budget_is_per_worker`, `make_budget`, `heuristic_effort_budget`, `patience_threshold`)

A tuned parameter vector is only valid at the worker count `N` it was
tuned at. Two distinct things vary with `N`, and they must not be
conflated: what a heuristic is **allowed** to spend, and what it
**actually** spends.

**What the budget arithmetic says.** Writing
`sized = heuristic_effort_budget(nnz, effort)`, `make_budget` yields:

| quantity | FJ (`budget_is_per_worker`) | FPR / LocalMIP / Scylla |
|---|---|---|
| dispatch `total` | `sized x N` — **scales with N** | `sized` — invariant |
| `per_worker` | `sized` — invariant | `sized / N` |
| runner gate `stale` | `N x min(patience x nnz << 10, sized / 4)` | `min(patience x nnz << 10, sized / 4)` — invariant |
| `worker_stale` | invariant | `stale / N` |
| `attempt_cap` | `sized / 10` — invariant | `sized / (10 N)` |

**The `stale` rows assume `patience > 0`.** At `patience = 0` the gate is
disabled and `patience_threshold` returns an unbounded threshold before any
clamp, in **both** columns — so the runner gate is `SIZE_MAX` rather than
`N x 0` or `0`, and `worker_stale` is `SIZE_MAX / N` everywhere. Read
literally the table would say zero, i.e. a gate retiring every worker
before it did any work, which is the exact misreading the `per_nnz == 0`
special case exists to prevent. The irace ranges include `0`, so this is
not a hypothetical corner.

So on **budget**, the three whole-dispatch heuristics are N-invariant in
both aggregates — total and runner gate — and only the per-worker slicing
moves. FJ is the mirror image: every per-worker quantity invariant, the
dispatch total growing with the pool.

**Charged spend is a different question, and Scylla answers it
differently from its budget.** Measured on `p0548` at the shipped
defaults, seed 0, charged presolve effort from `[Heur]`:

| heuristic | N=1 | N=8 | ratio |
|---|---|---|---|
| fj | 500,104 | 6,001,362 | 12.0x |
| fpr | 3,972,738 | 2,884,999 | 0.73x |
| local_mip | 16,889,772 | 18,316,837 | 1.08x |
| scylla | 684,860 | 5,478,908 | **8.0x** |

Scylla's budget is N-invariant exactly as the table above says, and its
spend still scales nearly linearly with the pool — 8.0x, reproducible to
the digit across repetitions at both counts. The cause is the granularity
floor documented under `mip_heuristic_scylla_patience`: one attempt charges
a whole PDLP solve (`iters x nnz`), and `attempt_cap` does not govern a
solve once started, so `N` workers each charge whole solves however small
the dispatch budget is. **"Budget is N-invariant" therefore does not
imply "spend is N-invariant", and Scylla is the counterexample** — tuning
`mip_heuristic_scylla_effort` at `N=1` and transferring to 16 workers
buys roughly 8–16x the spend the screen measured. FJ's 12x rather than
8x is its 500k callback granularity on an instance this small (see the
dead zone under `mip_heuristic_fj_effort`); FPR's 0.73x is the patience
gate firing on aggregate effort while each worker's share shrinks.

**The consequence is a shift in the balance *between* heuristics, not a
uniform rescale.** FJ's share against LocalMIP goes from 1:34 to 1:3 on
identical option values. A vector tuned at one worker count and deployed
at another allocates the chain differently, and nothing in a score
reveals it. **Record the worker count alongside any tuned vector**;
`bench/make_archive.py` already derives it per run from each log's
`Thread count N (of M threads)` line into `workers_observed` and warns
when a tree mixes two values, so reuse that rather than inventing a
second channel.

One quantity *is* transferable: the clamp boundary
`patience >= effort / 4` is N-independent for all four, because the N
factors cancel in FJ's case and are absent in the other three's. "This
configuration's gate is at its ceiling" therefore means the same thing at
every worker count, even though the effort it is measured against does
not.

**All four defaults are measured** (#113, see "The defaults are measured"
above), and they are the reason the constants became options in the first
place: a 64x sweep of the LocalMIP effort
option moved median presolve wall time by under 4%, because the gate, not
the budget, is what stops the search, and a `constexpr` cannot be swept
without a rebuild per point.

A third bound now competes with those two at the top of the effort range:
since #114 the solve's own `time_limit` stops every presolve heuristic
within one polling interval, so a heuristic given more budget than it can
spend in the time available is **limit**-bound rather than budget-bound.
That is the intended behaviour — before it, FeasibilityJump at
`effort=1.0` ran 1.4-2.0x past a 60 s limit and twice to an external
SIGKILL — but it changes what a high-effort sweep point measures: at
`effort=1.0` on gesa2 the four heuristics need 1.5-8.5 s each to exhaust
their budgets, so any limit below that measures the machine and not the
option. Size a calibration run's limit above the budget-bound time, or
read the reported effort rather than assuming the option was spent. They are registered in
`third_party/highs_patch/apply_patch.cmake` and pinned by
`tests/test_smoke.cpp`, which nothing else checks.

The units differ per heuristic and the values are **not** comparable
across them: FJ counts step units, FPR and LocalMIP coefficient accesses,
Scylla PDLP iterations x nnz. Do not align them numerically; align the
semantics (the same quantile of each heuristic's own inter-acceptance
effort-gap distribution).

### `mip_heuristic_fj_patience` — FeasibilityJump patience

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `0.71` (measured, #113; `727` before the unit change, #116)
- **Meaning**: Step units per worker without an incumbent improvement, as
  a multiple of `nnz << 10`. Scope is **per worker**, matching
  `mip_heuristic_fj_effort` — the only one of the four with that scope, so
  the runner-level gate is this times the worker count. The quarter-of-the
  -ceiling shape is FJ's own: `nnz << 8` against a `nnz << 10` budget is
  what it has always shipped, it was the one heuristic already using an
  absolute threshold, and it is the model the other three were moved onto
  and the clamp was taken from.
- **Least trustworthy of the four.** The p50 knee behind its *effort*
  rests on 8 completed dispatches out of 220 (115 were still improving at
  the 30 s cap), and its measured p95 wait is 4,446 — 1,566x its own
  ceiling — so this value is entirely the clamp. Read it as "FJ improves
  too rarely for a patience to express", not as a tuned number.
- **Suggested range**: 0.06–1.0, plus `0` for no gate. Values above
  `effort / 4` are clamped down to it.

---

### `mip_heuristic_fpr_patience` — FPR patience

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `1.918` (measured, #113; `1964` before the unit change, #116)
- **Meaning**: Coefficient accesses without an incumbent improvement, as a
  multiple of `nnz << 10`, **whole dispatch**. FPR had no worker-level
  gate at all before #111 (`FprWorker::finished()` returned false
  unconditionally); it now has one at this value divided by the worker
  count, and a retired FPR worker stays retired rather than being rebuilt.
- **The heuristic the #116 signal change matters most for**: ~3.3 M pool
  acceptances against 590 incumbent improvements over the probe's 233
  instances. `egout` is the instance it shows up on in this repo's own
  suite — 19.98x charged-effort growth over a 20x budget sweep both before
  and after #111, bounded only once the gate stopped counting
  acceptances (`tests/test_patience_gate.cpp`).
- **Suggested range**: 0.5–8, plus `0` for no gate. Values above
  `effort / 4` are clamped down to it.

---

### `mip_heuristic_local_mip_patience` — LocalMIP patience

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `7.308` (measured, #113; `7484` before the unit change, #116)
- **Meaning**: Coefficient accesses without an incumbent improvement, as a
  multiple of `nnz << 10`, **whole dispatch**. The only one of the four
  whose *measured* p95 (8.50) is within reach of its ceiling fraction
  (7.31), so it is the only one where the clamp is barely doing the work —
  and it is also the heuristic whose improvements most reliably close gap.
  Raising `kPatienceCeilingDivisor`'s quarter to a half would let that
  measurement stand and leave the other three clamped; that is the one
  knob here worth an ablation arm.
- **Residual**: LocalMIP is the least tightly bounded of the four, and
  unevenly so. With both halves of #111 in place, `p0548` at `threads=1`
  and effort 1.00 reaches the identical fifteen incumbents on 27.8M
  effort instead of 92.7M at seed 0 — but is unchanged at seeds 1–3 and
  halved at seed 4, giving 6.00x / 19.99x / 19.99x / 19.98x / 9.93x over
  a 20x sweep; `gt2` over the same five seeds gives 11.22x / 7.25x /
  6.50x / 13.36x / 5.73x. Quote a range, never one of these numbers on
  its own. The spread is not a bimodal flip — each solve is
  bit-reproducible at `threads=1`, and p0548 over seeds 0–9 is a
  continuous 20.7M–92.7M with three seeds at the ceiling — LocalMIP is
  legitimately earning acceptances wherever the ratio stays high. The
  residual is pool-fill and diversity accepts: `kPoolCapacity` offers are
  admitted unconditionally
  while the pool fills, and structurally diverse near-best solutions
  afterwards, both of which used to reset the gate. **#116 closed exactly
  that**: the gate now resets only on an offer that moved the incumbent,
  so those acceptances no longer clear staleness. The numbers above were
  measured before that change and are kept as the record of what the
  acceptance-driven gate did; re-measure them rather than quoting them.
- **Cost**: not free. At effort 1.00, `threads=1`, seed 0, LocalMIP
  alone, the final presolve-phase incumbent is unchanged on `p0548`
  (28271, and the same fifteen incumbents on the way) and `flugpl`
  (1201500), but worse on `gt2` (42355 → 45855, **+8.3%**), `rgn`
  (112.8 → 134.0, **+18.8%**) and `dcmulti` (212709 → 219964,
  **+3.4%**) — all minimisations, so higher is worse. The effort saved
  on those same five is 3.33x (p0548), 1.78x (gt2), 2.63x (rgn), 4.00x
  (dcmulti) and 1.00x (flugpl), **geomean 2.29x**. A worker that keeps
  improving its own solution without beating the pool's worst-of-ten now
  retires and is rebuilt from a pool restart, so it is redirected rather
  than killed. Whether that trade wins is a #106 question.
- **Suggested range**: 1–16, plus `0` for no gate. Values above
  `effort / 4` are clamped down to it.

---

### `mip_heuristic_scylla_patience` — Scylla patience

- **File**: `src/mode_dispatch.cpp` (`kChain`)
- **Default**: `0.284` (measured, #113; `291` before the unit change, #116)
- **Meaning**: PDLP-iteration x nnz units without an incumbent
  improvement, as a multiple of `nnz << 10`, **whole dispatch**. Small in
  absolute terms because one PDLP solve charges `iters x nnz`, so this is
  a handful of unproductive pump rounds rather than hundreds of matrix
  sweeps.
- **Granularity floor**: Scylla cannot honour a threshold below the cost
  of one attempt, and one attempt charges a whole PDLP solve
  (`iters x nnz`), which exceeds `512 x nnz` by 2–7x on the bundled
  instances. Its effective floor is therefore `N x one PDLP solve`
  regardless of what this constant says, and lowering it buys nothing.
  The per-attempt cap does not help: on `dcmulti` at effort 0.05,
  `attempt_cap` is already 45,192 — 15x below the 678k ceiling — and each
  attempt still charges ~1.35M, because `run_cap` does not govern a PDLP
  solve once it has started. Fixing this means bounding the solve itself
  (a PDLP iteration cap derived from the remaining patience room), not
  bounding the attempt. **Flagged for #106.**
- **Suggested range**: 0.12–2, plus `0` for no gate — though on Scylla
  `0` and any value below one PDLP solve are observationally the same,
  for the reason above. Values above `effort / 4` are clamped down to it.

---

## Presolve-Only Exit (issue #106)

### `mip_heuristic_presolve_only` — stop after the presolve chain

- **File**: `third_party/highs_patch/apply_patch.cmake`
- **Default**: `false`
- **Meaning**: When true, the solve exits after the presolve heuristic
  chain and **before the root LP**, keeping whatever incumbent the chain
  produced. It is a measurement mode, not a solver mode: inside a full
  solve a presolve heuristic runs for ~2 s of a 60 s limit and B&B owns
  the rest, so a primal-integral score over the whole solve dilutes the
  thing being tuned into seed noise. A presolve-only run scores exactly
  the chain.
- **Reported status**: `kSolutionLimit`, which maps to
  `HighsStatus::kWarning`. That is what HiGHS itself assigns when a
  user-configured search-size limit stops the solve (`mip_max_nodes`,
  `mip_max_leaves`, `mip_max_improving_sols`), and presolve-only is that
  kind of limit. It is also the shape of status that survives
  `cleanupSolve`, which overwrites only `kNotset` and `kInfeasible` —
  either of those would be rewritten to `kOptimal` whenever the chain
  found anything, claiming optimality for a solve that never computed a
  dual bound. The solution is still extracted; `Highs::callSolveMip` keys
  that off `solution_objective_`, never off the model status. With no
  solution the run reports an infinite primal bound and solution status
  `-`.
- **Not** `mip_max_nodes = 0` (checked inside the B&B loop, so the root
  LP and the dive heuristics run first) and **not**
  `mip_root_presolve_only` (controls where presolve is applied, not when
  the solve stops). Both were tried.
- **Side effects**: no root LP means no dual bound (`mip_dual_bound` is
  `-inf`), no B&B nodes, no LP iterations, and no dive-time heuristics —
  `fpr_lp` never runs under this option, and neither do RENS/RINS.

---

## Repair Search (`repair_search`)

### `kProgressThreshold` — no-progress trigger for best-open jump

- **File**: `src/repair_search.cpp` (anonymous namespace)
- **Default**: `10`
- **Meaning**: Number of consecutive RepairSearch DFS nodes without a
  violation improvement before the algorithm swaps to the lowest-
  violation open node (paper Fig. 5 best-first steering, line 27).
- **Suggested range**: 5–30.

---

## FJ Option Note

`mip_heuristic_run_feasibility_jump` is a **native HiGHS option**
(registered by HiGHS itself, default: `true`). It is **not** one of the
custom patch-added options. It keeps its meaning — `false` disables
FeasibilityJump — but *which* FJ it gates depends on
`mip_heuristic_suite`:

- at `suite=off` it gates HiGHS's own standalone single-threaded FJ,
  which the patch leaves in place so that `off` is a true
  vanilla-equivalent ablation;
- at every other suite value the native call site is off and this
  option gates our parallel FJ instead.

So an options file carrying both

    mip_heuristic_suite = off
    mip_heuristic_run_feasibility_jump = false

is the pure patch-overhead configuration: no heuristics of any kind.
Neither is a command-line flag — HiGHS's CLI accepts only its own fixed
flag set and rejects an unknown `--mip_heuristic_...` *without solving*,
so custom options are reachable only through `--options_file`.

`mip_heuristic_effort` is likewise **native to HiGHS**, not patch-added.
It is upstream's B&B heuristic knob: `moreHeuristicsAllowed()` admits
B&B-dive heuristics while `heuristic_lp_iterations < total_lp_iterations
* mip_heuristic_effort` (plus an initial 10000-iteration offset). The
patch keeps its vanilla default of `0.05` and its vanilla meaning, so a
patched binary at default options matches vanilla's B&B heuristic budget
exactly; it gates RENS/RINS and `fpr_lp` together. It was briefly raised
to `0.30` and overloaded as the presolve budget — that overload was split
out into a presolve-only option, which in turn became the four
per-heuristic options below.

The custom patch-added options are exactly five:

- `mip_heuristic_fj_effort` (default `0.0125`),
  `mip_heuristic_fpr_effort` (`0.0884`),
  `mip_heuristic_local_mip_effort` (`0.1821`),
  `mip_heuristic_scylla_effort` (`0.0296`) — one effort budget multiplier
  per presolve heuristic, each a double in `[0.0, 1e6]`. See
  "Per-Heuristic Effort Budgets" above for what each one sizes; FJ's is
  per worker, the other three are per dispatch.
- `mip_heuristic_suite` — which heuristics run (default `"all"`).
  The value is either one of the two whole-value aliases `off` (no
  heuristic) and `all` (every one), or a **comma-separated list** of the
  heuristic names `fj`, `fpr`, `local_mip`, `scylla` — so all fifteen
  non-empty subsets are expressible, e.g. `fj,fpr,local_mip` (#112).
  Order is irrelevant, whitespace around a name is ignored and repeats
  are harmless. `off` is an alias for the whole value only, never a token
  in a list: the patched HiGHS tree compares this option to `"off"`
  verbatim to hand back upstream's own FeasibilityJump call site, so a
  value that selected nothing without being that exact string would be a
  heuristic-free run that is *not* the vanilla-equivalent one. HiGHS does
  not validate string option values, so an unrecognised one is accepted
  by `setOptionValue` and caught at solve time: the dispatcher warns —
  naming the offending token, which is what makes a typo inside a list
  diagnosable — and falls back to running all four. The single place the
  string becomes four booleans is `heuristics::effective_flags` in
  `src/mode_dispatch.cpp`; the legal names are the presolve chain table's
  own, so they cannot drift from the `[Heur] name=<n>` traces.

`mip_heuristic_suite` also gates the B&B-dive `fpr_lp`, on the same bit
as presolve FPR. It therefore runs at any value naming `fpr` — `fpr`,
`all`, `fj,fpr`, and an unrecognised value, which fails open to all four
— while `off` and every subset that omits `fpr` disable it. That is
deliberate (a
per-heuristic attribution run must not leave a second FPR variant
running at dive time), but it means a dive-time result measured under
`suite=local_mip` or `suite=scylla` says nothing about `fpr_lp`.
