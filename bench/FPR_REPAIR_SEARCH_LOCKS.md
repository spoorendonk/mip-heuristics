# FprRepairSearchLocks first-fire latency on neos-3426085-ticino

## What the arm does

`FprRepairSearchLocks` is the portfolio arm mapped to
`(kStratLocks, FrameworkMode::kRepairSearch)` (see `src/portfolio.cpp:57`). After
FPR's Phase 1/2 DFS leaves some rows violated, `kRepairSearch` dispatches to
`repair_search()` (`src/repair_search.cpp:156`) instead of `walksat_repair`.
`repair_search` builds a **second** `PropEngine R` from global bounds
(`src/repair_search.cpp:270`) and runs a DFS over repair disjunctions (paper
Fig. 5): each node applies a branch to `R`, propagates `R` to fixpoint, calls
`sync_changes(E, R)` to transfer deductions and re-propagate `E`, picks a
WalkSAT move, and pushes two children.

## Evidence

Reproduced on `/tmp/miplib/neos-3426085-ticino.mps.gz` with the shipped
`build/bin/highs` in port/opp mode (`time_limit=5`, `log_dev_level=3`,
`random_seed=0`):

| arm                   | pulls | effort      | wall_ms  | effort/ms |
|-----------------------|-------|-------------|----------|-----------|
| FprDfsrepBadobjcl     | 1     | 7.40 M      |    61.9  | 119 k     |
| FprDfsBadobjcl        | 1     | 7.10 M      |    58.9  | 120 k     |
| FprDfsrepLocks        | 1     | 7.09 M      |    59.4  | 119 k     |
| Scylla                | 1     | 1.45 M      |    21.6  |  67 k     |
| LocalMIP              | 1     | 0.38 M      |     1.8  | 211 k     |
| **FprRepairSearchLocks** | **1** | **152.30 M** | **1360.9** | **112 k** |

Same arm on `liu.mps.gz` runs in 3–11 ms (effort 150 k–950 k); in port/det
mode on ticino the arm is never picked in 10 s because its first-pull cap
is throttled by epoch budgeting. So the 1.4 s blow-up is specific to
`opportunistic=true` + ticino's row/col structure (308 rows × 4688 integer
cols, 9083 nnz, binding "≥" capacity rows).

## Root cause hypothesis

`repair_search()` caps DFS nodes at `repair_iterations=50` (post-fix; was 200
when this report was written) and terminates on
`total_effort >= max_effort` (`src/repair_search.cpp:287`). **But
`total_effort` only accumulates `apply_move`/`walksat_select_move` counters
(lines 313, 320, 356)** — it does **not** include the two PropEngines'
propagation work. `E.effort()` and `R.effort()` are charged to `effort_out`
once at the end (line 389) but never consulted as a stopping condition.

On ticino the DFS therefore runs all 200 iterations and each iteration pays
for:

1. `apply_branch_to_r` → `R.propagate(-1)` — full AC-3 fixpoint on 308
   rows × 9083 nnz from a tight "≥" LP relaxation.
2. `sync_changes(E, R)` → `E.propagate(-1)` again
   (`src/repair_search.cpp:53`).
3. `rebuild_violated()` — O(nrow) scan per iteration
   (`src/repair_search.cpp:297,386`).
4. `best_solution = solution; best_lhs = lhs_cache;` — two full
   O(ncol+nrow) copies every time `total_viol` improves
   (`src/repair_search.cpp:336–337`).

152 M coef accesses / 200 iterations ≈ **760 k accesses per node** — two
propagate fixpoints on 9083 nnz is a plausible fit. The peer arms
(`FprDfsrepLocks`, `FprDfsrepBadobjcl`) run the WalkSAT repair path, which
has no secondary PropEngine and so respects the 7 M `max_effort` budget
honestly.

## Mitigation options

1. **Cap per-invocation wall_ms.** Add a `std::chrono` deadline check inside
   the `while (!Q.empty() && ...)` loop at `src/repair_search.cpp:287`
   (e.g. 100 ms for opportunistic first pulls). Cheapest fix, keeps the arm
   alive, loses no portfolio diversity. Downside: wall-clock caps are
   non-deterministic across machines and muddy the existing effort-budget
   contract; the fix belongs logically in `fpr_attempt`, but RepairSearch
   is the only caller where the `max_effort` contract is visibly broken.

2. **Arm-cost-aware Thompson priors.** Give known-slow arms a
   cost-weighted initial success prior so opportunistic first-pull cost is
   amortised by more optimistic priors on cheap arms. `compute_budget_cap`
   already uses `avg_effort` after the first pull, so only the cold start
   is at risk; a non-uniform Beta(α, β) per arm (e.g. `FprRepairSearchLocks`
   at α=0.5) would shift first picks toward the cheap arms. Downside:
   hand-tuned priors are fragile across instance families — ticino-tight
   instances may be exactly where RepairSearch eventually pays off.

3. **Drop the arm from the portfolio.** Remove the entry at
   `src/portfolio.cpp:57` and renumber. The reward row (`reward=0` on
   ticino, `reward=0` on liu in every log sample) argues the arm is not
   pulling its weight during presolve — its home is paper Class 2/3
   (LP-dependent, dive-time). Downside: portfolio regret if there is a
   MIPLIB subset where RepairSearch uniquely finds a first feasible;
   worth checking `bench/results_diag/` across the full sweep before
   dropping.

## Recommendation

Do **#1** now (one-line deadline check in `repair_search`'s DFS loop,
honoring a new `cfg.wall_deadline`) because the underlying bug is that
`max_effort` ignores PropEngine work, and then audit whether #3 is warranted
from MIPLIB-wide reward data — #2 is a band-aid that hides the effort-
accounting bug without fixing it.
