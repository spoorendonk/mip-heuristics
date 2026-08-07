# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.devkit/standards/cpp.md

**Testing override**: `.devkit/standards/cpp.md` currently documents the testing story in GoogleTest terms. Ignore that section — this project uses Catch2 v3 (`TEST_CASE(...)` with `[tag]` filters). Tracked upstream as [spoorendonk/devkit#1](https://github.com/spoorendonk/devkit/issues/1).

**Git workflow**: Trunk-based development. Commit directly to `main` and push when local gates pass. No long-lived feature branches.

## Project Overview

Custom MIP (Mixed-Integer Programming) heuristics integrated into the HiGHS solver via a patched fork. The heuristics run during HiGHS's presolve phase and are compiled as object files linked directly into the `highs` library target.

## Build Commands

```bash
# Configure (from repo root)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Run a single test by name
cd build && ctest -R "execution-mode: flugpl objective" --output-on-failure

# Run tests matching a Catch2 tag
cd build && ./mip_heuristics_tests "[mode-matrix]"
```

First build is slow (~5 min) because it fetches and builds HiGHS via FetchContent.

## Build & Test

Used by the devkit pre-push hook.  The `unset GIT_DIR GIT_WORK_TREE`
prefix is required: `git push` leaks `GIT_DIR=.git` into the hook
subshell, and CMake's nested `git clone` inside FetchContent then
treats that as the target git directory and fails with `fatal: invalid
reference: v1.15.1` when trying to check out the HiGHS tag.

TODO(devkit): this block is an in-project workaround for an upstream
devkit hook bug.  `.devkit/standards/common.md` says local hook
workarounds should be raised upstream — remove this whole section
once devkit's pre-push wrapper clears `GIT_DIR` before running nested
build commands.

```clean
rm -rf build
```

```build
unset GIT_DIR GIT_WORK_TREE && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"$(nproc)"
```

```test
ctest --test-dir build --output-on-failure -j"$(nproc)"
```

GPU acceleration: `-DMIP_HEURISTICS_CUDA=ON` enables CUDA for the PDLP solver used by Scylla. It **requires `CUDA_HOME`** to be exported (HiGHS's `FindCUDAConf.cmake` derives `CMAKE_CUDA_PATH` from it and uses that for the cudart/cublas/cusparse `find_library` hints and for HiGHS's CUDA include directory), and it **fails the configure** if nvcc or `CUDA_HOME` are missing, or if the `HConfig.h` HiGHS generates doesn't come out GPU-enabled — it does *not* fall back to CPU. That's deliberate: GPU vs CPU is a compile-time `#ifdef CUPDLP_CPU` in HiGHS's `CupdlpWrapper.cpp` with no runtime override, so a silent fallback yields a CPU binary indistinguishable from a GPU one at the command line. Build GPU into a separate tree (`cmake -B build-gpu -DMIP_HEURISTICS_CUDA=ON`) and confirm with `ldd build-gpu/bin/highs | grep -E 'cudart|cublas|cusparse'`. Omit the flag for a CPU build.

## Architecture

**Integration model**: Heuristics are compiled as a static object library (`mip_heuristics`) whose objects are injected into the HiGHS `highs` target. The HiGHS source is fetched at build time (v1.15.1) with patches applied from `third_party/highs_patch/`. Heuristics access HiGHS internals directly via `HighsMipSolver&`.

**Bumping the HiGHS tag**: HiGHS renames `advanced` options across minor versions with no deprecation shim (`pdlp_scaling` → `pdlp_scaling_mode` and `pdlp_e_restart_method` → `pdlp_cupdlpc_restart_method` at v1.14.0, which left two dead option writes in `contested_pdlp.cpp` for months). Every `Highs` instance we build sets `output_flag=false`, so a rejected `setOptionValue` is completely silent — it reports failure only through the return status. After a bump, grep for `setOptionValue` and check each name against `_deps/highs-src/highs/lp_data/HighsOptions.h`, and prefer routing writes through a status-checking helper (`set_option_or_die` in `contested_pdlp.cpp`). Verifying that an option *exists* is not enough: check that the code path you're on actually reads it (e.g. `pdlp_scaling_mode` is consumed only by HiPDLP, never by cuPDLP-C).

**Heuristic entry points** — each has a standalone `run()` that HiGHS calls during presolve:
- `fpr` — Fix, Propagate, and Repair. DFS tree search that fixes integers, propagates bounds, backtracks on infeasibility, then runs WalkSAT/RepairSearch to fix remaining violations. `fpr_core.cpp` exposes two APIs: the one-shot `fpr_attempt` (used by scylla / fpr_lp / one-shot tests) and the `fpr_attempt_begin` / `fpr_attempt_step` / `fpr_attempt_finish` lifecycle (issue #77) that lets `FprWorker` pause an in-flight DFS at the per-call budget gate and resume it on the next call with state in `FprAttemptState` + `FprScratch`. The pause/resume mechanic is what avoids discarding work on long DFS subtrees; multi-attempt looping inside `run_attempt` lets fast workers fill the slice with new attempts (rotated through `kInitialFprConfigs` per `(worker_idx + attempt_idx)`) instead of idling. Sub-algorithms: `prop_engine` (bound propagation, with both forward propagate and a domain PQ for dynamic-var strategies — `repair_search` requires `e_pq_mark` threading on `RepairSearchNode` to keep the PQ consistent across its secondary backtracks), `walksat`, `repair_search`, `fpr_strategies` (strategy variants).
- `fpr_lp` — LP-dependent FPR (paper Classes 2–3) using root LP solution. Called during B&B dive (after RENS/RINS), not presolve. Draws from the **same LP-iteration budget as RENS/RINS** (`mip_heuristic_effort` envelope) and charges its work back — see the options section below. Gated by `heuristics::effective_flags(...).fpr`, i.e. it runs at `mip_heuristic_suite=fpr` and `=all` only — `off`, `local_mip` and `scylla` all disable it. It runs arm-aligned parallel workers (`w % kNumLpArms`): each of `num_threads` workers is bound to an LP arm from the curated Class 2/3a/3b list; excess workers wrap around the arm list with distinct seeds for diversity.
- `fj` — Feasibility Jump. Thin wrapper that delegates to HiGHS's built-in FJ implementation, run on continuous parallel workers.
- `local_mip` — weighted local search (MIP neighborhood search), run on continuous parallel workers.
- `scylla` — feasibility pump: alternates PDLP approximate LP solves with FPR rounding, progressive objective blending, and cycling perturbation. Runs N independent pump chains sharing a single `ContestedPdlp` instance (mutex-guarded `Highs` PDLP wrapper) so only one PDLP solve is in flight at a time. Workers that can't grab the PDLP mutex fall back to rounding against the most-recent *stale* snapshot so N-1 chains stay productive during a peer's solve (bounded by `kMaxStaleRounds` per worker before forcing a blocking solve; issue #76). Each chain owns its own warm-start, α_K decay, cycle history, RNG, and static FPR rounding strategy (`kFprConfigs[w % N]`). Runs on continuous parallel workers.

**Dispatch and parallel infrastructure** (`src/`):
- `mode_dispatch` — top-level presolve entry point. Reads `mip_heuristic_*` options and always runs the fixed chain via `run_sequential`: FJ → FPR → LocalMIP → Scylla one after another with a weighted effort budget, each on continuous parallel workers. The chain is a `constexpr` four-entry `HeuristicConfig` table (`{name, source_tag, weight, fixed_budget, flag, run}`) and `run_sequential` is a filtered loop over it — adding or reordering a heuristic is a table edit (#94). There is one parallel runner; the deterministic epoch-gated mode was removed in #92. Reproducible runs come from `threads=1` plus a fixed `random_seed` — one worker per heuristic, no new option.
- `heuristic_context.h` — the common runner contract (#94). All four heuristics expose `size_t run(const ProblemView&, const HeuristicBudget&, ExecutionContext&, IncumbentSink&)`. `ProblemView` is the model + CSC transpose + derived `ncol`/`nrow`/`nnz`, built **once per dispatch** by `make_problem` in `run_sequential` and shared by the chain (each heuristic used to build its own identical copy; `mipdata`'s row-major buffers are frozen by `runSetup()` before dispatch). `HeuristicBudget` is one heuristic's `total`/`per_worker`/`attempt_cap`/`stale` split from `make_budget`. `ExecutionContext` carries worker count, base seed, and the single `terminated()` predicate. `fpr_lp` takes only `make_exec`/`make_budget` — it keeps its own `LpFprSetup` for LP references and the shared `ContestedPdlp`.
- `incumbent_sink` — the only path from a worker to a solution. Owns the `SolutionPool`, the mutex around HiGHS's non-thread-safe `trySolution`, and the `kSolutionSource*` tag; workers call `sink.offer(obj, sol)` and never name their own tag. `run_sequential` re-tags via `set_source` between heuristics (legal only there — every parallel region has joined).
- `effort_ledger` — the only place *in `src/`* that writes `heuristic_effort_used` or emits the `[Sequential]` line (the patch adds one further `heuristic_effort_used +=` inside HiGHS's own `feasibilityJump()` — see `apply_patch.cmake` — live only at `suite=off`, and emitting no log line). `charge_presolve` for the chain; `charge_dive` for `fpr_lp`, which additionally depletes the RENS/RINS LP-iteration envelope. Nothing else in `src/` writes an upstream `HighsMipSolverData` counter.
- `worker_base.h` — `AttemptResult` (one attempt's effort + improvement flag) and `WorkerBudgetState` (per-worker effort / staleness / budget bookkeeping, embedded by composition into FJ / LocalMIP / Scylla workers).
- `opportunistic_runner.h` — the generic continuous parallel loop every heuristic runs on.
- `contested_pdlp` — mutex-guarded `Highs` PDLP wrapper shared by all Scylla workers. One-shot `solve(modified_cost, warm_start, epsilon, time_limit)` API holds the mutex for the full `changeColsCost → setSolution → run → getSolution` path, guaranteeing at most one PDLP solve is in flight (critical for cuPDLP GPU state). Also exposes `try_solve_or_snapshot(...)`: `try_lock` + fresh solve on success, or a lock-free `std::atomic<std::shared_ptr<const Snapshot>>` read of the most-recent completed solve on contention — lets N-1 Scylla workers keep rounding while one holds the mutex (issue #76). The in-flight counter is debug-asserted == 1 inside the critical section.
- `scylla_worker` / `pump_common.h` — Scylla worker class and shared feasibility-pump primitives (Mexi et al. 2023). Each worker is one `ScyllaWorker` (see `worker_base.h`) and runs its own chain of PDLP→FPR iterations.
- `fj_worker` / `fpr_worker` (inside fpr) — the FJ and FPR worker classes. `FprWorker` is declared inline in `src/fpr.cpp`; there is no `fpr_worker.*` file.

**Shared utilities** (`src/`):
- `heuristic_common.h` — `HeuristicResult`, `CscMatrix`, row violation, clamping, deadline helpers.
- `solution_pool` — thread-safe top-K solution pool with crossover restarts.

**HiGHS options** added or touched by the patch:
- `mip_heuristic_effort` — **vanilla semantics, vanilla default (0.05)**. This is upstream's B&B heuristic knob: `moreHeuristicsAllowed()` admits B&B-dive heuristics while `heuristic_lp_iterations < total_lp_iterations * mip_heuristic_effort` (plus an initial 10000-iteration offset). It gates RENS/RINS *and* `fpr_lp`: `fpr_lp::run` sizes each call to the remaining LP-iteration headroom of that envelope (converted at `nnz` effort-units per LP iteration, capped at `heuristic_effort_budget(nnz, mip_heuristic_effort)`) and charges all consumed work — reference-LP solves plus worker effort — back to `heuristic_lp_iterations`/`total_lp_iterations`, so fpr_lp competes with RENS/RINS for the same budget instead of drawing unaccounted work (fpr_lp skips entirely while `parallelLockActive()` — multi-worker B&B under `parallel=on` — because those counters are shared and it has no worker-local flush path). A patched binary at default options therefore matches vanilla's B&B heuristic budget exactly. (History: the patch used to raise the default to 0.30 and overload this option as the presolve budget — split in July 2026.)
- `mip_heuristic_presolve_effort` — effort budget multiplier for the custom **presolve** heuristics (FPR/LocalMIP/Scylla), double in `[0.0, 1.0]`, default `0.30`; budget = `nnz<<12 * (value/0.05)`. Treat it as a **normalised, wall-clock-equivalent** budget: `src/mode_dispatch.cpp` assigns FPR/LocalMIP/Scylla by per-heuristic `kWeight*` constants calibrated against geomean `effort_per_ms` so equal weights produce *roughly* equal wall-clock spend averaged across MIPLIB (per-instance variance is bounded by the drift limit, not eliminated). **Exception**: FJ uses a fixed per-worker budget of `nnz*1024` steps (matching vanilla HiGHS's hardcoded single-thread FJ limit, which neither effort option scales); N parallel FJ workers each run for that budget, and the remaining presolve budget is split among FPR/LocalMIP/Scylla. What FJ *charges* against that envelope is floored so a quarter always survives for the other three — `fj_budget` scales with worker count and the envelope does not, so uncapped it took everything from N≥24 at the default effort and the other three silently got zero. FJ itself always runs its full per-worker allowance; only the charge is capped. Use `bench/check_effort_drift.py` on a MIPLIB results tree to recalibrate when a heuristic's `effort_per_ms` shifts. Issue #71 has the full story. Note: as of round 4, LocalMIP's reported effort folds in the cold-start construction sweep (Phase B greedy sweep from `local_mip_construction.cpp`, charged in coefficient-access units to match `WorkerCtx::effort`) on dispatches where pool and incumbent are both empty; the cheap Phase A column-write loop is excluded from the normal-path total for unit consistency, but a small `ncol` charge is still reported on the degenerate `nrow==0 || max_effort==0` early exit (round-5 R2-2 fix). `kWeight*` constants: round-5 base (30s, threads=16, epoch-gated, 0.30/`nnz<<12*6` budget) scaled in round 6 (#92) by a measured A/B — that changeset sped the heuristics up by different factors (fpr 1.27x, local_mip 1.36x, scylla only 1.03x, since Scylla is PDLP/mutex-bound), so the ratios moved and the constants were rescaled rather than re-measured. Absolute ratios are strongly machine- and thread-count-dependent — the same code and instances at 6 workers give local_mip:scylla = 2.81 against round 5's 4.68 — so a fresh calibration must run on the 16-worker benchmark machine, and `bench/instances_small.txt` (restored in #92) is the set the base numbers came from. Rerun `bench/check_effort_drift.py` whenever a heuristic's effort accounting changes.
- `mip_heuristic_suite` — **the** heuristic selector, a string in `off | fj | fpr | local_mip | scylla | all`, default `all`. Replaced `mip_heuristic_preset` and the three `mip_heuristic_run_*` bools in #93. It also gates `fpr_lp` (via `heuristics::effective_flags`), so `suite=local_mip` and `suite=scylla` disable the dive-time heuristic too. HiGHS does not validate string option *values*: an unknown one is accepted by `setOptionValue` and caught at solve time, where the dispatcher warns and fails open to all four. `suite=off` restores HiGHS's own standalone FeasibilityJump call site, which makes it a genuine vanilla-equivalent ablation on the patched binary — see the benchmarking note below.
- `mip_heuristic_run_feasibility_jump` — upstream's own FJ switch, and the one pre-existing option the patch still reads. `false` disables FeasibilityJump at every suite value: at `off` it gates HiGHS's native call site, elsewhere it gates ours. `suite=off` plus this set to false is the pure patch-overhead configuration.

**Observability**: at `log_dev_level=3` `EffortLedger::book` (see `src/effort_ledger.cpp`) emits one `[Sequential] heur=<name> effort=<N> wall_ms=<X> effort_per_ms=<R>` line per presolve-chain heuristic per solve, plus one per `fpr_lp` dive dispatch (new in #94 — it used to do the work and report nothing). `bench/check_effort_drift.py` aggregates only the four presolve heuristics, since `fpr_lp` has no `kWeight*` to calibrate and draws from a different envelope. `bench/parse_highs_log.py` parses it and `bench/check_effort_drift.py` aggregates the samples to recalibrate `kWeight*`. Recalibrate on `bench/instances_small.txt` at `threads=16` — both are part of the measurement. `effort_per_ms` is a throughput and Scylla scales sublinearly in workers (PDLP mutex) where FPR/LocalMIP scale near-linearly, so the worker-count factor does not cancel in the ratios: the same binary on the same set gives local_mip:scylla = 4.68 at 16 workers and 2.81 at 6.

`effort_per_ms` measures *charged* effort, not useful work, so it is **not** a valid before/after metric for a change that alters how much effort a unit of real work charges — only for comparing heuristics against each other at fixed code (its calibration job). Removing redundant work usually *lowers* it: the measured A/B for #89 (six `bench/correctness_check.py` instances, seeds 0-2, 12 workers) has the incremental structures at local_mip 400k effort/ms against 1.13M for a naive-sweep build, while the same runs do 19,079 LocalMIP steps/ms against 1,876 — a 10x throughput *gain* that the effort rate reports as a 2.8x loss, chiefly because a naive feasible step charges a full `nnz` recheck it did not need. Measure real work units (LocalMIP steps, FPR DFS nodes per ms) when A/B-ing a hot-path change.

**Testing**: Catch2 v3. Tests use `.mps` instances from HiGHS's own `check/instances/` directory (path injected via `INSTANCES_DIR` compile definition). Characterization tests verify known-optimal objectives. (See the testing override near the top of this file — we do not use GoogleTest.)

**Benchmarking**: `bench/` has scripts for MIPLIB benchmarks — `run_benchmark.py` runs instances, `analyze_results.py` parses results. Don't pass `--threads` (or set `threads=` in an `.opts` file) unless asked — let HiGHS use its default; forcing `threads=1` collapses each heuristic to a single worker (the right setting for reproducibility, the wrong one for a throughput benchmark). Since #93, `mip_heuristic_suite=off` on the patched binary **is** vanilla-equivalent — the patch hands HiGHS's standalone FJ call site back at that value, which it used to hard-disable in every configuration. `bench/check_vanilla_equivalence.py` proves it against a separately built unpatched binary at the same tag (identical objective, node count, total and heuristic LP iterations, and an empty normalized log diff; verified 12/12 on the six bundled instances × 2 seeds). Prefer `--vanilla-binary` with a real unpatched build for headline benchmark baselines anyway — it is the stronger claim and costs nothing but a second checkout. Patched binaries self-identify with a `mip-heuristics patch active` line right after the version banner (injected by `apply_patch.cmake`) — the version/githash banner alone is identical between patched and unpatched builds of the same tag.
