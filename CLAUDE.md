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
cd build && ctest -R "mode-matrix det: flugpl objective" --output-on-failure

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

GPU acceleration: `-DMIP_HEURISTICS_CUDA=ON` enables CUDA for the PDLP solver used by Scylla. Falls back to CPU if no CUDA compiler is found.

## Architecture

**Integration model**: Heuristics are compiled as a static object library (`mip_heuristics`) whose objects are injected into the HiGHS `highs` target. The HiGHS source is fetched at build time (v1.15.1) with patches applied from `third_party/highs_patch/`. Heuristics access HiGHS internals directly via `HighsMipSolver&`.

**Heuristic entry points** — each has a standalone `run()` that HiGHS calls during presolve:
- `fpr` — Fix, Propagate, and Repair. DFS tree search that fixes integers, propagates bounds, backtracks on infeasibility, then runs WalkSAT/RepairSearch to fix remaining violations. `fpr_core.cpp` exposes two APIs: the one-shot `fpr_attempt` (used by scylla / fpr_lp / one-shot tests) and the `fpr_attempt_begin` / `fpr_attempt_step` / `fpr_attempt_finish` lifecycle (issue #77) that lets `FprWorker` pause an in-flight DFS at the per-epoch budget gate and resume it next epoch with state in `FprAttemptState` + `FprScratch`. The pause/resume mechanic is what avoids discarding work on long DFS subtrees in seq/det; multi-attempt looping inside `run_epoch` lets fast workers fill the slice with new attempts (rotated through `kInitialFprConfigs` per `(worker_idx + attempt_idx)`) instead of idling at the runner barrier. Sub-algorithms: `prop_engine` (bound propagation, with both forward propagate and a domain PQ for dynamic-var strategies — `repair_search` requires `e_pq_mark` threading on `RepairSearchNode` to keep the PQ consistent across its secondary backtracks), `walksat`, `repair_search`, `fpr_strategies` (strategy variants).
- `fpr_lp` — LP-dependent FPR (paper Classes 2–3) using root LP solution. Called during B&B dive (after RENS/RINS), not presolve. Draws from the **same LP-iteration budget as RENS/RINS** (`mip_heuristic_effort` envelope) and charges its work back — see the options section below. Gated by `heuristics::effective_flags(...).fpr` so `mip_heuristic_preset` reaches it. `mip_heuristic_opportunistic` selects between the two variants: arm-aligned parallel workers (`w % kNumLpArms`) in either epoch-gated or continuous mode. Each of `num_threads` workers is bound to an LP arm from the curated Class 2/3a/3b list; excess workers wrap around the arm list with distinct seeds for diversity.
- `fj` — Feasibility Jump. Thin wrapper that delegates to HiGHS's built-in FJ implementation. Has both deterministic (epoch-gated) and opportunistic (continuous) parallel modes.
- `local_mip` — weighted local search (MIP neighborhood search). Has both deterministic (epoch-gated) and opportunistic (continuous) parallel modes.
- `scylla` — feasibility pump: alternates PDLP approximate LP solves with FPR rounding, progressive objective blending, and cycling perturbation. Runs N independent pump chains sharing a single `ContestedPdlp` instance (mutex-guarded `Highs` PDLP wrapper) so only one PDLP solve is in flight at a time. Workers that can't grab the PDLP mutex fall back to rounding against the most-recent *stale* snapshot so N-1 chains stay productive during a peer's solve (bounded by `kMaxStaleRounds` per worker before forcing a blocking solve; issue #76). Each chain owns its own warm-start, α_K decay, cycle history, RNG, and static FPR rounding strategy (`kFprConfigs[w % N]`). Has both deterministic (epoch-gated) and opportunistic (continuous) parallel modes.

**Dispatch and parallel infrastructure** (`src/`):
- `mode_dispatch` — top-level presolve entry point. Reads `mip_heuristic_*` options and always runs the fixed chain via `run_sequential`. One flag, `mip_heuristic_opportunistic`, picks the parallelism strategy:
  - **seq/det** (`opportunistic=false`): `run_sequential` runs FJ → FPR → LocalMIP → Scylla one after another with a weighted effort budget, each on epoch-gated parallel workers.
  - **seq/opp** (`opportunistic=true`): same sequence but each heuristic's `run_parallel` dispatches to its opportunistic variant (continuous parallel workers rather than epoch-gated).
  The flag is threaded to all four heuristics (FJ, FPR, LocalMIP, Scylla) and to `fpr_lp`; each picks its epoch-gated or continuous runner.
- `epoch_runner.h` — generic epoch loop: workers run in parallel within each epoch and synchronize at the barrier. `EpochWorker` concept defines the interface.
- `opportunistic_runner.h` — generic continuous parallel loop used when `mip_heuristic_opportunistic=true`.
- `contested_pdlp` — mutex-guarded `Highs` PDLP wrapper shared by all Scylla workers. One-shot `solve(modified_cost, warm_start, epsilon, time_limit)` API holds the mutex for the full `changeColsCost → setSolution → run → getSolution` path, guaranteeing at most one PDLP solve is in flight (critical for cuPDLP GPU state). Also exposes `try_solve_or_snapshot(...)`: `try_lock` + fresh solve on success, or a lock-free `std::atomic<std::shared_ptr<const Snapshot>>` read of the most-recent completed solve on contention — lets N-1 Scylla workers keep rounding while one holds the mutex (issue #76). The in-flight counter is debug-asserted == 1 inside the critical section.
- `scylla_worker` / `pump_common.h` — Scylla worker class and shared feasibility-pump primitives (Mexi et al. 2023). Each worker is one `ScyllaWorker` conforming to `EpochWorker` and runs its own chain of PDLP→FPR iterations.
- `fj_worker` / `fpr_worker` (inside fpr) — epoch-gated workers for FJ and FPR respectively.

**Shared utilities** (`src/`):
- `heuristic_common.h` — `HeuristicResult`, `CscMatrix`, row violation, clamping, deadline helpers.
- `solution_pool` — thread-safe top-K solution pool with crossover restarts.

**HiGHS options** added or touched by the patch:
- `mip_heuristic_effort` — **vanilla semantics, vanilla default (0.05)**. This is upstream's B&B heuristic knob: `moreHeuristicsAllowed()` admits B&B-dive heuristics while `heuristic_lp_iterations < total_lp_iterations * mip_heuristic_effort` (plus an initial 10000-iteration offset). It gates RENS/RINS *and* `fpr_lp`: `fpr_lp::run` sizes each call to the remaining LP-iteration headroom of that envelope (converted at `nnz` effort-units per LP iteration, capped at `heuristic_effort_budget(nnz, mip_heuristic_effort)`) and charges all consumed work — reference-LP solves plus worker effort — back to `heuristic_lp_iterations`/`total_lp_iterations`, so fpr_lp competes with RENS/RINS for the same budget instead of drawing unaccounted work (fpr_lp skips entirely while `parallelLockActive()` — multi-worker B&B under `parallel=on` — because those counters are shared and it has no worker-local flush path). A patched binary at default options therefore matches vanilla's B&B heuristic budget exactly. (History: the patch used to raise the default to 0.30 and overload this option as the presolve budget — split in July 2026.)
- `mip_heuristic_presolve_effort` — effort budget multiplier for the custom **presolve** heuristics (FPR/LocalMIP/Scylla), double in `[0.0, 1.0]`, default `0.30`; budget = `nnz<<12 * (value/0.05)`. Treat it as a **normalised, wall-clock-equivalent** budget: `src/mode_dispatch.cpp` assigns FPR/LocalMIP/Scylla by per-heuristic `kWeight*` constants calibrated against geomean `effort_per_ms` so equal weights produce *roughly* equal wall-clock spend averaged across MIPLIB (per-instance variance is bounded by the drift limit, not eliminated). **Exception**: FJ uses a fixed per-worker budget of `nnz*1024` steps (matching vanilla HiGHS's hardcoded single-thread FJ limit, which neither effort option scales); N parallel FJ workers each run for that budget, and the remaining presolve budget is split among FPR/LocalMIP/Scylla. Use `bench/check_effort_drift.py` on a MIPLIB results tree to recalibrate when a heuristic's `effort_per_ms` shifts. Issue #71 has the full story. Note: as of round 4, LocalMIP's reported effort folds in the cold-start construction sweep (Phase B greedy sweep from `local_mip_construction.cpp`, charged in coefficient-access units to match `WorkerCtx::effort`) on dispatches where pool and incumbent are both empty; the cheap Phase A column-write loop is excluded from the normal-path total for unit consistency, but a small `ncol` charge is still reported on the degenerate `nrow==0 || max_effort==0` early exit (round-5 R2-2 fix). `kWeight*` constants were last calibrated in round 5 (30s, threads=16, seq/det, at the same 0.30/`nnz<<12*6` budget — the calibration carries over unchanged across the option split). Rerun `bench/check_effort_drift.py` whenever a heuristic's effort accounting changes.
- `mip_heuristic_run_fpr`, `mip_heuristic_run_local_mip`, `mip_heuristic_run_scylla` — enable/disable individual heuristics. `mip_heuristic_run_fpr` also gates `fpr_lp` (via `heuristics::effective_flags`, so a preset overrides it — `preset=off` disables fpr_lp too; the raw option defaults to `true`).
- `mip_heuristic_run_feasibility_jump` — enable FJ.
- `mip_heuristic_opportunistic` — use continuous (opportunistic) parallelism rather than deterministic epoch-gated parallelism. The only dispatch knob for both presolve and `fpr_lp`.

**Observability**: at `log_dev_level=3` seq/det emits one `[Sequential] heur=<name> effort=<N> wall_ms=<X> effort_per_ms=<R>` line per heuristic per solve (see `src/mode_dispatch.cpp`). `bench/parse_highs_log.py` parses it and `bench/check_effort_drift.py` aggregates the samples to recalibrate `kWeight*`.

**Testing**: Catch2 v3. Tests use `.mps` instances from HiGHS's own `check/instances/` directory (path injected via `INSTANCES_DIR` compile definition). Characterization tests verify known-optimal objectives. (See the testing override near the top of this file — we do not use GoogleTest.)

**Benchmarking**: `bench/` has scripts for MIPLIB benchmarks — `run_benchmark.py` runs instances, `analyze_results.py` parses results. Don't pass `--threads` (or set `threads=` in an `.opts` file) unless asked — let HiGHS use its default; forcing `threads=1` collapses epoch-gated / opportunistic parallelism to one worker per epoch. **Vanilla baselines must use an unpatched HiGHS binary** (`--vanilla-binary`, e.g. the system `/usr/local/bin/highs` at the matching tag): `mip_heuristic_preset=off` on the patched binary is the *ablation* config, not vanilla — the patch hard-disables vanilla's standalone FJ call site, so preset=off lacks the FJ that real vanilla runs by default, and there is no option combo on the patched binary that reproduces vanilla's single-threaded FJ. Patched binaries self-identify with a `mip-heuristics patch active` line right after the version banner (injected by `apply_patch.cmake`) — the version/githash banner alone is identical between patched and unpatched builds of the same tag.
