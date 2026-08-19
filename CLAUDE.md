# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Navigation: LSP → narrow Grep → sliced Read.

# Standards

## Communication Style

Be terse. No preamble. No filler.

## Code Navigation

Prefer narrow queries over full-file reads:

1. **LSP** for symbol questions — `goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol`. Use before `Read`.
2. **Grep** with `-n` and a small `head_limit` (start at 20); raise only if inconclusive.
3. **Read** with `offset`/`limit` for a slice around the hit. Full-file `Read` is fine under ~200 lines or when structure matters.

Know the symbol → LSP. Know a string, not its location → Grep. Full-file Read is the last mile. This is a preference, not a prohibition: shelling out to `grep`/`rg` is fine when the built-in can't do the job (filtering a pipe like `git log | grep`, or a session without the `Grep` tool). What matters is bounding output, not which binary produces it.

Setup: install `clangd-lsp@claude-plugins-official` plus `clangd` (`apt install clangd`, or from LLVM). `.clangd` points at `build/compile_commands.json`, produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`.

## C++

- Target C++23. Use modern features (`std::expected`, concepts, ranges, `constexpr`).
- Use `#pragma once`. Minimize header includes; forward-declare where possible.
- **Formatting** is Google, via `.clang-format`. **Naming is not Google** — `.clang-tidy` enforces what this codebase actually does:

  | Kind | Style | Example |
  |---|---|---|
  | Functions (free and member) | `lower_case` | `run_sequential`, `charge_presolve` |
  | Locals, parameters, `const` locals | `lower_case` | `per_worker`, `max_effort` |
  | Private members | **trailing** `_` | `FjWorker::start_`, `seed_` |
  | File-scope / static / class / `constexpr` constants | `k` + `CamelCase` | `kWeightFpr`, `kNumLpArms` |
  | Classes | `CamelCase` | `ScyllaWorker`, `EffortLedger` |
  | Namespaces | `lower_case` | `heuristics`, `fpr` |
  | Files | `snake_case.h` / `.cpp` | |

  `ConstantMemberCase` is deliberately unset: it outranks `PrivateMemberSuffix`, so setting it would reject a `const` private member spelled with this project's `_` suffix. The `.clang-tidy` comments carry the reasoning for each narrowing — read them before changing one.

## Complexity

When a complexity warning fires, don't extract methods mechanically. Ask what the independent responsibilities are and split along those boundaries. If the function is genuinely complex because the domain is, add a comment explaining why and suppress the warning.

## CMake

- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` for clang-tidy.
- Use FetchContent for dependencies.
- A single root `CMakeLists.txt`; per-directory files would only add indirection.

## Testing (Catch2 v3)

- `TEST_CASE("name", "[tag]")` with `[tag]` filters, in `tests/`. Not GoogleTest.
- Instances come from HiGHS's own `check/instances/` (path injected as the `INSTANCES_DIR` compile definition).
- `ctest --progress` collapses the running list and `CMAKE_INSTALL_MESSAGE=LAZY` suppresses install chatter. Don't remove these.

## Development Workflow

```
plan (non-trivial) → implement → test → push to main
```

Nothing formats on save — there are no Claude Code hooks. `clang-format` runs at commit time: `pre-commit` formats the staged C++, applies safe `clang-tidy` fixes, and re-stages the result, so what you commit is canonical even though the file you just edited is not. Don't hand-tune formatting — let the hook normalize it, or run it yourself:

```
.venv/bin/clang-format -i <files>                       # normalize in place
.venv/bin/clang-format --dry-run --Werror <files>       # check only, non-zero if unformatted
```

**Use the venv's clang tools, not PATH.** The pinned 22.1.8 pair lives in `.venv/bin`, which is the exact path `cmake/Lint.cmake` searches; a different major version formats differently and fails the `clang_format` ctest gate. The hooks resolve the venv first for the same reason.

The hooks are **tracked in `.githooks/`** (`commit-msg`, `pre-commit`, `pre-push`, and the sourced `resolve-venv.sh`) — edit them there, not in `.git/hooks/`, which is empty of ours. Git only runs them when `core.hooksPath` points at that directory, and that setting is per-checkout config which cannot be tracked; `cmake -B build` sets it (option `MIP_HEURISTICS_INSTALL_GIT_HOOKS`, ON), leaving an existing `.githooks` value alone and warning rather than clobbering a hooksPath someone else set. To wire a clone up by hand: `git config core.hooksPath .githooks`. `.claude/` is gitignored, owned by this repo, and not part of the published artifact — edit it in place; it holds only permission rules and a statusline. A new Claude Code hook must read its file path and command from the hook JSON **on stdin**, never `$CLAUDE_FILE_PATH` (unset by Claude Code — a hook reading it no-ops silently rather than failing; the formatter this project inherited did exactly that). The clang-format and clang-tidy gates deliberately live outside the hooks, as ctest tests labelled `lint` plus the CI job, so a fresh clone still runs them. **Never `git push --no-verify` or `git commit --no-verify`** unless asked; a failing hook is a signal, so fix the root cause.

## Git Workflow

Trunk-based, linear history on main. Commit directly to main and push when local gates pass.

Feature branches are optional for larger changes: always branch from main (`git checkout main && git pull` first), never from another feature branch, keep them short-lived, and rebase or squash merge — no merge commits on main.

After a successful push:
- **Close any gh issue the work resolved**: `gh issue close <num> -c "<one-line note>"`, for every issue the push covers.
- **Delete the feature branch** if one was used: `git branch -d <branch>`, plus `git push origin --delete <branch>` if pushed.

## Commit Messages

Conventional Commits; the commit-msg hook enforces format.

- `type: description` or `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`, `build`, `ci`
- Subject ≤72 chars. Focus on **why**, not what.

## Issue Tracking

GitHub Issues, via the `gh` CLI.

- **Default to HTTPS** for GitHub remotes, not SSH.
- **Read an issue** with `gh issue view <num> --json title,body,labels,state,comments`; plain `gh issue view <num>` is deprecated for programmatic use.
- Don't defer work into a new issue unless it is substantial. Fix small follow-ups inline or leave them alone.

Issues get picked up cold, in fresh sessions, often by an agent with no access to this machine. So: keep the body **self-contained** (problem, motivation, acceptance criteria, repro steps); use **no local references** (`/home/user/...`, "see my other checkout" — dead links in a fresh session); prefer **stable external links** (GitHub permalinks, papers, RFCs); and **describe local code by concept, not path**, hinting that the agent can search under `..`, `../..`, or `~/code/`.

## Working Rules

- **CLAUDE.md discipline.** When Claude gets something wrong, fix CLAUDE.md in the same commit. It's a living document — update it whenever better instructions would have prevented the mistake.
- **Follow the agreed plan.** If a plan should change, stop and discuss — don't silently diverge. Same outside a written plan: if the current approach isn't working, say so rather than quietly switching strategies. Implement everything specified; no TODO placeholders or stubs unless asked.
- **Match references exactly.** Implementing from papers, pseudocode, or open source: no early exits, iteration limits, size caps, or "optimization" shortcuts that change behaviour. Introduce heuristic approximations only when asked. Implement the edge cases rather than simplifying them away. When in doubt, be faithful and let tests verify.
- **Don't invent APIs.** Verify functions, flags, and methods exist before using them.
- **Don't ignore type errors.** If clang-tidy or ruff flags something, fix the root cause — don't suppress.
- **Don't use deprecated patterns.** Check current docs, not training data.
- **Performance matters.** Most of this is solvers: profile before micro-optimizing, but don't sacrifice perf for "clean code".

# Project: mip-heuristics

## Project Overview

Custom MIP (Mixed-Integer Programming) heuristics integrated into the HiGHS solver via a patched fork. The heuristics run during HiGHS's presolve phase and are compiled as object files linked directly into the `highs` library target.

**Reader-facing docs**, kept in sync with this file — update both when the behaviour they describe moves:
- `README.md` — positioning, the `mip_heuristic_suite` table, recorded benchmark results, build options.
- `CONTRIBUTING.md` — build/test/lint commands, the git hooks and how a checkout gets them, the clean-rebuild rule for patch-script changes, the benchmarking rules, the standing code-hygiene bar.
- `docs/REPRODUCIBILITY.md` — what is reproducible and what is not, and the exact PLATO reproduction protocol.
- `docs/RELEASE.md` — how a version is cut and published: the gates, the artifact archive (`bench/make_archive.py`), the DOI wiring and its ordering constraint, and the release checklist. Publishing, not reproducing — it references `REPRODUCIBILITY.md` and `CONTRIBUTING.md` rather than restating them.
- `docs/PARAMETERS.md` — every tunable `constexpr`. **Verified by ctest** (`docs_parameter_references`, via `bench/check_docs_refs.py`): renaming a documented constant fails the suite. Entries name symbols, never line numbers — line numbers drifted on essentially every refactor, which is why they were dropped. Don't reintroduce them.

## Build Commands

```bash
# Lint tools, once per checkout.  `.venv/bin` is the exact path
# cmake/Lint.cmake searches; without this the gates are never registered and
# ctest reports green having linted nothing.  Include pytest — the root
# CMakeLists prefers `.venv/bin/python`, so a venv without it unregisters
# `bench_python_tests` rather than falling back to the system interpreter.
python3 -m venv .venv
.venv/bin/pip install clang-format==22.1.8 clang-tidy==22.1.8 pytest

# Configure (from repo root)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON

# Build
cmake --build build -j$(nproc)

# Run all tests, lint gates included
cd build && ctest --output-on-failure

# Fast inner loop: everything except the two lint gates
cd build && ctest -LE lint --output-on-failure

# Run a single test by name
cd build && ctest -R "execution-mode: flugpl objective" --output-on-failure

# Run tests matching a Catch2 tag
cd build && ./mip_heuristics_tests "[mode-matrix]"
```

First build is slow (~5 min) because it fetches and builds HiGHS via FetchContent.

**The lint gates are ctest tests** (#101), labelled `lint`: `clang_format` and
`clang_tidy`, registered by `cmake/Lint.cmake` over `src/` and `tests/` only.
They add roughly 30 s to a full `ctest` run — `ctest -LE lint` is the fast loop
while iterating, and the full suite is the gate. `MIP_HEURISTICS_REQUIRE_LINT=ON`
turns a missing tool or a wrong major version into a *configure* failure instead
of a warning that scrolls past; CI sets it, and so should you, because the
default failure mode is silent. Tool versions are part of the contract:
clang-format's output changes between major releases, so the pinned 22.1.8 pair
from PyPI is what the gate is written against. clang-tidy's own exit status is
unusable here — HiGHS's `HighsMipWorker.h` contains a construct clang rejects
as a parse error and GCC accepts — so `cmake/clang_tidy_gate.py` wraps it and
judges first-party diagnostics itself. Never work around a tidy finding by
widening that wrapper's filter.

## Build & Test

**`.githooks/pre-push` parses these three fenced blocks by tag** (`clean`,
`build`, `test`) and runs them as the push gate, so they are the single
definition of a full local run — keep them executable as written and keep the
tags.  The hook unsets `GIT_DIR` / `GIT_WORK_TREE` around each one: `git push`
exports `GIT_DIR=.git` into the hook environment, and CMake's nested `git clone`
inside FetchContent would then treat that as the repository it is cloning into,
failing with `fatal: invalid reference: v1.15.1` when it checks out the HiGHS
tag.  That belongs in the hook, not here, so the documented command stays the
one a human would type.

```clean
rm -rf build
```

```build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON && cmake --build build -j"$(nproc)"
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
- `heuristic_context.h` — the common runner contract (#94). All four heuristics expose `size_t run(const ProblemView&, const HeuristicBudget&, ExecutionContext&, IncumbentSink&)`. `ProblemView` is the model + CSC transpose + derived `ncol`/`nrow`/`nnz` + a copy of the incumbent, built **once per dispatch** by `make_problem` in `run_sequential` and shared by the chain (each heuristic used to build its own identical copy; `mipdata`'s row-major buffers are frozen by `runSetup()` before dispatch). **Workers must read `problem.incumbent`, never `mipdata->incumbent`** (#98): submission is immediate, so a peer's accepted solution runs `addIncumbent` — whose `incumbent = sol;` rewrites that buffer element-wise, and reallocates it outright on the empty-to-sized transition — while a worker indexes the live vector. Read it *after* the pool, never instead of it: `IncumbentSink` seeds the pool from the incumbent and every accepted solution goes through the pool first, so `copy_best` is the thread-safe way to see a peer's find, and a worker rebuilt mid-dispatch (FJ does this on staleness) that skipped the pool would silently lose it. Anything else a worker reads out of `mipdata` has to be either frozen by `runSetup()` or snapshotted the same way; `fpr_lp` keeps its own copies in `LpFprSetup`, and `seed_pool` reads the live vector only because it runs on the dispatching thread before any worker starts. `ProblemView::binary` is the second such snapshot (#99): `addIncumbent` also runs `getDomain().propagate()` and `redcostfixing.propagateRootRedcost`, which tighten the root domain bounds that `HighsDomain::isBinary` reads, so workers classify columns from the snapshot — LocalMIP via `WorkerCtx::is_binary`, FPR/Scylla/fpr_lp via `FprConfig::binary_mask`, which any caller inside a parallel region must set. The one live `isBinary` read left is `bucket_by_type` in `fpr_var_order.cpp`, reached only through `compute_var_order`, which every caller now runs on the dispatching thread — `fpr::precompute_var_orders`, `fpr_lp`'s `build_setup`, and `scylla::precompute_config_var_orders`. That last one was added by #99: `ScyllaWorker`'s constructor used to compute its own order, and `scylla::run` rebuilds retired workers *inside* the parallel loop, so a rebuild read the live domain and called `cliquePartition` concurrently with `addIncumbent`'s `extractObjCliques` — which reallocates the clique table. Any new `compute_var_order` caller must stay on the dispatching thread for the same reason. `HeuristicBudget` is one heuristic's `total`/`per_worker`/`attempt_cap`/`stale` split from `make_budget`. `ExecutionContext` carries worker count, base seed, and the single `terminated()` predicate. `fpr_lp` takes only `make_exec`/`make_budget` — it keeps its own `LpFprSetup` for LP references and the shared `ContestedPdlp`.
- `incumbent_sink` — the only path from a worker to a solution. Owns the `SolutionPool`, the mutex around HiGHS's non-thread-safe `trySolution`, and the `kSolutionSource*` tag; workers call `sink.offer(obj, sol)` and never name their own tag. `run_sequential` re-tags via `set_source` between heuristics (legal only there — every parallel region has joined). It also counts accepted offers (`accepted()`, atomic): that counter moving across one heuristic's dispatch is the `found` field of its `[Heur]` line, since a worker's return value is its effort and nothing else.
- `effort_ledger` — the only place *in `src/`* that writes `heuristic_effort_used` or emits the `[Sequential]` / `[Heur]` lines (the patch adds one further `heuristic_effort_used +=` inside HiGHS's own `feasibilityJump()` — see `apply_patch.cmake` — live only at `suite=off`, and emitting no log line). `charge_presolve` for the chain; `charge_dive` for `fpr_lp`, which additionally depletes the RENS/RINS LP-iteration envelope. `charge_presolve` also accumulates `presolve_heuristic_time`, a patch-added field with no upstream reader that feeds the `[Root]` line. `EffortLedger::now_s()` is the solver's own clock (`HighsMipSolver::timer_`), not a raw `steady_clock`, so `[Heur] start_s/end_s` share an origin with `[Root] lp_time_s` — that comparison is the whole point of #95's instrumentation. Nothing else in `src/` writes an upstream `HighsMipSolverData` counter.
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

Three further lines at the same level, added by #95 for the cannibalization analysis, all `key=value` with a stable leading token:

- `[Heur] name=<n> phase=<presolve|dive> start_s=<S> end_s=<E> effort=<N> wall_ms=<X> effort_per_ms=<R> found=<0|1>` — same emission site as `[Sequential]`, one alongside each. It carries what `[Sequential]` cannot: *when* the heuristic ran on the solver clock, which side of the patch boundary it ran on, and whether it produced anything. `[Sequential]` stays byte-identical alongside it because `check_effort_drift.py` and the `kWeight*` recalibration procedure parse it; retiring it is a separate, later decision (epic #88 coupling F).
- `[Native] rens=<N> rens_root=<N> rins=<N> rcfix=<N> heur_lp_iters=<N> total_lp_iters=<N> fpr_lp_lp_iters=<N>` and `[Root] lp_time_s=<S> presolve_heur_s=<S>` — once per solve, from `heuristics::log_solve_summary` (`src/mode_dispatch.cpp`), called from `HighsMipSolver::cleanupSolve` by the patch so it fires exactly once on every exit path. Sub-MIP solves return immediately (RENS/RINS build their own `HighsMipSolver`; their counters describe a different model). The call counts come from patch-added `std::atomic<size_t>` fields on `HighsMipSolverData`, incremented at the existing call sites — atomic because the B&B-dive RENS/RINS site runs concurrently across workers under `parallel=on`. **Both lines are emitted at `suite=off` too, deliberately**: that run is the vanilla reference the patched rows are compared against. They are invisible below `log_dev_level=3`, so `bench/check_vanilla_equivalence.py`'s normalized log diff is unaffected — it masks both tags anyway, so raising `log_dev_level` there stays safe.
- **`heur_lp_iters` and `total_lp_iters` are shared, not native.** `EffortLedger::charge_dive` writes both so `fpr_lp` competes with RENS/RINS for one envelope, which means reading them raw bills our dive work as HiGHS's: on flugpl at seed 1, `suite=local_mip` gives 169 and `suite=all` gives 1294 with *identical* rens/rins/rcfix counts. `fpr_lp_lp_iters` reports exactly what we charged to *each* of them; subtract it (`NativeCounters.native_heur_lp_iters` / `native_total_lp_iters`) before comparing an `off` row against a patched one. Likewise `rens` merges the root and dive call sites while `rens_root` isolates the root gate — the gate a presolve-found incumbent actually closes, and the one whose suppression the merged total can hide.
- `lp_time_s` is `-1.000` when the root LP was never reached; the parser reports `time_to_root_lp is None` rather than `t=0`. `presolve_heur_s` is the **full chain span** including the shared setup `run_sequential` hoisted out of the four heuristics (`make_problem` / `build_csc` / `seed_pool`), which is deliberately *not* the sum of the `[Heur]` windows — those stay scoped to what `kWeight*` calibrates, and the calibration basis must not move. `SolveResult.heuristic_wall_fraction` (sum of `[Heur]` wall time over `Timing`) is the cannibalization headline number; it returns `0.0` rather than `None` on an instrumented run with no `[Heur]` lines, so the `suite=off` baseline row is a real zero instead of being dropped by whatever filters `None`.
- The ledger times against HiGHS's own solver clock (`HighsMipSolver::timer_`) so `[Heur]` windows are comparable with `[Root] lp_time_s`. That clock is **not** monotonic — `HighsTimer` bottoms out in `high_resolution_clock`, which libstdc++ aliases to `system_clock` — so a wall-clock step can produce a negative `wall_ms`. Both bench regexes accept the sign so such a sample surfaces instead of silently vanishing.

`effort_per_ms` measures *charged* effort, not useful work, so it is **not** a valid before/after metric for a change that alters how much effort a unit of real work charges — only for comparing heuristics against each other at fixed code (its calibration job). Removing redundant work usually *lowers* it: the measured A/B for #89 (six `bench/correctness_check.py` instances, seeds 0-2, 12 workers) has the incremental structures at local_mip 400k effort/ms against 1.13M for a naive-sweep build, while the same runs do 19,079 LocalMIP steps/ms against 1,876 — a 10x throughput *gain* that the effort rate reports as a 2.8x loss, chiefly because a naive feasible step charges a full `nnz` recheck it did not need. Measure real work units (LocalMIP steps, FPR DFS nodes per ms) when A/B-ing a hot-path change.

**Testing**: Catch2 v3. Tests use `.mps` instances from HiGHS's own `check/instances/` directory (path injected via `INSTANCES_DIR` compile definition). Characterization tests verify known-optimal objectives. (See the testing override near the top of this file — we do not use GoogleTest.)

**Benchmarking**: `bench/` has scripts for MIPLIB benchmarks — `run_benchmark.py` runs instances, `analyze_results.py` parses results. The MIPLIB collection itself is **not** located at a fixed path: `bench/download_miplib.sh` and `run_benchmark.py --data-dir` share a search path — explicit argument, then `$MIPLIB_DIR`, then `~/data/miplib`, then `/tmp/miplib` — and the first directory holding more than 200 `.mps.gz` files wins. It is 3.5 GB, so it lives once per machine outside every checkout, and `/tmp` is probed (an existing copy is reused) but never written to (it does not survive a reboot). **That search path and its >200 threshold are defined twice, once in bash and once in Python**, and drift between them is silent and expensive — the downloader writes one directory while the benchmark reads another. Three tests in `bench/test_run_benchmark.py` parse the shell script and assert both against the Python constants; keep them passing rather than editing one side alone. Don't pass `--threads` (or set `threads=` in an `.opts` file) unless asked — let HiGHS use its default; forcing `threads=1` collapses each heuristic to a single worker (the right setting for reproducibility, the wrong one for a throughput benchmark). Since #93, `mip_heuristic_suite=off` on the patched binary **is** vanilla-equivalent — the patch hands HiGHS's standalone FJ call site back at that value, which it used to hard-disable in every configuration. `bench/check_vanilla_equivalence.py` proves it against a separately built unpatched binary at the same tag (identical objective, node count, total and heuristic LP iterations, and an empty normalized log diff; verified 12/12 on the six bundled instances × 2 seeds). Prefer `--vanilla-binary` with a real unpatched build for headline benchmark baselines anyway — it is the stronger claim and costs nothing but a second checkout. Patched binaries self-identify with a `mip-heuristics patch active` line right after the version banner (injected by `apply_patch.cmake`) — the version/githash banner alone is identical between patched and unpatched builds of the same tag.

`run_benchmark.py`'s config surface (#96) is one entry of `CONFIG_SUITES` per `mip_heuristic_suite` value plus `patched` as an alias for `all` — an alias, *not* a rename of the `all_opp` composition the recorded PLATO table was measured at, which the suite option cannot express. **`config_options` raises on an unknown config name**; it used to return `{}`, so `--configs patchd` produced a fully populated, plausible-looking, completely meaningless results tree. `--budget-sweep` crosses each config with `mip_heuristic_presolve_effort` values into `<output>/<config>@e<V>/seed<N>/`, which `analyze_results.py --configs` consumes unchanged (`@` needs no escaping in a path or in LaTeX text mode). `SWEEP_EXEMPT` names the three configs the option provably does not reach — `vanilla` and `off` (no presolve heuristic runs) and `fj` (fixed per-worker allowance; measured flat to 0.003% across the option's whole range, while the same sweep moves fpr/local_mip/scylla inside `all` by 22–231x) — which pass through the sweep once as anchor rows rather than becoming N identical trees; an explicit `vanilla@e0.30` raises. `resolve_config` is the single decomposer of a `<base>@e<V>` name and `build_plan` the single place that decomposition's two *consequences* — which binary, which options — are chosen together; choosing them at separate use sites is how a swept name picks one branch's binary and the other branch's options. A run that solved but ignored its configuration is routed to `<inst>.log.err` like a non-solving exit: HiGHS accepts an unknown `mip_heuristic_suite` *value* and `run_presolve` fails open to all four heuristics with a warning and exit 0, so an `off/` tree would otherwise record runs that executed `all`.

**Cannibalization tables** (#100): `bench/analyze_results.py --cannibalization` renders the internal-budget and wall-clock tables over a results tree. It needs a **patched `suite=off`** row as the baseline, not a `--vanilla-binary` row — the baseline must itself be instrumented, and an external unpatched binary emits none of the `[Native]` / `[Root]` / `[Heur]` lines, so it classifies as `not-instrumented` and every other row degrades to `no-baseline`. Auto-detection tries `vanilla`, `off`, `suite_off`, `baseline` in that order after a structural test; `--cannibalization-baseline NAME` overrides it. The classification is a triage label, not a statistical test: thresholds are the `CANNIBALIZATION_*` module constants. The category set is `CANNIBALIZATION_CATEGORIES` in `analyze_results.py` — `baseline`, `neutral`, `wall-clock`, `internal-budget`, `both`, `no-baseline`, `not-instrumented` — derived in #100 from the two axes epic #88 defines. Only four are cannibalization kinds; `baseline`, `no-baseline` and `not-instrumented` are data-availability states, so do not read the count as a taxonomy.

**`log_dev_level=3` is not free, so `run_benchmark.py` leaves it off** behind an opt-in `--dev-log`. Our instrumentation is `kVerbose`, which needs level 3, and HiGHS's own FeasibilityJump logs `Reached a local minimum.` at that same level from `updateWeights()` — once per weight bump, per parallel FJ worker, each with an `fflush`. Measured on five bundled instances at a 10 s limit: 97–750x log volume (bell5 16 KB → 3.5 MB) and 1.1–4.4x total solve wall time (egout 0.048 s → 0.212 s; flugpl 2.7x; p0548 only 1.1x), concentrated in the FJ phase — exactly the window `[Heur]` and `presolve_heur_s` report. Attribution runs and headline-timing runs are therefore different runs.

**Release archive** (#102): `bench/make_archive.py build <results-tree> --output <dir> --time-limit T` packages a results tree — logs, the `<instance>.opts` each run was given, generated tables, `MANIFEST.json` + `PROVENANCE.md` — and `verify` (wrapped by the archive's own `REGENERATE.sh`) re-derives every table from the archived logs and diffs it, so "regenerable from the archive alone" is checked rather than claimed. The provenance is *derived*, not asserted: which binary produced a config comes from the `mip-heuristics patch active` marker in each log, since patched and unpatched builds of the same tag have identical version/githash banners; the baseline's claim ("vanilla-equivalent setting on the patched binary" vs "separately built unpatched binary") follows from that same marker and they are not interchangeable; instrumentation is recorded both as requested (`log_dev_level` in the `.opts`) and as observed (`[Heur]`/`[Native]`/`[Root]`/`[Sequential]` tags), because a disagreement is the `--extra-options log_dev_level=1` failure — note that observed test is broader than `analyze_results.is_instrumented`, which keys on `[Native]` alone, so a pre-#95 log carrying only `[Sequential]` counts as instrumented here and `not-instrumented` there. The cannibalization table is offered when every *patched* config is instrumented, not every config: an externally built unpatched `vanilla` arm emits no tags by construction, and requiring it would drop the table from exactly the tree shape the docs recommend. `--time-limit` is required because HiGHS takes it on the command line, so it is not in any `.opts`. An unset `threads` warns: the harness deliberately does not pin it, so the effective worker count is the *run* machine's core count and the machine block is auto-detected on the *archive* host — hence `--machine-note`. A campaign yields two archives, headline-timing and `--dev-log` attribution; they are different runs. `docs/RELEASE.md` is the process around it, including the ordering constraint that Zenodo's GitHub integration must be enabled **before** the release is created (it does not backfill) and that `.zenodo.json` makes Zenodo ignore `CITATION.cff` entirely.
