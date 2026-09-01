# Contributing

## Build and test

```bash
# Lint tools, once per checkout.  `.venv/bin` is the exact path the CMake
# lint module searches, so this install is what registers the gates.  `ruff`
# belongs here and not only in the Python section below: both git hooks guard
# their ruff step on `[ -x "$VENV_BIN/ruff" ]`, so a venv without it skips
# Python linting *silently* while CI still gates on it — local hooks pass, CI
# goes red.
python3 -m venv .venv
.venv/bin/pip install clang-format==22.1.8 clang-tidy==22.1.8 ruff==0.16.3 pytest

# Also points core.hooksPath at .githooks/ — see "Git hooks" below.
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -j$(nproc)
```

The first build fetches HiGHS v1.15.1 via FetchContent and applies the patches
in `third_party/highs_patch/`; compiling that tree is most of the wall time and
how much depends on your core count.

**Install `ccache` if you have not** (`apt install ccache`). It is optional --
CMake reports `ccache: not found, compiling without a cache` and carries on --
but the `pre-push` gate is a *clean* rebuild by design, so without it every push
recompiles HiGHS from scratch. With it, a clean rebuild goes from ~48 s to under
a second on the same machine. Opt out with `-DMIP_HEURISTICS_CCACHE=OFF`.

Useful narrower invocations:

```bash
ctest --test-dir build -LE lint --output-on-failure       # skip the lint gates
ctest --test-dir build -R "execution-mode: flugpl objective" --output-on-failure
./build/mip_heuristics_tests "[mode-matrix]"              # Catch2 tag filter
```

Tests are Catch2 v3 (`TEST_CASE` with `[tag]` filters), not GoogleTest.

`[serial]` is a reserved tag, and the last resort rather than the first. It
marks the handful of cases that a loaded machine can fail with nothing wrong
in the code — the presolve-deadline cases, effort-zero's setup-free window,
the two `ContestedPdlp` cases that race a sleeping thread, and the
`[HeurSol]` coverage case, which needs real worker parallelism rather than a
clock. `CMakeLists.txt` registers them through a second
`catch_discover_tests` call carrying `RUN_SERIAL TRUE`, so ctest runs them
alone.

When a test flakes under `-j$(nproc)`, work down this list:

1. **Take the clock out of the fixture.** `test_deadline.cpp` was failing in
   `readModel`, because HiGHS's free-format MPS reader treats `time_limit` as
   its own *parse* budget — parsing `gesa2` costs 4-5 ms against a 0.1 s
   limit, and saturation closes that margin. Setting the option after the
   read fixes it outright, for every machine, including CI boxes with noisy
   neighbours that `RUN_SERIAL` cannot see.
2. **Assert the mechanism.** A starved runner spends *less* effort, so an
   effort bound holds under load where a time bound does not.
3. **Only then tag it**, for a wait nothing in the code can bound.

**Never widen the threshold.** A wall-clock bound tuned until it stops
failing is the same test with a longer fuse — and in the case above it would
not have worked at all, since the failure was not a near-miss.

### Build options

| Flag | Default | Meaning |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | — | Optimised build. Use it; the heuristics are unusable at `-O0`. |
| `-DMIP_HEURISTICS_REQUIRE_LINT=ON` | `OFF` | Missing or wrong-version clang tools fail the *configure* instead of warning. CI sets it; set it locally too. |
| `-DMIP_HEURISTICS_INSTRUMENT=OFF` | `ON` | Compiles out the LocalMIP warm-start branch counters. They are consumed by two tests, so leave them on unless you are measuring counter overhead. |
| `-DMIP_HEURISTICS_CUDA=ON` | `OFF` | cuPDLP GPU backend for Scylla. Requires `CUDA_HOME` exported and **fails the configure** rather than falling back to CPU. Build it into a separate tree. |

## Lint gates

`clang-format` and `clang-tidy` run over `src/` and `tests/` only — never over
the fetched HiGHS or Catch2 trees — and are registered as ctest tests labelled
`lint`. A full `ctest` run is roughly 30 s longer because of them.

**Tool versions are part of the contract.** clang-format's output changes
between major releases, so a gate is only reproducible if everyone runs the
same one. The pinned pair is `clang-format==22.1.8` / `clang-tidy==22.1.8` from
PyPI, installed into `.venv/` (which is gitignored). A different major version
is a configure-time warning, and a hard error under
`MIP_HEURISTICS_REQUIRE_LINT=ON`.

clang-tidy's raw exit status cannot be used on this project: one of HiGHS's own
headers (`HighsMipWorker.h`) contains a construct clang rejects as a parse
error and GCC accepts, so every invocation exits non-zero regardless of our
code. `cmake/clang_tidy_gate.py` wraps the tool and judges first-party
diagnostics itself. If a tidy finding is in the way, fix the root cause — do
not widen the wrapper's filter and do not add a blanket `NOLINT`.

`.clang-format`, `.clang-tidy` and `.clangd` are ordinary tracked files owned by
this repository — nothing generates or refreshes them. Two settings in
`.clang-tidy` are load-bearing and easy to undo by accident: a
`HeaderFilterRegex` anchored on a literal `/src/` or `/tests/` path segment (the
usual `.*` reports thousands of findings inside the fetched HiGHS headers), and
a `lower_case` function naming convention matching this codebase. Each
narrowing carries a comment explaining what it deviates from and why; read it
before widening one.

Nothing formats on save. `clang-format` runs at commit time, from the
pre-commit hook, over the staged C++ — so don't hand-fix formatting.

## Git hooks

Three hooks are tracked in `.githooks/`, plus the `resolve-venv.sh` they source:

| Hook | Does | Blocks? |
|---|---|---|
| `commit-msg` | Conventional Commits format, subject ≤72 chars | yes |
| `pre-commit` | formats staged C++ with the venv's pinned `clang-format`, applies safe `clang-tidy` fixes, re-stages, runs `ctest -LE lint` | yes, on test failure |
| `pre-push` | clean rebuild plus the full suite (lint gates included), then advisory `ruff` / `shellcheck` | yes, on build or test failure |

Git only runs hooks from `core.hooksPath`, which is per-checkout config and
cannot be tracked. `cmake -B build` sets it for you
(`-DMIP_HEURISTICS_INSTALL_GIT_HOOKS=OFF` opts out); it leaves an existing
`.githooks` value alone and warns rather than clobbering a hooksPath someone
else set. By hand:

```bash
git config core.hooksPath .githooks
```

`pre-push` reads the `clean` / `build` / `test` fenced blocks out of
`CLAUDE.md`'s `## Build & Test` section, so a full push costs a from-scratch
HiGHS build — about five to six minutes. That is deliberate: it is the same
sequence a release runs. A docs-only or hooks-only push skips it. Edit the
hooks in `.githooks/`; `.git/hooks/` holds none of ours.

## The clean-rebuild rule

**Any change to `third_party/highs_patch/apply_patch.cmake` requires a clean
rebuild of the HiGHS tree before it takes effect:**

```bash
rm -rf build/_deps/highs-src build/_deps/highs-subbuild build/CMakeCache.txt
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON
cmake --build build -j$(nproc)
```

The reason is that the patch script decides "is this tree already patched?" by
searching for text it previously inserted. An existing `build/_deps` tree was
patched by the *old* script, so a modified script either declines to run at all
or appends its new text to a tree that still carries the old layout — and the
failure then surfaces somewhere unrelated, typically as a compile error in
`src/mode_dispatch.cpp`.

Two consequences worth internalising:

- **Bump `PATCH_VERSION` whenever you change inserted text.** The script stamps
  `mip-heuristics patch version N` into `HighsOptions.h` and rejects any tree
  carrying our options without the current marker. That sentinel is what turns
  a silent mis-patch into a clear configure error. There is a second probe
  listing identifiers the script *used* to insert and no longer does, for trees
  that predate the marker; add to it whenever the script stops inserting a name.
- **CI's dependency cache key includes the patch script**, not just the file
  that pins the HiGHS tag, for exactly this reason. A cache entry keyed on the
  tag alone would restore a stale-layout tree that the sentinel then rejects.

All nineteen of the script's failure messages now end in one shared
`CLEAN_REBUILD` string that names this section and carries the command, rather
than nineteen copies free to drift apart.

## Benchmarking

Full detail is in `README.md` (per-heuristic ablation, budget sweep,
instance subsets and the config oracle) and `docs/REPRODUCIBILITY.md` (the
PLATO protocol).
Three rules that are easy to get wrong:

- **The vanilla baseline is a second binary.** `--vanilla-binary` must point at
  a separately built *unpatched* HiGHS of the tag in `cmake/FetchHiGHS.cmake`;
  the `vanilla` config requires it and the run is refused before its first
  solve if the binary carries the `mip-heuristics patch active` marker or
  reports a different version. `mip_heuristic_suite=off` on the patched binary
  is the ablation with our heuristics disabled, and is never a substitute.
- **Do not set `threads`.** Neither `--threads` on `bench/run_benchmark.py` nor
  `threads=` in an `.opts` file, unless you have been asked to. Forcing
  `threads=1` collapses every heuristic to a single worker: it is the right
  setting for reproducibility and the wrong one for a throughput benchmark, and
  it is not what the recorded numbers were measured at.
- **`--dev-log` is a different run, not a free extra.** Turning on
  `log_dev_level=3` costs 97–750x the log volume and 1.1–4.4x the wall time,
  concentrated in the FeasibilityJump phase — which is exactly the window the
  attribution numbers measure. Use it for attribution runs and leave it off for
  headline timings.

## Python (`bench/`)

The benchmark harness is Python with its own test suite, registered in ctest as
`bench_python_tests` and also runnable directly:

```bash
.venv/bin/ruff check bench cmake
.venv/bin/python -m pytest bench
```

`pytest` picks up `testpaths = ["bench"]` from `pyproject.toml`, so a bare
`pytest` from the repo root works too. Both run in CI as a separate fast job
with no C++ build.

Alongside the per-script unit tests, `bench/test_campaign_readiness.py` runs
the scripts end to end against a fake `highs` that records the argv and
options file it was handed: the tree layout, the options each run is given,
the `--wall-time-budget` chunk boundary and its resume, and the tables
`analyze_results.py` and `make_tuning_set.py` return. It is a readiness
suite, not a unit suite — a benchmark campaign stage costs a night of machine
time and reports its failures the morning after, so the pipeline is checked
before it is launched. Its last four tests use the real `build/bin/highs`
when one is present (they answer what a stand-in cannot: whether HiGHS
accepts the options a stage sets) and skip in the Python-only CI job.

**`ruff check bench cmake` gates.** It runs *after* the tests in the same job — a
failing step skips the rest of a job, so linting first would mean the bench
tests never ran in CI at all — but both fail the build.

The rule set is declared explicitly in `pyproject.toml` under
`[tool.ruff.lint]` rather than inherited from ruff's defaults, and CI pins the
ruff version to match. That pairing is what makes the gate safe to fail a
build on: ruff 0.16 widened its defaults and turned `bench/` red without a
line of `bench/` changing, and a gate whose rule set moves underneath it
fails for reasons unrelated to the change under review. Widen the set
deliberately — add a family, fix what it finds, land both together. Do not
loosen the version pin to make a finding go away.

## Code review bar

Every change is reviewed against this list. It is the standing bar the closeout
refactors were held to, not a one-off checklist.

- **Net LoC must go down** on a refactor. Adding indirection without deleting
  duplication is not a refactor.
- **No new abstraction with exactly one implementor.** Carve an existing type;
  do not invent a parallel one.
- **No O(n) scan** where an incremental structure already exists or is cheap to
  maintain. Keep an explicit index list beside the flag array and iterate that.
- **Hot loops use flat `std::vector` plus index arrays.** No `std::map`,
  `std::set`, `std::unordered_*` or `std::list` on a per-attempt or per-step
  path.
- **`reserve()` known sizes.** No allocation inside a worker's inner step loop.
- **Pass by `const&` or `std::span`.** No vector copies across the runner
  boundary.
- **Delete dead code in the commit that orphans it.** No `TODO` placeholders,
  no stubs, no "wire this up later".
- **clang-tidy clean.** Fix the root cause rather than suppressing.
- **Never mutate HiGHS solver state.** No `const_cast` on `HighsOptions`, and no
  writing an upstream `HighsMipSolverData` counter outside `src/effort_ledger`,
  the one place that deliberately charges the RENS/RINS envelope. Read solver
  options; do not reset, override or restore them. `mip_heuristic_suite=off`
  plus `mip_heuristic_run_feasibility_jump=false` — nothing of ours running,
  and FeasibilityJump out of the picture on both sides — has to stay
  byte-equivalent to an unpatched binary in the same configuration, which is
  what `bench/check_vanilla_equivalence.py` proves. `off` on its own is the
  "our four presolve heuristics disabled" ablation and is **not** a vanilla
  baseline: it still runs FeasibilityJump, and the FeasibilityJump it runs is
  ours.

Two further invariants that are easy to violate without a test noticing:

- **Workers read `problem.incumbent`, never `mipdata->incumbent`.** A peer's
  accepted solution rewrites the live vector, and reallocates it outright on the
  empty-to-sized transition, while a worker is indexing it.
- **The effort ledger is not thread-safe.** `charge_presolve` and `charge_dive`
  do plain non-atomic `+=` and must be called from the dispatching thread with
  every parallel region joined. Charging from inside a worker callback corrupts
  `heuristic_lp_iterations`, which decides whether RENS/RINS run at all —
  silent, non-reproducible budget corruption.

## Workflow

Trunk-based: commit to `main` and push when the local gates pass. Conventional
Commits (`type(scope): description`, subject ≤72 chars, focused on *why*); the
commit-msg hook enforces the format. Run the full suite before considering work
done — the pre-push hook is the final gate ([Git hooks](#git-hooks)), and
`--no-verify` is not an option. Close any GitHub issue the work resolved.

Cutting a version is a separate procedure with an ordering constraint a normal
push does not have: see [`docs/RELEASE.md`](docs/RELEASE.md).

When something in this file or `CLAUDE.md` would have prevented a mistake you
just made, fix it in the same commit.
