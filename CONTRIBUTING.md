# Contributing

## Build and test

```bash
# Lint tools, once per checkout.  `.venv/bin` is the exact path the CMake
# lint module searches, so this install is what registers the gates.
python3 -m venv .venv
.venv/bin/pip install clang-format==22.1.8 clang-tidy==22.1.8 pytest

cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -j$(nproc)
```

The first build takes about five minutes: it fetches HiGHS v1.15.1 via
FetchContent and applies the patches in `third_party/highs_patch/`.

Useful narrower invocations:

```bash
ctest --test-dir build -LE lint --output-on-failure       # skip the lint gates
ctest --test-dir build -R "execution-mode: flugpl objective" --output-on-failure
./build/mip_heuristics_tests "[mode-matrix]"              # Catch2 tag filter
```

Tests are Catch2 v3 (`TEST_CASE` with `[tag]` filters), not GoogleTest.

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

> **`.clang-tidy` and `.clang-format` are checked-in regular files here, not
> the symlinks devkit's `setup.sh` installs.** They carry project overrides
> devkit's shared copies do not: a `HeaderFilterRegex` scoped to `src/` and
> `tests/` (devkit's `.*` produces thousands of findings inside the fetched
> HiGHS headers) and a `lower_case` function naming convention matching this
> codebase. **Re-running devkit's `setup.sh` or `update.sh` will replace them**,
> and because `clang_tidy` is now a mandatory ctest test the suite then goes
> red for everyone with no obvious cause. If you re-run devkit setup, restore
> both files with `git checkout -- .clang-tidy .clang-format` and check
> `git diff --summary` for a `mode change` line.

Hooks auto-format on save, so don't hand-fix formatting.

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
cannibalization tables) and `docs/REPRODUCIBILITY.md` (the PLATO protocol).
Two rules that are easy to get wrong:

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
.venv/bin/pip install ruff==0.16.3 pytest
.venv/bin/ruff check bench cmake
.venv/bin/python -m pytest bench
```

`pytest` picks up `testpaths = ["bench"]` from `pyproject.toml`, so a bare
`pytest` from the repo root works too. Both run in CI as a separate fast job
with no C++ build.

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
  has to stay byte-equivalent to an unpatched binary — that is the row every
  other benchmark row is measured against, and
  `bench/check_vanilla_equivalence.py` is what proves it.

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
done — the pre-push hook is the final gate, and `--no-verify` is not an
option. Close any GitHub issue the work resolved.

Cutting a version is a separate procedure with an ordering constraint a normal
push does not have: see [`docs/RELEASE.md`](docs/RELEASE.md).

When something in this file or `CLAUDE.md` would have prevented a mistake you
just made, fix it in the same commit.
