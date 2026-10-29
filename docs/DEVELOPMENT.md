# Development details

Moved from CLAUDE.md (#202): the long form of its CMake, testing, development-workflow, code-navigation and build sections. CLAUDE.md keeps the short rules; the reasons and mechanisms are here.

## Code Navigation

Prefer narrow queries over full-file reads:

1. **LSP** for symbol questions — `goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol`. Use before `Read`.
2. **Grep** with `-n` and a small `head_limit` (start at 20); raise only if inconclusive.
3. **Read** with `offset`/`limit` for a slice around the hit. Full-file `Read` is fine under ~200 lines or when structure matters.

Know the symbol → LSP. Know a string, not its location → Grep. Full-file Read is the last mile. This is a preference, not a prohibition: shelling out to `grep`/`rg` is fine when the built-in can't do the job (filtering a pipe like `git log | grep`, or a session without the `Grep` tool). What matters is bounding output, not which binary produces it.

Setup: install `clangd-lsp@claude-plugins-official` plus `clangd` (`apt install clangd`, or from LLVM). `.clangd` points at `build/compile_commands.json`, produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`.


## CMake

- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` for clang-tidy.
- Use FetchContent for dependencies.
- A single root `CMakeLists.txt`; per-directory files would only add indirection.
- **Consumable as a subproject.** `highs` and `mip_heuristics_core` are the consumer-facing targets, and a parent's `FetchContent` / `add_subdirectory` gets no tests, lint gates or hooks of ours: tests, Catch2, the bench and docs checks and the lint gates sit behind `MIP_HEURISTICS_BUILD_TESTS` (default `PROJECT_IS_TOP_LEVEL`, via an early `return()` in the root file), hook installation defaults the same way, and `MIP_HEURISTICS_INSTRUMENT` follows `MIP_HEURISTICS_BUILD_TESTS`. HiGHS's own extra targets (`highs-bin`, `highs_extras`, CPack, install rules) still come along — they are upstream's. Consumer-facing targets go **above** that `return()`; anything development-only goes below it. Set HiGHS's switches (`BUILD_TESTING`, `BUILD_SHARED_LIBS`, …) as normal variables, never `CACHE FORCE`, which would overwrite the parent's. A parent's own HiGHS patches go through `MIP_HEURISTICS_EXTRA_HIGHS_PATCHES` (scripts run after ours in the same `PATCH_COMMAND`, each hashed into it, in `cmake/FetchHiGHS.cmake`); empty must leave the patch command byte-identical, and no consumer's patch content lives here.

## Testing (Catch2 v3)

- `TEST_CASE("name", "[tag]")` with `[tag]` filters, in `tests/`. Not GoogleTest.
- Instances come from HiGHS's own `check/instances/` (path injected as the `INSTANCES_DIR` compile definition).
- `ctest --progress` collapses the running list and `CMAKE_INSTALL_MESSAGE=LAZY` suppresses install chatter. Don't remove these.
- **`[serial]` is a reserved tag**, and it is the *last* resort for a timing-dependent test. `CMakeLists.txt` registers the tagged cases through their own `catch_discover_tests` call carrying `TEST_SPEC "[serial]"` and `PROPERTIES RUN_SERIAL TRUE`, beside a second call taking `TEST_SPEC "~[serial]"` for everything else; the two specs must partition the suite exactly (verify with `ctest -N | wc -l` and a `uniq -d` over the names). Give each call its own `TEST_LIST` name.

  Order of preference when a test flakes under `ctest -j$(nproc)`: **(1) take the clock out of the fixture** — e.g. HiGHS's free-format MPS reader spends `time_limit` as its own *parse* budget, so write the option *after* `readModel`, as `build_bare_mipsolver` in `tests/test_common.h` does; **(2) assert the mechanism** — a one-sided effort bound like `effort < attempt_cap` only gets easier under starvation, so it needs nothing; **(3) only then tag it**, for a wait nothing in the code can bound. **The load-safety of an effort bound is a property of its *shape*, not of the quantity it names**: the *differential* form `effort(T) < effort(5T)` that `tests/test_deadline.cpp`'s `require_clock_bound` uses is two-sided and fails if the longer run is descheduled harder, so it is a (3) case and carries `[serial]` even though it is an effort assertion.

  **Never widen a constant** to stop a flake — a tuned threshold is the same test with a longer fuse. `RUN_SERIAL`, not `RESOURCE_LOCK`: these need an unloaded machine, not mutual exclusion, and neither excludes load from outside ctest, which is why (1) and (2) beat (3) whenever available.

## Development Workflow

```
plan (non-trivial) → implement → test → /code-review → push to main
```

Nothing formats on save or at commit. `pre-commit` (wcp's) only **checks**: `clang-format --dry-run -Werror` over every tracked C/C++ file, against the major version `.clang-format-version` pins, plus `ruff format --check`, `ruff check` and `mypy` through `uv run`, each skipped loudly when it cannot run. It rewrites nothing and runs no tests, so format before committing:

```
.venv/bin/clang-format -i <files>                       # normalize in place
.venv/bin/clang-format --dry-run --Werror <files>       # check only, non-zero if unformatted
```

**Use the venv's clang tools, not PATH.** The pinned 22.1.8 pair lives in `.venv/bin`, which is the exact path `cmake/Lint.cmake` searches; a different major version formats differently and fails the `clang_format` ctest gate. Both hooks put the main checkout's `.venv/bin` first on PATH for the same reason. `pyproject.toml`'s `[project]` is never built (`package = false`): it carries `requires-python`, so the committed `uv.lock` is the same on every interpreter, and the dev dependency group is the whole Python toolchain. mypy reads `[tool.mypy]`, which names `bench` and `cmake` and excludes the build trees, whose `_deps` hold HiGHS's and Catch2's own Python; CI runs the same check.

The hooks are wcp's, in `.githooks/` (see CLAUDE.md's Hooks Are Files In This Repository). `cmake -B build` also points `core.hooksPath` at them (option `MIP_HEURISTICS_INSTALL_GIT_HOOKS`, ON as the top-level project), leaving an existing `.githooks` value alone and warning rather than clobbering a hooksPath someone else set. `.claude/` is gitignored and not part of the published artifact; `wcp init` writes its permission rules and statusline. A new Claude Code hook must read its file path and command from the hook JSON **on stdin**, never `$CLAUDE_FILE_PATH` (unset by Claude Code — a hook reading it no-ops silently rather than failing). The clang-format and clang-tidy gates deliberately live outside the hooks, as ctest tests labelled `lint` plus the CI job, so a fresh clone still runs them.

**A push is a clean build and the full suite, in a fresh worktree** under `~/.cache/local-ci`, not in this checkout: it leaves `build/` alone, so pushing during a benchmark no longer deletes the binary under it, but it does load the machine for the length of a HiGHS build. On a shared machine take the lock and cap it: `xk lock -- git push ...`, with `git config hooks.localCiCpus 0-3` in the clone so `-j"$(nproc)"` follows the pin. ccache makes the second worktree's build cheap only because no compile command carries a worktree-specific path: the user-level `base_dir` rewrites path *arguments*, not `-D` values, so `INSTANCES_DIR` is defined only on `tests/test_paths.cpp` and read through `instances_dir()`. Keep new path-valued defines off target-wide compile definitions.


## Build Commands

```bash
# Dev tools, once per checkout: pyproject.toml's dev group (the pinned clang
# pair, ruff, mypy, pytest) into `.venv`.  `.venv/bin` is the exact path
# cmake/Lint.cmake searches; without it the gates are never registered and
# ctest reports green having linted nothing.  pytest must be there too — the
# root CMakeLists prefers `.venv/bin/python`, so a venv without it unregisters
# `bench_python_tests` rather than falling back to the system interpreter.
# pre-commit's `uv run` installs the same group, so the hook and the setup
# cannot disagree; a venv made with pip instead is one `uv run` may rebuild.
uv sync

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

The first build fetches and compiles HiGHS via FetchContent, which dominates it. Two things keep the *clean* rebuild that `pre-push` forces cheap:

- **ccache**, wired in `CMakeLists.txt` **ahead of** `include(cmake/FetchHiGHS.cmake)` so HiGHS's own targets inherit the launcher — a target reads the property at creation, so setting it afterwards would cache our four object files and miss the tree worth caching. It does not weaken the clean-rebuild guarantee, because ccache keys on source content and a differently-patched tree misses. `-DMIP_HEURISTICS_CCACHE=OFF` opts out; a missing ccache is a status message, not an error. The launcher does **not** appear in `compile_commands.json`, so `cmake/clang_tidy_gate.py` and `.clangd` are unaffected — re-check that if the CMake version moves.
- **`GIT_SHALLOW ON`** on both `FetchContent_Declare` calls. One visible consequence: git sizes its abbreviated hash by object count, so a shallow tree's banner reads `Githash 04024d70` where a full clone reads `04024d701f` — the same commit at a different width. `same_build` in `check_vanilla_equivalence.py` compares the hash by prefix and `normalize_log` masks its width; `make_archive.py` records one hash per config. **Do not "tighten" either into an equality check.**

**The lint gates are ctest tests**, labelled `lint`: `clang_format` and `clang_tidy`, registered by `cmake/Lint.cmake` over `src/` and `tests/` only. They add roughly 30 s to a full `ctest` run — `ctest -LE lint` is the fast loop while iterating, and the full suite is the gate. `MIP_HEURISTICS_REQUIRE_LINT=ON` turns a missing tool or a wrong major version into a *configure* failure instead of a warning that scrolls past; CI sets it, and so should you, because the default failure mode is silent. Tool versions are part of the contract: clang-format's output changes between major releases, so the pinned 22.1.8 pair from PyPI is what the gate is written against. clang-tidy's own exit status is unusable here — HiGHS's `HighsMipWorker.h` contains a construct clang rejects as a parse error and GCC accepts — so `cmake/clang_tidy_gate.py` wraps it and judges first-party diagnostics itself. **Never work around a tidy finding by widening that wrapper's filter.**


## GPU build

GPU acceleration: `-DMIP_HEURISTICS_CUDA=ON` enables CUDA for the PDLP solver used by Scylla. It **requires `CUDA_HOME`** to be exported (HiGHS's `FindCUDAConf.cmake` derives `CMAKE_CUDA_PATH` from it and uses that for the cudart/cublas/cusparse `find_library` hints and for HiGHS's CUDA include directory), and it **fails the configure** if nvcc or `CUDA_HOME` are missing, or if the `HConfig.h` HiGHS generates doesn't come out GPU-enabled — it does *not* fall back to CPU. That's deliberate: GPU vs CPU is a compile-time `#ifdef CUPDLP_CPU` in HiGHS's `CupdlpWrapper.cpp` with no runtime override, so a silent fallback yields a CPU binary indistinguishable from a GPU one at the command line. Build GPU into a separate tree (`cmake -B build-gpu -DMIP_HEURISTICS_CUDA=ON`) and confirm with `ldd build-gpu/bin/highs | grep -E 'cudart|cublas|cusparse'`. Omit the flag for a CPU build.

