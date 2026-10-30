<!-- wcp: CLAUDE.md base=45f6a643367b type=cpp-python -->
<!-- wcp-declined: a6e5ec6d33d8 CLAUDE.md Discipline -->
<!-- wcp-declined: 1a2483f6e234 Plan Adherence -->
<!-- wcp-declined: e1a2b49b4c80 Reference Correctness -->
<!-- wcp-declined: 027569c8a6fc Common Mistakes -->
<!-- wcp-declined: e9feadcddcb5 Testing (GoogleTest) -->
<!-- wcp-declined: a87ec66c5246 Python -->
<!-- wcp-declined: cc3df5ee540d Testing (pytest) -->
<!-- wcp-declined: 004504610dd6 Dependencies -->
<!-- wcp-declined: 69dc0f786da4 nanobind Bindings -->
<!-- wcp-declined: c4fc130d1235 Ownership and Lifetime -->
<!-- wcp-declined: cf8d0fc1773f Type Conversions -->
<!-- wcp-declined: 7e7de94f1273 Testing Bindings -->
<!-- wcp-declined: 9070a09f6181 LSP -->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Navigation: LSP → narrow Grep → sliced Read.

# Standards

## Communication Style

Be terse. No preamble. No filler.

## Code Navigation

Prefer narrow queries over full-file reads: **LSP** for symbol questions (`goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol`), **Grep** with `-n` and a small `head_limit`, then **Read** with `offset`/`limit` around the hit. What matters is bounding output, not which binary produces it. clangd setup: `docs/DEVELOPMENT.md`.

## C++

- Target C++23. Use modern features (`std::expected`, concepts, ranges, `constexpr`).
- Use `#pragma once`. Minimize header includes; forward-declare where possible.
- **Formatting** is Google, via `.clang-format`. **Naming is not Google** — `.clang-tidy` enforces what this codebase actually does:

  | Kind | Style | Example |
  |---|---|---|
  | Functions (free and member) | `lower_case` | `run_sequential`, `charge_presolve` |
  | Locals, parameters, `const` locals | `lower_case` | `per_worker`, `max_effort` |
  | Private members | **trailing** `_` | `FjWorker::start_`, `seed_` |
  | File-scope / static / class / `constexpr` constants | `k` + `CamelCase` | `kMaxStaleRounds`, `kNumLpArms` |
  | Classes | `CamelCase` | `ScyllaWorker`, `EffortLedger` |
  | Namespaces | `lower_case` | `heuristics`, `fpr` |
  | Files | `snake_case.h` / `.cpp` | |

  `ConstantMemberCase` is deliberately unset: it outranks `PrivateMemberSuffix`, so setting it would reject a `const` private member spelled with this project's `_` suffix. The `.clang-tidy` comments carry the reasoning for each narrowing — read them before changing one.

## Complexity

When a complexity warning fires, don't extract methods mechanically. Ask what the independent responsibilities are and split along those boundaries. If the function is genuinely complex because the domain is, add a comment explaining why and suppress the warning.

## CMake

- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` for clang-tidy; FetchContent for dependencies; a single root `CMakeLists.txt`.
- **Consumable as a subproject.** `highs` and `mip_heuristics_core` are the consumer-facing targets and go **above** the early `return()` behind `MIP_HEURISTICS_BUILD_TESTS`; anything development-only goes below it. Set HiGHS's switches as normal variables, never `CACHE FORCE`. A parent's HiGHS patches go through `MIP_HEURISTICS_EXTRA_HIGHS_PATCHES`, and empty must leave the patch command byte-identical. Details: `docs/DEVELOPMENT.md`.
- `-Werror=unused-result` and `-Werror=switch` go on every library and test target of ours and stay off HiGHS's (why: `docs/RUNTIME.md`, Patience; `docs/ARCHITECTURE.md`, `fpr_lp`).

## Testing (Catch2 v3)

- `TEST_CASE("name", "[tag]")` in `tests/`, not GoogleTest. Instances come from HiGHS's own `check/instances/` (`instances_dir()`).
- Keep `ctest --progress` and `CMAKE_INSTALL_MESSAGE=LAZY`.
- **A test that flakes under `ctest -j$(nproc)`**: (1) take the clock out of the fixture; (2) assert the mechanism (a one-sided effort bound needs nothing); (3) only then tag it `[serial]`, a reserved tag with its own `catch_discover_tests` call and `RUN_SERIAL`, the two specs partitioning the suite exactly. A *differential* effort bound is two-sided, so it is a (3) case. **Never widen a constant** to stop a flake. Details: `docs/DEVELOPMENT.md`.

## Development Workflow

```
plan (non-trivial) → implement → test → /code-review → push to main
```

Nothing formats on save or at commit: `pre-commit` (wcp's) only **checks** (clang-format against `.clang-format-version`, `ruff format --check`, `ruff check`, `mypy` through `uv run`). Format first, with **the venv's clang tools, not PATH** — the pinned 22.1.8 pair is what `cmake/Lint.cmake` and the `clang_format` gate use:

```
.venv/bin/clang-format -i <files>
```

A push is a clean build and the full suite in a fresh worktree under `~/.cache/local-ci`; on a shared machine take the lock and cap it (`xk lock -- git push ...`, `git config hooks.localCiCpus 0-3`). A new Claude Code hook reads its input from the hook JSON on stdin, never `$CLAUDE_FILE_PATH`. Keep path-valued defines off target-wide compile definitions, or ccache misses across worktrees. Details: `docs/DEVELOPMENT.md`.

## Git Workflow

Trunk-based, linear history on main. Commit directly to main and push when local gates pass.

**Local CI is the gate**; GitHub Actions run at most weekly (`schedule` + `workflow_dispatch`, no `push` / `pull_request`). Nothing waits on a GitHub run.

Feature branches are optional for larger changes: always branch from main (`git checkout main && git pull` first), never from another feature branch, keep them short-lived, and rebase or squash merge — no merge commits on main.

After a successful push:
- **Close any gh issue the work resolved**: `gh issue close <num> -c "<one-line note>"`, for every issue the push covers.
- **Delete the feature branch** if one was used: `git branch -d <branch>`, plus `git push origin --delete <branch>` if pushed.

## Hooks Are Files In This Repository

The git hooks live in `.githooks/` as ordinary shell scripts, committed like any other file. `wcp init` copies them there and points `core.hooksPath` at them; wcp installs no Claude Code hooks at all. The gate is those files and nothing else, so a clone without wcp installed is still gated — including the review receipt, which is a line of plain git the Self-Review section documents rather than anything wcp runs.

- **They are yours to read, and yours to edit.** Each hook carries a `# wcp:` line recording the digest of the bytes wcp wrote. That is what lets `wcp doctor` tell an edit you made from a version wcp has moved past: an untouched hook is refreshed silently, an edited one is left alone, and one that moved on both sides is reported rather than overwritten. Edit them when the project needs something different — and say so upstream if the default was wrong for everyone.
- **A clone is not gated until someone runs `wcp init` in it.** git clones neither a hooks directory nor local config, so the files arriving with the clone is not the same as git running them. `wcp doctor` reports an unset `core.hooksPath` as an error for exactly this reason.
- **They call no wcp.** A hook that shelled out to a missing tool would exit 127, which git reads as a refusal — so a machine without wcp could not commit, and could not commit the fix either. What the hooks do call is the project's own toolchain, and there they fail open and say so on stderr.
- **Never use `git push --no-verify` or `git commit --no-verify`** unless explicitly asked. A failing hook is a signal — fix the root cause, or change the hook deliberately and commit that.

## This Document Carries A Provenance Line

The comment at the top of this file is wcp's record of the bytes it wrote, in the same scheme the hooks use. `wcp doctor` reads it to tell an edit you made from a version wcp has moved past, and it is where this project's type is recorded — there is no `.wcp/` directory and no config file beside it.

**The prose is yours.** wcp writes this document once and never rewrites it unasked; edit it freely. When wcp adds or changes a section you have not edited, `wcp doctor` names it and `wcp init --interactive` offers it, one section at a time; answering `n` declines that version for good. Leave the top line in place, and the tooling stays able to tell your changes from its own.

## Running wcp

wcp is a command-line tool installed once per machine, outside any repository. The hooks never call it, so a machine without it is still gated.

```sh
uv tool install git+https://github.com/spoorendonk/wcp   # once per machine; @vX.Y.Z for a release
uv tool upgrade wcp                                       # to update an install without a tag
uv tool install --force git+https://github.com/spoorendonk/wcp@vX.Y.Z   # to move a tagged one
wcp --help                                                # the commands; `wcp <command> --help` for each one
```

- **Run `wcp doctor`** after pulling, after updating wcp, and before editing the hooks or this file. It changes nothing, and every finding names the command that fixes it. It exits 1 only when the gate does not work, e.g. `core.hooksPath` is unset, which is how every fresh clone starts until someone runs `wcp init` in it.
- **Files wcp wrote carry a `wcp:` provenance line**: `# wcp:` in each hook, and the comment at the top of this file. Edit the files when the project needs to, but leave that line in place. It is how `wcp doctor` tells your edits apart from a newer wcp version.
- **If `wcp` is not on PATH**, say so rather than installing it unasked. Installing it is a machine-wide change.

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
- **Label work in progress** with `session:<name>`, your session's name: `wcp claim <num>` when you start, `wcp release <num>` when you stop, hand over or close it (by hand: `gh issue edit <num> --add-label` / `--remove-label "session:<name>"`). Every session acts as the same GitHub user, so the label is what tells them apart; one left on an issue nobody is working on reads as a claim.

Issues get picked up cold, in fresh sessions, often by an agent with no access to this machine. So: keep the body **self-contained** (problem, motivation, acceptance criteria, repro steps); use **no local references** (`/home/user/...`, "see my other checkout" — dead links in a fresh session); prefer **stable external links** (GitHub permalinks, papers, RFCs); and **describe local code by concept, not path**, hinting that the agent can search under `..`, `../..`, or `~/code/`.

## Working Through Issues

A session's goal is often a list of issues (`wcp goal`). Work through them in the order given, one at a time:

1. **Issue check.** Read the issue and what it links. If it is clear and implementable, say so in one line and go on; this should be fast. If not, write a plan as a comment on the issue (plans go on issues, never in git) and work from it. If a later issue's plan changes an earlier one, re-plan then.
2. **Implement.**
3. **Gate and commit.** Run the project's gate (`## Build & Test`) and commit: a review reads commits, never uncommitted changes (`## Self-Review`).
4. **Review by one cold reviewer**: a fresh teammate or subagent running `/code-review` on the commits not yet pushed, never several reviewers in parallel. On findings, fix, gate, commit, and review again with a fresh reviewer. **At most two rounds**; the fix after the second is confirmed, not reviewed again (`## Self-Review`). Anything real left after that, or found out of scope, becomes its own issue. The bar for filing is a concrete failure scenario or a missing acceptance criterion; nits are fixed within the round or dropped.
5. **Record, push, close.** Record the review on a clean tree, push it with the commit (`## Self-Review`), and close the issue.

Without agent teams, implement in the session itself and review with a fresh subagent; the caps and the bar are the same.

**A hygiene goal** (`/goal Hygiene round …`, set by `wcp hygiene`) is a task, not a list of issues: work it from its note. Its findings are edits to existing issues or new issues labelled `tech-debt`; nothing is fixed inside the round, one cold reviewer checks the findings (at most two rounds), and the task ends with the summary comment its note specifies, whose first line is the marker wcp looks for.

## Self-Review

**Code you produced gets reviewed before it lands on main.** Run `/code-review`
on your own changes and act on what it finds; the review is not a formality you
report having done. The pre-push hook is the backstop, not the review.

**Then record the review, and push it with the commit**, or `pre-push` will
refuse the push:

```sh
sha=$(git rev-parse HEAD)
branch=$(git symbolic-ref --short HEAD)
git update-ref "refs/reviews/$sha" "$sha"
xk lock -- git push --atomic origin "$sha:refs/heads/$branch" "refs/reviews/$sha"
```

The receipt is a ref named by the reviewed commit, `refs/reviews/<sha>`, and it
travels with the push, so a later check (here, the weekly `ci.yml` run, if it
adds wcp's receipt step) can audit every commit on the trunk — including one
pushed with `--no-verify`, or from a clone whose hooks never ran.
`--atomic` lands the branch and its receipt together or not at all.

**Push the reviewed sha by name, and commit nothing until the push returns.**
A `HEAD` in the push line is resolved when git gets to it, not when the
command is typed, and on a shared machine `xk lock` can hold it for minutes
before the local CI even starts; a commit made meanwhile in the same checkout
would go out with it, unreviewed, while its receipt names the older sha.

Plain git, on purpose. No wcp, no harness, nothing installed — so a reviewer on
any machine can write it, and a clone that has never heard of wcp is still able
to record a review and push. Every worktree of a clone sees the same receipts,
and each names one commit, so a review counts for that commit wherever it is
checked out and for nothing else.

**What it means, and what it does not.** It records that a review happened at
this commit — not what was reviewed. It carries exactly the trust of whoever ran
it, and it is a line anybody can write by hand; what it buys is that nobody can
push work no one has looked at *without noticing they are doing it*.

A later commit has no receipt, and the next push is refused again. That is
deliberate: a new commit is work nobody reviewed, including the commit that
fixes what the review found.

**The fix after the last round.** Reviews stop after two rounds, so the commit
that fixes the second round's findings gets no third review. That round's
reviewer reads the fix commit alone (`git show <sha>`) against its own
findings, and the receipt is recorded for the fix once it confirms they are
addressed. It looks for nothing new. What it does not confirm is taken out of
the fix, which makes a new commit: the same reviewer confirms that one before
its receipt is recorded, and what was taken out becomes its own issue.

**Record it only when the tree is clean.** With nothing new committed, a review
falls back to reading uncommitted changes — so a record written then names a
commit that does not contain what was read, and the push it clears is of work
nobody reviewed. Commit first, then review, then record. Untracked scratch files
do not matter; staged and unstaged changes to tracked files do.

## Working Rules

- **CLAUDE.md discipline.** When Claude gets something wrong, fix CLAUDE.md in the same commit. It's a living document — update it whenever better instructions would have prevented the mistake. Keep it **forward-looking**: rules and current behaviour, not a change log. Post-mortem detail, measurements of a past run, and "it used to be X" belong in the issue that owns them. Keep it under ~25 KB, since every session loads it whole: a mechanism's detail goes in the `docs/*.md` the Architecture map links, with only the rule here.
- **Follow the agreed plan.** If a plan should change, stop and discuss — don't silently diverge. Same outside a written plan: if the current approach isn't working, say so rather than quietly switching strategies. Implement everything specified; no TODO placeholders or stubs unless asked.
- **Match references exactly.** Implementing from papers, pseudocode, or open source: no early exits, iteration limits, size caps, or "optimization" shortcuts that change behaviour. Introduce heuristic approximations only when asked. Implement the edge cases rather than simplifying them away. When in doubt, be faithful and let tests verify. Where the reference is genuinely ambiguous, take a reading, implement it, and record the ambiguity in a comment beside the code (and in `docs/PARAMETERS.md` if it is a parameter) — the deviations recorded in `docs/ARCHITECTURE.md` are all of this shape.
- **Don't invent APIs.** Verify functions, flags, and methods exist before using them.
- **Don't ignore type errors.** If clang-tidy or ruff flags something, fix the root cause — don't suppress.
- **Don't use deprecated patterns.** Check current docs, not training data.
- **Performance matters.** Most of this is solvers: profile before micro-optimizing, but don't sacrifice perf for "clean code". Price any micro-optimization against the 4% anchor in `docs/RUNTIME.md`.

## Code Hygiene Bar

Every change is held to this list; it is not a one-off checklist.

- **Net LoC must go down** on a refactor. Adding indirection without deleting duplication is not a refactor.
- **No new abstraction with exactly one implementor.** Carve an existing type; don't invent a parallel one.
- **No O(n) scan** where an incremental structure already exists or is cheap to maintain. Keep an explicit index list beside the flag array and iterate that.
- **Hot loops use flat `std::vector` plus index arrays.** No `std::map`, `std::set`, `std::unordered_*` or `std::list` on a per-attempt or per-step path.
- **`reserve()` known sizes.** No allocation inside a worker's inner step loop.
- **Pass by `const&` or `std::span`.** No vector copies across the runner boundary.
- **Delete dead code in the commit that orphans it.** No `TODO` placeholders, no stubs, no "wire this up later".
- **Never mutate HiGHS solver state.** No `const_cast` on `HighsOptions`, and no writing an upstream `HighsMipSolverData` counter outside `src/effort_ledger`, the one place that deliberately charges the RENS/RINS envelope. Read solver options; don't reset, override or restore them.
- **The effort ledger is not thread-safe.** `charge_presolve` and `charge_dive` do plain non-atomic `+=` and must be called from the dispatching thread with every parallel region joined. Charging from inside a worker callback corrupts `heuristic_lp_iterations`, which decides whether RENS/RINS run at all — silent, non-reproducible budget corruption.

# Project: mip-heuristics

## Project Overview

Custom MIP (Mixed-Integer Programming) heuristics integrated into the HiGHS solver via a patched fork. The heuristics run during HiGHS's presolve phase and are compiled as object files linked directly into the `highs` library target.

**Reader-facing docs**, kept in sync with this file — update both when the behaviour they describe moves:
- `README.md` — positioning, the heuristic table and how each one is enabled, recorded benchmark results, build options.
- `docs/REPRODUCIBILITY.md` — what is reproducible and what is not, and the exact `mipfeas` reproduction protocol.
- `docs/PARAMETERS.md` — every tunable `constexpr`. **Verified by ctest** (`docs_parameter_references`, via `bench/check_docs_refs.py`): renaming a documented constant fails the suite. Entries name symbols, **never line numbers** — line numbers drift on essentially every refactor. Don't reintroduce them.

## Build Commands

```bash
uv sync                                             # once per checkout: the dev toolchain into .venv
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure               # the gate, lint included
cd build && ctest -LE lint --output-on-failure      # fast loop
cd build && ./mip_heuristics_tests "[mode-matrix]"  # by Catch2 tag
```

The lint gates (`clang_format`, `clang_tidy`, label `lint`) are ctest tests over `src/` and `tests/`; set `MIP_HEURISTICS_REQUIRE_LINT=ON`, since the default failure mode is silent. **Never work around a tidy finding by widening `cmake/clang_tidy_gate.py`'s filter.** ccache, `GIT_SHALLOW` and why the hash checks must not be tightened: `docs/DEVELOPMENT.md`.

## Build & Test

**`.githooks/pre-push` runs these three fenced blocks by tag** (`clean`, `build`, `test`) as the local CI, after the linear-history and review-receipt checks, in a fresh worktree at the tip being pushed, and skips them for a documentation-only push (`docs/PARAMETERS.md` counts as code: ctest checks it, and the hook carries that one project edit) — so they are the single definition of a full local run; keep them executable as written and keep the tags. The hook unsets `GIT_DIR` / `GIT_WORK_TREE` around them, which CMake's nested `git clone` inside FetchContent needs: with `GIT_DIR=.git` exported it treats that as the repository it is cloning into and fails checking out the HiGHS tag. The `test` block is the whole suite, lint gates included, rather than wcp's per-source `clang_tidy:<path>` scheme: `clang_tidy` is one gate (`cmake/clang_tidy_gate.py`), so the push re-tidies everything every time.

```clean
rm -rf build
```

```build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON && cmake --build build -j"$(nproc)"
```

```test
ctest --test-dir build --output-on-failure -j"$(nproc)"
```

GPU build (`-DMIP_HEURISTICS_CUDA=ON`, requires `CUDA_HOME`, fails rather than falling back to CPU, separate tree): `docs/DEVELOPMENT.md`.

## Architecture map

HiGHS v1.15.1 is fetched at build time and patched from `third_party/highs_patch/`. **Bump `PATCH_VERSION` on any change to inserted text and clean-rebuild** (`rm -rf build/_deps/highs-src build/_deps/highs-subbuild build/CMakeCache.txt`, then re-configure); older patch layouts are refused, not upgraded. Two layers:

- **`mip_heuristics_core`** (STATIC): FJ, LocalMIP, FPR's search, `SolutionPool`/`SolutionSink`, the runner scaffold and the model view (`heuristic_context.h`). It needs no `HighsMipSolver`; `core_boundary` and the whole-archive standalone test enforce that. A new core file, headers included, goes in the core target's list.
- **`mip_heuristics`** (OBJECT, injected into `highs`): the adapter — `mode_dispatch`, `IncumbentSink`, `effort_ledger`, `highs_context.h`, FPR's dispatch, `fpr_lp`, Scylla, `ContestedPdlp`.

The presolve chain is FJ → FPR → LocalMIP → Scylla (`kChain` in `src/mode_dispatch.cpp`), each on continuous parallel workers; `fpr_lp` runs at dive time inside the RENS/RINS LP-iteration envelope. A heuristic runs only if its `mip_heuristic_<name>_effort` is above zero (FJ also needs upstream's `mip_heuristic_run_feasibility_jump`).

Rules no change may break: workers read `problem.incumbent`, never `mipdata->incumbent`; `compute_var_order` and clique-table reads run on the dispatching thread only; `terminated()` is seat-holder-only while `past_deadline()` is safe on any thread; the effort ledger is charged from the dispatching thread only; headers inserted into HiGHS's own sources (`fpr_lp.h`, `mode_dispatch.h`) stay C++11.

**Read the matching doc before changing the code it covers** — its rules bind as this file's do:

| Doc | Covers |
|---|---|
| `docs/ARCHITECTURE.md` | integration model and core boundary; bumping the HiGHS tag (silently rejected option names); `fpr`, `fpr_lp`, `fj`, `local_mip`, `scylla`; dispatch and its threading rules; `contested_pdlp`; shared utilities |
| `docs/RUNTIME.md` | the patch's HiGHS options and their defaults; patience; the wall-clock deadline; `[Sequential]` / `[Heur]` / `[HeurSol]`; the ~4% anchor a micro-optimization is priced against |
| `docs/BENCHMARKING.md` | `mipfeas`, the vanilla baseline, killed runs, configs, tuning subsets, the oracle, the release archive |
| `docs/DEVELOPMENT.md` | the long form of the CMake, testing, workflow and build rules; the GPU build |
| `docs/PARAMETERS.md` | every tunable `constexpr` (checked by ctest) |

## Testing

Catch2 v3. Tests use `.mps` instances from HiGHS's own `check/instances/` directory (path injected via `INSTANCES_DIR`). Characterization tests verify known-optimal objectives. See the testing standards near the top of this file — we do not use GoogleTest.

## Benchmarking

`bench/run_benchmark.py` runs instances and `bench/analyze_results.py` scores them. The benchmark is **`mipfeas`** (233 MIPLIB 2017 instances, primal integral); `PLATO` is the results host, not a name for it. **Don't pass `--threads`** unless asked. **The vanilla baseline is always a separately built unpatched binary** (`--vanilla-binary`); zeroing every effort is an ablation, never a baseline. Never edit a byte of a generated output (instance lists, captured tables). Everything else: `docs/BENCHMARKING.md`.
