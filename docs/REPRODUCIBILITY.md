# Reproducibility

What this project guarantees, what it deliberately does not, and how to
reproduce the recorded results.

## The reproducible recipe

Three settings together:

```
threads = 1
random_seed = 42
mip_heuristic_suite = fpr
```

```bash
printf 'threads = 1\nrandom_seed = 42\nmip_heuristic_suite = fpr\n' > repro.opts
./build/bin/highs --options_file repro.opts model.mps
```

Two runs of that command on one binary produce the same objective, the same
node count, and the same per-heuristic effort trace.

**The custom options are not command-line flags.** HiGHS's CLI11 parser takes
only its own fixed flag set. `--mip_heuristic_suite fpr model.mps` makes `fpr`
a second positional argument and fails with `File does not exist: fpr`;
`--mip_heuristic_suite=fpr` fails with `The following argument was not
expected`. Both exit non-zero without solving. Anything scripted goes through
`--options_file`.

`random_seed` alone is not enough, and neither is `threads=1` alone. The seed
has to reach *our* workers, and a single worker per heuristic is what removes
the scheduling non-determinism.

### The executable specification

Prefer the tests to this document where they disagree — they run on every
commit and this file does not.

- `tests/test_fpr.cpp`, the `[fpr][resume][determinism]` cases (egout, bell5,
  flugpl): same seed reproduces the same objective at a small effort budget,
  with a guard asserting that the pause/resume path was actually engaged so the
  determinism assertion cannot pass vacuously. These pin *intra-worker*
  lifecycle determinism — a DFS paused at a budget gate and resumed on the next
  call must land in the same place.
- `tests/test_execution_modes.cpp`, the reproducible-mode cases:
  - `threads=1 same seed reproduces the run` — objective, node count and effort
    trace all match, plus a non-empty-trace guard.
  - `threads=1 different seeds take different search paths` — asserted on the
    effort trace, *not* on the node count. HiGHS consumes `random_seed` in its
    own branching, so a differing node count would be satisfied with zero
    contribution from our heuristics and would prove nothing about them. The
    trace is filtered to the presolve chain for the same reason: `fpr_lp`'s
    per-call budget is a function of `total_lp_iterations`, which HiGHS moves
    independently of our workers.
  - `threads=1 still finds the optimum` — the reproducible configuration must
    not be a degenerate one.

## What is not reproducible

**Any run with `threads > 1`.** This is by design, not a defect.

Every heuristic runs on the same continuous parallel loop
(`src/opportunistic_runner.h`): workers run until the global effort budget, a
staleness budget, or an external termination signal stops them, with no epoch
barrier between them. Effort accounting is therefore order-dependent — the
header states the bound explicitly: concurrent workers can overshoot
`budget.total` by up to `n * budget.attempt_cap`, because each worker tests the
atomic total *before* starting an attempt. Bounded overshoot is acceptable for
heuristic effort accounting, and removing it would mean reintroducing the
barrier the closeout deleted.

The guarantee is **deterministic algorithm behaviour, not deterministic parallel
scheduling.** Which worker wins a race, which solution reaches the pool first,
and how much a heuristic overshoots its slice are all scheduling facts.

A deterministic epoch-gated parallel mode used to exist and was removed in the
closeout: it carried substantial complexity, it was not the production mode, and
`threads=1` already provides the reproducibility contract without a new option.

## Why `threads=1` is not the benchmark configuration

It collapses each heuristic to a single worker. That is the right setting for a
determinism test and for debugging, and the wrong one for measuring anything:

- Scylla's whole design is N pump chains contending for one PDLP instance. At
  one worker there is no contention, no stale-snapshot path, and none of the
  behaviour the implementation exists to exercise.
- Throughput does not scale uniformly. Scylla is PDLP/mutex-bound and scales
  sublinearly in workers where FPR and LocalMIP scale near-linearly, so the
  worker count does not cancel out of a ratio: the same binary on the same
  instances gives `local_mip:scylla = 4.68` at 16 workers and `2.81` at 6.

Do not pass `--threads` to `bench/run_benchmark.py`, and do not put `threads=`
in a benchmark options file. Let HiGHS use its default.

## Exact PLATO reproduction

The recorded PLATO mipfeas table (233 MIPLIB 2017 instances, 600 s) is in
`README.md`, together with the provenance caveat that matters most: **it cannot
be reproduced on `HEAD`.** It was measured at `mip_heuristic_preset=all_opp` —
FJ + FPR + LocalMIP with Scylla deliberately excluded — and the single-valued
`mip_heuristic_suite` cannot express that combination. The closest value, `all`,
adds Scylla, which is a composition change and not a rename. The binary is gone
too: the numbers predate the runner cleanup. Treat the row as the last
full-campaign result, not as a claim about `HEAD`.

What *is* reproducible is the protocol.

**Solver version.** HiGHS `v1.15.1`, fetched at configure time by
`cmake/FetchHiGHS.cmake` and patched from `third_party/highs_patch/`.

**Reference objectives.** `bench/miplib2017-v36.solu`, a verbatim copy of
upstream MIPLIB 2017's current solution file
(<https://miplib.zib.de/downloads/miplib2017-v36.solu>, retrieved 2026-08-20).
It replaced a bundled `v22` copy that marked `supportcase22` `=inf=` while
`bench/instances_plato.txt` counted it among the 233 feasible instances —
upstream has since recorded it feasible at `=best= 110.0`. Over the 233 PLATO
instances the refresh moves exactly three entries: `supportcase22`, plus
corrected optima for `neos-3754480-nidda` (12941.738 → 12939.754) and
`binkar10_1` (6742.200 → 6741.380). The recorded README table predates the
refresh, which is one more reason it is a historical row rather than a claim
about `HEAD`. Do **not** pin an intermediate version to resolve the
`supportcase22` question: `v20`–`v35` carry `=opt= 111.0`, which upstream
itself retracted when a solution of 110 was submitted, and a reference worse
than achievable yields negative primal gaps.

An instance whose solution-file tag asserts no finite objective (`=inf=`,
`=unbd=`) is excluded from every table by `bench/analyze_results.py`, with the
exclusion printed. A gap against such an instance falls back to the best
*observed* primal, which is zero for whichever config found it — a
self-referential number that would enter the headline SGM looking like a real
one.

**Telling a patched binary from an unpatched one.** The version and githash
banners are identical between them — `highs --version` prints exactly the same
line either way. The distinguishing marker is printed by a *solve*, on the third
line of the log header:

```
Running HiGHS 1.15.1 (git hash: 04024d701f): Copyright (c) 2026 under MIT licence terms
Includes third-party software components, see THIRD_PARTY_NOTICES.md for full details
mip-heuristics patch active (custom MIP presolve heuristics; spoorendonk/mip-heuristics)
```

Check for that line before trusting any results tree's provenance.

**Vanilla-binary provenance.** `bench/run_plato.sh` defaults its vanilla binary
to the system HiGHS (`which highs`) and falls back to the *patched* build if
there is none — so an unset `PLATO_VANILLA_BINARY` on a machine without a system
HiGHS silently runs both arms on one binary. Set it explicitly:

```bash
export PLATO_VANILLA_BINARY=/path/to/unpatched/highs
```

It must be an unpatched build of the **same tag**; a different version makes the
comparison meaningless. Confirm it prints no patch marker.

**The chunking protocol.** A full campaign is roughly 77 hours (233 instances ×
600 s × 2 configs, run interleaved so partial results are always paired and
comparable). `bench/run_plato.sh` is built to be stopped and resumed:

```bash
bash bench/download_miplib.sh      # once per machine; stores to ~/data/miplib
bench/run_plato.sh next 8          # run for up to 8 hours, then stop
bench/run_plato.sh status          # progress and estimated time remaining
bench/run_plato.sh next 8          # resume; repeat until status shows 233/233
```

`next` takes an hour budget (default 1) and passes it through as a wall-time
budget, with `--skip-existing` so a resumed run never redoes completed
instances. Results accumulate in `bench/results/plato/`. When `status` reports
`COMPLETE` the analysis runs automatically; to run it by hand:

```bash
python3 bench/analyze_results.py bench/results/plato \
    --configs patched vanilla --time-limit 600 --baseline --summary
```

## `suite=off` is vanilla-equivalent — since August 2026

As of the option-surface collapse (issue #93, landed 2026-08-07),
`mip_heuristic_suite=off` on the *patched* binary is equivalent to an unpatched
build of the same tag. The patch hands HiGHS's standalone FeasibilityJump call
site back at that value; previously it was hard-disabled in every
configuration, which meant the patched binary had no vanilla-equivalent
setting at all.

This is proven rather than assumed:

```bash
python3 bench/check_vanilla_equivalence.py \
    --patched-binary ./build/bin/highs \
    --vanilla-binary /path/to/unpatched/highs
```

It compares status, primal bound, node count, and total and heuristic LP
iterations, and diffs the logs once wall-clock content is normalised away
(the timing block, the P-D integral, profiling seconds, the git-hash width, the
options-file echo, and the patch marker line). Verified 12/12 on the six
bundled instances × 2 seeds. Two residual differences are accepted and
documented: a `heuristic_effort_used` store inside HiGHS's own
`feasibilityJump()`, and the marker line itself.

> **Results produced with the older `mip_heuristic_preset=off` are not
> comparable.** That value did *not* restore native FeasibilityJump, so an old
> `preset=off` row is "vanilla minus FJ", not vanilla. Any baseline measured
> before 2026-08-07 has to be re-run.

For a headline benchmark baseline, prefer `--vanilla-binary` pointed at a real
unpatched build anyway. It is the stronger claim and costs nothing but a second
checkout. Note the corollary in `bench/run_benchmark.py`: on the patched binary
the `vanilla` and `off` configs are the *same run* unless `--vanilla-binary` is
given, so asking for both without it runs every instance twice for one data
point.

## The Thompson-sampling negative result

The project once selected among heuristics with a Thompson-sampling portfolio.
It is an original contribution and it **did not improve results**, so it was
removed from the mainline rather than kept as dead weight. It is retained as a
documented negative result.

The implementation is in git history, not in a branch or an archive tag:

- Last commit containing it: **`00c47c0`** (`src/thompson_sampler.cpp`,
  `src/thompson_sampler.h`).
- Removed by: **`d6fa834`** — *"refactor: delete the portfolio, ThompsonSampler
  and bandit runner"*.

```bash
git show 00c47c0:src/thompson_sampler.cpp
git checkout 00c47c0 -- src/thompson_sampler.cpp src/thompson_sampler.h  # to build against it
```

Anyone reproducing the negative result should start from `00c47c0` as a whole
tree; the sampler does not drop into `HEAD`, whose runner contract and option
surface both changed underneath it.

## Instrumentation caveat

The per-heuristic instrumentation needs `log_dev_level=3`, which
`bench/run_benchmark.py` exposes as `--dev-log` and leaves off by default.
**It is not free and it is not neutral:** HiGHS's own FeasibilityJump logs one
line per weight bump at exactly that level, from every parallel FJ worker, each
with an `fflush`. Measured on five bundled instances at a 10 s limit that is
97–750x the log volume (bell5: 16 KB → 3.5 MB) and 1.1–4.4x the total solve
wall time (egout: 0.048 s → 0.212 s), concentrated in the FJ phase — which is
the very window the attribution numbers report.

**Attribution runs and headline-timing runs are therefore different runs.** Do
not read a timing number off a `--dev-log` tree, and do not expect attribution
tables from one without it — without the flag the attribution tables come out
empty rather than wrong.
