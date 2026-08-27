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
and how much a heuristic overshoots its effort slice are all scheduling facts.

The *wall-clock* limit is a separate axis and is bounded at every worker count,
including `threads=1` (#114). Each of the four presolve heuristics polls
`ExecutionContext::past_deadline()` from inside its own inner loop, on a cadence
of its own — FeasibilityJump per upstream callback (every 500 000 effort units),
LocalMIP every `kTermCheckInterval` steps, FPR per inner attempt, Scylla per
pump iteration — and the runner polls it unconditionally on every iteration.
The overshoot is therefore one polling interval, not one *attempt*, and it no
longer grows with the effort option. Scylla keeps a documented floor of one
whole PDLP solve, which no constant can cross.

**The other side of that bound:** a run whose deadline actually binds is
wall-clock dependent, so it is not reproducible even at `threads=1` with a fixed
seed. This does not affect a normal solve — at the shipped effort defaults the
presolve chain finishes long inside any usable limit — but it does mean a
measurement run at a large effort option must be given a limit the chain does
not reach, or it measures the machine.

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
FJ + FPR + LocalMIP with Scylla deliberately excluded. That composition is
expressible again as `mip_heuristic_suite = fj,fpr,local_mip` (#112), having
been unnameable while the option took a single value, but the binary is gone:
the numbers predate the runner cleanup. Treat the row as the last full-campaign
result, not as a claim about `HEAD`.

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

**What a stage is.** The four campaign stages differ in what they run, not in
how they are launched, so each one is `run_plato.sh` with a different
environment rather than a hand-written `run_benchmark.py` command line:

| | |
|---|---|
| `PLATO_CONFIGS` | configs to run (default `vanilla all`) |
| `PLATO_SEEDS` | seeds per config (default `0`) |
| `PLATO_INSTANCES` | instance list (default `bench/instances_plato.txt`) |
| `PLATO_OUTPUT` | results tree (default `bench/results/plato`) |
| `PLATO_TIME_LIMIT` | seconds per solve (default 600, the PLATO limit) |
| `PLATO_BINARY` / `PLATO_VANILLA_BINARY` | the two binaries |

A config name is exactly a `mip_heuristic_suite` value and carries no budget
of its own; every heuristic runs at its shipped default. To move one for a
run, pass `run_benchmark.py --extra-options mip_heuristic_<name>_effort=<V>`.

```bash
# the headline: the selected configuration at three seeds, against vanilla
PLATO_CONFIGS="fj+fpr+local_mip vanilla" PLATO_SEEDS="0 1 2" \
  bench/run_plato.sh next 10
```

`status` counts an instance as done for a config only once *every* seed has
it, and the campaign as done at the least complete config — resume is per
(config, instance, seed), so a chunk boundary anywhere is harmless.

**The chunking protocol.** A full campaign is roughly 77 hours (233 instances ×
600 s × 2 configs, run interleaved so partial results are always paired and
comparable). `bench/run_plato.sh` is built to be stopped and resumed:

```bash
bash bench/download_miplib.sh      # once per machine; stores to ~/data/miplib
bench/run_plato.sh next 8          # run for up to 8 hours, then stop
bench/run_plato.sh status          # progress and estimated time remaining
bench/run_plato.sh next 8          # resume; repeat until status shows 233/233
```

`next` takes a *window* in hours (default 1) and hands the runner
`window - time_limit` as its wall-time budget, with `--skip-existing` so a
resumed run never redoes completed instances. The subtraction is not
cosmetic: the budget stops new instances being *launched*, and the one already
running still gets its full 600 s, so a chunk sized at the whole window
overruns it by up to ten minutes. Results accumulate in
`bench/results/plato/`. When `status` reports `COMPLETE` the analysis runs
automatically; to run it by hand:

```bash
python3 bench/analyze_results.py bench/results/plato \
    --configs all vanilla --time-limit 600 --baseline --summary
```

## The presolve-only screen and the tuning search

`mip_heuristic_presolve_only` exits after the presolve chain and before the root
LP, which is what makes the #107 tuning search affordable: roughly 25x cheaper
than a full solve, and it measures the chain rather than a chain diluted by
B&B. Such a run reports `Solution limit reached`, `Nodes 0`, `LP iterations 0`,
`Dual bound -inf` and CLI exit 1. **A presolve-only tree is therefore not
comparable to a full-solve tree on any dual-side metric, and its "gap" is
meaningless.** It is reproducible on the same terms as everything else here:
`threads=1` plus a fixed `random_seed`.

**Do not pin `threads` for a tuning run.** It is tempting — the presolve chain
races N workers to submit, so the presolve-exit objective varies ~3 % run to run
at a fixed seed, and `threads=1` is bit-stable. But pinning does not narrow that
distribution, it *moves* it: measured, the single-worker regime is steadier and
strictly worse, never sampling the outcome the multi-worker chain occasionally
wins. It is also a transfer error — FJ's budget is per-worker x N while the
other three are whole-dispatch, so changing N *reallocates* budget between
heuristics rather than rescaling it (measured p0548, N=1 -> N=8: local_mip
1.08x, fj 12.0x). Racing selects under noise; it cannot detect a reallocation.
A tuned vector is only valid at the worker count it was tuned at, which is why
`workers_observed` is recorded.

### Reading a trace

`[HeurSol]` and `[Heur] nnz=` both require `log_dev_level=3`, so they belong to
attribution runs, never to headline-timing runs. Note the trap that shaped the
design: level 3 *suppresses* the one-line model header, so `[Heur] nnz=` is the
only correct nonzero count available on any log that carries a trace.
`DispatchTrace.normalized_gaps()` is already scaled into the option's own unit —
do not rescale it. `stale_effort` is deliberately unavailable for Scylla: its
`[Heur]` total charges the full PDLP cost while its per-worker counter charges
it divided by N, and only the PDLP half is amortised, so no scalar corrects it.

### What the probe decides, and on what evidence

`bench/analyze_presolve_probe.py` **refuses a tree that is not a presolve-only
run of a patched binary**, so a list pinned by digest into a tuning-set header
cannot silently have come from a full-solve tree. Informativeness means *the
chain produced the incumbent* — a display row with one of the chain's own source
codes — not merely that a solution exists: HiGHS's own trivial heuristics run
inside `runSetup()`, before the chain, and their solutions route to the hard tier
as `trivial-only`. The verdict deliberately follows the incumbent rather than
the pool's accept signal, for three reasons: it is the predicate the search's
objective actually scores; it is the only one both probe passes can evaluate,
since the filtering pass runs without `--dev-log`; and `[Heur]` is written when a
dispatch *ends*, so a killed run — which the probe's own per-run cap produces by
design — has incumbent rows and no ledger at all.

The artifact chain is **probe tree -> informative list + hard tier -> tuning
list**, every link byte-identical for the same inputs and carrying no timestamp,
with each generated file recording its own `Regenerate with:` line.

### Running the probe

`bench/run_presolve_probe.sh` is `run_plato.sh` with the probe environment, so
it chunks and resumes the same way — `next <hours>` overnight, `status` to check
in. The launcher *is* the configuration: every heuristic at an effort that
cannot bind with its patience gate at 0 (which means no gate),
`mip_heuristic_presolve_only`, and a 60 s per-run cap the harness enforces as a
wall-clock kill.

```bash
bench/run_presolve_probe.sh filter next 8      # the instance screen
python3 bench/analyze_presolve_probe.py bench/results/probe/filter \
    --informative-output bench/results/probe/informative.txt \
    --hard-tier-output bench/results/probe/hard_tier.txt
bench/run_presolve_probe.sh trace next 4       # the trajectories
bench/run_presolve_probe.sh trace-low next 4   # the same, one decade down
```

It runs each heuristic **alone**, plus the chain: `run_sequential` is
sequential, so a wall-clock cap truncates the chain's *tail*, and at effort 1.0
with the gates off FJ's budget is large enough to consume the whole cap. On a
12-instance pilot the chained run alone would have filed three instances as
"produced nothing at a generous configuration" that a different heuristic
cracks. The informative set is therefore the union over `(single, seed)`, and
the chained arm is kept because it is the only one that measures the chain
interaction.

The filter pass runs at HiGHS's own thread default, which is the regime the
search runs in; the trace passes pin `threads=1` and `log_dev_level=3`, because
a trajectory is an effort *timeline* and multi-worker interleaving makes it
non-reproducible. `trace-low` exists to measure one confound: `attempt_cap` is
derived from the total budget, so a trajectory taken at effort 1.0 does not
exactly reproduce one taken at 0.1, and the two are compared over the effort
range they share.


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
