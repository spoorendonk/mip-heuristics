# Reproducibility

What this project guarantees, what it deliberately does not, and how to
reproduce the recorded results.

## The reproducible recipe

Two settings, and only two:

```
threads = 1
random_seed = 42
```

```bash
printf 'threads = 1\nrandom_seed = 42\n' > repro.opts
./build/bin/highs --options_file repro.opts model.mps
```

Two runs of that command on one binary produce the same objective, the same
node count, and the same per-heuristic effort trace. It holds at the shipped
defaults, with the whole presolve chain live — the contract is not a reduced
configuration, and `tests/test_execution_modes.cpp` pins it at exactly this
pair of options. Zeroing a heuristic's effort narrows *what runs*; it is not
part of what makes the run reproducible.

**The custom options are not command-line flags.** HiGHS's CLI11 parser takes
only its own fixed flag set. `--mip_heuristic_fpr_effort 3.0 model.mps` makes `3.0`
a second positional argument and fails with `--model_file: File does not exist: 3.0`;
`--mip_heuristic_fpr_effort=3.0` fails with `The following argument was not
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
LocalMIP every `kTermCheckWork` counted units, FPR per inner attempt and then
at four finer cadences inside one (every `kDeadlinePollNodes` DFS nodes, every
RepairSearch node, every `repair_walk` step, and every
`kPropagateDeadlinePollWork` counted units inside a propagation fixpoint,
#151), Scylla per pump iteration — and the runner polls it unconditionally on
every iteration.
The overshoot is therefore one polling interval, not one *attempt*, and it no
longer grows with the effort option. Scylla keeps a documented floor of one
whole PDLP solve, which no constant can cross.

**The other side of that bound:** a run whose deadline actually binds is
wall-clock dependent, so it is not reproducible even at `threads=1` with a fixed
seed. This does not affect a normal solve — at the shipped effort defaults the
presolve chain finishes long inside any usable limit — but it does mean a
measurement run at a large effort option must be given a limit the chain does
not reach, or it measures the machine.

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

## Exact mipfeas reproduction

The recorded mipfeas result (233 MIPLIB 2017 instances, 600 s) is in
`README.md`. The runs behind it are 16-worker and therefore non-deterministic by
design, so what a re-run reproduces is the *result*, not the logs. What is
reproducible exactly is the protocol.

**Solver version.** HiGHS `v1.15.1`, fetched at configure time by
`cmake/FetchHiGHS.cmake` and patched from `third_party/highs_patch/`.

**Reference objectives.** `bench/miplib2017-v36.solu`, a verbatim copy of
upstream MIPLIB 2017's current solution file
(<https://miplib.zib.de/downloads/miplib2017-v36.solu>, retrieved 2026-08-20).
Do **not** pin an intermediate version: `v20`–`v35` carry `=opt= 111.0` for
`supportcase22`, which upstream itself retracted when a solution of 110 was
submitted, and a reference worse than achievable yields negative primal gaps.
`v36` records it feasible at `=best= 110.0`, which is what
`bench/instances_mipfeas.txt` counts it as.

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

**Vanilla-binary provenance.** The baseline binary is always named, never
discovered. `bench/run_mipfeas.sh` has no default for it — no PATH search and no
fallback to the patched build — so a config list naming `vanilla` without
`MIPFEAS_VANILLA_BINARY` fails with a message rather than running whichever
`highs` happens to be installed. Set it:

```bash
export MIPFEAS_VANILLA_BINARY=/path/to/unpatched/highs
```

It must be an unpatched build of the **same tag**; a different version makes the
comparison meaningless. Both facts are checked for you —
`bench/run_benchmark.py` probes the binary before the first solve and refuses
one that prints the patch marker or a different version — but check them
yourself too if the tree came from elsewhere.

**What a stage is.** The campaign stages differ in what they run, not in how
they are launched: every one of them is `run_mipfeas.sh` under a different
environment — `run_presolve_probe.sh`, `run_finalists.sh`,
`run_ablation_c.sh` and `run_headline.sh` are each a wrapper that sets one —
rather than a hand-written `run_benchmark.py` command line:

| | |
|---|---|
| `MIPFEAS_CONFIGS` | configs to run (default `vanilla all`) |
| `MIPFEAS_SEEDS` | seeds per config (default `0`) |
| `MIPFEAS_INSTANCES` | instance list (default `bench/instances_mipfeas.txt`) |
| `MIPFEAS_OUTPUT` | results tree (default `bench/results/mipfeas`) |
| `MIPFEAS_TIME_LIMIT` | seconds per solve (default 600, the benchmark's limit) |
| `MIPFEAS_BINARY` / `MIPFEAS_VANILLA_BINARY` | the two binaries |

A config name lists the heuristics to run and zeroes the rest; it carries no
budget of its own, so every heuristic it names runs at its shipped default. To move one for a
run, pass `run_benchmark.py --extra-options mip_heuristic_<name>_effort=<V>`.

```bash
# the headline: the shipped configuration at three seeds, against vanilla
MIPFEAS_CONFIGS="all vanilla" MIPFEAS_SEEDS="0 1 2" \
  bench/run_mipfeas.sh next 10
```

`status` counts an instance as done for a config only once *every* seed has
it, and the campaign as done at the least complete config — resume is per
(config, instance, seed), so a chunk boundary anywhere is harmless.

**The chunking protocol.** A full campaign is roughly 77 hours (233 instances ×
600 s × 2 configs, run interleaved so partial results are always paired and
comparable). `bench/run_mipfeas.sh` is built to be stopped and resumed:

```bash
bash bench/download_miplib.sh      # once per machine; stores to ~/data/miplib
bench/run_mipfeas.sh next 8          # run for up to 8 hours, then stop
bench/run_mipfeas.sh status          # progress and estimated time remaining
bench/run_mipfeas.sh next 8          # resume; repeat until status shows 233/233
```

`next` takes a *window* in hours (default 1) and hands the runner
`window - time_limit` as its wall-time budget, with `--skip-existing` so a
resumed run never redoes completed instances. The subtraction is not
cosmetic: the budget stops new instances being *launched*, and the one already
running still gets its full 600 s, so a chunk sized at the whole window
overruns it by up to ten minutes. Results accumulate in
`bench/results/mipfeas/`. When `status` reports `COMPLETE` the analysis runs
automatically; to run it by hand:

```bash
python3 bench/analyze_results.py bench/results/mipfeas \
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
inside `runSetup()`, before the chain, and their solutions are excluded as
`trivial-only`. The verdict deliberately follows the incumbent rather than
the pool's accept signal, for three reasons: it is the predicate the search's
objective actually scores; it is the only one both probe passes can evaluate,
since the filtering pass runs without `--dev-log`; and `[Heur]` is written when a
dispatch *ends*, so a killed run — which the probe's own per-run cap produces by
design — has incumbent rows and no ledger at all.

The artifact chain is **probe tree -> informative list -> tuning list**, every
link byte-identical for the same inputs and carrying no timestamp,
with each generated file recording its own `Regenerate with:` line.

### Running the probe

`bench/run_presolve_probe.sh` is `run_mipfeas.sh` with the probe environment, so
it chunks and resumes the same way — `next <hours>` overnight, `status` to check
in. The launcher *is* the configuration: every heuristic at effort `1e6` (the
option's ceiling, a budget no run inside the cap can reach), every patience gate
at `0` (which means no gate), `mip_heuristic_presolve_only`, `log_dev_level=3`,
and a **30 s** per-run cap the harness enforces as a wall-clock kill as well as
through `time_limit` — HiGHS checks its clock between work units, and an
instance that does not return from its own presolve never looks at it. With both
retirement conditions disabled no worker ever retires, so the wall clock is the
single stopping rule, the same one for all four heuristics on every instance.

```bash
bench/run_presolve_probe.sh preprobe next 8    # the experiment, 233 x 4 arms
bench/run_presolve_probe.sh preprobe status    # progress
bench/derive_from_probe.sh                     # every artifact, one command
```

It runs each heuristic **alone**, and no chained arm: `run_sequential` is
sequential, so a wall-clock cap truncates the chain's *tail*, and at a budget
that cannot bind the first heuristic takes the entire cap on every instance — a
chained probe would report "produced nothing" for instances where three of the
four never executed. Membership in the informative set is the union over the
four singles, which dominates every subset: a heuristic that cracks an instance
inside some mix also cracks it running alone, earlier and with the whole cap to
itself.

Two controls answer questions about the experiment rather than about the
instances, and both run over a subset:

| mode | what it controls for |
|---|---|
| `budget` | the same probe at effort 1.0, where the budget binds on small models. `attempt_cap` is derived from the total budget, so a trace at one budget does not exactly reproduce another — and this also says whether membership moved |
| `serial` | the same probe at `threads=1`. The multi-worker regime is the one the search runs in, so it is what the experiment uses; this says whether its quantiles are an artifact of worker interleaving |

Read a finished tree with `bench/analyze_presolve_probe.py`, never with
`analyze_results.py`: a presolve-only run computes no dual bound, so its gap is
meaningless.


## Reproducing the campaign, stage by stage

Every stage is a tracked launcher plus a tracked reader, and every stage's
derived artifacts are committed even though its results tree is not
(`bench/results*` is gitignored, ~5 GB). So the numbers in each write-up are
readable without the logs, and regenerable from them.

| stage | launcher | reader | write-up |
|---|---|---|---|
| vanilla baseline (#105) | `bench/run_mipfeas.sh next <hours>` | `bench/analyze_results.py` | — |
| presolve probe (#113) | `bench/run_presolve_probe.sh preprobe next <hours>` | `bench/derive_from_probe.sh` | `bench/ablation_effort/` |
| joint search (#107) | `bench/run_irace.sh all` | `bench/analyze_irace.py` | `bench/ablation_search/` |
| finalist confirmation (#107) | `bench/run_finalists.sh confirm <hours>` | `bench/compare_finalists.sh confirm` | `bench/ablation_search/` |
| held-out validation (#107) | `bench/run_finalists.sh heldout` | `bench/compare_finalists.sh heldout` | `bench/ablation_search/` |
| `fpr_lp` capability (#165) | `bench/run_ablation_c.sh capability <hours>` | `bench/analyze_ablation_c.py capability` | `bench/ablation_fprlp/` |
| `fpr_lp` contribution (#165) | `bench/run_ablation_c.sh contribution <hours>` | `bench/analyze_ablation_c.py contribution` | `bench/ablation_fprlp/` |
| headline (#108) | `bench/run_headline.sh until <HH:MM>` | `bench/run_headline.sh report` | `bench/headline/` |
| archive | `bench/make_archive.py build <tree> --output <dir>` | the archive's own `REGENERATE.sh` | — |

`FINALISTS_ONLY` names one arm of `run_finalists.sh` for a stage that owns one
rather than the whole finalist set — which is how Ablation C's contribution arm
and the late-added `B'-mix-cheapest` held-out row were run without editing the
record of what the search selected.

### Two things a re-runner must know before starting

**A configuration's `.opts` does not identify it.** Both the headline arm and
its predecessor ran at *default options*, so their `.opts` files are
byte-identical — the seed and nothing else, since `all` zeroes nothing — and what differed
was the binary's built-in defaults. The results *directory* is the only record
of which configuration a run used, which is why `bench/results/mipfeas/` carries
`all` and `all-prev-vector` rather than two trees both called `all`. If you
re-run a stage after changing a default, give it a new directory or the
harness's `--skip-existing` will report the old runs as done.

**The headline arm passes no `--extra-options` at all**, deliberately. Passing
the eight effort and patience values explicitly would measure a vector that may
differ from the shipped defaults in the last decimal, so the headline would
describe a configuration nobody gets. `bench/run_headline.sh` therefore sets
only the config name and the seed, and the correctness of the measurement
depends on the binary being the one whose defaults are under test — which the
launcher checks by refusing a binary without the patch marker.

### What the campaign did not do, and why

* **One seed**, not the three the headline stage asks for. Extra seeds shrink
  only the within-instance variance, the between-instance component is what the
  paired sd is mostly made of, and the baseline is also single-seed, so
  averaging patched seeds removes at most half the seed noise. That bounds the
  gain but does not eliminate it: at the held-out sd, three seeds would resolve
  16.1-16.8% against the observed 16.4%, so they *might* have sharpened this —
  and with one seed the seed-noise share cannot be estimated to say. The
  decision stands on the effect already being separated at p = 0.009 rather
  than on the extra seeds being useless. Full arithmetic in
  `bench/headline/README.md`.
* **No `off` patch-overhead arm at scale**, no additional vanilla seeds, no
  offline best-of row. None would change what ships or what is claimed.
* **No share sweep for `fpr_lp`** — it produced zero accepted incumbents in
  ~100 paired runs across two presolve backgrounds, so there is no share at
  which it earns its slice of upstream's RENS/RINS envelope.

## Zeroing every heuristic is an ablation, not a vanilla baseline

Setting all five `mip_heuristic_<name>_effort` options to `0` disables our four
presolve heuristics and the dive-time `fpr_lp`. That makes it the reference row
for "what does the chain contribute on this binary?", and that is the only
thing it is. `bench/run_benchmark.py` spells it as the config `off`.

It runs **no FeasibilityJump at all**: upstream's standalone FJ call site never
fires on a patched build, at any configuration, because our chain owns FJ and a
live native site would double-run it.

It is **not** a vanilla measurement and must not be used as one. The binary is
still the patched one, and the patch modifies FeasibilityJump itself —
visibly so already at `log_dev_level=3`, where the per-bump `Reached a local
minimum.` line is absent, and in the search itself, where the patch corrects
two upstream FeasibilityJump defects — the negative-coefficient jump value and
the objective term's sign. A
vanilla baseline is always a **separately built unpatched binary** of the tag
in `cmake/FetchHiGHS.cmake`, and `bench/run_benchmark.py` enforces it: the
`vanilla` config requires `--vanilla-binary`, there is no fallback to the
patched build, and the binary is probed and refused *before the first solve* if
it carries the `mip-heuristics patch active` marker or reports a different
version.

What is proven against an unpatched binary is the **pure patch-overhead**
configuration — nothing of ours running, and FeasibilityJump disabled on both
sides, so the one component the patch deliberately changes is out of the
comparison rather than compared and forgiven:

```bash
python3 bench/check_vanilla_equivalence.py \
    --patched-binary ./build/bin/highs \
    --vanilla-binary /path/to/unpatched/highs
```

It hands the patched binary every `mip_heuristic_<name>_effort` at `0` plus
`mip_heuristic_run_feasibility_jump=false`, and the unpatched one
`mip_heuristic_run_feasibility_jump=false` (upstream's own option), then
compares status, primal bound, node count, and total and heuristic LP
iterations, and diffs the logs once wall-clock content is normalised away (the
timing block, the P-D integral, profiling seconds, the git-hash width, the
options-file echo, and the patch marker line). What that establishes is that
injecting the heuristics does not move HiGHS's presolve, its B&B or its LP path
by a node or an iteration. Two residual differences are accepted and
documented: the marker line itself, and the instrumentation lines only the
patched side can print. One consequence is deliberate and worth stating: with
FeasibilityJump disabled on both sides, nothing in this project compares FJ's
*search behaviour* against upstream's. FJ is what the patch changes on purpose,
so there is no behaviour to hold equal there — an equivalence claim that
quietly included it would be the false one. That HiGHS's own FeasibilityJump still runs at `off`
and still charges its effort is pinned separately, by
`tests/test_native_fj.cpp` — presence and accounting, not bit-identity.

## Instrumentation caveat

The per-heuristic instrumentation needs `log_dev_level=3`, which
`bench/run_benchmark.py` exposes as `--dev-log` and leaves off by default.
**It is not free and it is not neutral:** HiGHS's own FeasibilityJump logs one
line per weight bump at exactly that level, from every parallel FJ worker, each
with an `fflush` — 99.8% of a traced run's volume, and enough that a clock-bound
traced FJ dispatch was not the same heuristic as an untraced one. **The patch
drops that line**, so what remains at level 3 is FJ's periodic table plus our
own two lines: a few per cent of wall time at campaign limits, and
proportionally more on very short solves, where a fixed logging cost dominates
the solve it is timing.

No ratio has been measured at a campaign time limit, and a ratio taken on a
short solve does not transfer: a fixed logging cost against a 48 ms solve says
nothing about a 600 s one.

**Attribution runs and headline-timing runs are therefore different runs.** Do
not read a timing number off a `--dev-log` tree, and do not expect attribution
tables from one without it — without the flag the attribution tables come out
empty rather than wrong.
