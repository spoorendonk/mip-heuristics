# `fpr_lp` — what it contributes, and why it ships off (#165)

Ablation C. Companion to `bench/ablation_effort/` (Ablation A, which produced
the four presolve effort and patience defaults) and `bench/ablation_search/`
(Ablation B, the joint search over the presolve mix).

**Headline: `fpr_lp` keeps its shipped default of `0` — disabled — on evidence
rather than on caution.** It is capable of producing accepted incumbents when
it owns the LP-iteration envelope, produces none when RENS and RINS compete for
it, and costs **27%** of the primal integral on the instances where it engages
and finds nothing. That is 57% of the benchmark set.

## Reproducing

| step | command | artifacts |
|---|---|---|
| C0, capability | `bench/run_ablation_c.sh capability 3` | `bench/results/ablation_c/capability/` |
| read it | `bench/analyze_ablation_c.py capability <tree>/fpr_lp/seed0` | `capability.txt` here |
| C1, contribution | `bench/run_ablation_c.sh contribution 6` | `bench/results/finalists/confirm/E-shipped-plus-fprlp/` |
| read it | `bench/analyze_ablation_c.py contribution <D>/all <E>/all --labels <C0>/fpr_lp/seed0` | `contribution.txt` here |

Results trees are gitignored (`bench/results*`); every derived artifact needed
to read the result is in this directory.

Both stages ran on the 16-core / 32-thread benchmark machine at HiGHS's default
worker count — every log reads `Thread count 16 (of 32 threads). Using 1 max
workers. Parallel search off` — on `bench/results/irace/bin/highs`, the same
binary Ablation B's arms used. One seed.

## The question, and why it needed asking

`fpr_lp` is the LP-dependent fix-and-propagate heuristic, run during the B&B
dive rather than in presolve. It is the one heuristic of the five that Ablation
A never measured: #113's probe is presolve-only, and `fpr_lp` does not exist
there.

It is not a free rider either. `fpr_lp` charges its work back to
`heuristic_lp_iterations` / `total_lp_iterations`, the counters
`moreHeuristicsAllowed()` reads to decide whether RENS and RINS run — so its
budget is **zero-sum against upstream's own two dive heuristics**. Every LP
iteration it takes is one they do not get. Shipping it unmeasured is therefore
not neutral in either direction.

Until #164 the question could not even be asked on the shipped binary: one
suite token gated presolve FPR and `fpr_lp` together, so "presolve FPR without
`fpr_lp`" had no spelling. #164 gave it its own token and its own effort
option; this is the measurement.

## Design: a capability gate before a contrast

The issue's original design opened with a fire-rate measurement at shipped
defaults. That is the wrong first experiment, because the shipped chain
squeezes `fpr_lp` from three sides — RENS/RINS drain the envelope first,
`headroom_iters <= 0`, and `max_effort < nnz << 8` — so a null there is
answerable with "you never gave it a chance" and cannot end the exercise.

**C0 inverts it.** `fpr_lp` runs alone (`suite=fpr_lp`), owns the whole
envelope (`mip_heuristic_run_rens=false`, `mip_heuristic_run_rins=false`,
`mip_heuristic_effort=1.0`), and gets a per-call budget that cannot bind
(`mip_heuristic_fpr_lp_effort=100`). Its output is a **count** — dispatches,
and accepted incumbents — so it needs no power calculation and a zero is
decisive at any n.

That ordering is a direct response to Ablation B, whose methodological finding
was that a mean-difference experiment sized to what was affordable produces a
null that cannot be read as an answer. A count sidesteps that entirely.

**`threads=1` is deliberately not set.** The design called for it so
`parallelLockActive()` would not skip the heuristic; `src/fpr_lp.cpp` says the
lock is held only under multi-worker B&B, which is not HiGHS's default, and
every log in both stages confirms it. Pinning it would have measured a regime
nothing ships in.

## Stage C0 — capability

49 instances (`bench/instances_confirm48.txt`, the stratified confirmation set),
120 s, one seed. Full table in `capability.txt`.

| | |
|---|---|
| dispatches | **87** |
| setup bails (`abandoned_setup=1`) | **0** |
| fired on | **38 / 48 observable instances (79%)** |
| accepted incumbents (`D`-sourced) | **60**, on **10 / 49** instances |
| killed runs | 1 (`germanrr`, excluded from the fire-rate denominator) |

**The gate does not close.** Given every advantage `fpr_lp` reaches dive nodes
on four instances in five and produces accepted incumbents on one in five;
`markshare2` alone yields 20 from 9 dispatches.

**What C0 does not establish is value.** RENS and RINS are disabled here, so a
`D`-sourced incumbent is not necessarily an *incremental* solution — it may be
one RENS or RINS would have found from the same envelope. C0 answers "can it",
which is what a gate is for.

## Stage C1 — contribution at the campaign metric

The shipped configuration with and without `fpr_lp`, paired on identical
instances at the 600 s campaign limit: `D-shipped` (Ablation B's confirmed
incumbent) against `E-shipped-plus-fprlp`, which is `D-shipped` plus
`mip_heuristic_fpr_lp_effort=1.0` and nothing else. Full output in
`contribution.txt`.

That pairing resolves better than anything in Ablation B, because the two arms
differ in exactly one option, so configuration cancels along with instance
difficulty.

### Headline: a null

| set | n | E/D | 95% CI | t |
|---|---|---|---|---|
| **overall** | **49** | **1.137** | **[0.99, 1.31]** | +1.75 |
| pilot (20, run early) | 20 | 1.179 | [1.055, 1.317] | +2.92 |
| pilot-free (29) | 29 | 1.108 | [0.880, 1.396] | +0.87 |

**The pilot is reported separately because it was run before this design
existed** — the arm was enabled in `dfa90d2` and disabled again in `abe1043`,
and 20 instances landed in between. Its separation did not survive: same
direction on the new 29, but the paired sd more than doubles (0.252 → 0.635).
Those 20 were the quiet instances. `bench/ablation_search/` carries the same
convention for the same reason (`analysis-set-pilotfree29.txt`).

### The result: a mixture, not a flat landscape

Partitioned by C0's independent labels:

| subgroup | n | ratio | 95% CI | t |
|---|---|---|---|---|
| C0 yielded | 10 | 0.952 | [0.58, 1.55] | −0.20 |
| **C0 fired, never yielded** | **28** | **1.270** | **[1.07, 1.51]** | **+2.66** |
| C0 never fired (**null control**) | 10 | 1.007 | [0.99, 1.02] | +1.02 |

**The null control is the load-bearing row.** On the ten instances where the
dive never dispatched the two arms agree to within 1% — CI [0.99, 1.02]. That
is what says the pairing is tight and the 1.270 is a real effect rather than
run-to-run nondeterminism at 16 workers. `analyze_ablation_c.py` performs that
check itself and refuses to let the other subgroups be read if the control ever
separates.

So the overall null is an average over three populations: a 27% cost where
`fpr_lp` engages and finds nothing, level where it finds something, exactly
neutral where it never runs.

**Caveats, stated rather than discovered.** The labels come from C0's
configuration, not from these runs, so they are a proxy for "an instance where
`fpr_lp` engages"; the split is post-hoc and hypothesis-generating. It is *not*
outcome selection — the partition is fixed by an independent run rather than by
the metric being compared — but it is not a second headline either.

### The aggregate effect is unresolvable on this benchmark

At the measured paired sd of 0.511, `n = 8 sd^2 / ln(1+delta)^2` gives:

| n | smallest detectable difference |
|---|---|
| 49 (this stage) | 22.9% |
| 81 (the pre-registered escalation) | 17.4% |
| 97 (confirm + held-out, one seed) | 15.8% |
| **128** | **13.6% — the observed difference** |

The whole available pool is 97 instances at one seed, so no affordable
extension resolves 13.7%. Reported rather than spent around: **the aggregate
contribution of `fpr_lp` cannot be resolved on this benchmark at one seed.**
What is resolved is the conditional effect, and it has a mechanism.

## Verdict

**`fpr_lp` is dominated by RENS/RINS inside the shared envelope.** It produces
accepted incumbents when nothing competes for the LP iterations (C0: 60, on 10
of 49 instances) and none when they do (the pilot: 0 across 20 instances at 600
s, five times the wall clock) — while costing 27% wherever it engages.

**Shipped setting: `mip_heuristic_fpr_lp_effort = 0`.** The value does not
move; its justification does, from "unmeasured, so off" to "measured, and off
because it costs".

**No share sweep, and this is a decision rather than a deferral.** Raising the
share hands more of the envelope to a heuristic that already fires on 79% of
instances and yields on 20%, which by this mechanism makes the 1.270 worse;
lowering it shrinks cost and yield together. There is no share at which "costs
27% on 28 instances, level on 10" becomes a win, and one dimension gives a
search nothing to navigate while costing >= 504 experiments (~85 h at 600 s).

**The lever the data points at is a gate, not a budget** — do not dispatch the
dive where it will not yield. That is a code change, out of scope for a
measurement, and recorded here as the finding rather than as an action.

## For a write-up

Three things here are worth stating in a paper, and two of them are about
method rather than about `fpr_lp`.

1. **A capability gate is cheaper than a contrast and can end the question.**
   C0 cost 1.3 h and would have closed the whole campaign had it come back
   zero. It needs no power calculation because its output is a count. Ablation
   B spent ~30 h establishing a null it could have predicted.
2. **An untuned subgroup with an independent label beats a mean.** The
   aggregate says nothing (1.137, CI spanning 1); the partition says 1.270 with
   a null control at [0.99, 1.02]. A heuristic that fires on some instances and
   not others cannot be summarised by its mean effect, and the instances where
   it cannot fire are a free control that validates the pairing.
3. **A zero-sum budget makes "does it help" the wrong question.** `fpr_lp`
   competes with RENS/RINS for one envelope, so the finding is not "it is a bad
   heuristic" but "it is worse than what it displaces". That distinction is
   invisible to a measurement that does not disable the competitors, which is
   exactly what C0 does and C1 does not.

## What was fixed while running this

The reader initially inferred "untraced" from the absence of `[Heur]` lines and
reported 11 of 49 C0 runs as unreadable — every one of which carried
`log_dev_level = 3`. At `suite=fpr_lp` the only heuristic that can emit such a
line is the one under measurement, so their absence is *data*: the dive never
dispatched. It put the fire rate at 38/38 instead of 38/48. Tracing is now read
from the run's `.opts`, which is the request rather than a consequence of it,
with the observed signal surviving as a one-directional fallback.

This is #165's own trap 1 in mirror image — that one was an untraced arm read
as a null. Both directions are now pinned by tests.
