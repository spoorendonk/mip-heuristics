# `fpr_lp` — what it contributes, and why it ships off (#165)

Ablation C. Companion to `bench/ablation_effort/` (Ablation A, which produced
the four presolve effort and patience defaults) and `bench/ablation_search/`
(Ablation B, the joint search over the presolve mix).

**Headline: `fpr_lp` keeps its shipped default of `0` — disabled — on evidence
rather than on caution.** It is capable of producing accepted incumbents when
it owns the LP-iteration envelope, and produces **none** when RENS and RINS
compete for it: zero accepted incumbents across ~100 paired runs on two
different presolve backgrounds.

**Stage C1 was measured twice, and the second reading is the one that counts.**
The first was taken against the four-heuristic vector that shipped at the time,
where `fpr_lp` cost a separated 27% on the instances where it engaged without
producing anything. #107 then retired that vector, and against the
three-heuristic configuration that actually ships the cost is gone — 0.955
overall, CI [0.87, 1.05]. So the harm was an interaction with a heavier
presolve chain, not a property of `fpr_lp`. The conclusion is unchanged and its
reasoning is weaker: not "it costs" but **"it does nothing, in either
direction"**.

## Reproducing

| step | command | artifacts |
|---|---|---|
| C0, capability | `bench/run_ablation_c.sh capability 3` | `bench/results/ablation_c/capability/` |
| read it | `bench/analyze_ablation_c.py capability <tree>/fpr_lp/seed0` | `capability.txt` here |
| C1, contribution | `bench/run_ablation_c.sh contribution 6` | `bench/results/finalists/confirm/F-selected-plus-fprlp/` |
| read it | `bench/analyze_ablation_c.py contribution <control>/<suite> <arm>/<suite> --labels <C0>/fpr_lp/seed0` | `contribution.txt` here |
| the superseded reading | `ABLATION_C_ARM=E-shipped-plus-fprlp bench/run_ablation_c.sh contribution 6` | `.../E-shipped-plus-fprlp/` (already complete) |

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

## Stage C1 — contribution at the campaign metric, measured twice

The shipped configuration with and without `fpr_lp`, paired on identical
instances at the 600 s campaign limit. Both arms differ in exactly one option,
so configuration cancels along with instance difficulty — which is why n=49
resolves 14% here against the 20.6% of #107's arm-vs-arm comparisons.

Full output of the current reading in `contribution.txt`.

### The reading that counts: against `B'-mix-cheapest`, the shipped vector

| subgroup | n | ratio | 95% CI | t |
|---|---|---|---|---|
| overall | 49 | **0.955** | [0.87, 1.05] | −0.99 |
| C0 yielded | 10 | 1.060 | [0.90, 1.25] | +0.70 |
| C0 fired, never yielded | 28 | 0.905 | [0.78, 1.05] | −1.33 |
| C0 never fired (**null control**) | 10 | **0.996** | **[0.99, 1.00]** | −1.73 |

**Zero accepted incumbents across all 49 runs.** Nothing separates, in either
direction. The null control is tight to within 0.4%, which is what says these
are real nulls rather than a measurement too noisy to see anything.

Resolving the observed 4.5% would need **n=403** against a 49-instance set, so
this is unresolvable rather than merely unresolved — the same shape as #107's
finding and #108's held-out interval.

### The superseded reading: against the previous four-heuristic vector

Kept because the contrast between the two is the informative part, and because
the arm (`E-shipped-plus-fprlp`, 49 runs) is on disk.

| subgroup | n | ratio | 95% CI | t |
|---|---|---|---|---|
| overall | 49 | 1.137 | [0.99, 1.31] | +1.75 |
| C0 yielded | 10 | 0.952 | [0.58, 1.55] | −0.20 |
| **C0 fired, never yielded** | **28** | **1.270** | **[1.07, 1.51]** | **+2.66** |
| C0 never fired (null control) | 10 | 1.007 | [0.99, 1.02] | +1.02 |

Also zero accepted incumbents, across its own 49 runs.

### What moved, and why it was worth re-running

The 27% cost was **separated** on the old background and is **gone** on the
new one. `fpr_lp` draws from upstream's RENS/RINS envelope at dive time and
none of that involves the presolve chain, so the prior was that the verdict
would carry over unchanged. It did not, in the one respect that mattered.

The mechanism is the one #107 measured: the previous chain spent **3259 ms**
of median presolve wall clock against the shipped one's **438 ms**. On that
heavier background the dive starts later, from a different incumbent, with the
envelope in a different state — and adding a heuristic that produces nothing
was measurably harmful. On the cheaper chain it is merely inert.

**This is the case against closing a measurement on a mechanism argument.**
The argument was sound and the prediction was wrong, and 3.2 h of machine time
is what separated the two.

### A pilot, recorded rather than discarded

The first reading's arm began as 20 runs taken before this ablation had a
design (`dfa90d2` enabled it, `abe1043` disabled it again). At n=20 it showed
1.179, CI [1.055, 1.317] — separated. At n=49 the same arm gives 1.137, CI
[0.99, 1.31] — not separated, with the paired sd more than doubling. The 20
were the quiet instances. Reported separately at the time, and the reason to
keep doing so.

## Verdict

**`fpr_lp` is dominated by RENS/RINS inside the shared envelope.** It produces
accepted incumbents when nothing competes for the LP iterations (C0: 60, on 10
of 49 instances) and none when they do (the pilot: 0 across 20 instances at 600
s, five times the wall clock) — while costing 27% wherever it engages.

**Shipped setting: `mip_heuristic_fpr_lp_effort = 0`.** The value does not
move; its justification does, from "unmeasured, so off" to **"measured, and off
because it does nothing"** — a weaker case than the first reading suggested,
and the honest one. A heuristic that produces no accepted solution in ~100
paired runs across two backgrounds, and moves the campaign metric by 4.5% ±
noise, has not earned a share of upstream's RENS/RINS envelope.

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
