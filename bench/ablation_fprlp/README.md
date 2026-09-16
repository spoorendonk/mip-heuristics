# Ablation C — what `fpr_lp` contributes, and why it ships off

Companion to `bench/ablation_effort/` (Ablation A, the per-heuristic effort and
patience measurement) and `bench/ablation_search/` (Ablation B, the joint
search over the presolve mix).

**Headline: `fpr_lp` ships at `mip_heuristic_fpr_lp_effort = 0` — disabled — on
evidence rather than on caution.** Given the whole LP-iteration envelope to
itself it *can* produce accepted incumbents. In the shipped chain it produces
**none**: zero accepted incumbents across 49 paired runs, with nothing
separating in either direction on the campaign metric.

The verdict rests on that zero, not on the metric. At n=49 the observed 4.5%
decrease would need n=403 to resolve — this n resolves a 12.4% decrease at 80%
power — so the campaign metric settles nothing here on its own. (Both figures
are read on the same side of the ratio; see `bench/headline/README.md` for why
that matters.)

## Reproducing

| step | command | artifacts |
|---|---|---|
| C0, capability | `bench/run_ablation_c.sh capability 3` | `bench/results/ablation_c/capability/` |
| read it | `bench/analyze_ablation_c.py capability <tree>/fpr_lp/seed0` | `capability.txt` here |
| C1, contribution | `bench/run_ablation_c.sh contribution 6` | `bench/results/finalists/confirm/F-selected-plus-fprlp/` |
| read it | `bench/analyze_ablation_c.py contribution <control>/<config> <arm>/<config> --labels <C0>/fpr_lp/seed0` | `contribution.txt` here |

Results trees are gitignored (`bench/results*`); every derived artifact needed
to read the result is in this directory.

Both stages ran on the 16-core / 32-thread benchmark machine at HiGHS's default
worker count — every log reads `Thread count 16 (of 32 threads). Using 1 max
workers. Parallel search off` — on `bench/results/irace/bin/highs`, the same
binary Ablation B's arms used. One seed.

## The question, and why it needs asking

`fpr_lp` is the LP-dependent fix-and-propagate heuristic, run during the B&B
dive rather than in presolve. It is the one heuristic of the five Ablation A
cannot measure: that probe is presolve-only, and `fpr_lp` does not exist there.

It is not a free rider either. `fpr_lp` charges its work back to
`heuristic_lp_iterations` / `total_lp_iterations`, the counters
`moreHeuristicsAllowed()` reads to decide whether RENS and RINS run — so its
budget is **zero-sum against upstream's own two dive heuristics**. Every LP
iteration it takes is one they do not get. Shipping it unmeasured is therefore
not neutral in either direction.

## Design: a capability gate before a contrast

A fire-rate measurement at shipped defaults is the wrong first experiment,
because the shipped chain squeezes `fpr_lp` from three sides — RENS/RINS drain
the envelope first, `headroom_iters <= 0`, and `max_effort < nnz << 8` — so a
null there is answerable with "you never gave it a chance" and cannot end the
exercise.

**C0 inverts it.** `fpr_lp` runs alone (the four presolve efforts at `0`), owns
the whole envelope (`mip_heuristic_run_rens=false`,
`mip_heuristic_run_rins=false`, `mip_heuristic_effort=1.0`), and gets a
per-call budget that cannot bind (`mip_heuristic_fpr_lp_effort=100`). Its
output is a **count** — dispatches, and accepted incumbents — so it needs no
power calculation and a zero is decisive at any n.

That ordering answers Ablation B's methodological finding directly: a
mean-difference experiment sized to what is affordable produces a null that
cannot be read as an answer. A count sidesteps that entirely.

**`threads=1` is deliberately not set.** Pinning it would avoid
`parallelLockActive()` skipping the heuristic, but the lock is held only under
multi-worker B&B, which is not HiGHS's default, and every log in both stages
confirms it. Pinning would measure a regime nothing ships in.

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

Reading C0 has one trap worth knowing: with only `fpr_lp` enabled, the absence
of `[Heur]` lines on a traced run is *data* — the dive never dispatched — not a
sign that tracing was off. `analyze_ablation_c.py` therefore reads tracing from
the run's `.opts`, which is the request rather than a consequence of it, with
the observed signal surviving as a one-directional fallback. Both directions of
that confusion are pinned by tests.

## Stage C1 — contribution at the campaign metric

The shipped configuration with and without `fpr_lp`, paired on identical
instances at the 600 s campaign limit — `B'-mix-cheapest` against the same
vector plus `mip_heuristic_fpr_lp_effort=1.0`, with RENS and RINS live. The
arms differ in exactly one option, so configuration cancels along with instance
difficulty, which is why n=49 resolves 14% here against the 20.6% of the
arm-vs-arm comparisons in `bench/ablation_search/`.

Full output in `contribution.txt`.

| subgroup | n | ratio | 95% CI | t |
|---|---|---|---|---|
| overall | 49 | **0.955** | [0.87, 1.05] | −0.99 |
| C0 yielded | 10 | 1.060 | [0.90, 1.25] | +0.70 |
| C0 fired, never yielded | 28 | 0.905 | [0.78, 1.05] | −1.33 |
| C0 never fired (**null control**) | 10 | **0.996** | **[0.99, 1.00]** | −1.73 |

**Zero accepted incumbents across all 49 runs.** Nothing separates, in either
direction. The null control is tight to within 0.4%, which is what says these
are real nulls rather than a measurement too noisy to see anything.

Subgroups are labelled by C0's independent run, so the partition is fixed by
something other than the metric being compared — but the labels come from C0's
configuration rather than from these runs, so they are a proxy for "an instance
where `fpr_lp` engages", and the split is post-hoc. Read it as a mechanism, not
as a second headline.

**The comparison must be made against the configuration that ships**, and that
is a measurement rather than a matter of taste. The presolve background moves
the dive: a chain spending 3259 ms of median presolve time against the shipped
chain's 438 ms starts the dive later, from a different incumbent, with the
envelope in a different state, and the cost half of the verdict changes with
it. `fpr_lp` draws only from upstream's dive-time envelope, so the mechanism
argument predicts the presolve chain is irrelevant — and it is not. A sound
mechanism argument is not a measurement.

## Verdict

**`fpr_lp` is dominated by RENS/RINS inside the shared envelope.** It produces
accepted incumbents when nothing competes for the LP iterations (C0: 60, on 10
of 49 instances) and **none** when they do — zero across 49 paired runs at the
campaign limit, five times C0's wall clock.

What carries the verdict is that zero, not the campaign metric: nothing
separates in either direction (0.955, CI [0.87, 1.05]), and resolving the
observed 4.5% would need n=403 against a 49-instance set.

**Shipped setting: `mip_heuristic_fpr_lp_effort = 0`** — measured, and off
because it does nothing. A heuristic that produces no accepted solution in 49
paired runs at the campaign limit, and moves the metric by 4.5% ± noise, has
not earned a share of upstream's RENS/RINS envelope.

**No share sweep, and this is a decision rather than a deferral.** The lever
changes how much of the envelope `fpr_lp` takes from RENS and RINS, and it
produces nothing to show for what it already takes — so a larger share buys
more of nothing at their expense, and a smaller one converges on the shipped
`0`. One dimension also gives a search nothing to navigate, at >= 504
experiments (~85 h at 600 s).

**The lever the data points at is a gate, not a budget** — do not dispatch the
dive where it will not yield. That is a code change, out of scope for a
measurement, and recorded here as the finding rather than as an action.

## For a write-up

Three things here are worth stating in a paper, and two of them are about
method rather than about `fpr_lp`.

1. **A capability gate is cheaper than a contrast and can end the question.**
   C0 costs 1.3 h and would have closed the whole campaign had it come back
   zero. It needs no power calculation because its output is a count.
2. **The instances a heuristic cannot reach are a free control.** The ten
   where the dive never fires come back at 0.996, CI [0.99, 1.00], and that is
   what licenses reading the other subgroups at all — without it, "nothing
   separates" is indistinguishable from "the measurement is too noisy to see
   anything". `analyze_ablation_c.py` refuses to print the subgroups if that
   control ever separates.
3. **A zero-sum budget makes "does it help" the wrong question.** `fpr_lp`
   competes with RENS/RINS for one envelope, so the finding is not "it is a bad
   heuristic" but "it is worse than what it displaces". That distinction is
   invisible to a measurement that does not disable the competitors, which is
   exactly what C0 does and C1 does not.
