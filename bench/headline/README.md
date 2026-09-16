# The headline: the shipped configuration against vanilla HiGHS (#108)

`mipfeas`, all 233 instances, 600 s, one seed, 16 workers, CPU build.
Companion to `bench/ablation_effort/` (A), `bench/ablation_search/` (B) and
`bench/ablation_fprlp/` (C), which chose the configuration this measures.

**Headline: 16.4% better on the primal-integral SGM over instances never used
for tuning** — ratio 0.836, 95% CI [0.732, 0.956], p = 0.009 — and a first
feasible solution on 3 more instances. **Final solution quality is a wash.**
The contribution is *sooner*, not *better*, which is what the benchmark's
metric measures and what a feasibility heuristic should claim.

## Reproducing

| step | command |
|---|---|
| run | `bench/run_headline.sh until 08:00` (or `next <hours>`), repeat until `status` shows 233/233 |
| the three tables | `bench/run_headline.sh report` -> `tables.txt` |
| paired statistics | the script embedded in the commit that added `paired.txt` |

Results trees are gitignored; everything needed to read the result is here.

## Configuration

The arm is the **shipped binary at default options**. Its `.opts` carries the
seed and nothing else — the `all` config sets no option, and no
`--extra-options` is passed — so this measures what a user gets rather than a
hand-assembled vector. The defaults are `B'-mix-cheapest`
(`bench/ablation_search/`): fj 0.3317, fpr 3.161, local_mip 3.2865,
**scylla 0 (disabled)**, patiences 0 / 0.3372 / 3.1943 / 0, with
`mip_heuristic_fpr_lp_effort = 0` (`bench/ablation_fprlp/`).

`mip_heuristic_effort` is at upstream's own default of 0.05 and is asserted at
configure time, so the B&B dive-heuristic budget — what RENS and RINS draw
from — is bit-identical to vanilla's. The only difference between the arms is
what runs during presolve.

**The baseline is a separately built unpatched binary** of the same tag (#105),
not a setting on the patched one.

## Result

| | patched (B') | vanilla |
|---|---|---|
| SGM primal integral, 233 | **19.18** | 26.57 |
| SGM primal integral, held-out 143 | **22.89** | 27.36 |
| #Feasible | **214** | 211 |

Paired per instance (full output in `paired.txt`):

| set | n | ratio | 95% CI | t | p | better/tied/worse |
|---|---|---|---|---|---|---|
| all 233 | 233 | 0.722 | [0.633, 0.823] | −4.87 | <0.001 | 107/32/94 |
| **held-out 143** | **143** | **0.836** | **[0.732, 0.956]** | **−2.62** | **0.009** | 57/25/61 |
| tuning 90 | 90 | 0.572 | [0.441, 0.741] | −4.23 | <0.001 | 50/7/33 |

**The held-out number is the result.** The tuning set is 43% better and the
held-out 16% — the selection bias this split exists to quantify, and it is
large. Reporting the tuning figure as the headline would overstate the effect
by a factor of nearly three.

## Three qualifications that travel with it

**1. The win is in magnitude, not frequency.** 57 better against 61 worse on
held-out; 107/94 over all 233. A sign test finds nothing (p = 0.71 and 0.36).
The heuristics do not win more often — they win *bigger*: `comp07-2idx` 600 ->
7.8 and `sorrell3` 162 -> 2.3 against `fast0507` 14.4 -> 248. Both measures
belong in any write-up; quoting the SGM alone would misrepresent the shape of
the result.

**2. Final quality is level.** Paired final primal gap at 600 s: better on 45,
tied on 130, worse on 34, and vanilla nominally leads on `#Win`. HiGHS's own
machinery holds **188 of 214** final answers. The claim is that HiGHS reaches a
good solution sooner, not that it reaches a better one.

**3. The effect sits at the edge of what 143 instances can resolve.** At
sd 0.815 the held-out n=143 resolves a **17.5% decrease** at 80% power, and the
observed effect is a **16.4% decrease** — a margin of about one point. Quote
the interval, not the point: significance at this power implies the point
estimate is likely overstated.

*Both percentages are read on the same side of the ratio, and that is not a
pedantic detail.* A minimum detectable log effect `d = sqrt(8 sd^2 / n)` has
two percentage readings — `exp(d) - 1 = 21.3%`, how much bigger vanilla is than
patched, and `1 - exp(-d) = 17.5%`, how much smaller patched is than vanilla.
Quoting the 21.3% beside an observed 16.4% *decrease* compares two different
baselines and makes the margin look like five points rather than one.

## Statistical conventions, stated rather than inherited

Each has a textbook alternative that gives different numbers, so which one
produced a figure is part of the figure.

* **Paired on the log-ratio** of the primal integral, `log((arm + s) / (control
  + s))` with the SGM shift `s = 1e-3`. The integral spans orders of magnitude
  across instances, and pairing cancels instance difficulty — the dominant
  variance component in cross-instance MIP benchmarking.
* **Intervals are normal**, `exp(mean +- 1.96 * se)`, not Student-t.
* **The sign test is the normal approximation**, without continuity
  correction.
* **Minimum detectable effect** is `1 - exp(-sqrt(8 sd^2 / n))` for an
  improvement and `exp(sqrt(8 sd^2 / n)) - 1` for a regression — always read on
  the same side of the ratio as the effect it is compared with.

## Where the gain comes from

Partitioning the 233 instances by which heuristic produced the patched arm's
*first* incumbent:

| first incumbent from | n | ratio vs vanilla | 95% CI | t |
|---|---|---|---|---|
| **FJ** | **112** | **0.563** | **[0.451, 0.702]** | **−5.08** |
| FPR | 25 | 0.732 | [0.409, 1.309] | −1.05 |
| HiGHS/other | 68 | 0.956 | [0.722, 1.267] | −0.31 |
| **no incumbent** | **20** | **1.005** | **[0.995, 1.014]** | +1.00 |
| LocalMIP | 8 | 2.114 | [0.794, 5.630] | +1.50 |

(Computed on the previous vector's 233-run tree, which has the same shape.)

The effect is concentrated where FeasibilityJump gets there first, and the
"no incumbent" row is a clean internal control: where nothing is found, the two
arms are identical to within 1%, so the pairing is tight and the 0.563 is an
effect rather than noise.

**That does not make this a paper about parallelism.** Our FJ differs from
vanilla's in three confounded ways: 16 opportunistic workers against one call,
a per-worker budget that totals ~5x vanilla's single FJ allowance, and two
corrected upstream defects (the negative-coefficient jump value and the sign of
the objective term in the move score). Vanilla runs its own FJ, so the
comparison already includes FJ-vs-FJ; separating the three is not attempted.

## Attribution

| | patched #First | #Best | vanilla #First | #Best |
|---|---|---|---|---|
| FJ | 114 | 20 | 97 | 10 |
| FPR | 16 | 2 | — | — |
| LocalMIP | 5 | 4 | — | — |
| HiGHS/other | 79 | **188** | 114 | **201** |

Ours find the first feasible solution on 135 of 214 instances against vanilla's
97 of 211, and hold the final best on 26 against 10.

## The second arm: does the simplification cost anything?

`all-prev-vector` is the four-heuristic vector Ablation B replaced, measured
over the same 233 against the same baseline — so this is a 233-instance paired
comparison of the two patched configurations, at ~10% resolution against the
20.6% at which Ablation B's n=49 tuning-set comparison returned a null.

```
B' / prev-4-heuristic   n=233  ratio 0.972  CI [0.898, 1.052]  t=-0.71  p=0.48
                               better 133 / tied 33 / worse 67   sign p=8e-7
```

**The two measures disagree, and both are reported.** B′ is not separated on
magnitude, but wins twice as often. The reading: dropping Scylla and cutting
total effort from 29.85 to 6.78 costs nothing and helps slightly and often, by
amounts too small to move an SGM.

## Deviations from the specified design, stated

**One seed, not the three this issue asks for.** Seed 0 was run as a gate and
the campaign stopped there.

Seeds shrink only the within-instance variance component. The between-instance
component does not move, because the instances are the same 143 — and the
baseline is *also* a single seed, so averaging `k` patched seeds removes at
most half the seed noise even as `k` grows. Writing `f` for the share of
variance that is seed noise, and assuming it splits evenly between the two
arms, `Var(k) = sd^2 * [1 - f/2 + f/(2k)]`. At the held-out sd of 0.815 and
n = 143, on the decrease side:

| `f` | k=1 | k=2 | **k=3** | k=∞ |
|---|---|---|---|---|
| 25% | 17.5% | 17.0% | **16.8%** | 16.5% |
| 50% | 17.5% | 16.5% | **16.1%** | 15.4% |
| 100% (not attainable) | 17.5% | 15.4% | **14.6%** | 12.7% |

**Three seeds reach the observed 16.4% once `f` is above roughly 40%** — so
the honest statement is that extra seeds *might* have resolved this, not that
they could not. With one seed `f` is unestimable: there are no replicates, so
the table is bracketing rather than measurement.

What that makes the stopping decision is a judgement rather than an
impossibility argument. It stands on what a tighter interval would have bought:
the held-out effect is already separated at p = 0.009, nothing about what ships
turns on the width, and two further passes (~48 h) would at best have moved a
16.4% point estimate inside a slightly narrower band. Two seeds would have been
the informative spend — that is the smallest `k` from which `f` can be
estimated at all — and it was not made.

**Two binaries.** `all-prev-vector` was produced by the PATCH_VERSION 23
build; `all` by PATCH_VERSION 24. They differ in the default option values,
which is the thing under comparison. The PATCH_VERSION 23 binary was not
retained — its configuration is recoverable from commit `7056a0f`, and its logs
carry the patch marker.

**Both patched arms ran at default options**, so their `.opts` files are
byte-identical and the directory name is the only thing distinguishing them.
That is why the previous arm is `all-prev-vector` rather than `all`.

**Killed runs.** One per arm (`germanrr` patched, `neos-5114902-kasavu`
vanilla), kept as truncated logs. They score correctly: T1st and the primal
integral read incumbent lines, not the final report.

## Comparability

Adopting the `mipfeas` instance list, time limit and metric makes this
*definitionally* the same benchmark. It does **not** make the absolute numbers
comparable with the rankings published on Mittelmann's server at
[plato.asu.edu](https://plato.asu.edu/bench.html), which are measured on
different hardware. The defensible claim is patched versus vanilla on one
machine under the `mipfeas` definition.
