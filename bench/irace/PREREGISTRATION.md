# #107 joint search — pre-registration

**Status: SIGNED OFF 2026-09-06. In effect for every run recorded against it.**

#107 requires the objective to be fixed *before* the first run, because the
tables report several quantities and choosing among them afterwards is
post-hoc selection. This file is that commitment. It is tracked, so the
version that was in effect when a search ran is recoverable from git.

Sign-off: **Simon Spoorendonk, 2026-09-06** — commit at sign-off: `01780d0`

Reviewed and accepted as written, before the first experiment ran.

---

## 1. What is being searched

Eight parameters: four presolve effort budgets and four patience thresholds,
for `fj`, `fpr`, `local_mip`, `scylla`. Inclusion is **not** a ninth
dimension — effort 0 means the heuristic does not run, so the fifteen
non-empty subsets plus `off` are exactly the zero-patterns of the four
efforts. `bench/irace/parameters.txt` is the authoritative domain; two
discrete switches per heuristic (`<h>`, `<h>_gated`) exist only so a
log-sampled real can reach the exact values 0 (no run) and 0 (no gate).

**`fpr_lp` is not searched and is off.** It ships disabled
(`mip_heuristic_fpr_lp_effort = 0`, 15871b6) because it has never been
calibrated and its budget is zero-sum against RENS/RINS. The search, the
shipped binary and #108's headline arm therefore agree; measuring whether it
earns its share is #165.

### Domains, and where they come from

From the #113 calibration probe, re-run 2026-09-05/06, shipped in b212155,
derived in `bench/ablation_effort/README.md`.

| heuristic | knee (p50) | censored hi | effort domain | patience domain |
|---|---|---|---|---|
| fj | 0.5665 | 6.6125 | `(0.08, 9.92)` | `(0.00221, 2.48)` |
| fpr | 12.2559 | 93.7124 | `(1.53, 141.0)` | `(0.0479, 35.3)` |
| local_mip | 13.9607 | 50.8226 | `(1.75, 112.0)` | `(0.0545, 28.0)` |
| scylla | 3.0680 | 17.8144 | `(0.384, 26.7)` | `(0.012, 6.68)` |

Effort: `[knee/8, max(8*knee, 1.5*censored_hi)]`, log-sampled, with FJ's
bottom raised to the documented `0.08` granularity floor. Patience:
`[knee/256, effort_hi/4]`, the top being the clamp
(`patience_threshold` caps at a quarter of that heuristic's own budget).

## 2. The primary objective — pre-registered

Per `(configuration, instance, seed)`, `bench/run_target.py` prints

```
cost = gap + lambda * tau
```

* `gap` — primal gap of the **presolve-exit** incumbent against the reference
  objective, capped at 1, resolved through
  `analyze_results.resolve_reference`. A run that found nothing scores
  `--no-solution-penalty` (default `2.0`), which exceeds the cap by
  construction so that finding nothing is never the cheapest outcome.
  Instances tagged `=inf=` / `=unbd=` are excluded, never scored.
* `tau` — the **heuristics' own** `[Heur] wall_ms`, in seconds, summed over
  the presolve chain. Not total presolve time: HiGHS's own presolve dominates
  on large models and is not ours to spend.
* `lambda` — the cost weight, default `1/600`.

irace minimises this. Aggregation across instances is irace's own, over the
paired instance sequence; the reported headline is the shifted geometric mean
of `gap` over the tuning set at the selected configuration.

**Rationale for lambda.** Primal integral is `∫ gap(t) dt`; spending `tau`
extra seconds shifts the trajectory right by `tau`, costing about
`tau * g(0)`, so `lambda ≈ g(0)/T` per second — at the campaign's 600 s limit
roughly 0.17% of the integral per extra presolve second. Measured presolve
times are ~2 s median, so quality dominates and cost is a small correction.

**lambda sweep, also pre-registered:** the search runs at
`lambda ∈ {1/1200, 1/600, 1/300}` — the derived value and one octave either
side — and the *family* of resulting configurations is reported, not just the
middle one. If the selection is stable across the three, say so; if it is not,
that instability is the result.

## 3. Secondary quantities — reported, never selected on

Reported for the selected configuration and its runner-up, and explicitly
**not** eligible to change the choice: time to first accepted solution; count
of instances with any solution; the per-heuristic productive/stale effort
split; and the hard-tier verdict (did any configuration crack an instance no
configuration cracked in #113).

## 4. Selection rule — pre-registered

1. irace's own statistical elimination decides survival; the scenario's
   `firstTest` and `elitist` settings are in `scenario.txt` and are part of
   this registration.
2. The output is the **surviving set**, not an argmax. If several
   configurations are statistically indistinguishable, that is stated and the
   **simpler** one is chosen — fewer heuristics enabled, then lower total
   effort. Fewer heuristics is less to defend and a tie is a legitimate
   result.
3. Finalists are validated on **held-out instances**: the PLATO list minus the
   tuning set, via `--instances plato --exclude-instances tuning`.
4. 3–5 finalists — the scalarised winner, one clearly cheaper, one clearly
   more generous — are confirmed by **full timed solves** at the campaign's
   600 s limit on the tuning set. Ranking stability is recorded either way,
   including if the ranking flips.

### Amendment, 2026-09-07: the confirmation runs on a stratified subset

**Signed off before any confirmation run.** Commit at amendment: `16ddd1b`.

The confirmation as written costs **~9.7 h per finalist** over the 90-instance
tuning set — measured from the #105 vanilla tree, where the median solve hits
the 600 s cap and the mean is 394 s — so four finalists is ~39 h. It runs
instead on a **stratified subset of 20 instances**, `bench/instances_confirm.txt`,
drawn by `bench/make_tuning_set.py` from the same vanilla tree with `--seed 0`
and therefore reproducible from tree plus seed. ~8.8 h.

Three things make this sound, and one makes it honest.

* **The design is paired.** Every finalist runs the *same* instances, so
  instance difficulty — the dominant variance component in cross-instance MIP
  benchmarking, spanning orders of magnitude — cancels in the per-instance
  difference. A subset costs precision on the margin between arms, not
  correctness of the comparison. The subset is drawn **once** and shared: this
  registration already forbids resampling per configuration, which would break
  pairing and inject instance-selection variance into every estimate.
* **The stage's job is coarse.** It asks whether the presolve-exit ranking
  survives a 600 s primal integral. A ranking flip is a large effect; a margin
  too small for 20 paired observations to see is not the thing this stage
  exists to catch. **#108 is the headline** — full 233 instances, three seeds —
  and no part of the shipped claim rests on this subset's precision.
* **Reduced power degrades into an outcome already specified.** Rule 2 above
  says that configurations which cannot be separated are reported as such and
  the simpler one is chosen. Less power means resolving to that clause more
  often, which is a pre-registered result rather than an ambiguity.
* **What is given up, stated rather than discovered:** the power to resolve
  close rankings. With four arms there are six pairwise comparisons, so at
  n = 20 a *clear* flip is believable and a *narrow* ordering is not, and the
  report must not present one as the other.

Stratified rather than randomly sampled, deliberately: the effect under test is
plausibly difficulty-dependent — a heavy presolve chain should cost most on
instances B&B would have cracked quickly — so a simple random draw could
over-weight the 50 of 89 instances that hit the cap and hide exactly that.
The draw preserves the tuning set's own strata, with `--min-per-stratum`
reserving seats for the small ones, which mildly over-weights the hard end.

**Sequential by design.** n = 20 is a starter. If the top arms land inside
noise the subset is extended rather than the verdict forced; the harness
resumes per `(config, instance, seed)`, so nothing already spent is repeated.
Any extension is recorded here with its size and the reason.

## 5. Known limitations — stated now, not discovered later

* **Only the root dispatch is screened.** The presolve chain is re-entered for
  sub-MIPs during a full solve; a presolve-only run sees the first dispatch.
* **18% of the PLATO set is invisible** to a presolve screen. #113's
  informative set is 191 of 233; the other 42 are the separately scored hard
  tier.
* **Selection optimism remains.** The winner of a search over `n`
  configurations is biased upward by roughly `sigma * sqrt(2 ln n)` — ~3.5
  sigma at 500 candidates. The 90-instance tuning set and the held-out
  validation are the mitigations; #108's held-out complement quantifies what
  is left.
* **Valid at the deployment worker count.** The four budgets are not
  worker-count invariant: FJ's is per worker and the other three per
  dispatch, so changing the count reallocates budget between heuristics
  rather than rescaling it. The search must run at the count #108 will use.
* **The patience domain is over-represented at its loose end.** For a
  configuration whose effort is below the top of its domain, every patience
  above `effort/4` clamps to `effort/4` and maps onto one real configuration
  (the loosest gate). That costs sampling resolution, not correctness.
* **The presolve four and `fpr_lp` are each calibrated with the other
  absent.** They do not compete for the same budget — different envelopes —
  but they do compete for wall clock and can find the same solutions. No
  stage claims a joint optimum over all five; see #165.

## 6. What would invalidate a run against this file

Any of: a change to `run_target.py`'s cost function; a change to the domains
in `parameters.txt`; a change to the shipped defaults the domains are centred
on; a different worker count from #108's; or a different tuning set. Each
means re-registering, not reinterpreting.
