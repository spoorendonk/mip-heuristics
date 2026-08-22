#!/usr/bin/env python3
"""Analyze benchmark results: compute metrics, generate tables and plots.

Two distinct notions of "best" live in this module and must not be conflated:

*virtual best* (`resolve_reference`) is about the **reference objective** of an
instance — when some config observed a primal that beats the published `.solu`
value, that observed value replaces it, so a config is never punished for
finding something better than the library knew about.  It is a property of an
instance, it produces a number in objective units, and it says nothing about
which config is good.

*oracle* (`build_oracle_config`) is about **config selection** — the
hypothetical selector that, per instance, runs whichever participating config
scores best on the headline metric.  It is a property of a set of configs, it
produces a synthetic extra row in the tables, and it is the ceiling any real
selection mechanism (e.g. the Thompson-sampling selector) could reach.

They are unrelated.  Do not describe the oracle as a "virtual best solver",
however standard that term is elsewhere in the benchmarking literature — in
this file the phrase is already taken.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_highs_log import SolveResult, parse_log_file


def parse_solu_file(path: str) -> dict[str, tuple[str, float | None]]:
    """Parse a MIPLIB .solu file into {instance: (tag, value)}.

    Tags: "=opt=", "=best=", "=unkn=", "=fea=".  Value is None for =unkn=.
    """
    refs: dict[str, tuple[str, float | None]] = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or not parts[0].startswith("="):
                continue
            tag, name = parts[0], parts[1]
            val: float | None = None
            if len(parts) >= 3:
                try:
                    val = float(parts[2])
                except ValueError:
                    val = None
            refs[name] = (tag, val)
    return refs


def resolve_reference(
    solu_value: float | None,
    observed_primals: list[float],
    sense: str = "min",
) -> float | None:
    """Pick reference objective for primal-gap computation.

    When observed primals beat the published =best=, use the virtual best
    instead so we don't punish configs that found better solutions.
    `sense` is "min" (default, MIPLIB convention) or "max".

    NOTE: `build_best_known` below only calls this with sense='min'.  That
    matches the current MIPLIB benchmark subsets (head-to-head instances
    are all minimization); adding a maximization instance to a benchmark
    requires threading sense detection from the HiGHS log or .mps file.
    The assumption is guarded here so a future caller can opt in.
    """
    finite = [p for p in observed_primals if math.isfinite(p)]
    if solu_value is None:
        return (min(finite) if sense == "min" else max(finite)) if finite else None
    if not finite:
        return solu_value
    return min(solu_value, *finite) if sense == "min" else max(solu_value, *finite)


def build_best_known(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    configs: list[str],
    instances: list[str],
    solu_refs: dict[str, tuple[str, float | None]],
) -> dict[str, float | None]:
    """Build {instance: reference_objective} from .solu + observed primals."""
    refs: dict[str, float | None] = {}
    for inst in instances:
        observed: list[float] = []
        for c in configs:
            for seed_data in results.get(c, {}).values():
                r = seed_data.get(inst)
                if r and r.primal_bound != float("inf"):
                    observed.append(r.primal_bound)
        solu_value = None
        if inst in solu_refs:
            tag, val = solu_refs[inst]
            # =opt= is proven optimal; =best= is best known (may be beatable);
            # =fea= is a feasible objective that is NOT optimal.  We still
            # treat =fea= as a reference because `resolve_reference` takes
            # min(solu_value, min(observed)) — any config that finds better
            # raises the virtual best — so a gap computed against a =fea=
            # reference is a pessimistic lower bound on the true gap.  Skip
            # =unkn= / =inf= / =unbd=.
            if tag in ("=opt=", "=best=", "=fea=") and val is not None:
                solu_value = val
        refs[inst] = resolve_reference(solu_value, observed)
    return refs


# `.solu` tags that positively assert the instance has no finite optimal
# objective.  A primal gap measured against one is not a small error, it is a
# category error: `build_best_known` falls back to the best *observed* primal,
# which makes the gap self-referential — zero for whichever config found it and
# positive for the rest — and that number then enters the headline SGM looking
# exactly like a real one.
#
# This is not hypothetical.  Until 2026-08 the bundled solution file marked
# `supportcase22` `=inf=` while `bench/instances_plato.txt` counted it among the
# 233 feasible PLATO instances; upstream had since found it feasible.  The data
# is fixed (see the note in `bench/instances_plato.txt`), but the class of bug
# must not be able to recur silently, so the contradiction is now detected
# rather than averaged in.
CONTRADICTED_REFERENCE_TAGS: tuple[str, ...] = ("=inf=", "=unbd=")

# Tags carrying a usable objective.  `=fea=` is a feasible-but-not-optimal
# value; `build_best_known` documents why it is still a sound reference.
USABLE_REFERENCE_TAGS: tuple[str, ...] = ("=opt=", "=best=", "=fea=")


def classify_reference_status(
    instances: list[str],
    solu_refs: dict[str, tuple[str, float | None]],
) -> dict[str, str]:
    """Per instance: "published" | "contradicted" | "unpublished".

    "contradicted" means the solution file asserts no finite objective exists
    (`=inf=` / `=unbd=`) — a reference that cannot be used and must not be
    silently folded into an aggregate.

    "unpublished" means the file simply has nothing usable (absent, `=unkn=`,
    or a tag with no value).  That is *not* an error: it is the ordinary case
    the virtual-best fallback in `resolve_reference` exists to cover, and when
    no config found anything either, every config scores gap 1.0 on it, which
    is PLATO's own convention rather than a distortion.
    """
    status: dict[str, str] = {}
    for inst in instances:
        entry = solu_refs.get(inst)
        if entry is None:
            status[inst] = "unpublished"
            continue
        tag, val = entry
        if tag in CONTRADICTED_REFERENCE_TAGS:
            status[inst] = "contradicted"
        elif tag in USABLE_REFERENCE_TAGS and val is not None:
            status[inst] = "published"
        else:
            status[inst] = "unpublished"
    return status


def contradicted_reference_instances(
    instances: list[str],
    solu_refs: dict[str, tuple[str, float | None]],
) -> list[str]:
    """Instances whose solution-file status rules out any valid reference."""
    status = classify_reference_status(instances, solu_refs)
    return [i for i in instances if status[i] == "contradicted"]


def print_reference_guard(dropped: list[str], solu_path: str | None) -> None:
    """Announce instances excluded for having no usable reference objective."""
    if not dropped:
        return
    print("\n## Unusable reference objectives\n")
    print(
        f"{len(dropped)} instance(s) are marked "
        f"{' / '.join(CONTRADICTED_REFERENCE_TAGS)} in "
        f"{solu_path or 'the solution file'}, which asserts that no finite\n"
        "optimal objective exists — so no primal gap or primal integral "
        "against them is meaningful.\nThey are EXCLUDED from every table "
        "below and the counts reflect that."
    )
    print(f"Excluded: {', '.join(dropped)}")
    print(
        "\nIf these instances are in fact feasible, the solution file is "
        "stale: refresh it from\nhttps://miplib.zib.de/download.html rather "
        "than re-including them."
    )


def load_results(
    results_dir: str,
    configs: list[str],
    config_dirs: dict[str, str] | None = None,
) -> dict[str, dict[int, dict[str, SolveResult]]]:
    """Load all parsed results.

    Returns {config: {seed: {instance: SolveResult}}}.
    Layout is `results/{config}/seed{N}/*.log`, which is what
    `bench/run_benchmark.py` writes.

    `config_dirs` optionally maps a config name to an explicit directory,
    overriding the default `results_dir/<config>`.  This lets one report
    pull its anchor configs from a different results tree than the rest.
    """
    config_dirs = config_dirs or {}
    results: dict[str, dict[int, dict[str, SolveResult]]] = {}
    for config in configs:
        config_dir = config_dirs.get(config, os.path.join(results_dir, config))
        if not os.path.isdir(config_dir):
            print(f"Warning: config directory not found: {config_dir}", file=sys.stderr)
            continue
        results[config] = {}

        # Check for seed subdirectories
        seed_dirs = sorted(Path(config_dir).glob("seed*"))
        if not seed_dirs:
            print(
                f"Warning: no seed*/ subdirectory under {config_dir}",
                file=sys.stderr,
            )
        for sd in seed_dirs:
            if not sd.is_dir():
                continue
            seed_num = int(sd.name.removeprefix("seed"))
            results[config][seed_num] = {}
            for log_file in sorted(sd.glob("*.log")):
                name = log_file.stem
                results[config][seed_num][name] = parse_log_file(str(log_file))
    return results


def read_instance_list(path: str) -> list[str]:
    """Read an instance-name list file into a de-duplicated, ordered list.

    Same format as `bench/instances_plato.txt` and `bench/instances_small.txt`:
    one bare instance name per line, `#` comments and blank lines ignored.  A
    name is the log stem, i.e. the `.mps.gz` basename without suffixes.
    """
    names: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or line in seen:
                continue
            seen.add(line)
            names.append(line)
    return names


@dataclass
class InstanceFilter:
    """What an include/exclude restriction did to a loaded results tree.

    `kept` is the surviving instance set.  `unknown_included` /
    `unknown_excluded` are names the list files asked for that the tree never
    contained — reported because a typo in a list file otherwise restricts a
    report to silence, and an empty table looks the same as a clean one.
    """

    kept: list[str]
    dropped: list[str]
    unknown_included: list[str]
    unknown_excluded: list[str]
    include_path: str | None = None
    exclude_path: str | None = None

    @property
    def active(self) -> bool:
        return self.include_path is not None or self.exclude_path is not None


def filter_results(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    include_path: str | None = None,
    exclude_path: str | None = None,
) -> tuple[dict[str, dict[int, dict[str, SolveResult]]], InstanceFilter]:
    """Restrict a loaded results tree to an instance list, and/or remove one.

    Applied to the *raw* tree before aggregation, so every downstream table —
    existing ones included — covers the restricted set and reports its own
    count without needing to know a filter was applied.

    Include runs first, then exclude, which is what makes a held-out
    complement expressible as `--instances plato --exclude-instances tuning`
    without materialising the complement as a third file that can drift out of
    sync with the tuning set it is defined against.
    """
    present: set[str] = set()
    for config_data in results.values():
        for seed_data in config_data.values():
            present.update(seed_data.keys())

    keep = set(present)
    unknown_included: list[str] = []
    unknown_excluded: list[str] = []
    if include is not None:
        wanted = set(include)
        unknown_included = sorted(wanted - present)
        keep &= wanted
    if exclude is not None:
        unwanted = set(exclude)
        unknown_excluded = sorted(unwanted - present)
        keep -= unwanted

    filtered: dict[str, dict[int, dict[str, SolveResult]]] = {}
    for config, config_data in results.items():
        filtered[config] = {
            seed: {i: r for i, r in seed_data.items() if i in keep}
            for seed, seed_data in config_data.items()
        }
    return filtered, InstanceFilter(
        kept=sorted(keep),
        dropped=sorted(present - keep),
        unknown_included=unknown_included,
        unknown_excluded=unknown_excluded,
        include_path=include_path,
        exclude_path=exclude_path,
    )


def print_instance_selection(
    filt: InstanceFilter,
    reference_dropped: list[str] | None = None,
    final_count: int | None = None,
) -> None:
    """State which instance list a report covers, so a restricted run is obvious.

    `reference_dropped` and `final_count` exist so the closing sentence is not
    a lie.  The list filter is only the first of two things that shrink the
    set — the unusable-reference guard runs after it — and a block that
    announced its own retained count would be immediately contradicted by
    tables reporting a smaller one.  The number stated here is the number the
    tables actually cover.
    """
    if not filt.active:
        return
    print("\n## Instance selection\n")
    if filt.include_path:
        print(f"Restricted to: {filt.include_path}")
    if filt.exclude_path:
        print(f"Excluding:     {filt.exclude_path}")
    print(f"{len(filt.kept)} instances retained, {len(filt.dropped)} removed by list.")
    if reference_dropped:
        print(
            f"{len(reference_dropped)} further removed for having no usable "
            "reference objective (see above)."
        )
    covered = final_count if final_count is not None else len(filt.kept)
    print(
        f"Every table below covers {covered} instance(s) common to the "
        "requested configs — its\nstated count is not the full tree's."
    )
    for label, missing, path in (
        ("include", filt.unknown_included, filt.include_path),
        ("exclude", filt.unknown_excluded, filt.exclude_path),
    ):
        if missing:
            print(
                f"WARNING: {len(missing)} name(s) in the {label} list "
                f"({path}) are absent from this results tree: "
                f"{', '.join(missing[:10])}" + (" ..." if len(missing) > 10 else "")
            )


def get_seeds(results: dict[str, dict[int, dict[str, SolveResult]]]) -> list[int]:
    """Get sorted list of all seeds across configs."""
    seeds: set[int] = set()
    for config_data in results.values():
        seeds.update(config_data.keys())
    return sorted(seeds)


def get_common_instances(
    results: dict[str, dict[int, dict[str, SolveResult]]], configs: list[str]
) -> list[str]:
    """Get instances present in all configs and at least one seed."""
    per_config: list[set[str]] = []
    for config in configs:
        if config not in results:
            continue
        inst_set: set[str] = set()
        for seed_data in results[config].values():
            inst_set.update(seed_data.keys())
        per_config.append(inst_set)
    if not per_config:
        return []
    return sorted(set.intersection(*per_config))


def aggregate_results(
    results: dict[str, dict[int, dict[str, SolveResult]]], configs: list[str]
) -> dict[str, dict[str, SolveResult]]:
    """Aggregate across seeds using median for each metric.

    Returns {config: {instance: SolveResult}} with median values.
    For incumbents, uses the seed with the median primal_bound (upper-middle for even N).
    """
    aggregated: dict[str, dict[str, SolveResult]] = {}
    for config in configs:
        if config not in results:
            continue
        aggregated[config] = {}
        instances = get_common_instances(results, [config])
        seeds = sorted(results[config].keys())

        for inst in instances:
            seed_results = [
                results[config][s][inst] for s in seeds if inst in results[config][s]
            ]
            if not seed_results:
                continue
            if len(seed_results) == 1:
                aggregated[config][inst] = seed_results[0]
                continue

            # Pick the median-performing seed based on primal_bound
            # (lower is better for minimization; use the middle one)
            by_obj = sorted(seed_results, key=lambda r: r.primal_bound)
            median_r = by_obj[len(by_obj) // 2]

            aggregated[config][inst] = median_r
    return aggregated


# ── Offline config oracle (issue #104, part 1) ───────────────────────────────
#
# The best-of-N-configs ceiling: per instance, the score a selector would get
# if it always ran whichever participating config turns out to score best.  Any
# real selection mechanism — the Thompson-sampling selector this project
# records as a negative result — is bounded above by it, which is what makes it
# worth reporting: a low ceiling says per-instance selection was never going to
# pay, which is a far stronger statement than "our selector did not help".
#
# Read the module docstring before renaming anything here: "virtual best" means
# something else in this file and the two must stay separable.

ORACLE_DEFAULT_NAME = "oracle"

# An oracle over one config is that config, relabelled.  Presenting a
# byte-identical clone of an existing row as a "ceiling" is worse than
# printing nothing, so it is refused.
ORACLE_MIN_PARTICIPANTS = 2


@dataclass
class OracleReport:
    """What `build_oracle_config` formed, and what it had to leave out."""

    name: str
    participants: list[str]
    missing_participants: list[str]
    seeds: list[int]
    instances: list[str]
    dropped_incomplete: list[str]
    dropped_not_common: list[str]
    row_picks: dict[str, str]
    seed_picks: dict[tuple[int, str], str]
    time_limit: float
    refused: str | None = None

    @property
    def dropped(self) -> list[str]:
        """Every candidate that did not make it into the oracle row."""
        return sorted({*self.dropped_incomplete, *self.dropped_not_common})

    @property
    def pick_counts(self) -> dict[str, int]:
        """How often each participant supplied the oracle's row, per instance."""
        counts = {c: 0 for c in self.participants}
        for chosen in self.row_picks.values():
            counts[chosen] = counts.get(chosen, 0) + 1
        return counts

    @property
    def seed_pick_counts(self) -> dict[str, int]:
        """Per-(instance, seed) selection counts — the finer-grained diagnostic."""
        counts = {c: 0 for c in self.participants}
        for chosen in self.seed_picks.values():
            counts[chosen] = counts.get(chosen, 0) + 1
        return counts

    @property
    def formed(self) -> bool:
        return bool(self.instances) and self.refused is None


def _oracle_score(r: SolveResult | None, time_limit: float, ref: float | None) -> float:
    """The headline metric, lower-is-better, with a missing row worst-possible."""
    if r is None:
        return float("inf")
    return r.primal_integral(time_limit, ref)


def build_oracle_config(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    participants: list[str],
    best_known: dict[str, float | None] | None,
    time_limit: float,
    name: str = ORACLE_DEFAULT_NAME,
    instances: list[str] | None = None,
) -> tuple[dict[int, dict[str, SolveResult]], OracleReport]:
    """Build a synthetic best-of-participants config.

    **The ceiling property is the whole point**, so it is guaranteed by
    construction rather than hoped for: the oracle's row at an instance is the
    row of whichever participant scores best there *as the tables report that
    participant*.  Since every table reads `aggregate_results`' seed-collapsed
    rows, selecting among exactly those rows makes
    `SGM(oracle) <= min_c SGM(c)` hold identically — the shifted geometric
    mean is monotone, and the oracle dominates every participant instance by
    instance.

    Getting this wrong is subtle and was the first version's bug.  Selecting
    per `(instance, seed)` on primal integral and then letting
    `aggregate_results` collapse the picks by *median primal bound* mixes two
    criteria: the per-seed wins are chosen on one metric and then discarded by
    an aggregation reading another, and the resulting row can score **worse**
    than a participant while still being labelled a ceiling.  No collapse of
    per-seed picks can avoid this in general, because a participant's own
    representative seed may be its best one; the only rule that would dominate
    it is "take the best seed", which is precisely the lucky-seed cherry-pick
    the issue rules out.  So the seed collapse is not re-derived here at all —
    it is inherited, identically, from the rows the comparison is against.

    That also settles what the oracle sees across seeds, which is the rule the
    issue asks to be stated: **it never sees an individual seed.**  Each
    participant is represented by its own median-seed row, so the oracle can
    no more pick a lucky seed than a real selector could.  This is strictly
    more conservative than per-seed selection, which is the direction the
    issue's own reasoning points.

    Per-seed selection is still computed, as `seed_picks` — a diagnostic for
    how stable the choice is across seeds, reported alongside but never used
    to build the row.

    `instances` restricts candidates to the set the surrounding tables cover
    (their cross-config common set).  Scoring anything outside it is what made
    the first version mis-select: `best_known` is built over that same common
    set, so an instance outside it has no reference, `primal_integral` falls
    back to the dual bound, and the tie-break hands the instance to whichever
    participant happens to be named first.  Such instances are dropped and
    counted instead — a config the oracle does not even include cannot be
    allowed to silently shrink the table underneath it.

    Ties are broken by `participants` order, so the result is deterministic.
    """
    present = [c for c in participants if c in results]
    missing = [c for c in participants if c not in results]

    def _empty(refused: str | None) -> tuple[dict, OracleReport]:
        return {}, OracleReport(
            name=name,
            participants=present,
            missing_participants=missing,
            seeds=[],
            instances=[],
            dropped_incomplete=[],
            dropped_not_common=[],
            row_picks={},
            seed_picks={},
            time_limit=time_limit,
            refused=refused,
        )

    if len(present) < ORACLE_MIN_PARTICIPANTS:
        return _empty(
            f"an oracle needs at least {ORACLE_MIN_PARTICIPANTS} participating "
            f"configs; {len(present)} of the requested "
            f"{len(participants)} were loaded"
        )

    seed_sets = [set(results[c].keys()) for c in present]
    seeds = sorted(set.intersection(*seed_sets))
    if not seeds:
        return _empty("the participating configs share no common seed")

    # Candidates are the UNION over participants, not the intersection: an
    # instance one participant never ran is exactly the incompleteness the
    # report has to surface, and intersecting first would delete the evidence
    # before it could be counted.
    candidates: set[str] = set()
    for c in present:
        for seed_data in results[c].values():
            candidates.update(seed_data.keys())
    ordered = sorted(candidates)

    agg = aggregate_results(results, present)
    in_scope = set(instances) if instances is not None else set(ordered)

    complete: list[str] = []
    dropped_incomplete: list[str] = []
    dropped_not_common: list[str] = []
    for inst in ordered:
        # Incomplete means either "a participant never ran it" or "a
        # participant ran it at only some of the shared seeds".  The second is
        # the same defect as the first wearing a different hat: it would put a
        # row aggregated over two seeds next to one aggregated over one, under
        # a single heading.
        everywhere = all(inst in agg.get(c, {}) for c in present) and all(
            inst in results[c][s] for c in present for s in seeds
        )
        if not everywhere:
            dropped_incomplete.append(inst)
        elif inst not in in_scope:
            dropped_not_common.append(inst)
        else:
            complete.append(inst)

    row_picks: dict[str, str] = {}
    chosen: dict[str, SolveResult] = {}
    for inst in complete:
        ref = best_known.get(inst) if best_known else None
        best_cfg = min(
            present, key=lambda c: _oracle_score(agg[c].get(inst), time_limit, ref)
        )
        row_picks[inst] = best_cfg
        chosen[inst] = agg[best_cfg][inst]

    # Diagnostic only — never used to build the row.  See the docstring.
    seed_picks: dict[tuple[int, str], str] = {}
    for s in seeds:
        for inst in complete:
            if not all(inst in results[c][s] for c in present):
                continue
            ref = best_known.get(inst) if best_known else None
            seed_picks[(s, inst)] = min(
                present,
                key=lambda c: _oracle_score(results[c][s].get(inst), time_limit, ref),
            )

    # The chosen row is replicated across the shared seeds so that per-seed
    # readers (`count_feasible`'s per-seed column) see the oracle at every seed
    # it is defined for, and so `aggregate_results` collapsing identical rows
    # returns that same row unchanged.
    tree: dict[int, dict[str, SolveResult]] = {s: dict(chosen) for s in seeds}

    return tree, OracleReport(
        name=name,
        participants=present,
        missing_participants=missing,
        seeds=seeds,
        instances=complete,
        dropped_incomplete=dropped_incomplete,
        dropped_not_common=dropped_not_common,
        row_picks=row_picks,
        seed_picks=seed_picks,
        time_limit=time_limit,
    )


def print_oracle_report(report: OracleReport) -> None:
    """State how the oracle row was formed, before any table uses it."""
    print("\n## Config oracle\n")
    for miss in report.missing_participants:
        print(
            f"WARNING: oracle participant '{miss}' was not loaded — it must "
            "also be listed in --configs\n         (that is what reads a "
            "config off disk); it is skipped."
        )
    if report.refused is not None:
        print(f"(no oracle row: {report.refused})")
        return
    if not report.participants:
        print("(no participating configs found — no oracle row)")
        return
    print(f"Row '{report.name}' = best of: {', '.join(report.participants)}")
    print(
        "Selection: per instance, the participant with the lowest primal "
        f"integral at {report.time_limit:.0f}s\n"
        "           (the headline metric); ties go to the first named.\n"
        "Seeds:     the oracle never sees an individual seed. Each "
        "participant is represented\n"
        "           by the same seed-collapsed row the tables show for it, so "
        "the oracle can no\n"
        "           more pick a lucky seed than a real selector could, and "
        "the row is a true\n"
        "           ceiling: its headline SGM is <= every participant's, "
        "instance by instance.\n"
        "Other columns report whatever the selected config scored on them; "
        "only the\n"
        "primal integral is being optimised."
    )
    if not report.formed:
        print("(no instance is common to every participant — no oracle row)")
        return
    seeds = ", ".join(str(s) for s in report.seeds)
    print(
        f"\nCoverage: {len(report.instances)} instances "
        f"[seeds {seeds}]; {len(report.dropped)} dropped."
    )
    for label, names in (
        ("absent from at least one participant", report.dropped_incomplete),
        ("outside the common set the tables cover", report.dropped_not_common),
    ):
        if names:
            shown = ", ".join(names[:10])
            print(
                f"  {len(names)} dropped, {label}: {shown}"
                + (" ..." if len(names) > 10 else "")
            )
    counts = report.pick_counts
    print("\nSupplied the oracle row (per instance):")
    for c in report.participants:
        print(f"  {c:<22} {counts.get(c, 0):>6}")
    if len(report.seeds) > 1 and report.seed_picks:
        seed_counts = report.seed_pick_counts
        print(
            "\nPer-(instance, seed) winner, for reference only — this does "
            "NOT build the row:"
        )
        for c in report.participants:
            print(f"  {c:<22} {seed_counts.get(c, 0):>6}")


def shifted_geomean(values: list[float], shift: float = 1.0) -> float:
    """Shifted geometric mean: exp(mean(log(x + shift))) - shift."""
    if not values:
        return float("nan")
    log_sum = sum(math.log(max(v + shift, 1e-12)) for v in values)
    return math.exp(log_sum / len(values)) - shift


def format_float(v: float | None, width: int = 10, prec: int = 4) -> str:
    if v is None:
        return "-".rjust(width)
    if v == float("inf"):
        return "inf".rjust(width)
    if abs(v) < 1e-8:
        return "0".rjust(width)
    return f"{v:.{prec}f}".rjust(width)


def format_int(v: int | None, width: int = 8, signed: bool = False) -> str:
    """Right-align an integer, rendering a missing value as '-'."""
    if v is None:
        return "-".rjust(width)
    return (f"{v:+d}" if signed else str(v)).rjust(width)


def count_feasible(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    config: str,
    instances: list[str],
) -> dict[str, int]:
    """Count #Feas: instances with at least one feasible solution across all seeds.

    Returns {"per_seed": {seed: count}, "any": count_any_seed}.
    """
    seeds = sorted(results.get(config, {}).keys())
    per_seed = {}
    any_seed_count = 0

    for inst in instances:
        found_any = False
        for s in seeds:
            r = results.get(config, {}).get(s, {}).get(inst)
            if r and r.incumbents:
                per_seed[s] = per_seed.get(s, 0) + 1
                found_any = True
        if found_any:
            any_seed_count += 1

    return {"per_seed": per_seed, "any": any_seed_count}


def count_first(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
    tie_tol: float = 0.1,
    synthetic: set[str] | None = None,
) -> dict[str, float]:
    """Count #First: instances where config finds first feasible strictly
    earliest (all others later by > tie_tol seconds). Ties within tie_tol
    split the credit evenly across the tied configs. Returns fractional
    counts per config.

    `synthetic` names rows that are not competitors — the config oracle is a
    relabelled copy of whichever participant it selected, so leaving it in
    would have it tie with that participant on every instance and halve a real
    config's credit.  A reporting row must not move the numbers it sits next
    to; synthetic rows are scored 0.0 and excluded from the field.
    """
    synthetic = synthetic or set()
    contenders = [c for c in configs if c not in synthetic]
    firsts: dict[str, float] = {c: 0.0 for c in configs}
    for inst in instances:
        times: dict[str, float] = {}
        for c in contenders:
            r = agg_results.get(c, {}).get(inst)
            if r and r.time_to_first_feasible is not None:
                times[c] = r.time_to_first_feasible
        if not times:
            continue
        best = min(times.values())
        leaders = [c for c, t in times.items() if t <= best + tie_tol]
        share = 1.0 / len(leaders)
        for c in leaders:
            firsts[c] += share
    return firsts


def count_wins(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
    synthetic: set[str] | None = None,
) -> dict[str, int]:
    """Count #Win: instances where one config finds a strictly better primal bound.

    Ties (all best-bound holders within 1e-6 of each other) are not credited
    to any config. Returns {config: win_count}.

    `synthetic` excludes non-competitor rows; see `count_first` for why a
    reporting row left in the field silently rewrites real configs' counts.
    """
    synthetic = synthetic or set()
    contenders = [c for c in configs if c not in synthetic]
    wins = {c: 0 for c in configs}
    for inst in instances:
        bounds = {}
        for c in contenders:
            r = agg_results.get(c, {}).get(inst)
            if r and r.primal_bound != float("inf"):
                bounds[c] = r.primal_bound
        if not bounds:
            continue
        best = min(bounds.values())
        winners = [c for c, b in bounds.items() if abs(b - best) < 1e-6]
        if len(winners) == 1:
            wins[winners[0]] += 1
    return wins


# Incumbent source-char → heuristic label.  In the PATCHED build the custom
# heuristics emit these display chars (see the "Src:" legend printed in every
# patched log): A=FPR, D=FPR_LP, M=LocalMIP, G=Scylla, J=FJ.  HiGHS's built-in
# Feasibility-Jump dispatch is patched off, so a 'J' in a patched log is OUR FJ.
# Every other char is a HiGHS built-in source (branching, rounding, sub-MIP, the
# trivial rounders, the presolve 'P' seed, …) and is bucketed as "HiGHS/other".
CUSTOM_SOURCE_LABELS: dict[str, str] = {
    "A": "FPR",
    "D": "FPR_LP",
    "M": "LocalMIP",
    "G": "Scylla",
    "J": "FJ",
}

# Order used when printing attribution rows (custom heuristics first, then the
# built-in bucket).
ATTRIBUTION_ORDER: list[str] = [
    "FPR",
    "FPR_LP",
    "LocalMIP",
    "Scylla",
    "FJ",
    "HiGHS/other",
]


def source_label(src: str) -> str:
    """Map an incumbent source char to a heuristic label (see CUSTOM_SOURCE_LABELS)."""
    return CUSTOM_SOURCE_LABELS.get(src, "HiGHS/other")


def heuristic_attribution(
    agg_results: dict[str, dict[str, SolveResult]],
    config: str,
    instances: list[str],
) -> dict[str, object]:
    """Attribute, for one config, which heuristic found the first feasible
    incumbent and which produced the best (final) incumbent, per instance.

    Returns {"first": {label: count}, "best": {label: count}, "n_feasible": int},
    where labels come from `source_label`.  Only instances with at least one
    recorded incumbent are counted, so the per-dict totals equal n_feasible.
    """
    first: dict[str, int] = {}
    best: dict[str, int] = {}
    n_feasible = 0
    for inst in instances:
        r = agg_results.get(config, {}).get(inst)
        if not r or not r.incumbents:
            continue
        n_feasible += 1
        f = source_label(r.incumbents[0].source)
        b = source_label(r.incumbents[-1].source)
        first[f] = first.get(f, 0) + 1
        best[b] = best.get(b, 0) + 1
    return {"first": first, "best": best, "n_feasible": n_feasible}


def print_attribution(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
) -> None:
    """Print, per config, the first-feasible and best-held heuristic breakdown.

    Reads only the already-parsed incumbent sources — no solves.  Most
    meaningful for the multi-heuristic configs (e.g. all_opp); single-heuristic
    or vanilla configs simply attribute everything to one source.
    """
    instances = get_common_instances(results, configs)
    print(f"\n## Heuristic attribution ({len(instances)} instances)\n")
    print(
        "Columns: #First = found the first feasible incumbent; "
        "#Best = held the best incumbent at termination.\n"
    )

    for c in configs:
        attr = heuristic_attribution(agg_results, c, instances)
        first = attr["first"]  # type: ignore[assignment]
        best = attr["best"]  # type: ignore[assignment]
        n_feas = attr["n_feasible"]
        print(f"### {c}  ({n_feas} feasible)")
        print(f"{'Heuristic':<14} {'#First':>8} {'#Best':>8}")
        print("-" * 32)
        labels = [lbl for lbl in ATTRIBUTION_ORDER if first.get(lbl) or best.get(lbl)]  # type: ignore[union-attr]
        for lbl in labels:
            print(f"{lbl:<14} {first.get(lbl, 0):>8} {best.get(lbl, 0):>8}")  # type: ignore[union-attr]
        print()


def print_comparison_table(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_cutoffs: list[float] | None = None,
    best_known: dict[str, float | None] | None = None,
    time_limit: float = 600.0,
) -> None:
    """Print per-instance comparison table using seed-aggregated values."""
    if time_cutoffs is None:
        time_cutoffs = [10.0, 60.0, 600.0]

    if len(configs) < 2:
        print("Need at least 2 configs for comparison")
        return

    c1, c2 = configs[0], configs[1]
    instances = sorted(
        set(agg_results.get(c1, {}).keys()) & set(agg_results.get(c2, {}).keys())
    )

    if not instances:
        print("No common instances found between configs")
        return

    # Determine which time cutoffs are relevant
    max_solve_time = max(
        max((r.solve_time for r in agg_results[c1].values()), default=0),
        max((r.solve_time for r in agg_results[c2].values()), default=0),
    )
    active_cutoffs = [tc for tc in time_cutoffs if tc <= max_solve_time + 1]

    # Header.  The instance count leads the table as well as closing it: a
    # restricted run must not be readable as a full one from the top of the
    # output alone.
    print(f"\n## Per-instance comparison: {c1} vs {c2} ({len(instances)} instances)\n")
    print(f"{'Instance':<25} ", end="")
    print(f"{'T1st(' + c1 + ')':<10} {'T1st(' + c2 + ')':<10} ", end="")
    for tc in active_cutoffs:
        print(
            f"{'Gap@' + str(int(tc)) + '(' + c1[:3] + ')':<12} {'Gap@' + str(int(tc)) + '(' + c2[:3] + ')':<12} ",
            end="",
        )
    print(
        f"{'PD(' + c1[:3] + ')':<12} {'PD(' + c2[:3] + ')':<12} {'Status(' + c1[:3] + ')':<15} {'Status(' + c2[:3] + ')':<15}"
    )
    print("-" * 180)

    # Per-instance rows
    wins = {"t1st": 0, "gap": {tc: 0 for tc in time_cutoffs}, "pd": 0}
    losses = {"t1st": 0, "gap": {tc: 0 for tc in time_cutoffs}, "pd": 0}
    ties = {"t1st": 0, "gap": {tc: 0 for tc in time_cutoffs}, "pd": 0}

    t1st_vals = {c1: [], c2: []}
    gap_vals = {tc: {c1: [], c2: []} for tc in time_cutoffs}
    pd_vals = {c1: [], c2: []}
    n_both = n_c1_only = n_c2_only = n_neither = 0

    for inst in instances:
        r1, r2 = agg_results[c1][inst], agg_results[c2][inst]

        print(f"{inst:<25} ", end="")

        # Time to first feasible
        t1 = r1.time_to_first_feasible
        t2 = r2.time_to_first_feasible
        print(f"{format_float(t1, 10, 2)} {format_float(t2, 10, 2)} ", end="")
        if t1 is not None or t2 is not None:
            tl = time_limit
            if t1 is not None:
                t1st_vals[c1].append(min(t1, tl))
            if t2 is not None:
                t1st_vals[c2].append(min(t2, tl))
            if t1 is not None and t2 is not None:
                if t1 < t2 - 0.01:
                    wins["t1st"] += 1
                elif t2 < t1 - 0.01:
                    losses["t1st"] += 1
                else:
                    ties["t1st"] += 1
            elif t1 is not None:
                wins["t1st"] += 1
            else:
                losses["t1st"] += 1

        # Track coverage
        if t1 is not None and t2 is not None:
            n_both += 1
        elif t1 is not None:
            n_c1_only += 1
        elif t2 is not None:
            n_c2_only += 1
        else:
            n_neither += 1

        # Gap at cutoffs — treat no solution as gap=1.0; count when at least
        # one config found a solution, skip only when neither did.
        ref = best_known.get(inst) if best_known else None
        for tc in active_cutoffs:
            g1 = r1.primal_gap_at(tc, ref)
            g2 = r2.primal_gap_at(tc, ref)
            print(f"{format_float(g1, 12, 6)} {format_float(g2, 12, 6)} ", end="")
            if g1 is not None or g2 is not None:
                g1c = g1 if g1 is not None else 1.0
                g2c = g2 if g2 is not None else 1.0
                gap_vals[tc][c1].append(g1c)
                gap_vals[tc][c2].append(g2c)
                if g1c < g2c - 1e-6:
                    wins["gap"][tc] += 1
                elif g2c < g1c - 1e-6:
                    losses["gap"][tc] += 1
                else:
                    ties["gap"][tc] += 1

        # P-D integral (from HiGHS)
        pd1 = r1.pd_integral if r1.pd_integral != float("inf") else None
        pd2 = r2.pd_integral if r2.pd_integral != float("inf") else None
        print(f"{format_float(pd1, 12, 4)} {format_float(pd2, 12, 4)} ", end="")
        if pd1 is not None and pd2 is not None:
            pd_vals[c1].append(pd1)
            pd_vals[c2].append(pd2)
            if pd1 < pd2 - 1e-6:
                wins["pd"] += 1
            elif pd2 < pd1 - 1e-6:
                losses["pd"] += 1
            else:
                ties["pd"] += 1

        print(f"{r1.status:<15} {r2.status:<15}")

    # Summary
    print("-" * 180)
    print(f"\n## Summary: {c1} vs {c2} ({len(instances)} instances)\n")
    print(
        f"Coverage: {n_both} both solved  |  "
        f"{n_c1_only} {c1}-only  |  {n_c2_only} {c2}-only  |  {n_neither} neither\n"
    )

    print(
        f"{'Metric':<25} {c1 + ' wins':<12} {c2 + ' wins':<12} {'Ties':<8} "
        f"{'SGM(' + c1[:3] + ')':<12} {'SGM(' + c2[:3] + ')':<12}"
    )
    print("-" * 80)

    print(
        f"{'Time to 1st feasible':<25} {wins['t1st']:<12} {losses['t1st']:<12} {ties['t1st']:<8} "
        f"{format_float(shifted_geomean(t1st_vals[c1], 1.0), 12, 4)} "
        f"{format_float(shifted_geomean(t1st_vals[c2], 1.0), 12, 4)}"
    )

    for tc in time_cutoffs:
        if gap_vals[tc][c1]:
            label = f"Gap @ {int(tc)}s"
            print(
                f"{label:<25} {wins['gap'][tc]:<12} {losses['gap'][tc]:<12} {ties['gap'][tc]:<8} "
                f"{format_float(shifted_geomean(gap_vals[tc][c1], 0.001), 12, 6)} "
                f"{format_float(shifted_geomean(gap_vals[tc][c2], 0.001), 12, 6)}"
            )

    print(
        f"{'P-D integral':<25} {wins['pd']:<12} {losses['pd']:<12} {ties['pd']:<8} "
        f"{format_float(shifted_geomean(pd_vals[c1], 1.0), 12, 4)} "
        f"{format_float(shifted_geomean(pd_vals[c2], 1.0), 12, 4)}"
    )


def _categorize_instances(
    agg_results: dict[str, dict[str, SolveResult]],
    instances: list[str],
) -> dict[str, str]:
    """Return {instance: category} using the first config that parsed model stats."""
    cats: dict[str, str] = {}
    for inst in instances:
        for cfg_results in agg_results.values():
            r = cfg_results.get(inst)
            if r and r.category is not None:
                cats[inst] = r.category
                break
    return cats


def _print_category_breakdown(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    instances: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
    synthetic: set[str] | None = None,
) -> None:
    """Local-MIP Table 1 style: #Feas / #Win per category × per time cutoff."""
    synthetic = synthetic or set()
    cats = _categorize_instances(agg_results, instances)
    ordered = ["BP", "IP", "MBP", "MIP"]
    buckets = {c: [i for i in instances if cats.get(i) == c] for c in ordered}
    uncls = [i for i in instances if i not in cats]
    if uncls:
        buckets["?"] = uncls

    print(f"\n### Category breakdown (#Feas / #Win) ({len(instances)} instances)\n")
    header = f"{'Category':<8} {'#Inst':>6}"
    for c in configs:
        header += f"  {'Feas(' + c[:4] + ')':>11} {'Win(' + c[:4] + ')':>11}"
    print(header)
    print("-" * len(header))

    for cat, insts_in_cat in buckets.items():
        if not insts_in_cat:
            continue
        feas = {c: count_feasible(results, c, insts_in_cat)["any"] for c in configs}
        wins = count_wins(agg_results, configs, insts_in_cat, synthetic=synthetic)
        row = f"{cat:<8} {len(insts_in_cat):>6}"
        for c in configs:
            win_cell = "-" if c in synthetic else str(wins[c])
            row += f"  {feas[c]:>11} {win_cell:>11}"
        print(row)


def print_paper_metrics(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None = None,
    synthetic: set[str] | None = None,
) -> None:
    """Print paper-standard metrics: #Feas, #Win, SGM of T1st, SGM of gap@cutoff.

    `synthetic` names reporting-only rows (the config oracle).  They keep
    their metric columns, which are meaningful, but are held out of the
    head-to-head counters, which are not: an oracle row is a copy of the
    participant it selected and would tie with it on every instance.
    """
    synthetic = synthetic or set()
    instances = get_common_instances(results, configs)
    seeds = get_seeds(results)

    print(f"\n## Paper Metrics ({len(instances)} instances, {len(seeds)} seed(s))\n")

    # --- #Feas per seed and aggregated ---
    print(f"{'#Feas':<25}", end="")
    for c in configs:
        print(f" {c:<12}", end="")
    print()
    print("-" * (25 + 13 * len(configs)))

    feas_data = {c: count_feasible(results, c, instances) for c in configs}
    for s in seeds:
        print(f"  seed {s:<19}", end="")
        for c in configs:
            count = feas_data[c]["per_seed"].get(s, 0)
            print(f" {count:<12}", end="")
        print()
    if len(seeds) > 1:
        print(f"  {'any seed':<21}", end="")
        for c in configs:
            print(f" {feas_data[c]['any']:<12}", end="")
        print()

    # --- #Win (on aggregated) ---
    win_counts = count_wins(agg_results, configs, instances, synthetic=synthetic)
    print(f"\n{'#Win (best obj)':<25}", end="")
    for c in configs:
        cell = "-" if c in synthetic else str(win_counts[c])
        print(f" {cell:<12}", end="")
    print()

    # --- #First (fastest to feasible, tie-split within 0.1s) ---
    first_counts = count_first(agg_results, configs, instances, synthetic=synthetic)
    print(f"{'#First (fastest T1st)':<25}", end="")
    for c in configs:
        cell = "-" if c in synthetic else f"{first_counts[c]:.1f}"
        print(f" {cell:<12}", end="")
    print()

    # --- SGM of time-to-first-feasible (shift=1s, matching FJ/FPR) ---
    # Clamp at time_limit: HiGHS occasionally reports an incumbent found
    # fractionally after the wall-clock limit (node completing mid-timeout).
    print(f"\n{'SGM T1st (s=1)':<25}", end="")
    for c in configs:
        t1st = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r and r.time_to_first_feasible is not None:
                t1st.append(min(r.time_to_first_feasible, time_limit))
        print(f" {format_float(shifted_geomean(t1st, 1.0), 12, 4)}", end="")
    print()

    # --- SGM of time-to-best incumbent (shift=1s) ---
    print(f"{'SGM Tbest (s=1)':<25}", end="")
    for c in configs:
        tbest = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r and r.time_to_best is not None:
                tbest.append(min(r.time_to_best, time_limit))
        print(f" {format_float(shifted_geomean(tbest, 1.0), 12, 4)}", end="")
    print()

    # --- SGM of primal gap at cutoff (shift=0.001, matching PLATO) ---
    # Infeasible instances contribute gap=1.0 so all instances are counted
    # (matching Mittelmann's PLATO methodology).
    print(f"{'SGM Gap@' + str(int(time_limit)) + 's (s=0.001)':<25}", end="")
    for c in configs:
        gaps = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r:
                ref = best_known.get(inst) if best_known else None
                g = r.primal_gap_at(time_limit, ref)
                gaps.append(g if g is not None else 1.0)
        print(f" {format_float(shifted_geomean(gaps, 0.001), 12, 6)}", end="")
    print()

    # --- SGM of primal integral (shift=1.0) ---
    print(f"{'SGM Primal Integral':<25}", end="")
    for c in configs:
        pis = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r:
                ref = best_known.get(inst) if best_known else None
                pi = r.primal_integral(time_limit, ref)
                pis.append(pi)
        print(f" {format_float(shifted_geomean(pis, 1.0), 12, 4)}", end="")
    print()

    # --- SGM of HiGHS-reported primal-dual integral (shift=1.0) ---
    print(f"{'SGM P-D Integral':<25}", end="")
    for c in configs:
        pdis = []
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r and math.isfinite(r.pd_integral):
                pdis.append(r.pd_integral)
        print(f" {format_float(shifted_geomean(pdis, 1.0), 12, 4)}", end="")
    print()

    # --- Category breakdown (Local-MIP Table 1 style) ---
    if len(configs) >= 1:
        _print_category_breakdown(
            results,
            agg_results,
            configs,
            instances,
            time_limit,
            best_known,
            synthetic=synthetic,
        )

    # --- Reference coverage ---
    if best_known is not None:
        covered = sum(1 for inst in instances if best_known.get(inst) is not None)
        print(
            f"\n(reference objective available for {covered}/{len(instances)} instances)"
        )


def generate_survival_plot(
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    output_path: str,
    gap_threshold: float = 0.01,
) -> None:
    """Generate survival plot: fraction of instances solved to gap% over time."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping survival plot", file=sys.stderr)
        return

    instances = sorted(set.intersection(*(set(agg_results[c].keys()) for c in configs)))
    if not instances:
        return

    _fig, ax = plt.subplots(figsize=(10, 6))

    for config in configs:
        # For each instance, find time when gap <= threshold
        solve_times = []
        for inst in instances:
            r = agg_results[config][inst]
            # Find first incumbent where gap <= threshold
            found = False
            for inc in r.incumbents:
                denom = max(abs(inc.dual_bound), 1.0)
                gap = abs(inc.objective - inc.dual_bound) / denom
                if gap <= gap_threshold:
                    solve_times.append(inc.time)
                    found = True
                    break
            if not found:
                # Check final status — may have reached gap after last logged incumbent
                if r.gap <= gap_threshold:
                    solve_times.append(r.solve_time)
                else:
                    solve_times.append(float("inf"))

        # Sort and create survival curve
        solve_times.sort()
        n = len(solve_times)
        times = [0.0]
        fractions = [0.0]
        for solved, t in enumerate(solve_times, start=1):
            if t == float("inf"):
                break
            times.append(t)
            fractions.append(solved / n)

        # Extend to max time
        max_time = max(r.solve_time for r in agg_results[config].values())
        times.append(max_time)
        fractions.append(fractions[-1])

        ax.step(times, fractions, where="post", label=config, linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Fraction solved to {gap_threshold * 100:.0f}% gap")
    ax.set_title(
        f"Survival Plot (gap threshold = {gap_threshold * 100:.0f}%, "
        f"{len(instances)} instances)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Survival plot saved to {output_path}")


def print_plato_summary(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
) -> None:
    """Print PLATO-style headline metrics: primal-integral SGM ratio and feasibility counts.

    The PLATO benchmark uses primal integral (area under primal-gap curve, 600s
    window, shift=0.001) as the primary metric with shifted geometric mean across
    all 233 instances, and counts the number of instances where a feasible solution
    was found within the time limit.
    """
    instances = get_common_instances(results, configs)
    if not instances:
        return

    # PLATO shift matches Mittelmann's published methodology (sh=0.001)
    plato_shift = 0.001

    print(
        f"\n## PLATO Headline Metrics ({len(instances)} instances, {time_limit:.0f}s, SGM shift={plato_shift})\n"
    )

    pi_per_config: dict[str, list[float]] = {}
    feas_per_config: dict[str, int] = {}
    for c in configs:
        pis = []
        feas_count = 0
        for inst in instances:
            r = agg_results.get(c, {}).get(inst)
            if r:
                ref = best_known.get(inst) if best_known else None
                pi = r.primal_integral(time_limit, ref)
                pis.append(pi)
                if r.incumbents:
                    feas_count += 1
        pi_per_config[c] = pis
        feas_per_config[c] = feas_count

    # Print per-config row
    print(
        f"{'Config':<20} {'SGM(PrimalIntegral)':<22} {'#Feasible':>10} {'#Instances':>12}"
    )
    print("-" * 68)
    for c in configs:
        sgm = shifted_geomean(pi_per_config[c], plato_shift)
        print(f"{c:<20} {sgm:<22.6f} {feas_per_config[c]:>10} {len(instances):>12}")

    # Print pairwise ratios
    if len(configs) >= 2:
        print()
        c1, c2 = configs[0], configs[1]
        sgm1 = shifted_geomean(pi_per_config[c1], plato_shift)
        sgm2 = shifted_geomean(pi_per_config[c2], plato_shift)
        if sgm2 > 0 and math.isfinite(sgm2) and math.isfinite(sgm1):
            ratio = sgm1 / sgm2
            winner = c1 if ratio < 1.0 else c2
            print(
                f"SGM ratio {c1}/{c2}: {ratio:.4f}  (lower is better; winner: {winner})"
            )
        feas1, feas2 = feas_per_config[c1], feas_per_config[c2]
        print(
            f"Feasible delta {c1}-{c2}: {feas1 - feas2:+d}  ({c1}={feas1}, {c2}={feas2})"
        )


def _config_metrics(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    config: str,
    instances: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
) -> dict[str, float]:
    """Compute the one-row-per-config ablation metrics for a single config.

    Returns #Feasible (any seed) plus shifted-geometric-mean of T1st (shift=1),
    primal gap @time_limit (shift=0.001), primal integral (shift=1), and the
    PLATO headline (primal integral, shift=0.001).  All reuse the same helpers
    as the headline tables so an ablation row at a horizon T is directly
    comparable to the anchors analyzed at the same T.
    """
    feas = count_feasible(results, config, instances)["any"]
    t1st: list[float] = []
    gaps: list[float] = []
    pis: list[float] = []
    for inst in instances:
        r = agg_results.get(config, {}).get(inst)
        if not r:
            continue
        if r.time_to_first_feasible is not None:
            t1st.append(min(r.time_to_first_feasible, time_limit))
        ref = best_known.get(inst) if best_known else None
        g = r.primal_gap_at(time_limit, ref)
        gaps.append(g if g is not None else 1.0)
        pis.append(r.primal_integral(time_limit, ref))
    return {
        "feasible": float(feas),
        "sgm_t1st": shifted_geomean(t1st, 1.0),
        "sgm_gap": shifted_geomean(gaps, 0.001),
        "sgm_pi": shifted_geomean(pis, 1.0),
        "plato_sgm": shifted_geomean(pis, 0.001),
    }


def print_ablation_summary(
    results: dict[str, dict[int, dict[str, SolveResult]]],
    agg_results: dict[str, dict[str, SolveResult]],
    configs: list[str],
    time_limit: float,
    best_known: dict[str, float | None] | None,
    latex_path: str | None = None,
) -> None:
    """Print a one-row-per-config ablation table (and optionally write LaTeX).

    Unlike `print_comparison_table` (pairwise), this lists every config on its
    own row, the shape needed for the per-component ablation in the paper.
    """
    instances = get_common_instances(results, configs)
    metrics = {
        c: _config_metrics(results, agg_results, c, instances, time_limit, best_known)
        for c in configs
    }

    print(
        f"\n## Ablation summary ({len(instances)} instances, {time_limit:.0f}s horizon)\n"
    )
    header = (
        f"{'Config':<22} {'#Feas':>6} {'SGM T1st':>10} "
        f"{'SGM Gap':>10} {'SGM PI':>10} {'PLATO SGM':>11}"
    )
    print(header)
    print("-" * len(header))
    for c in configs:
        m = metrics[c]
        print(
            f"{c:<22} {int(m['feasible']):>6} {m['sgm_t1st']:>10.4f} "
            f"{m['sgm_gap']:>10.6f} {m['sgm_pi']:>10.4f} {m['plato_sgm']:>11.4f}"
        )

    if latex_path:
        with open(latex_path, "w") as f:
            f.write(latex_ablation_table(configs, metrics, len(instances), time_limit))
        print(f"\nLaTeX ablation table written to {latex_path}")


# Config names that stand for "no custom heuristic", most-preferred first.
# Used to auto-detect the reference arm of a tree. `bench/make_tuning_set.py`
# imports this to find the vanilla arm it stratifies on, and
# `bench/make_archive.py` keeps its own copy for provenance classification.
BASELINE_CONFIG_NAMES: tuple[str, ...] = (
    "vanilla",
    "off",
    "suite_off",
    "baseline",
)


def latex_ablation_table(
    configs: list[str],
    metrics: dict[str, dict[str, float]],
    n_instances: int,
    time_limit: float,
) -> str:
    """Render the ablation metrics as a booktabs LaTeX table (string).

    Config labels are emitted with underscores escaped for LaTeX.  Kept here
    (rather than in the paper repo) so the table regenerates directly from the
    committed logs, matching the cptp-paper convention.
    """

    def esc(s: str) -> str:
        return s.replace("_", r"\_")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            rf"\caption{{Per-component ablation on {n_instances} PLATO instances "
            rf"at a {time_limit:.0f}\,s horizon (single seed). SGM = shifted "
            r"geometric mean; PI = primal integral; PLATO SGM is the headline "
            r"primal-integral SGM (shift $0.001$). Lower is better except \#Feas.}"
        ),
        r"\label{tbl:ablation}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Config & \#Feas & SGM T1st & SGM Gap & SGM PI & PLATO SGM \\",
        r"\midrule",
    ]
    for c in configs:
        m = metrics[c]
        lines.append(
            f"{esc(c)} & {int(m['feasible'])} & {m['sgm_t1st']:.4f} & "
            f"{m['sgm_gap']:.6f} & {m['sgm_pi']:.4f} & {m['plato_sgm']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HiGHS benchmark results")
    parser.add_argument(
        "results_dir", help="Directory with config subdirectories of log files"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all", "vanilla"],
        help="Configs to compare (default: all vanilla). An entry of the "
        "form NAME=DIR loads that config from an explicit directory instead "
        "of results_dir/NAME (used to pull ablation anchors from "
        "bench/results/plato).",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Path to save survival plot (e.g., bench/survival.png)",
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.01,
        help="Gap threshold for survival plot (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=600.0,
        help="Time limit used in the benchmark (for gap@cutoff metric)",
    )
    parser.add_argument(
        "--solu",
        default=os.path.join(os.path.dirname(__file__), "miplib2017-v36.solu"),
        help="MIPLIB .solu file with reference objectives",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Print PLATO headline metrics: primal-integral SGM (shift=0.001, "
            "matching Mittelmann's published methodology) and feasibility counts. "
            "Appended after the standard paper metrics table."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print only the SGM/paper-metrics summary and PLATO headline; "
            "skip the per-instance comparison table. Implies --baseline."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Print a one-row-per-config ablation table (every config on its own "
            "row) instead of the pairwise comparison/paper-metrics tables. Use "
            "for the per-component ablation across many configs."
        ),
    )
    parser.add_argument(
        "--attribution",
        action="store_true",
        help=(
            "Print the per-config heuristic attribution: which heuristic found "
            "the first feasible incumbent and which held the best at termination "
            "(from the recorded incumbent source chars; no solves)."
        ),
    )
    parser.add_argument(
        "--latex",
        default=None,
        metavar="PATH",
        help="With --ablation, also write the ablation table as LaTeX to PATH.",
    )
    parser.add_argument(
        "--instances",
        default=None,
        metavar="FILE",
        help=(
            "Restrict every report to the instance names listed in FILE (one "
            "per line, '#' comments ignored — the format of "
            "bench/instances_plato.txt). Applied to the loaded tree before "
            "aggregation, so every table covers and reports the restricted "
            "count."
        ),
    )
    parser.add_argument(
        "--exclude-instances",
        default=None,
        metavar="FILE",
        help=(
            "Remove the instance names listed in FILE from every report. "
            "Applied after --instances, so the held-out complement of a "
            "tuning set is '--instances instances_plato.txt "
            "--exclude-instances <tuning>' with no third file to drift."
        ),
    )
    parser.add_argument(
        "--oracle",
        nargs="+",
        default=None,
        metavar="CONFIG",
        help=(
            "Add a best-of-these-configs oracle row: per instance, the "
            "participant with the lowest primal integral at --time-limit, "
            "selected among exactly the seed-collapsed rows the tables show, "
            "which makes the row a true ceiling (its headline SGM is <= every "
            "participant's). The oracle never sees an individual seed, so it "
            "cannot pick a lucky one. Reported alongside the individual "
            "configs under the same headline metric, so the gap between "
            "best-single, combined and ceiling is readable in one table. It "
            "is additive: it does not move any existing row's numbers. "
            "Instances absent from any participant at any shared seed, or "
            "outside the common set the tables cover, are dropped with the "
            "count reported. Needs at least 2 participants, which must also "
            "appear in --configs since that is what gets loaded. NOTE: "
            "unrelated to the 'virtual best' reference-objective handling — "
            "see the module docstring."
        ),
    )
    parser.add_argument(
        "--oracle-name",
        default=ORACLE_DEFAULT_NAME,
        metavar="NAME",
        help=(
            f"Row label for the --oracle row (default: {ORACLE_DEFAULT_NAME}). "
            "Change it only to avoid colliding with a real config name."
        ),
    )
    args = parser.parse_args()

    # Parse NAME=DIR config overrides so ablation anchors can be loaded from a
    # different results directory than the positional results_dir.
    config_dirs: dict[str, str] = {}
    config_names: list[str] = []
    for entry in args.configs:
        if "=" in entry:
            name, path = entry.split("=", 1)
            name, path = name.strip(), path.strip()
            config_dirs[name] = path
            config_names.append(name)
        else:
            config_names.append(entry)

    results = load_results(args.results_dir, config_names, config_dirs)
    if not results:
        print("No results found", file=sys.stderr)
        sys.exit(1)

    active_configs = [c for c in config_names if c in results]

    solu_refs: dict[str, tuple[str, float | None]] = {}
    if args.solu and os.path.exists(args.solu):
        solu_refs = parse_solu_file(args.solu)

    # Instance selection, then the reference guard, then the oracle — all on
    # the raw tree, before aggregation, so every table downstream reports the
    # set it actually covers without having to know any of this happened.
    try:
        include = read_instance_list(args.instances) if args.instances else None
        exclude = (
            read_instance_list(args.exclude_instances)
            if args.exclude_instances
            else None
        )
    except OSError as exc:
        print(f"Error: cannot read instance list: {exc}", file=sys.stderr)
        sys.exit(1)

    filt: InstanceFilter | None = None
    if include is not None or exclude is not None:
        results, filt = filter_results(
            results,
            include,
            exclude,
            include_path=args.instances,
            exclude_path=args.exclude_instances,
        )

    # An instance the solution file says has no finite objective cannot carry a
    # primal gap; folding one into a 233-instance SGM distorts the headline
    # silently.  Drop it loudly instead.  This runs before the selection block
    # is printed so that block can state the count that actually survives —
    # otherwise it announces a retained count the tables below contradict.
    unusable = contradicted_reference_instances(
        get_common_instances(results, active_configs), solu_refs
    )
    if unusable:
        results, _ = filter_results(results, exclude=unusable)

    common = get_common_instances(results, active_configs)
    print_reference_guard(unusable, args.solu)
    if filt is not None:
        print_instance_selection(
            filt, reference_dropped=unusable, final_count=len(common)
        )

    best_known = build_best_known(results, active_configs, common, solu_refs)

    # The oracle is additive reporting: it gets its own row and must not move
    # any existing one.  `synthetic` carries that through to the head-to-head
    # counters, which would otherwise treat a
    # relabelled copy of a real run as a competitor to it.
    synthetic: set[str] = set()
    if args.oracle:
        if args.oracle_name in active_configs:
            print(
                f"Error: --oracle-name '{args.oracle_name}' collides with a "
                "real config in this tree; pass a different --oracle-name.",
                file=sys.stderr,
            )
            sys.exit(1)
        oracle_tree, oracle_report = build_oracle_config(
            results,
            args.oracle,
            best_known,
            args.time_limit,
            name=args.oracle_name,
            instances=common,
        )
        print_oracle_report(oracle_report)
        if oracle_report.formed:
            results[oracle_report.name] = oracle_tree
            active_configs = [*active_configs, oracle_report.name]
            synthetic.add(oracle_report.name)

    agg_results = aggregate_results(results, active_configs)

    if args.ablation:
        # Per-component ablation: one row per config, plus optional attribution.
        print_ablation_summary(
            results,
            agg_results,
            active_configs,
            args.time_limit,
            best_known,
            latex_path=args.latex,
        )
        if args.attribution:
            print_attribution(results, agg_results, active_configs)
        if args.plot:
            generate_survival_plot(
                agg_results, active_configs, args.plot, args.gap_threshold
            )
        return

    if not args.summary:
        print_comparison_table(
            agg_results,
            active_configs,
            best_known=best_known,
            time_limit=args.time_limit,
        )
    print_paper_metrics(
        results,
        agg_results,
        active_configs,
        args.time_limit,
        best_known=best_known,
        synthetic=synthetic,
    )

    if args.attribution:
        print_attribution(results, agg_results, active_configs)

    if args.baseline or args.summary:
        print_plato_summary(
            results, agg_results, active_configs, args.time_limit, best_known
        )

    if args.plot:
        generate_survival_plot(
            agg_results, active_configs, args.plot, args.gap_threshold
        )


if __name__ == "__main__":
    main()
