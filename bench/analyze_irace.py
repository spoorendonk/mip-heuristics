#!/usr/bin/env python3
"""Turn irace's output into #107's selected configuration.

The search produces one `irace.Rdata` per pre-registered lambda.  This module
is the analysis that reads them and applies the **selection rule fixed in
`bench/irace/PREREGISTRATION.md`** — it implements that rule and decides
nothing of its own.  It is tracked tooling with tests, not a scratch script,
because #107 requires the selection to be re-derivable from the run log rather
than re-run.

Why a separate module rather than a few lines of R
--------------------------------------------------
Three things the pre-registration asks for are not what `getFinalElites`
returns:

* **A surviving set, not an argmax.**  irace's elites are already a set, but
  the rule says that when several are statistically indistinguishable the
  *simpler* one is chosen — fewer heuristics enabled, then lower total effort.
  That tie-break is a decision procedure and belongs somewhere testable.
* **A family across lambda.**  The deliverable is whether the selection is
  stable across `{1/1200, 1/600, 1/300}`, not the middle one alone.  Instability
  is a result, so it has to be reported rather than resolved.
* **The eight numbers.**  irace reports its own 16-parameter encoding,
  including the `<h>` and `<h>_gated` sampling switches.  A configuration is
  eight numbers; `collapse` performs the same reduction `run_target.py` does,
  so what is reported is what would be shipped.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HEURISTICS = ("fj", "fpr", "local_mip", "scylla")


class NotFinished(RuntimeError):
    """The search has not completed an iteration, so it has no elites yet."""


# The pre-registered lambda sweep, in the order the launcher runs them.
LAMBDA_TAGS = ("1_1200", "1_600", "1_300")


@dataclass(frozen=True)
class Configuration:
    """One configuration as the eight numbers that define it."""

    efforts: dict[str, float]
    patiences: dict[str, float]
    irace_id: int | None = None

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(h for h in HEURISTICS if self.efforts.get(h, 0.0) > 0.0)

    @property
    def suite(self) -> str:
        return ",".join(self.enabled) if self.enabled else "off"

    @property
    def total_effort(self) -> float:
        return sum(self.efforts.get(h, 0.0) for h in HEURISTICS)

    def simplicity_key(self) -> tuple[int, float]:
        """The pre-registered tie-break: fewer heuristics, then less effort.

        "Fewer heuristics is less to defend and a tie is a legitimate result."
        Sorting ascending on this key puts the simplest first.
        """
        return (len(self.enabled), self.total_effort)

    def as_dict(self) -> dict:
        return {
            "irace_id": self.irace_id,
            "suite": self.suite,
            "efforts": {h: self.efforts.get(h, 0.0) for h in HEURISTICS},
            "patiences": {h: self.patiences.get(h, 0.0) for h in HEURISTICS},
        }


def collapse(row: dict) -> Configuration:
    """irace's 16-parameter encoding down to the eight numbers.

    `<h>` selects effort 0 ("does not run") and `<h>_gated` selects patience 0
    ("no staleness gate at all"); both exist only because a log-sampled real
    cannot land on 0.  irace also writes `NA` for a conditional parameter whose
    condition did not hold, which means the same thing.  This is the same
    reduction `run_target.parameters_from_args` performs, and it has to stay
    the same or the reported configuration is not the one that ran.
    """

    def number(value) -> float:
        if value is None:
            return 0.0
        if isinstance(value, str):
            if value.strip().upper() in {"NA", ""}:
                return 0.0
            return float(value)
        return float(value)

    efforts: dict[str, float] = {}
    patiences: dict[str, float] = {}
    for h in HEURISTICS:
        on = number(row.get(h))
        efforts[h] = 0.0 if on == 0 else number(row.get(f"{h}_effort"))
        gated = number(row.get(f"{h}_gated"))
        patiences[h] = 0.0 if gated == 0 else number(row.get(f"{h}_patience"))
    ident = row.get("irace_id")
    return Configuration(
        efforts=efforts,
        patiences=patiences,
        irace_id=int(ident) if ident not in (None, "", "NA") else None,
    )


_R_READ_ELITES = r"""
suppressMessages(library(irace))
l <- read_logfile("%s")
e <- getFinalElites(l)
ids <- e$.ID.
e <- removeConfigurationsMetaData(e)
e$irace_id <- ids
write.csv(e, row.names = FALSE, na = "NA")
"""


def read_elites(rdata: Path) -> list[dict]:
    """The surviving set irace selected, straight from its own state.

    Read from `irace.Rdata` rather than from a text summary so the report and
    the run log cannot disagree.

    CSV rather than JSON on the R side deliberately: `jsonlite` is not part of
    a base R install and is not needed here, and a bench script that fails on
    a fresh machine for want of an optional package is a bad trade for the
    two lines of parsing it saves.
    """
    out = subprocess.run(
        ["Rscript", "-e", _R_READ_ELITES % rdata],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        # `getFinalElites` indexes the last entry of `allElites`, which is
        # empty until the first iteration finishes, so this is what an
        # in-progress search looks like rather than a broken one.  Worth
        # distinguishing: the state file exists and grows from the first
        # experiment, so "the file is there" is not evidence of a result.
        if "less than one element" in out.stderr:
            raise NotFinished(
                f"{rdata}: no iteration has completed yet, so irace has "
                "selected no elites"
            )
        raise RuntimeError(f"reading {rdata} failed:\n{out.stderr.strip()}")
    reader = csv.DictReader(io.StringIO(out.stdout))
    return [dict(row) for row in reader]


@dataclass
class LambdaResult:
    tag: str
    survivors: list[Configuration]
    selected: Configuration
    tie: bool = False
    notes: list[str] = field(default_factory=list)


def select(survivors: list[Configuration], tag: str) -> LambdaResult:
    """Apply the pre-registered rule to one lambda's surviving set.

    irace's elites are ordered best-first and are, by construction, the
    configurations its statistical test could not separate.  So more than one
    survivor *is* the "statistically indistinguishable" case the rule covers,
    and the tie-break applies rather than taking irace's first.
    """
    if not survivors:
        raise ValueError(f"{tag}: irace returned no elites")
    ordered = sorted(survivors, key=Configuration.simplicity_key)
    chosen = ordered[0]
    tie = len(survivors) > 1
    notes = []
    if tie:
        notes.append(
            f"{len(survivors)} configurations survived; the rule takes the "
            f"simplest ({len(chosen.enabled)} heuristics, total effort "
            f"{chosen.total_effort:.4g}), not irace's first."
        )
    return LambdaResult(
        tag=tag, survivors=survivors, selected=chosen, tie=tie, notes=notes
    )


def stability(results: list[LambdaResult]) -> tuple[bool, str]:
    """Whether the selection is stable across the lambda sweep.

    Pre-registered as a *result*, not a diagnostic: if the three lambdas
    disagree, that instability is what gets reported, not resolved by picking
    the middle one.
    """
    suites = {r.selected.suite for r in results}
    if len(suites) == 1:
        return True, f"stable: every lambda selects the same mix ({suites.pop()})"
    listing = ", ".join(f"{r.tag} -> {r.selected.suite}" for r in results)
    return False, f"UNSTABLE across the cost weight: {listing}"


def report(results: list[LambdaResult]) -> str:
    lines = ["#107 joint search — selection", ""]
    for r in results:
        lines.append(f"lambda {r.tag}: {len(r.survivors)} survivor(s)")
        for note in r.notes:
            lines.append(f"  note: {note}")
        c = r.selected
        lines.append(f"  selected: suite={c.suite}")
        for h in HEURISTICS:
            lines.append(
                f"    {h:<10} effort {c.efforts[h]:>10.4f}   "
                f"patience {c.patiences[h]:>10.4f}"
            )
        lines.append("")
    ok, message = stability(results)
    lines.append(message)
    if not ok:
        lines.append(
            "  The pre-registration treats this as the result: report the "
            "family, do not pick one and call it stable."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_dir",
        type=Path,
        help="bench/results/irace — the directory holding one subdir per lambda",
    )
    parser.add_argument("--tags", nargs="+", default=list(LAMBDA_TAGS))
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args(argv)

    results: list[LambdaResult] = []
    for tag in args.tags:
        rdata = args.results_dir / tag / "irace.Rdata"
        if not rdata.exists():
            print(f"skipping {tag}: no irace.Rdata yet", file=sys.stderr)
            continue
        try:
            survivors = [collapse(row) for row in read_elites(rdata)]
        except NotFinished as exc:
            print(f"skipping {tag}: {exc}", file=sys.stderr)
            continue
        results.append(select(survivors, tag))

    if not results:
        print("no completed lambda found", file=sys.stderr)
        return 2

    text = report(results)
    print(text)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(
                {
                    "lambdas": [
                        {
                            "tag": r.tag,
                            "survivors": [c.as_dict() for c in r.survivors],
                            "selected": r.selected.as_dict(),
                            "tie": r.tie,
                        }
                        for r in results
                    ],
                    "stable": stability(results)[0],
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
