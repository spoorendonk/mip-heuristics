# Release process

How to cut and publish a version of this project. Written so the next
maintainer can do v1.1 without reverse-engineering v1.0.

This document is about *publishing*. It is not about reproducing a run —
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) owns the reproducible recipe, what
is deliberately not reproducible, the PLATO protocol and the `suite=off`
equivalence — nor about the day-to-day gates, which
[`CONTRIBUTING.md`](../CONTRIBUTING.md) owns. Both are referenced below rather
than reproduced. Two things are deliberately restated: the gate commands,
because a release runs them as one clean-tree sequence and a maintainer should
not have to assemble it from two documents, and the clean-rebuild rule, which
is stronger here than in day-to-day work. Everything else is a link.

## What a release consists of

Four things, in this order. The order is not a preference: two of the steps
cannot be done after the one that follows them.

| # | Artifact | Made by | Reversible? |
|---|---|---|---|
| 1 | Archive-service integration enabled for the repository | maintainer, once ever | yes, but only affects *future* releases |
| 2 | Annotated git tag `vX.Y.Z` | maintainer | in principle; not once anything resolves to it |
| 3 | GitHub release from that tag | maintainer | yes |
| 4 | DOI + deposit, minted from the release | archive service, automatically | **no** — a DOI is permanent |

Step 1 must happen before step 3, because the integration only archives
releases created *after* it was switched on. Step 4 has no undo, so nothing
about it should be discovered during the release.

The **artifact archive** — logs, per-run configurations, generated tables and
provenance — is built before step 2 and published alongside steps 3 and 4. It
has its own section below, including the reason it cannot simply be a GitHub
release asset.

## Three version numbers, only one of which is ours

A reader who sees `v1.0.0` and `HiGHS 1.15.1` in the same sentence needs to
know which one moves when.

| Number | Where it lives | What it means |
|---|---|---|
| `vX.Y.Z` | the git tag; `version:` in `CITATION.cff` | this project's own version. Nothing else derives from it. |
| `v1.15.1` | `GIT_TAG` in `cmake/FetchHiGHS.cmake` | the upstream solver the heuristics are compiled into, fetched and patched at configure time. |
| `PATCH_VERSION` | `third_party/highs_patch/apply_patch.cmake` | the revision of *our inserted text*. Stamped into the fetched tree and checked as a sentinel on every configure. |

The relationship a release has to state: **a tag pins a project version against
one upstream solver tag.** A HiGHS bump is release-visible even when no
first-party line changes — upstream renames `advanced` options across minor
versions with no deprecation shim, and every `Highs` instance we build sets
`output_flag=false`, so a rejected `setOptionValue` is silent. `CLAUDE.md`'s
"Bumping the HiGHS tag" note is the procedure; a release that includes a bump
does not go out without it having been followed.

`PATCH_VERSION` is independent of both and moves whenever inserted text
changes. **A change to `apply_patch.cmake` requires a clean rebuild before it
takes effect** — the script decides "already patched?" by searching for text it
previously inserted, so an existing `build/_deps` tree was patched by the *old*
script. The remedy, and why it presents as an unrelated compile error, are in
[`CONTRIBUTING.md` § The clean-rebuild rule](../CONTRIBUTING.md#the-clean-rebuild-rule).
For a release the rule is stronger than in day-to-day work: **release binaries
are built from a clean tree regardless**, because the whole point of the
archive is that its provenance can be trusted, and a half-patched tree produces
a binary that is not the one the tag describes.

## Gates that must be green before tagging

All of them, on a clean rebuild, from the tag's exact tree:

```bash
rm -rf build
python3 -m venv .venv                                   # if not already present
.venv/bin/pip install clang-format==22.1.8 clang-tidy==22.1.8 pytest ruff==0.16.3
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMIP_HEURISTICS_REQUIRE_LINT=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure -j"$(nproc)"
.venv/bin/python -m pytest -q bench
.venv/bin/ruff check bench cmake
```

Three things about that list are easy to get wrong:

- **`ctest` is not just the C++ tests.** `clang_format` and `clang_tidy` are
  registered as ctest tests labelled `lint`, and `bench_python_tests` and
  `docs_parameter_references` are in there too. A release run never uses
  `ctest -LE lint`.
- **The lint gates only exist if `.venv/bin` holds the pinned tools.**
  `cmake/Lint.cmake` searches that exact path; without it the gates are not
  registered at all and `ctest` reports green having linted nothing.
  `-DMIP_HEURISTICS_REQUIRE_LINT=ON` is what turns that silence into a
  configure failure, which is why it is not optional here.
- **`ruff check bench cmake` gates and its rule set is pinned** in
  `pyproject.toml`. The `cmake/` half is not optional — that is where the
  clang-tidy gate's own wrapper lives. Do not widen the select list to make a
  release pass.

Plus the one gate that is not part of `ctest`, because it needs a second
binary:

```bash
python3 bench/check_vanilla_equivalence.py \
    --patched-binary ./build/bin/highs \
    --vanilla-binary /path/to/unpatched/highs
```

`mip_heuristic_suite=off` on the patched binary is the row every benchmark row
is measured against, so a release that has not re-proved that equivalence is
publishing an unverified baseline. Build the unpatched binary from the same
HiGHS tag; see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md#suiteoff-is-vanilla-equivalent--since-august-2026)
for what the check compares and which two residual differences are accepted.

## The artifact archive

`bench/make_archive.py` packages a results tree so that every published table
can be regenerated from the archive alone, and so that a reader can tell which
binary, which baseline, which seeds, which machine and which instrumentation
state produced each row.

```bash
# Build it.  --time-limit is required: it is a HiGHS command-line argument
# rather than an options-file entry, so it is not recoverable from the tree.
bench/make_archive.py build bench/results/plato \
    --output dist/mip-heuristics-v1.0.0-archive \
    --time-limit 600 \
    --machine-note "16-core benchmark host, Xeon ..., 64 GB, otherwise idle" \
    --note "PLATO mipfeas campaign for v1.0.0" \
    --tar

# Prove it regenerates.  Runs every recorded table command against the
# archived logs and diffs the result; also re-checks every sha256.
dist/mip-heuristics-v1.0.0-archive/REGENERATE.sh
```

`verify` establishes that the logs, the analysis code and `MANIFEST.json`
inside one archive agree with each other. It cannot establish that the archive
is the one the release published — the manifest travels with the archive, so a
rebuilt archive verifies clean too. What pins it to the release is the archive
service's own checksum on the deposited tarball, which is why the dataset
record below is not optional.

Write the archive **outside the working tree**, or into the gitignored `dist/`.
`make_archive.py` records whether the checkout was dirty, and an archive
written into an untracked directory inside the repository reports its own
source tree as dirty.

The archive holds:

```
MANIFEST.json    provenance + table index + sha256 of every archived file
PROVENANCE.md    the same, rendered for a human
REGENERATE.sh    wrapper around `make_archive.py verify .`
bench/           analyze_results.py, parse_highs_log.py, the .solu file and
                 make_archive.py — no checkout needed to regenerate
results/         the tree verbatim: <config>/seed<N>/<instance>.log, plus the
                 <instance>.opts each run was given, plus any .log.err
tables/          one file per recorded table command
```

### What the provenance covers, and where each field comes from

| Field | Source |
|---|---|
| Repository commit, `describe`, dirty flag | `git` in the checkout the archive is built from |
| Upstream solver tag | `GIT_TAG` in `cmake/FetchHiGHS.cmake` |
| Patch version | `PATCH_VERSION` in `third_party/highs_patch/apply_patch.cmake` |
| Lint tool pins | the `==` pins in `.github/workflows/ci.yml` |
| Patched vs unpatched binary, **per config** | the `mip-heuristics patch active` marker line in each log |
| HiGHS version and git hash | the banner line in each log |
| Baseline claim | which of the two the baseline config's logs support (see below) |
| Seeds | the `seed<N>/` directory names |
| Instances | the log file names |
| Per-run options | each `<instance>.opts`, archived verbatim; summarised per config |
| `threads` | the options files, if pinned there at all |
| Instrumentation, requested | `log_dev_level` in the options files |
| Instrumentation, observed | `[Heur]` / `[Native]` / `[Root]` / `[Sequential]` tags in the logs |
| Machine | auto-detected on the archive host, plus `--machine-note` |
| Time limit | `--time-limit`, because HiGHS takes it on the command line |

Four of those deserve their reasoning stated, because getting them wrong
produces an archive that looks complete and cannot be interpreted:

- **The binary.** A patched and an unpatched build of the same HiGHS tag print
  *identical* version and githash banners; `highs --version` cannot tell them
  apart. Only the `mip-heuristics patch active` line does, and it is printed by
  a solve, not by `--version`. The tool reads it out of every log, so a config
  whose logs disagree is rejected rather than archived.
- **The baseline.** "Vanilla-equivalent setting on the patched binary" and
  "separately built unpatched binary" are two different claims and only one of
  them rests on the build. The first rests on
  `bench/check_vanilla_equivalence.py`. The manifest names which one this
  archive supports and cites the right evidence for it.
- **Instrumentation.** `--dev-log` costs 97–750x the log volume and 1.1–4.4x
  the wall time, concentrated in the FeasibilityJump phase, so an attribution
  run and a headline-timing run are *different runs* whose timings are not
  comparable. Both the requested and the observed state are recorded, and a
  disagreement — the failure mode where `--extra-options log_dev_level=1`
  cancels the flag — is a warning rather than a silent mis-label.
- **Thread count.** Throughput ratios here do not cancel across worker counts:
  the same binary on the same instances gives `local_mip:scylla = 4.68` at 16
  workers and `2.81` at 6. The harness deliberately does not set `threads`, so
  the effective count is the *run* machine's core count — which is why an unset
  `threads` warns and why `--machine-note` matters when the archive is not
  built on the machine that ran the campaign.

A campaign therefore normally produces **two archives**, not one: the
headline-timing tree and the `--dev-log` attribution tree. Publishing one and
labelling it as both is the mistake the instrumentation fields exist to
prevent.

### Size

A `--dev-log` tree is large — measured at 27 MB for 24 runs (4 bundled
instances × 3 configs × 2 seeds) at a 10 s limit, and a 600 s PLATO campaign is
orders of magnitude past that. Two limits bound where it can go:

- A GitHub release asset must be **under 2 GiB** per file.
- A Zenodo record takes **at most 100 files and 50 GB total**, which is why
  `--tar` exists: upload the single `.tar.gz`, never the unpacked directory.

If a `--dev-log` archive does not fit, archive a *subset* of instances rather
than dropping the provenance — a partial tree with intact provenance is
citable, a complete tree without it is not.

## DOI wiring

### Current state, as verified

**The archive-service integration is not configured for this repository.**
Verified rather than assumed, at the time of writing:

- `gh api repos/spoorendonk/mip-heuristics/hooks` returns `[]`. Zenodo's
  integration works by installing a webhook, and it installs that webhook only
  when the repository is toggled on. No webhook means no integration.
- A Zenodo search for `mip-heuristics` / `spoorendonk` returns no record for
  this repository.
- The repository has no tags and no releases.

So **a release cut today would mint no DOI**, and neither `CITATION.cff` nor
`.zenodo.json` currently carries one. That is not a defect in those files —
they were written for the closeout positioning by the documentation issue and
are correct as far as they go — it is a step nobody has taken yet.

### What the maintainer must do, exactly

This is a browser task with no CLI equivalent; the integration requires an
OAuth grant that `gh` cannot make.

1. Sign in at <https://zenodo.org/> (use "Log in with GitHub" if you want the
   accounts linked in one step).
2. Profile menu → **GitHub**.
3. Press **Sync now** so the repository list is current.
4. Find `spoorendonk/mip-heuristics` and flip its toggle **On**. This is what
   installs the webhook; `gh api .../hooks` will list it afterwards, which is
   how you confirm the step took.
5. **Do this before creating the release.** Releases made before the toggle was
   flipped are not archived, and there is no backfill.

Two consequences that decide the rest of the process:

- **`.zenodo.json` wins outright.** When a repository has one, Zenodo uses it
  and ignores `CITATION.cff` entirely. So `.zenodo.json` is the deposit
  metadata and `CITATION.cff` is what GitHub's own "Cite this repository"
  widget reads. Both need to stay true; only one of them reaches the DOI.
- **A DOI cannot be pre-reserved through the GitHub integration.** The DOI does
  not exist until the release is archived, so the tagged tree cannot contain
  it. Add it in the first commit *after* the release — see the checklist.

### Which DOI to cite

Zenodo mints two: a **version DOI** for that specific release, and a **concept
DOI** that always resolves to the newest version. Put the *concept* DOI in
`CITATION.cff` and the README badge — it stays correct across v1.1 — and cite
the version DOI when you need to pin exactly what was run.

`CITATION.cff` takes it as an `identifiers:` entry:

```yaml
identifiers:
  - type: doi
    value: 10.5281/zenodo.XXXXXXX
    description: Concept DOI — always resolves to the latest release
```

### The archive is not covered by the software DOI

Zenodo's GitHub integration archives **the repository source zipball only**.
Assets attached to a GitHub release are not deposited. The artifact archive is
therefore not in the software record, and attaching it to the GitHub release
does not make it citable.

Deposit it as its own Zenodo record:

1. Upload `mip-heuristics-vX.Y.Z-archive.tar.gz` as a new Zenodo upload, type
   **Dataset**.
2. Title it after the release it belongs to, and paste `PROVENANCE.md` into the
   description — a dataset record whose provenance is only inside the tarball
   is one nobody can evaluate before downloading it.
3. Link the two records: on the dataset, add a related identifier
   `isSupplementTo` → the software concept DOI. Optionally add the reverse
   (`isSupplementedBy` → the dataset DOI) to `.zenodo.json` for the *next*
   release; it cannot be added to this one, because the dataset DOI does not
   exist until after the software record is published.

## Release checklist

Copy this into the release issue and tick it through. Steps 1–7 are the ones
that must not be reordered.

**Preconditions**

- [ ] Every issue in the closeout epic (#88) is merged to `main` — for v1.0 the
      tag is blocked on all of them, not only on the ones the release notes
      name.
- [ ] `git status` is clean and `main` is up to date.
- [ ] `README.md`'s benchmark tables and their provenance caveats describe the
      tree being tagged, not an earlier campaign.
- [ ] `CITATION.cff` `version:` and `date-released:` updated for this release.
- [ ] `.zenodo.json` description still matches the positioning (it is the
      metadata the DOI record gets; `CITATION.cff` is not).

**Gates**

- [ ] Clean rebuild from scratch (`rm -rf build`), `REQUIRE_LINT=ON`.
- [ ] `ctest --test-dir build --output-on-failure -j"$(nproc)"` fully green,
      lint labels included.
- [ ] `.venv/bin/python -m pytest -q bench` and
      `.venv/bin/ruff check bench cmake` green.
- [ ] `bench/check_vanilla_equivalence.py` green against a separately built
      unpatched binary at the pinned HiGHS tag.
- [ ] CI green on the commit being tagged.

**Archive**

- [ ] Headline-timing archive built with `bench/make_archive.py build ... --tar`.
- [ ] `--dev-log` attribution archive built the same way, if the release
      publishes attribution or cannibalization tables.
- [ ] `REGENERATE.sh` exits 0 on each archive.
- [ ] `PROVENANCE.md` read end to end: baseline claim, thread count, seeds,
      instrumentation and machine all say what you believe they say.
- [ ] Every `Warning:` the build printed is either resolved or understood and
      acceptable.

**Publish — order matters**

1. [ ] Zenodo integration enabled for the repository, webhook confirmed with
       `gh api repos/spoorendonk/mip-heuristics/hooks`.
2. [ ] Annotated tag: `git tag -a vX.Y.Z -m "..."` then `git push origin vX.Y.Z`.
3. [ ] GitHub release created from that tag, notes naming the pinned HiGHS tag
       and the `PATCH_VERSION`.
4. [ ] Archive tarball(s) attached to the GitHub release.
5. [ ] Zenodo software record appeared; note both the version and the concept
       DOI.
6. [ ] Archive tarball(s) uploaded as a separate Zenodo **Dataset** record,
       with `isSupplementTo` → the software concept DOI.
7. [ ] Post-release commit on `main`: concept DOI added to `CITATION.cff`
       `identifiers:`, a DOI badge added to `README.md` (there is none today —
       this release is the first thing to badge), and the archive's dataset DOI
       referenced wherever the tables are published.

**After**

- [ ] `https://doi.org/<concept DOI>` resolves, and the record's metadata
      matches `.zenodo.json` (title, description, license, ORCID, the four
      referenced papers).
- [ ] GitHub's "Cite this repository" widget renders `CITATION.cff` without a
      parse warning.
- [ ] Close the release issue, noting both DOIs.

## Cutting the next one

v1.1 differs in three places only:

- If the HiGHS tag moved, follow `CLAUDE.md`'s "Bumping the HiGHS tag" note and
  clean-rebuild; the archive's provenance will carry the new tag automatically.
- Zenodo integration is already on, so step 1 of the publish sequence collapses
  to confirming the webhook is still there.
- `CITATION.cff` already carries the concept DOI, so step 7 shrinks to updating
  `version:` and `date-released:` — which the preconditions already ask for.
