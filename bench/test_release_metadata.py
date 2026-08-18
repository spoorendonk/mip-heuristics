"""Structural gate on the metadata a release deposit is built from.

`.zenodo.json` is the deposit metadata: when a repository has one, Zenodo uses
it and ignores `CITATION.cff` entirely, so this file alone decides what the DOI
record says.  It is hand-edited, never parsed by anything in the repository,
and a malformed or under-specified one fails at deposit time — after the tag is
pushed and the release is created, which is the one point in the release
process with no undo.

This checks *shape*, not wording.  The content is owned by the documentation
issue; see `docs/RELEASE.md` for the process around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ZENODO = REPO_ROOT / ".zenodo.json"
CITATION = REPO_ROOT / "CITATION.cff"

# Zenodo's legacy deposit schema requires these for an `upload_type: software`
# record; a missing one is rejected at deposit time.
REQUIRED_KEYS = ("upload_type", "title", "description", "creators", "license")

# The `relation` enum from Zenodo's deposit schema, restricted to the values
# this project has any use for.  A typo here is accepted by the JSON parser and
# rejected by the deposit.
KNOWN_RELATIONS = {
    "references",
    "isReferencedBy",
    "isSupplementTo",
    "isSupplementedBy",
    "isPartOf",
    "hasPart",
    "isNewVersionOf",
    "isPreviousVersionOf",
    "isDocumentedBy",
    "documents",
    "cites",
    "isCitedBy",
    "isDerivedFrom",
    "isSourceOf",
}


@pytest.fixture(scope="module")
def zenodo() -> dict:
    return json.loads(ZENODO.read_text())


def test_zenodo_json_parses(zenodo: dict):
    assert isinstance(zenodo, dict)


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_zenodo_json_carries_the_required_deposit_keys(zenodo: dict, key: str):
    assert zenodo.get(key), f".zenodo.json is missing a non-empty {key!r}"


def test_zenodo_creators_are_well_formed(zenodo: dict):
    """A creator without a name is a record with no author on it."""
    for creator in zenodo["creators"]:
        assert creator.get("name"), f"creator without a name: {creator!r}"


def test_zenodo_related_identifiers_use_schema_relations(zenodo: dict):
    for entry in zenodo.get("related_identifiers", []):
        assert entry.get("identifier"), f"related identifier with no id: {entry!r}"
        assert entry.get("relation") in KNOWN_RELATIONS, (
            f"relation {entry.get('relation')!r} is not in Zenodo's enum"
        )


def test_citation_cff_exists_and_agrees_on_the_licence(zenodo: dict):
    """The two files describe one artifact to two different consumers —
    Zenodo reads `.zenodo.json`, GitHub's citation widget reads `CITATION.cff`
    — so a licence that differs between them is visibly wrong in one of them.
    Compared case-insensitively: Zenodo's identifiers are lowercase, CFF uses
    the SPDX spelling."""
    text = CITATION.read_text()
    cff_license = next(
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("license:")
    )
    assert cff_license.lower() == zenodo["license"].lower()
