"""Private cohort-role construction and validation for the GFC-v2 study."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


ROLE_MAP_VERSION = "healthgait-gfc-v2-roles-v1"
DEVELOPMENT_ROLE = "development"
LOCKED_OUTCOME_ROLE = "locked_outcome"
ROLE_MAP_COLUMNS = ("subject_id", "role")
EXPECTED_ASSIGNED_COUNTS = {DEVELOPMENT_ROLE: 80, LOCKED_OUTCOME_ROLE: 318}
EXPECTED_COMPLETE_COUNTS = {DEVELOPMENT_ROLE: 76, LOCKED_OUTCOME_ROLE: 308}
HISTORICAL_SPLIT_ROLES = {"val": DEVELOPMENT_ROLE, "train": LOCKED_OUTCOME_ROLE}


def _validated_identifiers(values: Iterable[object], *, label: str) -> list[str]:
    identifiers: list[str] = []
    folded: dict[str, str] = {}
    for raw in values:
        if not isinstance(raw, str) or not raw or raw != raw.strip():
            raise ValueError(f"{label} must contain nonempty text without outer whitespace")
        canonical = raw.casefold()
        previous = folded.setdefault(canonical, raw)
        if previous != raw:
            raise ValueError(
                f"{label} contains case-colliding identifiers {previous!r} and {raw!r}"
            )
        identifiers.append(raw)
    return identifiers


def validate_role_map(
    table: pd.DataFrame,
    *,
    expected_subject_ids: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Validate the exact private role-map contract and return canonical row order."""

    if list(table.columns) != list(ROLE_MAP_COLUMNS):
        raise ValueError(
            "role map must have exactly these columns in order: "
            + ",".join(ROLE_MAP_COLUMNS)
        )
    result = table.copy()
    result["subject_id"] = _validated_identifiers(
        result["subject_id"].tolist(), label="role-map subject_id"
    )
    if result["subject_id"].duplicated().any():
        duplicates = sorted(result.loc[result["subject_id"].duplicated(), "subject_id"].unique())
        raise ValueError("role map contains duplicate participants: " + ", ".join(duplicates))
    roles = _validated_identifiers(result["role"].tolist(), label="role")
    unknown = sorted(set(roles) - set(EXPECTED_ASSIGNED_COUNTS))
    if unknown:
        raise ValueError("role map contains unknown roles: " + ", ".join(unknown))
    result["role"] = roles
    counts = result["role"].value_counts().to_dict()
    if counts != EXPECTED_ASSIGNED_COUNTS:
        raise ValueError(
            f"role map assigned counts must be {EXPECTED_ASSIGNED_COUNTS}, got {counts}"
        )

    if expected_subject_ids is not None:
        expected = _validated_identifiers(
            expected_subject_ids, label="feature-table subject_id"
        )
        if len(expected) != len(set(expected)):
            raise ValueError("feature-table participant identifiers must be unique")
        assigned = set(result["subject_id"])
        required = set(expected)
        missing = sorted(required - assigned)
        extra = sorted(assigned - required)
        if missing or extra:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing[:10]))
            if extra:
                details.append("extra=" + ",".join(extra[:10]))
            raise ValueError("role map does not exactly cover feature participants: " + "; ".join(details))
    return result.sort_values("subject_id", kind="stable").reset_index(drop=True)


def load_role_map(
    path: Path,
    *,
    expected_subject_ids: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Load a private role map without retaining its path in result metadata."""

    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read role map: {error}") from error
    return validate_role_map(table, expected_subject_ids=expected_subject_ids)


def build_role_map(manifest: Path, output: Path) -> pd.DataFrame:
    """Build the versioned private role map from historical manifest splits."""

    if output.exists():
        raise FileExistsError(f"role-map output already exists: {output}")
    try:
        table = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"could not read candidate manifest: {error}") from error
    required = {"subject_id", "split"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError("candidate manifest is missing columns: " + ", ".join(missing))
    subjects = _validated_identifiers(table["subject_id"].tolist(), label="manifest subject_id")
    splits = _validated_identifiers(table["split"].tolist(), label="manifest split")
    assignments: dict[str, str] = {}
    folded: dict[str, str] = {}
    for subject_id, split in zip(subjects, splits):
        previous_case = folded.setdefault(subject_id.casefold(), subject_id)
        if previous_case != subject_id:
            raise ValueError(
                f"manifest contains case-colliding identifiers {previous_case!r} and {subject_id!r}"
            )
        try:
            role = HISTORICAL_SPLIT_ROLES[split]
        except KeyError as error:
            raise ValueError(f"manifest contains unsupported historical split {split!r}") from error
        previous_role = assignments.setdefault(subject_id, role)
        if previous_role != role:
            raise ValueError(f"manifest participant {subject_id!r} spans historical splits")
    result = validate_role_map(
        pd.DataFrame(sorted(assignments.items()), columns=ROLE_MAP_COLUMNS)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, lineterminator="\n")
    return result


def role_lookup(table: pd.DataFrame) -> dict[str, str]:
    """Return a subject-to-role lookup from an already validated map."""

    return dict(zip(table["subject_id"], table["role"]))


__all__ = [
    "DEVELOPMENT_ROLE",
    "EXPECTED_ASSIGNED_COUNTS",
    "EXPECTED_COMPLETE_COUNTS",
    "HISTORICAL_SPLIT_ROLES",
    "LOCKED_OUTCOME_ROLE",
    "ROLE_MAP_COLUMNS",
    "ROLE_MAP_VERSION",
    "build_role_map",
    "load_role_map",
    "role_lookup",
    "validate_role_map",
]
