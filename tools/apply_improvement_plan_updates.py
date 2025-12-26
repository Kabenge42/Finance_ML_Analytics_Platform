"""
Utility to apply structured updates to IMPROVEMENT_PLAN.md.

Implements the user story: Revise_the_IMPROVEMENT_PLAN.md_by_integr.md
by inserting a new section that documents a robust SQLite ingestion path
and parity with PostgreSQL, plus related tasks and testing notes.

The updater is idempotent and safe to run multiple times.
"""

from __future__ import annotations

from pathlib import Path

SQLITE_SECTION_TITLE = "### New Section: Robust SQLite Ingestion Path and Parity With PostgreSQL"

# Minimal curated content distilled from the provided scratch document to keep
# the change focused and testable, while remaining useful to readers.
SQLITE_SECTION_BODY = f"""
{SQLITE_SECTION_TITLE}

#### Problem summary
- SQLite shell `.import` treats the first row as data (no automatic header skip).
- Empty strings are not coerced to NULL by default, causing downstream issues.
- Errors are not isolated across regions; limited validation and mapping safeguards.

#### Proposed approach
- Use per-region TEMP staging tables with generic `col1..colN` schema.
- Delete the header row explicitly and map with `NULLIF(colN, '')` into `equities`.
- Wrap each region import in its own transaction with `.bail on` for fail-fast.
- Default missing `"Region"` per file (US/EU/APAC/ROTW) and rely on `UNIQUE("Ticker","Region")`.
- Provide a Python importer alternative with chunking using pandas for reliability.

#### Tasks (to be tracked under Phase 2 — Data Ingestion and Validation)
1) SQLite import hardening (shell-based)
- Add `import_equities_data_sqlite.sql` with:
  - `.bail on`, `.echo on`
  - TEMP staging `col1..colN` per region
  - Explicit header-row deletion
  - `NULLIF` mapping to `equities` and default Region per file
  - `INSERT OR IGNORE` for deduplication via `UNIQUE("Ticker","Region")`
  - Per-region transactions and basic validation summaries

2) Python import alternative for SQLite
- Create `tools/import_sqlite.py` that:
  - Reads CSVs with `dtype=str`, normalizes empty strings to `None`
  - Backfills Region, supports `--chunksize` and per-region selection
  - Appends with de-duplication via the unique index or temp-table merge

3) Validation utilities parity
- Validate header matches expected columns, sample numeric fields, per-region counts.
- Emit a machine-readable JSON report for CI.

4) Documentation updates
- README: add a “SQLite local path” subsection with exact commands and caveats
  (header handling, NULLs, `.bail on`).

5) Tests for SQLite path
- Add `tests/test_sqlite_import.py`:
  - Build a temp SQLite DB, apply schema, import tiny CSV fixtures
  - Assert header removed, empty strings mapped to NULL, Region backfilled
  - Ensure `UNIQUE("Ticker","Region")` prevents duplicates

#### Rationale
- Eliminates header-as-data issues; enforces consistent NULL semantics.
- Improves debuggability via explicit mapping and per-region transactions.
- Python importer provides a robust, cross-platform alternative with chunked loads.
""".strip()


def has_sqlite_section(text: str) -> bool:
    """Return True if the IMPROVEMENT_PLAN.md text already contains the section.

    We search for the section title to keep the check simple and robust.
    """
    return SQLITE_SECTION_TITLE in text


def apply_sqlite_parity_section(plan_path: str | Path) -> bool:
    """Insert the SQLite parity section into the given IMPROVEMENT_PLAN.md file.

    Returns True if the file was modified, False if the section already existed.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"IMPROVEMENT_PLAN.md not found: {path}")

    original = path.read_text(encoding="utf-8")
    if has_sqlite_section(original):
        return False

    # Append the new section at the end with a separating newline.
    updated = original.rstrip() + "\n\n" + SQLITE_SECTION_BODY + "\n"
    path.write_text(updated, encoding="utf-8")
    return True


__all__ = [
    "apply_sqlite_parity_section",
    "has_sqlite_section",
    "SQLITE_SECTION_TITLE",
    "SQLITE_SECTION_BODY",
]
