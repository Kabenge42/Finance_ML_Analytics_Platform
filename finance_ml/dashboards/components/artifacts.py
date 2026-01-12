from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from dash import html

from .constants import FONT_FAMILY
from .data_utils import PROJECT_ROOT, ARTIFACTS_DIR


def _list_artifacts() -> List[Dict[str, str]]:
    """List all available artifacts from various outputs directories.

    Scans for HTML and JSON files in standard artifact locations.
    """
    items: List[Dict[str, str]] = []

    # Map of category label to search path
    artifact_sources = [
        ("Earnings", PROJECT_ROOT / "outputs" / "eda" / "earnings_analytics"),
        ("Dashboard", ARTIFACTS_DIR),
        ("Category", PROJECT_ROOT / "outputs" / "eda" / "dashboards" / "categories"),
        ("Portfolio", PROJECT_ROOT / "outputs" / "portfolio"),
        ("Analytics", PROJECT_ROOT / "outputs" / "analytics"),
        ("Governance", PROJECT_ROOT / "outputs" / "governance"),
        ("Safety", PROJECT_ROOT / "outputs" / "safety_rails"),
        ("Evaluation", PROJECT_ROOT / "outputs" / "evaluation"),
    ]

    for label, base_path in artifact_sources:
        if base_path.exists():
            # Recursively find all HTML and JSON files
            for p in sorted(base_path.rglob("*")):
                if p.suffix.lower() not in {".html", ".json"}:
                    continue

                # Create a descriptive label including subdirectories if present
                try:
                    rel_to_base = p.relative_to(base_path)
                    # Convert to posix style and remove extension for display
                    name_display = (
                        rel_to_base.with_suffix("")
                        .as_posix()
                        .replace("_", " ")
                        .replace("/", " > ")
                        .title()
                    )
                except ValueError:
                    name_display = p.stem.replace("_", " ").title()

                items.append({"label": f"[{label}] {name_display}", "value": str(p)})

    return items


def _render_artifact(path_str: str) -> Any:
    """Render artifact content for the Artifacts tab.

    Styling aligned with code_guidelines.md Section 17.
    """
    if not path_str:
        return html.Div("Select an artifact to view", style={"padding": "20px"})

    p = Path(path_str)
    if not p.exists():
        return html.Div(
            f"Artifact not found: {p.name}", style={"color": "red", "padding": "20px"}
        )

    if p.suffix.lower() == ".html":
        # Resolve the path relative to outputs to map to /app_assets/
        try:
            rel_path = p.relative_to(PROJECT_ROOT / "outputs")
            src = f"/app_assets/{rel_path.as_posix()}"
        except ValueError:
            # Fallback if path is not under outputs (should not happen normally)
            src = ""

        return html.Iframe(
            src=src,
            style={"width": "100%", "height": "800px", "border": "none"},
        )
    elif p.suffix.lower() == ".json":
        import json

        try:
            content = json.loads(p.read_text(encoding="utf-8"))
            return html.Pre(
                json.dumps(content, indent=2),
                style={
                    "backgroundColor": "#111",
                    "color": "#ffffff",
                    "padding": "10px",
                    "fontFamily": FONT_FAMILY,
                },
            )
        except Exception as e:
            return html.Div(f"Error loading JSON: {e}", style={"color": "red"})

    return html.Div("Unsupported artifact type", style={"padding": "10px"})
