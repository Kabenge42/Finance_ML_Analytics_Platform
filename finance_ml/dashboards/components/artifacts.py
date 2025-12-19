from __future__ import annotations
from typing import Any, Dict, List
from pathlib import Path
from dash import html
from .data_utils import PROJECT_ROOT, ARTIFACTS_DIR, ARTIFACTS_METADATA_PATH
from .constants import FONT_FAMILY


def _list_artifacts() -> List[Dict[str, str]]:
    """List all available artifacts from earnings_analytics and dashboard artifacts dirs."""
    items: List[Dict[str, str]] = []

    # Include artifacts from earnings_analytics directory
    base1 = PROJECT_ROOT / "outputs" / "eda" / "earnings_analytics"
    if base1.exists():
        for p in sorted(base1.glob("*")):
            if p.suffix.lower() not in {".html", ".json"}:
                continue
            items.append({"label": f"[Earnings] {p.name}", "value": str(p)})

    # Include artifacts from dashboard artifacts directory
    if ARTIFACTS_DIR.exists():
        for p in sorted(ARTIFACTS_DIR.glob("*")):
            if p.suffix.lower() not in {".html", ".json"}:
                continue
            items.append({"label": f"[Dashboard] {p.name}", "value": str(p)})

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
        # For security, we should ideally serve via an iframe or sanitized component
        # Here we just show a link or simple iframe
        return html.Iframe(
            src=f"/app_assets/{p.relative_to(PROJECT_ROOT / 'outputs').as_posix()}",
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
