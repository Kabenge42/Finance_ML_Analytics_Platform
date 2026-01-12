import ast
import json
import unittest
from pathlib import Path


def _load_zeppelin_note(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_pythonish_code(paragraph_text: str) -> str:
    """Return paragraph text with Zeppelin interpreter directives removed."""
    lines = paragraph_text.splitlines()
    stripped = []
    for line in lines:
        # Zeppelin directives like: %python, %pyspark, %sql
        if line.lstrip().startswith("%"):
            continue
        stripped.append(line)
    return "\n".join(stripped).strip() + "\n"


class TestEtlDataExplorerZeppelinNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parent.parent
        cls.zpln_path = project_root / "etl_data_explorer.zpln"
        cls.note = _load_zeppelin_note(cls.zpln_path)

    def test_notebook_is_valid_json_object(self):
        self.assertIsInstance(self.note, dict)

    def test_has_expected_top_level_keys(self):
        self.assertIn("paragraphs", self.note)
        self.assertIn("name", self.note)
        self.assertIn("config", self.note)

    def test_paragraphs_is_non_empty_list(self):
        paragraphs = self.note.get("paragraphs")
        self.assertIsInstance(paragraphs, list)
        self.assertGreaterEqual(
            len(paragraphs), 5, "Expected >= 5 paragraphs for the ETL + analytics flow"
        )

    def test_notebook_is_not_nested_json_blob(self):
        """The old file embedded a second JSON note inside a single paragraph text."""
        paragraphs = self.note.get("paragraphs", [])
        first_text = (paragraphs[0].get("text") or "") if paragraphs else ""
        self.assertFalse(
            first_text.lstrip().startswith('{\n  "paragraphs"'),
            "Notebook appears to embed another Zeppelin JSON note inside paragraph text",
        )

    def test_has_required_sections(self):
        paragraphs = self.note.get("paragraphs", [])
        haystack = "\n".join(
            (p.get("title") or "") + "\n" + (p.get("text") or "") for p in paragraphs
        ).lower()

        required = [
            "configuration",
            "etl",
            "data loading",
            "validation",
            "feature engineering",
            "analytics",
        ]
        missing = [r for r in required if r not in haystack]
        self.assertFalse(missing, f"Missing required sections in notebook: {missing}")

    def test_reuses_schema_normalize_column_name(self):
        paragraphs = self.note.get("paragraphs", [])
        all_text = "\n".join((p.get("text") or "") for p in paragraphs)

        self.assertIn(
            "from finance_ml.core.schema import normalize_column_name",
            all_text,
            "Notebook should reuse canonical normalize_column_name from finance_ml.core.schema",
        )
        self.assertNotIn(
            "def normalize_column_name",
            all_text,
            "Notebook should not redefine normalize_column_name (use finance_ml.core.schema)",
        )

    def test_python_paragraphs_are_syntax_valid(self):
        """Ensure code blocks are at least parseable Python after stripping % directives."""
        paragraphs = self.note.get("paragraphs", [])
        pythonish = []
        for p in paragraphs:
            text = p.get("text") or ""
            if "%python" in text or "%pyspark" in text:
                pythonish.append(text)

        self.assertGreaterEqual(
            len(pythonish), 2, "Expected at least two python/pyspark paragraphs"
        )

        for idx, text in enumerate(pythonish):
            code = _extract_pythonish_code(text)
            try:
                ast.parse(code)
            except SyntaxError as e:
                raise AssertionError(
                    f"Paragraph {idx} is not valid Python after stripping directives: {e}"
                )


if __name__ == "__main__":
    unittest.main()
