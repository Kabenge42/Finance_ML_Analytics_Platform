import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from finance_ml.dashboards.components.artifacts import _list_artifacts, _render_artifact


class TestDashboardArtifacts(unittest.TestCase):
    @patch("finance_ml.dashboards.components.artifacts.PROJECT_ROOT")
    @patch("finance_ml.dashboards.components.artifacts.ARTIFACTS_DIR")
    def test_list_artifacts(self, mock_artifacts_dir, mock_project_root):
        mock_artifacts_dir.exists.return_value = True
        mock_artifacts_dir.glob.return_value = [Path("art1.html"), Path("art2.json")]

        # Mock base1 as well
        base1 = MagicMock()
        base1.exists.return_value = False
        mock_project_root.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = base1

        items = _list_artifacts()
        # Filtered by extension, art1.html and art2.json should be in it
        self.assertTrue(any(i["label"].startswith("[Dashboard]") for i in items))

    def test_render_artifact_empty(self):
        res = _render_artifact("")
        self.assertIsNotNone(res)


if __name__ == "__main__":
    unittest.main()
