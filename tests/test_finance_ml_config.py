"""
Tests for finance_ml.config module.

Following strict TDD methodology to achieve ≥80% coverage.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finance_ml.config import (
    FinanceMLConfig,
    load_config,
    get_config,
    set_config,
    reset_config,
    )


class TestFinanceMLConfig(unittest.TestCase):
    """Test FinanceMLConfig dataclass."""

    def test_default_config_creation(self):
        """Should create config with default values."""
        config = FinanceMLConfig()

        self.assertIsInstance(config.data_dir, Path)
        self.assertEqual(config.data_dir, Path("data"))
        self.assertIsInstance(config.model_dir, Path)
        self.assertEqual(config.model_dir, Path("../outputs/regression"))
        self.assertIsInstance(config.cache_dir, Path)
        self.assertEqual(config.cache_dir, Path(".cache"))
        self.assertIsInstance(config.output_dir, Path)
        self.assertEqual(config.output_dir, Path("outputs"))

        self.assertIsNone(config.db_url)
        self.assertEqual(config.db_table, "equities")
        self.assertEqual(config.model_version, "v8_2")
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.n_jobs, -1)

    def test_post_init_converts_string_paths(self):
        """Should convert string paths to Path objects in __post_init__."""
        config = FinanceMLConfig(
            data_dir="my_data", model_dir="my_models", cache_dir="my_cache", output_dir="my_outputs"
        )

        self.assertIsInstance(config.data_dir, Path)
        self.assertEqual(config.data_dir, Path("my_data"))
        self.assertIsInstance(config.model_dir, Path)
        self.assertEqual(config.model_dir, Path("my_models"))
        self.assertIsInstance(config.cache_dir, Path)
        self.assertEqual(config.cache_dir, Path("my_cache"))
        self.assertIsInstance(config.output_dir, Path)
        self.assertEqual(config.output_dir, Path("my_outputs"))

    def test_custom_config_values(self):
        """Should allow custom configuration values."""
        config = FinanceMLConfig(
            db_url="postgresql://localhost:5432/testdb",
            model_version="v9_0",
            random_seed=123,
            n_jobs=4,
        )

        self.assertEqual(config.db_url, "postgresql://localhost:5432/testdb")
        self.assertEqual(config.model_version, "v9_0")
        self.assertEqual(config.random_seed, 123)
        self.assertEqual(config.n_jobs, 4)


class TestConfigFromEnv(unittest.TestCase):
    """Test FinanceMLConfig.from_env() method."""

    def setUp(self):
        """Save original environment."""
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_from_env_with_defaults(self):
        """Should use default values when env vars not set."""
        # Clear relevant env vars
        for key in [
            "DATA_DIR",
            "MODEL_DIR",
            "CACHE_DIR",
            "OUTPUT_DIR",
            "DB_URL",
            "MODEL_VERSION",
            "RANDOM_SEED",
            "N_JOBS",
        ]:
            os.environ.pop(key, None)

        config = FinanceMLConfig.from_env()

        self.assertEqual(config.data_dir, Path("data"))
        self.assertEqual(config.model_dir, Path("../outputs/regression"))
        self.assertEqual(config.cache_dir, Path(".cache"))
        self.assertEqual(config.output_dir, Path("outputs"))
        self.assertIsNone(config.db_url)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.n_jobs, -1)

    def test_from_env_with_custom_values(self):
        """Should use environment variable values when set."""
        os.environ["DATA_DIR"] = "test_data"
        os.environ["MODEL_DIR"] = "test_models"
        os.environ["CACHE_DIR"] = "test_cache"
        os.environ["OUTPUT_DIR"] = "test_outputs"
        os.environ["DB_URL"] = "postgresql://test:5432/db"
        os.environ["MODEL_VERSION"] = "v10_0"
        os.environ["RANDOM_SEED"] = "999"
        os.environ["N_JOBS"] = "8"
        os.environ["MEMORY_LIMIT"] = "4GB"

        config = FinanceMLConfig.from_env()

        self.assertEqual(config.data_dir, Path("test_data"))
        self.assertEqual(config.model_dir, Path("test_models"))
        self.assertEqual(config.cache_dir, Path("test_cache"))
        self.assertEqual(config.output_dir, Path("test_outputs"))
        self.assertEqual(config.db_url, "postgresql://test:5432/db")
        self.assertEqual(config.model_version, "v10_0")
        self.assertEqual(config.random_seed, 999)
        self.assertEqual(config.n_jobs, 8)
        self.assertEqual(config.memory_limit, "4GB")


class TestConfigToDict(unittest.TestCase):
    """Test FinanceMLConfig.to_dict() method."""

    def test_to_dict_converts_paths_to_strings(self):
        """Should convert Path objects to strings in dictionary."""
        config = FinanceMLConfig(data_dir=Path("my_data"), model_version="v8_3", random_seed=100)

        result = config.to_dict()

        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["data_dir"], str)
        self.assertEqual(result["data_dir"], "my_data")
        self.assertEqual(result["model_version"], "v8_3")
        self.assertEqual(result["random_seed"], 100)

    def test_to_dict_preserves_none_values(self):
        """Should preserve None values in dictionary."""
        config = FinanceMLConfig(db_url=None, memory_limit=None)

        result = config.to_dict()

        self.assertIsNone(result["db_url"])
        self.assertIsNone(result["memory_limit"])


class TestConfigJsonSerialization(unittest.TestCase):
    """Test FinanceMLConfig JSON serialization (from_json/to_json)."""

    def test_from_json_loads_config(self):
        """Should load configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "data_dir": "json_data",
                "model_version": "v9_0",
                "random_seed": 777,
                "n_jobs": 4,
            }
            json.dump(config_data, f)
            json_path = f.name

        try:
            config = FinanceMLConfig.from_json(json_path)

            self.assertEqual(config.data_dir, Path("json_data"))
            self.assertEqual(config.model_version, "v9_0")
            self.assertEqual(config.random_seed, 777)
            self.assertEqual(config.n_jobs, 4)
        finally:
            os.unlink(json_path)

    def test_from_json_raises_on_missing_file(self):
        """Should raise FileNotFoundError for non-existent file."""
        with self.assertRaises(FileNotFoundError) as cm:
            FinanceMLConfig.from_json("nonexistent.json")

        self.assertIn("Config file not found", str(cm.exception))

    def test_to_json_saves_config(self):
        """Should save configuration to JSON file."""
        config = FinanceMLConfig(data_dir="test_data", model_version="v8_5", random_seed=555)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_config.json"
            config.to_json(json_path)

            self.assertTrue(json_path.exists())

            # Load and verify
            with open(json_path, "r") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["data_dir"], "test_data")
            self.assertEqual(loaded["model_version"], "v8_5")
            self.assertEqual(loaded["random_seed"], 555)

    def test_to_json_creates_parent_directory(self):
        """Should create parent directory if it doesn't exist."""
        config = FinanceMLConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "subdir" / "config.json"
            config.to_json(json_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(json_path.parent.exists())


class TestConfigYamlSerialization(unittest.TestCase):
    """Test FinanceMLConfig YAML serialization (from_yaml/to_yaml)."""

    def test_from_yaml_raises_import_error_when_yaml_not_available(self):
        """Should raise ImportError when PyYAML is not installed."""
        with patch.dict("sys.modules", {"yaml": None}):
            with self.assertRaises(ImportError) as cm:
                # Force import error by making yaml unavailable
                import sys

                yaml_module = sys.modules.get("yaml")
                sys.modules["yaml"] = None
                try:
                    FinanceMLConfig.from_yaml("test.yaml")
                finally:
                    if yaml_module is not None:
                        sys.modules["yaml"] = yaml_module

            self.assertIn("PyYAML is required", str(cm.exception))

    def test_from_yaml_raises_on_missing_file(self):
        """Should raise FileNotFoundError for non-existent file."""
        try:
            import yaml  # noqa: F401

            yaml_available = True
        except ImportError:
            yaml_available = False

        if yaml_available:
            with self.assertRaises(FileNotFoundError) as cm:
                FinanceMLConfig.from_yaml("nonexistent.yaml")

            self.assertIn("Config file not found", str(cm.exception))
        else:
            self.skipTest("PyYAML not installed")

    def test_from_yaml_loads_config(self):
        """Should load configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "data_dir": "yaml_data",
                "model_version": "v9_1",
                "random_seed": 888,
                "n_jobs": 2,
            }
            yaml.dump(config_data, f)
            yaml_path = f.name

        try:
            config = FinanceMLConfig.from_yaml(yaml_path)

            self.assertEqual(config.data_dir, Path("yaml_data"))
            self.assertEqual(config.model_version, "v9_1")
            self.assertEqual(config.random_seed, 888)
            self.assertEqual(config.n_jobs, 2)
        finally:
            os.unlink(yaml_path)

    def test_to_yaml_saves_config(self):
        """Should save configuration to YAML file."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")

        config = FinanceMLConfig(data_dir="test_yaml_data", model_version="v8_6", random_seed=666)

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(yaml_path)

            self.assertTrue(yaml_path.exists())

            # Load and verify using from_yaml
            loaded_config = FinanceMLConfig.from_yaml(yaml_path)
            self.assertEqual(loaded_config.data_dir, Path("test_yaml_data"))
            self.assertEqual(loaded_config.model_version, "v8_6")
            self.assertEqual(loaded_config.random_seed, 666)

    def test_to_yaml_raises_import_error_when_yaml_not_available(self):
        """Should raise ImportError when PyYAML is not installed."""
        config = FinanceMLConfig()

        with patch.dict("sys.modules", {"yaml": None}):
            with self.assertRaises(ImportError) as cm:
                import sys

                yaml_module = sys.modules.get("yaml")
                sys.modules["yaml"] = None
                try:
                    config.to_yaml("test.yaml")
                finally:
                    if yaml_module is not None:
                        sys.modules["yaml"] = yaml_module

            self.assertIn("PyYAML is required", str(cm.exception))


class TestConfigApplyToEnv(unittest.TestCase):
    """Test FinanceMLConfig.apply_to_env() method."""

    def setUp(self):
        """Save original environment."""
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_apply_to_env_sets_environment_variables(self):
        """Should set environment variables from config."""
        config = FinanceMLConfig(
            data_dir=Path("env_data"),
            model_dir=Path("env_models"),
            cache_dir=Path("env_cache"),
            output_dir=Path("env_outputs"),
            db_url="postgresql://localhost/testdb",
            model_version="v10_0",
            random_seed=123,
            n_jobs=4,
            memory_limit="8GB",
        )

        config.apply_to_env()

        self.assertEqual(os.environ["DATA_DIR"], "env_data")
        self.assertEqual(os.environ["MODEL_DIR"], "env_models")
        self.assertEqual(os.environ["CACHE_DIR"], "env_cache")
        self.assertEqual(os.environ["OUTPUT_DIR"], "env_outputs")
        self.assertEqual(os.environ["DB_URL"], "postgresql://localhost/testdb")
        self.assertEqual(os.environ["MODEL_VERSION"], "v10_0")
        self.assertEqual(os.environ["RANDOM_SEED"], "123")
        self.assertEqual(os.environ["N_JOBS"], "4")
        self.assertEqual(os.environ["MEMORY_LIMIT"], "8GB")

    def test_apply_to_env_skips_none_db_url(self):
        """Should not set DB_URL when it's None."""
        config = FinanceMLConfig(db_url=None)

        # Clear DB_URL if set
        os.environ.pop("DB_URL", None)

        config.apply_to_env()

        # DB_URL should not be set
        self.assertNotIn("DB_URL", os.environ)

    def test_apply_to_env_skips_none_memory_limit(self):
        """Should not set MEMORY_LIMIT when it's None."""
        config = FinanceMLConfig(memory_limit=None)

        # Clear MEMORY_LIMIT if set
        os.environ.pop("MEMORY_LIMIT", None)

        config.apply_to_env()

        # MEMORY_LIMIT should not be set
        self.assertNotIn("MEMORY_LIMIT", os.environ)


class TestLoadConfig(unittest.TestCase):
    """Test load_config() function."""

    def setUp(self):
        """Save original environment."""
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_load_config_from_json(self):
        """Should load config from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "data_dir": "load_json_data",
                "model_version": "v11_0",
                "random_seed": 321,
            }
            json.dump(config_data, f)
            json_path = f.name

        try:
            config = load_config(json_path)

            self.assertEqual(config.data_dir, Path("load_json_data"))
            self.assertEqual(config.model_version, "v11_0")
            self.assertEqual(config.random_seed, 321)
        finally:
            os.unlink(json_path)

    def test_load_config_from_yaml(self):
        """Should load config from YAML file."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "data_dir": "load_yaml_data",
                "model_version": "v11_1",
                "random_seed": 456,
            }
            yaml.dump(config_data, f)
            yaml_path = f.name

        try:
            config = load_config(yaml_path)

            self.assertEqual(config.data_dir, Path("load_yaml_data"))
            self.assertEqual(config.model_version, "v11_1")
            self.assertEqual(config.random_seed, 456)
        finally:
            os.unlink(yaml_path)

    def test_load_config_raises_on_unsupported_format(self):
        """Should raise ValueError for unsupported file format."""
        with self.assertRaises(ValueError) as cm:
            load_config("config.txt")

        self.assertIn("Unsupported config format", str(cm.exception))

    def test_load_config_from_env_when_no_path(self):
        """Should load from environment when no path provided and use_env=True."""
        os.environ["DATA_DIR"] = "env_load_data"
        os.environ["MODEL_VERSION"] = "v12_0"

        config = load_config(use_env=True)

        self.assertEqual(config.data_dir, Path("env_load_data"))
        self.assertEqual(config.model_version, "v12_0")

    def test_load_config_returns_default_when_no_path_and_no_env(self):
        """Should return default config when no path and use_env=False."""
        config = load_config(use_env=False)

        self.assertEqual(config.data_dir, Path("data"))
        self.assertEqual(config.model_version, "v8_2")
        self.assertEqual(config.random_seed, 42)


class TestGlobalConfigManagement(unittest.TestCase):
    """Test global config management (get_config, set_config, reset_config)."""

    def tearDown(self):
        """Reset global config after each test."""
        reset_config()

    def test_get_config_returns_instance(self):
        """Should return a FinanceMLConfig instance."""
        config = get_config()

        self.assertIsInstance(config, FinanceMLConfig)

    def test_get_config_returns_same_instance(self):
        """Should return the same instance on subsequent calls."""
        config1 = get_config()
        config2 = get_config()

        self.assertIs(config1, config2)

    def test_set_config_changes_global_instance(self):
        """Should change the global config instance."""
        custom_config = FinanceMLConfig(model_version="v_custom", random_seed=999)

        set_config(custom_config)
        retrieved = get_config()

        self.assertIs(retrieved, custom_config)
        self.assertEqual(retrieved.model_version, "v_custom")
        self.assertEqual(retrieved.random_seed, 999)

    def test_reset_config_clears_global_instance(self):
        """Should clear global config instance."""
        # Get initial config
        config1 = get_config()
        config1_id = id(config1)

        # Reset and get again
        reset_config()
        config2 = get_config()
        config2_id = id(config2)

        # Should be a new instance
        self.assertNotEqual(config1_id, config2_id)

    def test_reset_and_get_reloads_from_env(self):
        """Should reload from environment after reset."""
        # Save original env
        original_env = os.environ.copy()

        try:
            # Set environment variable
            os.environ["MODEL_VERSION"] = "v_reset_test"

            # Get config (should load from env)
            config1 = get_config()
            self.assertEqual(config1.model_version, "v_reset_test")

            # Change environment
            os.environ["MODEL_VERSION"] = "v_reset_test_2"

            # Get config again (should return cached)
            config2 = get_config()
            self.assertEqual(config2.model_version, "v_reset_test")  # Still cached

            # Reset and get again (should reload from new env)
            reset_config()
            config3 = get_config()
            self.assertEqual(config3.model_version, "v_reset_test_2")  # New value
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)
            reset_config()


if __name__ == "__main__":
    unittest.main()
