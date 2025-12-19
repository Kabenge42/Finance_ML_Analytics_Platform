"""
Tests for Data Versioning and Lineage Tracking (Phase 9.1 - TDD)

This module tests data versioning capabilities including:
- Timestamp-based version tracking
- Hash-based content verification
- Version comparison and rollback
- Lineage tracking across transformations
"""

import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from finance_ml.data_versioning import (
    DataVersion,
    DataVersionManager,
    calculate_dataframe_hash,
    compare_versions,
    create_version_snapshot,
)


class TestDataFrameHashing(unittest.TestCase):
    """Test suite for DataFrame content hashing"""

    def setUp(self):
        """Create sample DataFrames for testing"""
        self.df1 = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "price": [150.0, 2800.0, 300.0],
                "volume": [1000000, 500000, 800000],
            }
        )

        self.df2 = self.df1.copy()

        self.df3 = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "price": [150.0, 2800.0, 301.0],  # Different price
                "volume": [1000000, 500000, 800000],
            }
        )

    def test_hash_identical_dataframes(self):
        """Test that identical DataFrames produce the same hash"""
        hash1 = calculate_dataframe_hash(self.df1)
        hash2 = calculate_dataframe_hash(self.df2)

        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA256 produces 64-char hex string

    def test_hash_different_dataframes(self):
        """Test that different DataFrames produce different hashes"""
        hash1 = calculate_dataframe_hash(self.df1)
        hash3 = calculate_dataframe_hash(self.df3)

        self.assertNotEqual(hash1, hash3)

    def test_hash_empty_dataframe(self):
        """Test hashing of empty DataFrame"""
        empty_df = pd.DataFrame()
        hash_val = calculate_dataframe_hash(empty_df)

        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)


class TestDataVersion(unittest.TestCase):
    """Test suite for DataVersion class"""

    def setUp(self):
        """Create sample data for versioning"""
        self.df = pd.DataFrame({"ticker": ["AAPL", "GOOGL"], "price": [150.0, 2800.0]})

        self.metadata = {"source": "CSV", "region": "US", "rows": 2, "columns": 2}

    def test_create_data_version(self):
        """Test creating a DataVersion instance"""
        version = DataVersion(data=self.df, version_id="v1", metadata=self.metadata)

        self.assertEqual(version.version_id, "v1")
        self.assertIsInstance(version.timestamp, datetime)
        self.assertIsInstance(version.content_hash, str)
        self.assertEqual(len(version.content_hash), 64)
        self.assertEqual(version.metadata, self.metadata)
        pd.testing.assert_frame_equal(version.data, self.df)

    def test_version_auto_timestamp(self):
        """Test that version automatically captures timestamp"""
        version = DataVersion(data=self.df, version_id="v1")

        self.assertIsInstance(version.timestamp, datetime)
        # Should be very recent (within last second)
        time_diff = (datetime.now() - version.timestamp).total_seconds()
        self.assertLess(time_diff, 1.0)

    def test_version_auto_hash(self):
        """Test that version automatically calculates content hash"""
        version = DataVersion(data=self.df, version_id="v1")

        self.assertIsNotNone(version.content_hash)
        # Hash should match manual calculation
        expected_hash = calculate_dataframe_hash(self.df)
        self.assertEqual(version.content_hash, expected_hash)

    def test_version_equality(self):
        """Test comparing two versions with same content"""
        version1 = DataVersion(data=self.df, version_id="v1")
        version2 = DataVersion(data=self.df.copy(), version_id="v2")

        # Same content hash even with different version IDs
        self.assertEqual(version1.content_hash, version2.content_hash)


class TestDataVersionManager(unittest.TestCase):
    """Test suite for DataVersionManager"""

    def setUp(self):
        """Create temporary directory and sample data"""
        self.temp_dir = tempfile.mkdtemp()
        self.version_dir = Path(self.temp_dir) / "versions"
        self.version_dir.mkdir(exist_ok=True)

        self.df1 = pd.DataFrame({"ticker": ["AAPL", "GOOGL"], "price": [150.0, 2800.0]})

        self.df2 = pd.DataFrame(
            {"ticker": ["AAPL", "GOOGL", "MSFT"], "price": [150.0, 2800.0, 300.0]}
        )

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_create_version_manager(self):
        """Test creating a DataVersionManager"""
        manager = DataVersionManager(version_dir=self.version_dir)

        self.assertEqual(manager.version_dir, self.version_dir)
        self.assertTrue(self.version_dir.exists())

    def test_save_version(self):
        """Test saving a data version"""
        manager = DataVersionManager(version_dir=self.version_dir)

        version = manager.save_version(
            data=self.df1, version_id="v1", metadata={"source": "test"}
        )

        self.assertIsInstance(version, DataVersion)
        self.assertEqual(version.version_id, "v1")

        # Check that version is registered
        self.assertIn("v1", manager.list_versions())

    def test_load_version(self):
        """Test loading a saved version"""
        manager = DataVersionManager(version_dir=self.version_dir)

        # Save version
        original_version = manager.save_version(
            data=self.df1, version_id="v1", metadata={"source": "test"}
        )

        # Load version
        loaded_version = manager.load_version("v1")

        self.assertIsInstance(loaded_version, DataVersion)
        self.assertEqual(loaded_version.version_id, "v1")
        self.assertEqual(loaded_version.content_hash, original_version.content_hash)
        pd.testing.assert_frame_equal(loaded_version.data, self.df1)

    def test_list_versions(self):
        """Test listing all saved versions"""
        manager = DataVersionManager(version_dir=self.version_dir)

        # Initially empty
        self.assertEqual(len(manager.list_versions()), 0)

        # Save multiple versions
        manager.save_version(self.df1, "v1")
        manager.save_version(self.df2, "v2")

        versions = manager.list_versions()
        self.assertEqual(len(versions), 2)
        self.assertIn("v1", versions)
        self.assertIn("v2", versions)

    def test_get_version_info(self):
        """Test getting version metadata"""
        manager = DataVersionManager(version_dir=self.version_dir)

        metadata = {"source": "CSV", "region": "US", "rows": 2}
        manager.save_version(self.df1, "v1", metadata=metadata)

        info = manager.get_version_info("v1")

        self.assertIsInstance(info, dict)
        self.assertIn("version_id", info)
        self.assertIn("timestamp", info)
        self.assertIn("content_hash", info)
        self.assertIn("metadata", info)
        self.assertEqual(info["metadata"], metadata)

    def test_compare_versions(self):
        """Test comparing two versions"""
        manager = DataVersionManager(version_dir=self.version_dir)

        manager.save_version(self.df1, "v1")
        manager.save_version(self.df2, "v2")

        comparison = manager.compare_versions("v1", "v2")

        self.assertIsInstance(comparison, dict)
        self.assertIn("same_content", comparison)
        self.assertIn("rows_added", comparison)
        self.assertIn("rows_removed", comparison)
        self.assertFalse(comparison["same_content"])
        self.assertEqual(comparison["rows_added"], 1)  # MSFT added

    def test_delete_version(self):
        """Test deleting a version"""
        manager = DataVersionManager(version_dir=self.version_dir)

        manager.save_version(self.df1, "v1")
        self.assertIn("v1", manager.list_versions())

        manager.delete_version("v1")
        self.assertNotIn("v1", manager.list_versions())


class TestVersionComparison(unittest.TestCase):
    """Test suite for version comparison utilities"""

    def setUp(self):
        """Create sample versions for comparison"""
        self.df1 = pd.DataFrame({"ticker": ["AAPL", "GOOGL"], "price": [150.0, 2800.0]})

        self.df2 = pd.DataFrame(
            {"ticker": ["AAPL", "GOOGL", "MSFT"], "price": [150.0, 2800.0, 300.0]}
        )

        self.version1 = DataVersion(self.df1, "v1")
        self.version2 = DataVersion(self.df2, "v2")

    def test_compare_identical_versions(self):
        """Test comparing identical versions"""
        version1 = DataVersion(self.df1, "v1")
        version1_copy = DataVersion(self.df1.copy(), "v1_copy")

        result = compare_versions(version1, version1_copy)

        self.assertIsInstance(result, dict)
        self.assertTrue(result["same_content"])
        self.assertEqual(result["rows_added"], 0)
        self.assertEqual(result["rows_removed"], 0)

    def test_compare_different_versions(self):
        """Test comparing versions with different content"""
        result = compare_versions(self.version1, self.version2)

        self.assertFalse(result["same_content"])
        self.assertGreater(result["rows_added"], 0)


class TestVersionSnapshot(unittest.TestCase):
    """Test suite for creating version snapshots"""

    def setUp(self):
        """Create sample data"""
        self.df = pd.DataFrame({"ticker": ["AAPL", "GOOGL"], "price": [150.0, 2800.0]})

    def test_create_snapshot(self):
        """Test creating a version snapshot"""
        snapshot = create_version_snapshot(data=self.df, source="CSV", region="US")

        self.assertIsInstance(snapshot, DataVersion)
        self.assertIsNotNone(snapshot.version_id)
        self.assertIsNotNone(snapshot.timestamp)
        self.assertIsNotNone(snapshot.content_hash)

        # Check metadata
        self.assertEqual(snapshot.metadata["source"], "CSV")
        self.assertEqual(snapshot.metadata["region"], "US")
        self.assertEqual(snapshot.metadata["rows"], 2)
        self.assertEqual(snapshot.metadata["columns"], 2)

    def test_snapshot_auto_version_id(self):
        """Test that snapshot generates version ID automatically"""
        snapshot = create_version_snapshot(self.df)

        self.assertIsNotNone(snapshot.version_id)
        self.assertTrue(snapshot.version_id.startswith("v_"))


if __name__ == "__main__":
    unittest.main()
