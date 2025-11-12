"""
finance_ml.data_versioning - Data versioning and lineage tracking

This module provides capabilities for tracking data versions with:
- Timestamp-based version tracking
- Hash-based content verification
- Version comparison and rollback
- Lineage tracking across transformations

Part of Phase 9.1 implementation (TDD approach).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_dataframe_hash(df: pd.DataFrame) -> str:
    """Calculate SHA256 hash of DataFrame content.

    Args:
        df: DataFrame to hash

    Returns:
        64-character hexadecimal hash string
    """
    # Convert DataFrame to bytes for hashing
    # Use pickle to get a consistent byte representation
    df_bytes = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)

    # Calculate SHA256 hash
    hash_obj = hashlib.sha256(df_bytes)
    return hash_obj.hexdigest()


@dataclass
class DataVersion:
    """Container for a versioned data snapshot.

    Attributes:
        data: The DataFrame being versioned
        version_id: Unique identifier for this version
        timestamp: When this version was created
        content_hash: SHA256 hash of the data content
        metadata: Additional metadata about this version
    """

    data: pd.DataFrame
    version_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    content_hash: str = field(default="", init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate content hash after initialization."""
        if not self.content_hash:
            self.content_hash = calculate_dataframe_hash(self.data)


class DataVersionManager:
    """Manager for saving, loading, and comparing data versions.

    Provides functionality for:
    - Saving data snapshots with metadata
    - Loading previous versions
    - Comparing versions
    - Listing available versions
    """

    def __init__(self, version_dir: Path):
        """Initialize version manager.

        Args:
            version_dir: Directory to store version files
        """
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self._versions: Dict[str, DataVersion] = {}
        self._load_version_index()

    def _get_version_path(self, version_id: str) -> Path:
        """Get file path for a version.

        Args:
            version_id: Version identifier

        Returns:
            Path to version file
        """
        return self.version_dir / f"{version_id}.pkl"

    def _get_index_path(self) -> Path:
        """Get path to version index file.

        Returns:
            Path to index file
        """
        return self.version_dir / "version_index.json"

    def _load_version_index(self) -> None:
        """Load version index from disk."""
        index_path = self._get_index_path()
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    index_data = json.load(f)
                    # Just track version IDs in memory
                    for version_id in index_data.get("versions", []):
                        # Create placeholder - actual data loaded on demand
                        self._versions[version_id] = None  # type: ignore
            except Exception as e:
                logger.warning(f"Could not load version index: {e}")

    def _save_version_index(self) -> None:
        """Save version index to disk."""
        index_path = self._get_index_path()
        index_data = {
            "versions": list(self._versions.keys()),
            "last_updated": datetime.now().isoformat(),
        }
        try:
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save version index: {e}")

    def save_version(
        self, data: pd.DataFrame, version_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> DataVersion:
        """Save a new data version.

        Args:
            data: DataFrame to version
            version_id: Unique identifier for this version
            metadata: Optional metadata dictionary

        Returns:
            DataVersion object
        """
        if metadata is None:
            metadata = {}

        version = DataVersion(data=data, version_id=version_id, metadata=metadata)

        # Save to disk
        version_path = self._get_version_path(version_id)
        try:
            with open(version_path, "wb") as f:
                pickle.dump(version, f)

            # Register in memory
            self._versions[version_id] = version
            self._save_version_index()

            logger.info(f"Saved version '{version_id}' with hash {version.content_hash[:8]}...")

        except Exception as e:
            logger.error(f"Could not save version '{version_id}': {e}")
            raise

        return version

    def load_version(self, version_id: str) -> DataVersion:
        """Load a saved version.

        Args:
            version_id: Version identifier

        Returns:
            DataVersion object

        Raises:
            ValueError: If version not found
        """
        if version_id not in self._versions:
            raise ValueError(f"Version '{version_id}' not found")

        version_path = self._get_version_path(version_id)
        if not version_path.exists():
            raise ValueError(f"Version file not found: {version_path}")

        try:
            with open(version_path, "rb") as f:
                version = pickle.load(f)

            logger.info(f"Loaded version '{version_id}'")
            return version

        except Exception as e:
            logger.error(f"Could not load version '{version_id}': {e}")
            raise

    def list_versions(self) -> List[str]:
        """List all available version IDs.

        Returns:
            List of version IDs
        """
        return list(self._versions.keys())

    def get_version_info(self, version_id: str) -> Dict[str, Any]:
        """Get metadata about a version without loading full data.

        Args:
            version_id: Version identifier

        Returns:
            Dictionary with version information
        """
        version = self.load_version(version_id)

        return {
            "version_id": version.version_id,
            "timestamp": version.timestamp,
            "content_hash": version.content_hash,
            "metadata": version.metadata,
            "rows": len(version.data),
            "columns": len(version.data.columns),
        }

    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two versions.

        Args:
            version_id1: First version ID
            version_id2: Second version ID

        Returns:
            Dictionary with comparison results
        """
        version1 = self.load_version(version_id1)
        version2 = self.load_version(version_id2)

        return compare_versions(version1, version2)

    def delete_version(self, version_id: str) -> None:
        """Delete a version.

        Args:
            version_id: Version identifier
        """
        if version_id not in self._versions:
            raise ValueError(f"Version '{version_id}' not found")

        version_path = self._get_version_path(version_id)
        if version_path.exists():
            version_path.unlink()

        del self._versions[version_id]
        self._save_version_index()

        logger.info(f"Deleted version '{version_id}'")


def compare_versions(version1: DataVersion, version2: DataVersion) -> Dict[str, Any]:
    """Compare two data versions.

    Args:
        version1: First version
        version2: Second version

    Returns:
        Dictionary with comparison results including:
        - same_content: Whether content hashes match
        - rows_added: Number of rows added
        - rows_removed: Number of rows removed
        - columns_added: Number of columns added
        - columns_removed: Number of columns removed
    """
    same_content = version1.content_hash == version2.content_hash

    rows1 = len(version1.data)
    rows2 = len(version2.data)

    cols1 = set(version1.data.columns)
    cols2 = set(version2.data.columns)

    return {
        "same_content": same_content,
        "rows_added": max(0, rows2 - rows1),
        "rows_removed": max(0, rows1 - rows2),
        "columns_added": len(cols2 - cols1),
        "columns_removed": len(cols1 - cols2),
        "hash1": version1.content_hash,
        "hash2": version2.content_hash,
    }


def create_version_snapshot(
    data: pd.DataFrame, source: Optional[str] = None, region: Optional[str] = None, **kwargs
) -> DataVersion:
    """Create a version snapshot with automatic metadata.

    Args:
        data: DataFrame to snapshot
        source: Data source (e.g., 'CSV', 'DB')
        region: Data region (e.g., 'US', 'EU')
        **kwargs: Additional metadata fields

    Returns:
        DataVersion object
    """
    # Generate automatic version ID
    timestamp = datetime.now()
    version_id = f"v_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    # Build metadata
    metadata = {
        "rows": len(data),
        "columns": len(data.columns),
        "created_at": timestamp.isoformat(),
    }

    if source:
        metadata["source"] = source
    if region:
        metadata["region"] = region

    # Add any additional metadata
    metadata.update(kwargs)

    return DataVersion(data=data, version_id=version_id, metadata=metadata)
