"""
finance_ml.data_catalog - Data catalog with metadata tracking

This module provides capabilities for tracking dataset metadata including:
- Schema information (columns, types, row/column counts)
- Statistical profiles (numeric and categorical statistics)
- Data quality metrics
- Catalog persistence and search

Part of Phase 9.1 implementation (TDD approach).
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SchemaInfo:
    """Container for dataset schema information.
    
    Attributes:
        num_rows: Number of rows in dataset
        num_columns: Number of columns in dataset
        column_names: List of column names
        column_types: Dictionary mapping column names to data types
    """
    
    num_rows: int
    num_columns: int
    column_names: List[str]
    column_types: Dict[str, str]


@dataclass
class StatisticalProfile:
    """Container for dataset statistical profile.
    
    Attributes:
        numeric_stats: Statistics for numeric columns
        categorical_stats: Statistics for categorical columns
    """
    
    numeric_stats: Dict[str, Dict[str, Any]]
    categorical_stats: Dict[str, Dict[str, Any]]


@dataclass
class DatasetMetadata:
    """Container for complete dataset metadata.
    
    Attributes:
        dataset_id: Unique identifier for dataset
        name: Human-readable name
        description: Dataset description
        schema: Schema information
        profile: Statistical profile
        source: Data source (e.g., 'CSV', 'PostgreSQL')
        region: Data region (e.g., 'US', 'EU')
        created_at: Timestamp when metadata was created
        updated_at: Timestamp when metadata was last updated
        tags: List of tags for categorization
    """
    
    dataset_id: str
    name: str
    schema: SchemaInfo
    profile: StatisticalProfile
    description: str = ""
    source: Optional[str] = None
    region: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


def extract_schema_info(df: pd.DataFrame) -> SchemaInfo:
    """Extract schema information from DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        SchemaInfo object with schema details
    """
    return SchemaInfo(
        num_rows=len(df),
        num_columns=len(df.columns),
        column_names=list(df.columns),
        column_types={col: str(df[col].dtype) for col in df.columns}
    )


def create_statistical_profile(df: pd.DataFrame) -> StatisticalProfile:
    """Create statistical profile of DataFrame.
    
    Args:
        df: DataFrame to profile
        
    Returns:
        StatisticalProfile with numeric and categorical statistics
    """
    numeric_stats = {}
    categorical_stats = {}
    
    # Process numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col]
        numeric_stats[col] = {
            'mean': float(series.mean()) if not series.isna().all() else None,
            'std': float(series.std()) if not series.isna().all() else None,
            'min': float(series.min()) if not series.isna().all() else None,
            'max': float(series.max()) if not series.isna().all() else None,
            'median': float(series.median()) if not series.isna().all() else None,
            'missing_count': int(series.isna().sum())
        }
    
    # Process categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        series = df[col]
        value_counts = series.value_counts()
        categorical_stats[col] = {
            'unique_count': int(series.nunique()),
            'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'top_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            'missing_count': int(series.isna().sum())
        }
    
    return StatisticalProfile(
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats
    )


class DataCatalog:
    """Manager for dataset catalog with metadata tracking.
    
    Provides functionality for:
    - Registering datasets with metadata
    - Retrieving dataset metadata
    - Searching datasets by tags, source, region
    - Updating and removing datasets
    - Catalog persistence
    """
    
    def __init__(self, catalog_dir: Path):
        """Initialize data catalog.
        
        Args:
            catalog_dir: Directory to store catalog metadata
        """
        self.catalog_dir = Path(catalog_dir)
        self.catalog_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: Dict[str, DatasetMetadata] = {}
        self._load_catalog_index()
    
    def _get_metadata_path(self, dataset_id: str) -> Path:
        """Get file path for dataset metadata.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Path to metadata file
        """
        return self.catalog_dir / f"{dataset_id}_metadata.pkl"
    
    def _get_index_path(self) -> Path:
        """Get path to catalog index file.
        
        Returns:
            Path to index file
        """
        return self.catalog_dir / "catalog_index.json"
    
    def _load_catalog_index(self) -> None:
        """Load catalog index from disk."""
        index_path = self._get_index_path()
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                    # Load metadata for each dataset
                    for dataset_id in index_data.get('datasets', []):
                        try:
                            metadata = self._load_metadata(dataset_id)
                            self._metadata[dataset_id] = metadata
                        except Exception as e:
                            logger.warning(f"Could not load metadata for {dataset_id}: {e}")
            except Exception as e:
                logger.warning(f"Could not load catalog index: {e}")
    
    def _save_catalog_index(self) -> None:
        """Save catalog index to disk."""
        index_path = self._get_index_path()
        index_data = {
            'datasets': list(self._metadata.keys()),
            'last_updated': datetime.now().isoformat()
        }
        try:
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save catalog index: {e}")
    
    def _load_metadata(self, dataset_id: str) -> DatasetMetadata:
        """Load metadata from disk.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            DatasetMetadata object
        """
        metadata_path = self._get_metadata_path(dataset_id)
        with open(metadata_path, 'rb') as f:
            return pickle.load(f)
    
    def _save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save metadata to disk.
        
        Args:
            metadata: DatasetMetadata to save
        """
        metadata_path = self._get_metadata_path(metadata.dataset_id)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def register_dataset(
        self,
        data: pd.DataFrame,
        dataset_id: str,
        name: str,
        description: str = "",
        source: Optional[str] = None,
        region: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> DatasetMetadata:
        """Register a dataset in the catalog.
        
        Args:
            data: DataFrame to register
            dataset_id: Unique identifier for dataset
            name: Human-readable name
            description: Dataset description
            source: Data source
            region: Data region
            tags: List of tags
            
        Returns:
            DatasetMetadata object
        """
        if tags is None:
            tags = []
        
        # Extract schema and profile
        schema = extract_schema_info(data)
        profile = create_statistical_profile(data)
        
        # Create metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            description=description,
            schema=schema,
            profile=profile,
            source=source,
            region=region,
            tags=tags
        )
        
        # Save metadata
        self._save_metadata(metadata)
        self._metadata[dataset_id] = metadata
        self._save_catalog_index()
        
        logger.info(f"Registered dataset '{dataset_id}' in catalog")
        
        return metadata
    
    def get_metadata(self, dataset_id: str) -> DatasetMetadata:
        """Get metadata for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            DatasetMetadata object
            
        Raises:
            ValueError: If dataset not found
        """
        if dataset_id not in self._metadata:
            raise ValueError(f"Dataset '{dataset_id}' not found in catalog")
        
        return self._metadata[dataset_id]
    
    def list_datasets(self) -> List[str]:
        """List all dataset IDs in catalog.
        
        Returns:
            List of dataset IDs
        """
        return list(self._metadata.keys())
    
    def search_by_tag(self, tag: str) -> List[str]:
        """Search datasets by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching dataset IDs
        """
        return [
            dataset_id
            for dataset_id, metadata in self._metadata.items()
            if tag in metadata.tags
        ]
    
    def search_by_source(self, source: str) -> List[str]:
        """Search datasets by source.
        
        Args:
            source: Source to search for
            
        Returns:
            List of matching dataset IDs
        """
        return [
            dataset_id
            for dataset_id, metadata in self._metadata.items()
            if metadata.source == source
        ]
    
    def update_metadata(
        self,
        dataset_id: str,
        **kwargs
    ) -> DatasetMetadata:
        """Update dataset metadata.
        
        Args:
            dataset_id: Dataset identifier
            **kwargs: Fields to update
            
        Returns:
            Updated DatasetMetadata
        """
        if dataset_id not in self._metadata:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        
        metadata = self._metadata[dataset_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # Update timestamp
        metadata.updated_at = datetime.now()
        
        # Save updated metadata
        self._save_metadata(metadata)
        self._metadata[dataset_id] = metadata
        
        logger.info(f"Updated metadata for dataset '{dataset_id}'")
        
        return metadata
    
    def remove_dataset(self, dataset_id: str) -> None:
        """Remove a dataset from catalog.
        
        Args:
            dataset_id: Dataset identifier
        """
        if dataset_id not in self._metadata:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        
        # Remove metadata file
        metadata_path = self._get_metadata_path(dataset_id)
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from memory
        del self._metadata[dataset_id]
        self._save_catalog_index()
        
        logger.info(f"Removed dataset '{dataset_id}' from catalog")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get catalog summary statistics.
        
        Returns:
            Dictionary with summary information
        """
        total_rows = sum(
            metadata.schema.num_rows
            for metadata in self._metadata.values()
        )
        
        total_columns = sum(
            metadata.schema.num_columns
            for metadata in self._metadata.values()
        )
        
        return {
            'total_datasets': len(self._metadata),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'datasets': list(self._metadata.keys())
        }
