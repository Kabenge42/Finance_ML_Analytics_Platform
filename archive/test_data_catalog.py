"""
Tests for Data Catalog with Metadata (Phase 9.1 - TDD)

This module tests data catalog capabilities including:
- Schema tracking and validation
- Statistical metadata (row counts, column types, value ranges)
- Data quality metrics integration
- Catalog persistence and retrieval
"""

import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from finance_ml.data_catalog import (
    DatasetMetadata,
    SchemaInfo,
    StatisticalProfile,
    DataCatalog,
    extract_schema_info,
    create_statistical_profile,
    )


class TestSchemaInfo(unittest.TestCase):
    """Test suite for schema information extraction"""

    def setUp(self):
        """Create sample DataFrame"""
        self.df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'price': [150.0, 2800.0, 300.0],
            'volume': [1000000, 500000, 800000],
            'sector': ['Tech', 'Tech', 'Tech']
        })

    def test_extract_schema_info(self):
        """Test extracting schema information from DataFrame"""
        schema = extract_schema_info(self.df)
        
        self.assertIsInstance(schema, SchemaInfo)
        self.assertEqual(schema.num_rows, 3)
        self.assertEqual(schema.num_columns, 4)
        self.assertEqual(len(schema.column_names), 4)
        self.assertEqual(len(schema.column_types), 4)
        self.assertIn('ticker', schema.column_names)
        self.assertIn('price', schema.column_names)

    def test_schema_column_types(self):
        """Test that column types are correctly identified"""
        schema = extract_schema_info(self.df)
        
        # Check data types mapping
        self.assertEqual(schema.column_types['ticker'], 'object')
        self.assertEqual(schema.column_types['price'], 'float64')
        self.assertEqual(schema.column_types['volume'], 'int64')

    def test_schema_empty_dataframe(self):
        """Test schema extraction from empty DataFrame"""
        empty_df = pd.DataFrame()
        schema = extract_schema_info(empty_df)
        
        self.assertEqual(schema.num_rows, 0)
        self.assertEqual(schema.num_columns, 0)
        self.assertEqual(len(schema.column_names), 0)


class TestStatisticalProfile(unittest.TestCase):
    """Test suite for statistical profiling"""

    def setUp(self):
        """Create sample data"""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'price': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(100000, 1000000, 100),
            'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], 100)
        })

    def test_create_statistical_profile(self):
        """Test creating statistical profile"""
        profile = create_statistical_profile(self.df)
        
        self.assertIsInstance(profile, StatisticalProfile)
        self.assertIsInstance(profile.numeric_stats, dict)
        self.assertIsInstance(profile.categorical_stats, dict)

    def test_numeric_statistics(self):
        """Test numeric column statistics"""
        profile = create_statistical_profile(self.df)
        
        # Check price statistics
        self.assertIn('price', profile.numeric_stats)
        price_stats = profile.numeric_stats['price']
        
        self.assertIn('mean', price_stats)
        self.assertIn('std', price_stats)
        self.assertIn('min', price_stats)
        self.assertIn('max', price_stats)
        self.assertIn('median', price_stats)
        
        # Values should be within expected range
        self.assertGreaterEqual(price_stats['mean'], 100)
        self.assertLessEqual(price_stats['mean'], 200)

    def test_categorical_statistics(self):
        """Test categorical column statistics"""
        profile = create_statistical_profile(self.df)
        
        # Check sector statistics
        self.assertIn('sector', profile.categorical_stats)
        sector_stats = profile.categorical_stats['sector']
        
        self.assertIn('unique_count', sector_stats)
        self.assertIn('top_value', sector_stats)
        self.assertIn('top_frequency', sector_stats)
        
        self.assertEqual(sector_stats['unique_count'], 3)

    def test_missing_value_detection(self):
        """Test detection of missing values in profile"""
        # Add missing values
        df_with_missing = self.df.copy()
        df_with_missing.loc[0:10, 'price'] = np.nan
        
        profile = create_statistical_profile(df_with_missing)
        
        # Check that missing count is tracked
        self.assertIn('missing_count', profile.numeric_stats['price'])
        self.assertEqual(profile.numeric_stats['price']['missing_count'], 11)


class TestDatasetMetadata(unittest.TestCase):
    """Test suite for DatasetMetadata container"""

    def setUp(self):
        """Create sample metadata"""
        self.df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0]
        })
        
        self.schema = extract_schema_info(self.df)
        self.profile = create_statistical_profile(self.df)

    def test_create_dataset_metadata(self):
        """Test creating dataset metadata"""
        metadata = DatasetMetadata(
            dataset_id='test_dataset',
            name='Test Dataset',
            description='Test description',
            schema=self.schema,
            profile=self.profile,
            source='CSV',
            region='US'
        )
        
        self.assertEqual(metadata.dataset_id, 'test_dataset')
        self.assertEqual(metadata.name, 'Test Dataset')
        self.assertIsInstance(metadata.created_at, datetime)
        self.assertIsInstance(metadata.schema, SchemaInfo)
        self.assertIsInstance(metadata.profile, StatisticalProfile)

    def test_metadata_auto_timestamp(self):
        """Test that metadata automatically captures timestamp"""
        metadata = DatasetMetadata(
            dataset_id='test',
            name='Test',
            schema=self.schema,
            profile=self.profile
        )
        
        self.assertIsInstance(metadata.created_at, datetime)
        time_diff = (datetime.now() - metadata.created_at).total_seconds()
        self.assertLess(time_diff, 1.0)

    def test_metadata_tags(self):
        """Test metadata tags functionality"""
        metadata = DatasetMetadata(
            dataset_id='test',
            name='Test',
            schema=self.schema,
            profile=self.profile,
            tags=['finance', 'stocks', 'US']
        )
        
        self.assertEqual(len(metadata.tags), 3)
        self.assertIn('finance', metadata.tags)


class TestDataCatalog(unittest.TestCase):
    """Test suite for DataCatalog"""

    def setUp(self):
        """Create temporary directory and sample data"""
        self.temp_dir = tempfile.mkdtemp()
        self.catalog_dir = Path(self.temp_dir) / 'catalog'
        
        self.df1 = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0]
        })
        
        self.df2 = pd.DataFrame({
            'ticker': ['MSFT', 'AMZN', 'TSLA'],
            'price': [300.0, 3300.0, 700.0],
            'volume': [1000000, 800000, 1200000]
        })

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_create_catalog(self):
        """Test creating a data catalog"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        self.assertEqual(catalog.catalog_dir, self.catalog_dir)
        self.assertTrue(self.catalog_dir.exists())

    def test_register_dataset(self):
        """Test registering a dataset in catalog"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        metadata = catalog.register_dataset(
            data=self.df1,
            dataset_id='us_stocks',
            name='US Stocks',
            description='US stock data',
            source='CSV',
            region='US'
        )
        
        self.assertIsInstance(metadata, DatasetMetadata)
        self.assertEqual(metadata.dataset_id, 'us_stocks')
        self.assertIn('us_stocks', catalog.list_datasets())

    def test_get_dataset_metadata(self):
        """Test retrieving dataset metadata"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        # Register dataset
        catalog.register_dataset(
            data=self.df1,
            dataset_id='test_data',
            name='Test'
        )
        
        # Retrieve metadata
        metadata = catalog.get_metadata('test_data')
        
        self.assertIsInstance(metadata, DatasetMetadata)
        self.assertEqual(metadata.dataset_id, 'test_data')
        self.assertEqual(metadata.schema.num_rows, 2)

    def test_list_datasets(self):
        """Test listing all datasets in catalog"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        # Initially empty
        self.assertEqual(len(catalog.list_datasets()), 0)
        
        # Register datasets
        catalog.register_dataset(self.df1, 'dataset1', 'Dataset 1')
        catalog.register_dataset(self.df2, 'dataset2', 'Dataset 2')
        
        datasets = catalog.list_datasets()
        self.assertEqual(len(datasets), 2)
        self.assertIn('dataset1', datasets)
        self.assertIn('dataset2', datasets)

    def test_search_by_tag(self):
        """Test searching datasets by tag"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        # Register with tags
        catalog.register_dataset(
            self.df1, 'dataset1', 'Dataset 1',
            tags=['finance', 'US']
        )
        catalog.register_dataset(
            self.df2, 'dataset2', 'Dataset 2',
            tags=['finance', 'EU']
        )
        
        # Search by tag
        finance_datasets = catalog.search_by_tag('finance')
        self.assertEqual(len(finance_datasets), 2)
        
        us_datasets = catalog.search_by_tag('US')
        self.assertEqual(len(us_datasets), 1)
        self.assertEqual(us_datasets[0], 'dataset1')

    def test_search_by_source(self):
        """Test searching datasets by source"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        catalog.register_dataset(
            self.df1, 'csv_data', 'CSV Data',
            source='CSV'
        )
        catalog.register_dataset(
            self.df2, 'db_data', 'DB Data',
            source='PostgreSQL'
        )
        
        csv_datasets = catalog.search_by_source('CSV')
        self.assertEqual(len(csv_datasets), 1)
        self.assertEqual(csv_datasets[0], 'csv_data')

    def test_update_metadata(self):
        """Test updating dataset metadata"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        # Register dataset
        catalog.register_dataset(
            self.df1, 'test', 'Test',
            description='Original description'
        )
        
        # Update metadata
        updated = catalog.update_metadata(
            'test',
            description='Updated description',
            tags=['new_tag']
        )
        
        self.assertEqual(updated.description, 'Updated description')
        self.assertIn('new_tag', updated.tags)

    def test_remove_dataset(self):
        """Test removing a dataset from catalog"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        catalog.register_dataset(self.df1, 'test', 'Test')
        self.assertIn('test', catalog.list_datasets())
        
        catalog.remove_dataset('test')
        self.assertNotIn('test', catalog.list_datasets())

    def test_get_catalog_summary(self):
        """Test getting catalog summary statistics"""
        catalog = DataCatalog(catalog_dir=self.catalog_dir)
        
        catalog.register_dataset(self.df1, 'dataset1', 'Dataset 1')
        catalog.register_dataset(self.df2, 'dataset2', 'Dataset 2')
        
        summary = catalog.get_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_datasets'], 2)
        self.assertIn('total_rows', summary)
        self.assertIn('total_columns', summary)


class TestCatalogPersistence(unittest.TestCase):
    """Test suite for catalog persistence"""

    def setUp(self):
        """Create temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.catalog_dir = Path(self.temp_dir) / 'catalog'
        
        self.df = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL'],
            'price': [150.0, 2800.0]
        })

    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.temp_dir)

    def test_catalog_persistence(self):
        """Test that catalog persists across instances"""
        # Create catalog and register dataset
        catalog1 = DataCatalog(catalog_dir=self.catalog_dir)
        catalog1.register_dataset(self.df, 'test', 'Test Dataset')
        
        # Create new catalog instance pointing to same directory
        catalog2 = DataCatalog(catalog_dir=self.catalog_dir)
        
        # Should load existing catalog
        datasets = catalog2.list_datasets()
        self.assertIn('test', datasets)
        
        # Should be able to retrieve metadata
        metadata = catalog2.get_metadata('test')
        self.assertEqual(metadata.name, 'Test Dataset')


if __name__ == '__main__':
    unittest.main()
