"""
Tests for enhanced logging module with file rotation (TDD implementation).

This test module follows strict TDD methodology:
1. Write failing tests first (RED)
2. Implement minimal code to pass (GREEN)
3. Refactor and optimize (REFACTOR)

Logging features to implement:
- Setup file logging with rotation
- Configure log levels
- Add custom formatters
- Support multiple handlers (console + file)
- Rotation based on size and time
"""
import unittest
import logging
import tempfile
import shutil
from pathlib import Path
from finance_ml.logging_config import (
    setup_file_logging,
    get_logger,
    configure_logging,
    add_file_handler,
    remove_file_handlers,
    get_log_level,
    set_log_level,
)


class TestLoggingSetup(unittest.TestCase):
    """Test basic logging setup and configuration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for log files
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
        
    def tearDown(self):
        """Clean up test environment."""
        # Remove all handlers to avoid conflicts
        remove_file_handlers()
        # Remove temporary directory
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_setup_file_logging_creates_log_file(self):
        """Test that setup_file_logging creates a log file."""
        setup_file_logging(str(self.log_file))
        logger = get_logger("test")
        logger.info("Test message")
        self.assertTrue(self.log_file.exists())
        
    def test_setup_file_logging_writes_messages(self):
        """Test that log messages are written to file."""
        setup_file_logging(str(self.log_file))
        logger = get_logger("test")
        test_message = "Test log message"
        logger.info(test_message)
        
        # Read log file and check content
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertIn(test_message, content)
        
    def test_configure_logging_with_level(self):
        """Test configuring logging with specific level."""
        configure_logging(level=logging.DEBUG, log_file=str(self.log_file))
        logger = get_logger("test")
        logger.debug("Debug message")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertIn("Debug message", content)
        
    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a logging.Logger instance."""
        logger = get_logger("test")
        self.assertIsInstance(logger, logging.Logger)
        
    def test_get_logger_with_same_name_returns_same_instance(self):
        """Test that get_logger returns the same instance for same name."""
        logger1 = get_logger("test")
        logger2 = get_logger("test")
        self.assertIs(logger1, logger2)


class TestFileRotation(unittest.TestCase):
    """Test file rotation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "rotating.log"
        
    def tearDown(self):
        """Clean up test environment."""
        remove_file_handlers()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_setup_logging_with_rotation_by_size(self):
        """Test that logging rotates files based on size."""
        # Setup with small max_bytes to trigger rotation
        setup_file_logging(
            str(self.log_file),
            max_bytes=1024,  # 1KB
            backup_count=3
        )
        logger = get_logger("test")
        
        # Write enough data to trigger rotation
        for i in range(100):
            logger.info(f"Log message number {i} with some padding text to increase size")
        
        # Check that rotation occurred (backup files should exist)
        log_dir = Path(self.log_file).parent
        log_files = list(log_dir.glob("rotating.log*"))
        # Should have main file plus at least one backup
        self.assertGreater(len(log_files), 1)
        
    def test_rotation_respects_backup_count(self):
        """Test that rotation respects backup_count limit."""
        setup_file_logging(
            str(self.log_file),
            max_bytes=500,  # Small size
            backup_count=2
        )
        logger = get_logger("test")
        
        # Write enough data to create multiple backups
        for i in range(200):
            logger.info(f"Message {i} with padding text")
        
        # Check number of backup files
        log_dir = Path(self.log_file).parent
        log_files = list(log_dir.glob("rotating.log*"))
        # Should have at most backup_count + 1 (main file + backups)
        self.assertLessEqual(len(log_files), 3)


class TestLogLevels(unittest.TestCase):
    """Test log level management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "level_test.log"
        
    def tearDown(self):
        """Clean up test environment."""
        remove_file_handlers()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_set_log_level_changes_level(self):
        """Test that set_log_level changes the logging level."""
        configure_logging(level=logging.INFO, log_file=str(self.log_file))
        set_log_level(logging.DEBUG)
        current_level = get_log_level()
        self.assertEqual(current_level, logging.DEBUG)
        
    def test_get_log_level_returns_current_level(self):
        """Test that get_log_level returns the current logging level."""
        configure_logging(level=logging.WARNING, log_file=str(self.log_file))
        level = get_log_level()
        self.assertEqual(level, logging.WARNING)
        
    def test_debug_messages_filtered_at_info_level(self):
        """Test that DEBUG messages are filtered when level is INFO."""
        configure_logging(level=logging.INFO, log_file=str(self.log_file))
        logger = get_logger("test")
        logger.debug("Debug message - should not appear")
        logger.info("Info message - should appear")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertNotIn("Debug message", content)
        self.assertIn("Info message", content)


class TestMultipleHandlers(unittest.TestCase):
    """Test multiple log handlers (console + file)."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "multi_handler.log"
        
    def tearDown(self):
        """Clean up test environment."""
        remove_file_handlers()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_add_file_handler_adds_handler(self):
        """Test that add_file_handler adds a new file handler."""
        logger = get_logger("test")
        initial_handler_count = len(logger.handlers)
        
        add_file_handler(logger, str(self.log_file))
        
        self.assertEqual(len(logger.handlers), initial_handler_count + 1)
        
    def test_remove_file_handlers_removes_all_file_handlers(self):
        """Test that remove_file_handlers removes all file handlers."""
        logger = get_logger("test")
        add_file_handler(logger, str(self.log_file))
        
        # Should have at least one handler now
        self.assertGreater(len(logger.handlers), 0)
        
        remove_file_handlers(logger)
        
        # Check that file handlers are removed
        file_handlers = [h for h in logger.handlers 
                        if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(file_handlers), 0)
        
    def test_multiple_loggers_can_write_to_same_file(self):
        """Test that multiple loggers can write to the same log file."""
        setup_file_logging(str(self.log_file))
        logger1 = get_logger("test1")
        logger2 = get_logger("test2")
        
        logger1.info("Message from logger1")
        logger2.info("Message from logger2")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertIn("Message from logger1", content)
        self.assertIn("Message from logger2", content)


class TestLogFormatting(unittest.TestCase):
    """Test log message formatting."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "format_test.log"
        
    def tearDown(self):
        """Clean up test environment."""
        remove_file_handlers()
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_log_format_includes_timestamp(self):
        """Test that log format includes timestamp."""
        setup_file_logging(str(self.log_file))
        logger = get_logger("test")
        logger.info("Test message")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        # Check for timestamp pattern (e.g., "2025-")
        self.assertRegex(content, r'\d{4}-\d{2}-\d{2}')
        
    def test_log_format_includes_level(self):
        """Test that log format includes log level."""
        setup_file_logging(str(self.log_file))
        logger = get_logger("test")
        logger.warning("Warning message")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertIn("WARNING", content)
        
    def test_log_format_includes_logger_name(self):
        """Test that log format includes logger name."""
        setup_file_logging(str(self.log_file))
        logger = get_logger("my_module")
        logger.info("Test message")
        
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertIn("my_module", content)


if __name__ == '__main__':
    unittest.main()
