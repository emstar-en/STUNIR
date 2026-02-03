"""STUNIR Platform Compatibility Tests.

Tests for cross-platform functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.platform import (
    get_platform,
    is_windows,
    is_macos,
    is_linux,
    is_unix,
    get_platform_info,
    normalize_line_endings,
    to_unix_endings,
    to_windows_endings,
    set_executable,
    is_executable,
    find_executable,
    get_temp_dir,
    create_temp_file,
    create_temp_dir,
    get_env_path_separator,
    split_env_path,
)


class TestPlatformDetection(unittest.TestCase):
    """Test platform detection functions."""
    
    def test_get_platform_returns_valid(self):
        """Test that get_platform returns a valid platform."""
        platform = get_platform()
        self.assertIn(platform, ['windows', 'macos', 'linux', 'unknown'])
    
    def test_platform_flags_consistent(self):
        """Test that platform flags are mutually exclusive."""
        flags = [is_windows(), is_macos(), is_linux()]
        # Exactly one should be true (or none if unknown)
        true_count = sum(flags)
        self.assertLessEqual(true_count, 1)
    
    def test_unix_includes_linux_and_macos(self):
        """Test that is_unix is true for linux and macos."""
        if is_linux() or is_macos():
            self.assertTrue(is_unix())
        if is_windows():
            self.assertFalse(is_unix())
    
    def test_get_platform_info_structure(self):
        """Test platform info dictionary structure."""
        info = get_platform_info()
        expected_keys = [
            'platform', 'platform_detail', 'python_version',
            'architecture', 'path_separator', 'line_separator',
            'env_separator', 'cwd'
        ]
        for key in expected_keys:
            self.assertIn(key, info)


class TestLineEndings(unittest.TestCase):
    """Test line ending handling."""
    
    def test_to_unix_endings(self):
        """Test conversion to Unix line endings."""
        windows_text = "line1\r\nline2\r\nline3"
        unix_text = to_unix_endings(windows_text)
        self.assertEqual(unix_text, "line1\nline2\nline3")
        self.assertNotIn('\r', unix_text)
    
    def test_to_windows_endings(self):
        """Test conversion to Windows line endings."""
        unix_text = "line1\nline2\nline3"
        windows_text = to_windows_endings(unix_text)
        self.assertEqual(windows_text, "line1\r\nline2\r\nline3")
        self.assertEqual(windows_text.count('\r\n'), 2)
    
    def test_normalize_handles_mixed(self):
        """Test normalization of mixed line endings."""
        mixed = "line1\r\nline2\nline3\rline4"
        normalized = normalize_line_endings(mixed, to_unix=True)
        self.assertEqual(normalized, "line1\nline2\nline3\nline4")
    
    def test_idempotent_normalization(self):
        """Test that normalization is idempotent."""
        text = "line1\nline2\nline3"
        once = to_unix_endings(text)
        twice = to_unix_endings(once)
        self.assertEqual(once, twice)


class TestFilePermissions(unittest.TestCase):
    """Test file permission handling."""
    
    def test_set_executable(self):
        """Test setting executable permission."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"#!/bin/bash\necho hello")
            filepath = f.name
        
        try:
            # Set executable
            result = set_executable(filepath, True)
            self.assertTrue(result)
            
            # Check if executable (Unix only)
            if is_unix():
                self.assertTrue(is_executable(filepath))
            
            # Unset executable
            result = set_executable(filepath, False)
            self.assertTrue(result)
            
            if is_unix():
                self.assertFalse(is_executable(filepath))
        finally:
            os.unlink(filepath)


class TestExecutableDetection(unittest.TestCase):
    """Test executable detection."""
    
    def test_find_python(self):
        """Test finding Python executable."""
        python = find_executable('python3') or find_executable('python')
        self.assertIsNotNone(python)
        self.assertTrue(Path(python).exists())
    
    def test_find_nonexistent(self):
        """Test finding non-existent executable."""
        result = find_executable('definitely_not_a_real_program_xyz123')
        self.assertIsNone(result)


class TestTempDirectories(unittest.TestCase):
    """Test temporary file/directory handling."""
    
    def test_get_temp_dir_exists(self):
        """Test that temp dir exists."""
        temp = get_temp_dir()
        self.assertTrue(Path(temp).exists())
        self.assertTrue(Path(temp).is_dir())
    
    def test_create_temp_file(self):
        """Test creating temporary file."""
        fd, path = create_temp_file(suffix='.txt', prefix='test_')
        try:
            self.assertTrue(Path(path).exists())
            self.assertTrue(path.endswith('.txt'))
            self.assertIn('test_', path)
        finally:
            os.close(fd)
            os.unlink(path)
    
    def test_create_temp_dir(self):
        """Test creating temporary directory."""
        temp_dir = create_temp_dir(prefix='test_')
        try:
            self.assertTrue(Path(temp_dir).exists())
            self.assertTrue(Path(temp_dir).is_dir())
        finally:
            os.rmdir(temp_dir)


class TestEnvironment(unittest.TestCase):
    """Test environment path handling."""
    
    def test_path_separator_valid(self):
        """Test that path separator is valid."""
        sep = get_env_path_separator()
        if is_windows():
            self.assertEqual(sep, ';')
        else:
            self.assertEqual(sep, ':')
    
    def test_split_env_path(self):
        """Test splitting PATH variable."""
        paths = split_env_path()
        self.assertIsInstance(paths, list)
        # Should have at least one path
        self.assertGreater(len(paths), 0)


if __name__ == '__main__':
    unittest.main()
