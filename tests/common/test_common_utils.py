"""STUNIR Common Utilities Tests.

Tests for shared utility functions.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.common import (
    canonical_json,
    canonical_json_pretty,
    parse_json,
    compute_sha256,
    compute_file_hash,
    safe_read_file,
    safe_write_file,
    atomic_write,
    scan_directory,
    ensure_directory,
    normalize_path,
    join_paths,
    get_extension,
    change_extension,
)
from tools.common.cache import (
    LRUCache,
    cached_file_hash,
    lru_cache,
    get_cache_stats,
    clear_all_caches,
)
from tools.common.batch_ops import (
    batch_hash_files,
    batch_read_json,
    efficient_directory_walk,
)


class TestJsonUtils(unittest.TestCase):
    """Test JSON utility functions."""
    
    def test_canonical_json_sorting(self):
        """Test that keys are sorted."""
        data = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(data)
        self.assertEqual(result, '{"a":2,"m":3,"z":1}')
    
    def test_canonical_json_no_whitespace(self):
        """Test that no unnecessary whitespace is added."""
        data = {"key": "value", "nested": {"a": 1}}
        result = canonical_json(data)
        self.assertNotIn(' ', result.replace('"key"', '').replace('"value"', '').replace('"nested"', ''))
    
    def test_canonical_json_deterministic(self):
        """Test that output is deterministic."""
        data = {"b": 2, "a": 1, "c": 3}
        result1 = canonical_json(data)
        result2 = canonical_json(data)
        self.assertEqual(result1, result2)
    
    def test_canonical_json_pretty(self):
        """Test pretty printing."""
        data = {"key": "value"}
        result = canonical_json_pretty(data)
        self.assertIn('\n', result)
        self.assertIn('  ', result)  # Indentation
    
    def test_parse_json_valid(self):
        """Test parsing valid JSON."""
        result = parse_json('{"key": "value"}')
        self.assertEqual(result, {"key": "value"})
    
    def test_parse_json_bytes(self):
        """Test parsing JSON from bytes."""
        result = parse_json(b'{"key": "value"}')
        self.assertEqual(result, {"key": "value"})
    
    def test_parse_json_invalid_strict(self):
        """Test parsing invalid JSON in strict mode."""
        with self.assertRaises(json.JSONDecodeError):
            parse_json('invalid json', strict=True)
    
    def test_parse_json_invalid_lenient(self):
        """Test parsing invalid JSON in lenient mode."""
        result = parse_json('invalid json', strict=False)
        self.assertIsNone(result)


class TestHashUtils(unittest.TestCase):
    """Test hash utility functions."""
    
    def test_compute_sha256_string(self):
        """Test hashing a string."""
        result = compute_sha256("hello")
        # Known SHA256 of "hello"
        self.assertEqual(result, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    
    def test_compute_sha256_bytes(self):
        """Test hashing bytes."""
        result = compute_sha256(b"hello")
        self.assertEqual(result, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    
    def test_compute_sha256_deterministic(self):
        """Test that hashing is deterministic."""
        data = "test data"
        result1 = compute_sha256(data)
        result2 = compute_sha256(data)
        self.assertEqual(result1, result2)
    
    def test_compute_file_hash(self):
        """Test file hashing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello")
            filepath = f.name
        
        try:
            result = compute_file_hash(filepath)
            self.assertEqual(result, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
        finally:
            os.unlink(filepath)
    
    def test_hash_length(self):
        """Test hash output length."""
        result = compute_sha256("test")
        self.assertEqual(len(result), 64)  # SHA256 is 64 hex chars


class TestFileUtils(unittest.TestCase):
    """Test file utility functions."""
    
    def test_safe_read_file_exists(self):
        """Test reading existing file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            filepath = f.name
        
        try:
            content = safe_read_file(filepath)
            self.assertEqual(content, "test content")
        finally:
            os.unlink(filepath)
    
    def test_safe_read_file_not_exists(self):
        """Test reading non-existent file."""
        content = safe_read_file('/nonexistent/path/file.txt')
        self.assertIsNone(content)
    
    def test_safe_write_file(self):
        """Test writing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'subdir', 'test.txt')
            result = safe_write_file(filepath, "test content", create_dirs=True)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(filepath))
    
    def test_atomic_write(self):
        """Test atomic file writing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.txt')
            
            with atomic_write(filepath) as f:
                f.write("atomic content")
            
            with open(filepath, 'r') as f:
                content = f.read()
            self.assertEqual(content, "atomic content")
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'new', 'nested', 'dir')
            result = ensure_directory(new_dir)
            self.assertTrue(result.exists())
            self.assertTrue(result.is_dir())
    
    def test_scan_directory(self):
        """Test directory scanning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for name in ['test1.txt', 'test2.json', 'test3.py']:
                with open(os.path.join(tmpdir, name), 'w') as f:
                    f.write('test')
            
            # Scan all files
            files = list(scan_directory(tmpdir))
            self.assertEqual(len(files), 3)
            
            # Scan with extension filter
            txt_files = list(scan_directory(tmpdir, extensions=['.txt']))
            self.assertEqual(len(txt_files), 1)


class TestPathUtils(unittest.TestCase):
    """Test path utility functions."""
    
    def test_normalize_path(self):
        """Test path normalization."""
        # Result should use forward slashes
        result = normalize_path('/home/user/./file.txt')
        self.assertIn('/', result)
        self.assertNotIn('./', result)
    
    def test_join_paths(self):
        """Test path joining."""
        result = join_paths('/home', 'user', 'file.txt')
        self.assertIn('home', result)
        self.assertIn('user', result)
        self.assertIn('file.txt', result)
    
    def test_get_extension(self):
        """Test extension extraction."""
        self.assertEqual(get_extension('file.txt'), '.txt')
        self.assertEqual(get_extension('file.tar.gz'), '.gz')
        self.assertEqual(get_extension('noext'), '')
    
    def test_change_extension(self):
        """Test extension changing."""
        result = change_extension('file.txt', '.json')
        self.assertTrue(result.endswith('.json'))
        
        result = change_extension('file.txt', 'json')  # Without dot
        self.assertTrue(result.endswith('.json'))


class TestCache(unittest.TestCase):
    """Test caching utilities."""
    
    def test_lru_cache_basic(self):
        """Test basic LRU cache operations."""
        cache = LRUCache(maxsize=3)
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)
        
        self.assertEqual(cache.get('a'), 1)
        self.assertEqual(cache.get('b'), 2)
        self.assertEqual(cache.get('c'), 3)
    
    def test_lru_cache_eviction(self):
        """Test LRU cache eviction."""
        cache = LRUCache(maxsize=2)
        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)  # Should evict 'a'
        
        self.assertIsNone(cache.get('a'))
        self.assertEqual(cache.get('b'), 2)
        self.assertEqual(cache.get('c'), 3)
    
    def test_lru_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(maxsize=10)
        cache.set('a', 1)
        cache.get('a')  # Hit
        cache.get('b')  # Miss
        
        stats = cache.stats
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
    
    def test_lru_cache_decorator(self):
        """Test lru_cache decorator."""
        call_count = [0]
        
        @lru_cache(maxsize=10)
        def expensive_func(x):
            call_count[0] += 1
            return x * 2
        
        result1 = expensive_func(5)
        result2 = expensive_func(5)  # Should be cached
        
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count[0], 1)  # Only called once
    
    def test_cached_file_hash(self):
        """Test cached file hashing."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            filepath = f.name
        
        try:
            # First call computes hash
            hash1 = cached_file_hash(filepath)
            # Second call should use cache
            hash2 = cached_file_hash(filepath)
            
            self.assertEqual(hash1, hash2)
        finally:
            os.unlink(filepath)
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        clear_all_caches()
        stats = get_cache_stats()
        
        for cache_name, cache_stats in stats.items():
            self.assertEqual(cache_stats['size'], 0)


class TestBatchOps(unittest.TestCase):
    """Test batch operations."""
    
    def test_batch_hash_files(self):
        """Test batch file hashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(3):
                filepath = os.path.join(tmpdir, f'file{i}.txt')
                with open(filepath, 'w') as f:
                    f.write(f'content {i}')
                files.append(filepath)
            
            results = batch_hash_files(files)
            self.assertEqual(len(results), 3)
            for path in files:
                self.assertIn(path, results)
                self.assertEqual(len(results[path]), 64)  # SHA256 length
    
    def test_batch_read_json(self):
        """Test batch JSON reading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = []
            for i in range(3):
                filepath = os.path.join(tmpdir, f'file{i}.json')
                with open(filepath, 'w') as f:
                    json.dump({"id": i}, f)
                files.append(filepath)
            
            results = batch_read_json(files)
            self.assertEqual(len(results), 3)
            for path in files:
                self.assertIn(path, results)
                self.assertIsNotNone(results[path])
    
    def test_efficient_directory_walk(self):
        """Test efficient directory walking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            os.makedirs(os.path.join(tmpdir, 'sub1'))
            for name in ['a.txt', 'b.py', 'sub1/c.txt']:
                with open(os.path.join(tmpdir, name), 'w') as f:
                    f.write('test')
            
            all_files = list(efficient_directory_walk(tmpdir))
            self.assertEqual(len(all_files), 3)
            
            txt_files = list(efficient_directory_walk(tmpdir, extensions=['.txt']))
            self.assertEqual(len(txt_files), 2)


if __name__ == '__main__':
    unittest.main()
