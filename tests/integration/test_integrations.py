"""STUNIR External Integrations Tests.

Tests for Git, CI/CD, and package manager integrations.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.integrations import (
    GitIntegration,
    get_git_info,
    get_commit_hash,
    get_branch_name,
    is_dirty,
    CIIntegration,
    detect_ci_environment,
    is_ci,
    get_ci_metadata,
    generate_github_actions_workflow,
    generate_gitlab_ci_config,
    CargoIntegration,
    PipIntegration,
    CabalIntegration,
    detect_package_manager,
)
from tools.integrations.ci_integration import CIPlatform


class TestGitIntegration(unittest.TestCase):
    """Test Git integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.git = GitIntegration()
    
    def test_git_available(self):
        """Test Git availability check."""
        # Git should be available on most systems
        available = self.git.is_git_available()
        self.assertIsInstance(available, bool)
    
    def test_commit_hash_format(self):
        """Test commit hash format."""
        if not self.git.is_git_repo():
            self.skipTest("Not in a git repository")
        
        full_hash = self.git.get_commit_hash()
        if full_hash:
            self.assertEqual(len(full_hash), 40)
            self.assertTrue(all(c in '0123456789abcdef' for c in full_hash))
        
        short_hash = self.git.get_commit_hash(short=True)
        if short_hash:
            self.assertLessEqual(len(short_hash), 12)
    
    def test_branch_name(self):
        """Test branch name retrieval."""
        if not self.git.is_git_repo():
            self.skipTest("Not in a git repository")
        
        branch = self.git.get_branch_name()
        if branch:
            self.assertIsInstance(branch, str)
            self.assertGreater(len(branch), 0)
    
    def test_is_dirty(self):
        """Test dirty working tree detection."""
        if not self.git.is_git_repo():
            self.skipTest("Not in a git repository")
        
        dirty = self.git.is_dirty()
        self.assertIsInstance(dirty, bool)
    
    def test_get_info(self):
        """Test comprehensive git info."""
        if not self.git.is_git_repo():
            self.skipTest("Not in a git repository")
        
        info = self.git.get_info()
        if info:
            self.assertIsNotNone(info.commit_hash)
            self.assertIsNotNone(info.branch)
            self.assertIsInstance(info.is_dirty, bool)
    
    def test_metadata_for_receipt(self):
        """Test metadata generation for receipts."""
        if not self.git.is_git_repo():
            self.skipTest("Not in a git repository")
        
        metadata = self.git.get_metadata_for_receipt()
        self.assertIsInstance(metadata, dict)
        if metadata:  # Only check if we got metadata
            self.assertIn('git_commit', metadata)
            self.assertIn('git_branch', metadata)


class TestCIIntegration(unittest.TestCase):
    """Test CI/CD integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ci = CIIntegration()
    
    def test_detect_platform(self):
        """Test CI platform detection."""
        platform = self.ci.detect_platform()
        self.assertIsInstance(platform, CIPlatform)
    
    def test_is_ci_returns_bool(self):
        """Test is_ci returns boolean."""
        result = self.ci.is_ci()
        self.assertIsInstance(result, bool)
    
    def test_get_metadata(self):
        """Test CI metadata retrieval."""
        metadata = self.ci.get_metadata()
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata.platform, CIPlatform)
    
    def test_metadata_dict(self):
        """Test CI metadata as dictionary."""
        metadata_dict = self.ci.get_metadata_dict()
        self.assertIsInstance(metadata_dict, dict)
        self.assertIn('ci_platform', metadata_dict)
    
    def test_generate_github_actions(self):
        """Test GitHub Actions workflow generation."""
        workflow = generate_github_actions_workflow()
        self.assertIn('name:', workflow)
        self.assertIn('on:', workflow)
        self.assertIn('jobs:', workflow)
        self.assertIn('pytest', workflow)
    
    def test_generate_gitlab_ci(self):
        """Test GitLab CI config generation."""
        config = generate_gitlab_ci_config()
        self.assertIn('stages:', config)
        self.assertIn('test:', config)
        self.assertIn('build:', config)


class TestPackageIntegration(unittest.TestCase):
    """Test package manager integration functionality."""
    
    def test_cargo_integration_available(self):
        """Test Cargo availability check."""
        cargo = CargoIntegration()
        available = cargo.is_available()
        self.assertIsInstance(available, bool)
    
    def test_pip_integration_available(self):
        """Test pip availability check."""
        pip = PipIntegration()
        available = pip.is_available()
        self.assertTrue(available)  # pip should always be available in Python
    
    def test_cabal_integration_available(self):
        """Test Cabal availability check."""
        cabal = CabalIntegration()
        available = cabal.is_available()
        self.assertIsInstance(available, bool)
    
    def test_detect_package_manager(self):
        """Test package manager detection."""
        managers = detect_package_manager()
        self.assertIsInstance(managers, dict)
        self.assertIn('pip', managers)
        self.assertIn('cargo', managers)
        self.assertIn('cabal', managers)
    
    def test_cargo_build_config(self):
        """Test Cargo build configuration."""
        cargo = CargoIntegration()
        config = cargo.get_build_config()
        self.assertIsInstance(config, dict)
        self.assertIn('profile.release', config)
        self.assertIn('profile.dev', config)
    
    def test_pip_project_detection(self):
        """Test pip project detection."""
        repo_root = Path(__file__).parent.parent.parent
        pip = PipIntegration(str(repo_root))
        # We just created pyproject.toml
        self.assertTrue(pip.has_pyproject())


class TestConvenienceFunctions(unittest.TestCase):
    """Test module-level convenience functions."""
    
    def test_get_git_info_function(self):
        """Test get_git_info function."""
        info = get_git_info()
        # May be None if not in a git repo
        if info:
            self.assertIsNotNone(info.commit_hash)
    
    def test_detect_ci_environment_function(self):
        """Test detect_ci_environment function."""
        platform = detect_ci_environment()
        self.assertIsInstance(platform, CIPlatform)
    
    def test_is_ci_function(self):
        """Test is_ci function."""
        result = is_ci()
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
