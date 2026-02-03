"""STUNIR Git Integration.

Provides Git repository integration for:
- Commit hash tracking
- Branch detection
- Dirty working tree detection
- Git metadata in receipts
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class GitInfo:
    """Git repository information."""
    commit_hash: str
    short_hash: str
    branch: str
    is_dirty: bool
    author: str
    author_email: str
    commit_date: str
    commit_message: str
    tags: List[str]
    remote_url: Optional[str]


class GitIntegration:
    """Git repository integration."""
    
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize Git integration.
        
        Args:
            repo_path: Path to repository root (default: current directory)
        """
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self._git_available: Optional[bool] = None
    
    def _run_git(self, *args: str, check: bool = True) -> Optional[str]:
        """Run a git command and return output.
        
        Args:
            *args: Git command arguments
            check: If True, return None on error; if False, raise exception
            
        Returns:
            Command output or None on error
        """
        try:
            result = subprocess.run(
                ['git', '-C', str(self.repo_path)] + list(args),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            elif check:
                return None
            else:
                raise RuntimeError(f"Git command failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None
    
    def is_git_available(self) -> bool:
        """Check if git is available."""
        if self._git_available is None:
            result = self._run_git('--version')
            self._git_available = result is not None
        return self._git_available
    
    def is_git_repo(self) -> bool:
        """Check if current path is a git repository."""
        return self._run_git('rev-parse', '--git-dir') is not None
    
    def get_commit_hash(self, short: bool = False) -> Optional[str]:
        """Get current commit hash.
        
        Args:
            short: If True, return short (7-char) hash
            
        Returns:
            Commit hash or None if not in a git repo
        """
        if short:
            return self._run_git('rev-parse', '--short', 'HEAD')
        return self._run_git('rev-parse', 'HEAD')
    
    def get_branch_name(self) -> Optional[str]:
        """Get current branch name.
        
        Returns:
            Branch name or None if detached HEAD or not in git repo
        """
        # Try symbolic-ref first (works for normal branches)
        branch = self._run_git('symbolic-ref', '--short', 'HEAD')
        if branch:
            return branch
        # Fallback for detached HEAD
        return self._run_git('describe', '--tags', '--always')
    
    def is_dirty(self) -> bool:
        """Check if working tree has uncommitted changes.
        
        Returns:
            True if there are uncommitted changes
        """
        # Check for staged and unstaged changes
        status = self._run_git('status', '--porcelain')
        return bool(status)
    
    def get_author(self) -> Optional[str]:
        """Get commit author name."""
        return self._run_git('log', '-1', '--format=%an')
    
    def get_author_email(self) -> Optional[str]:
        """Get commit author email."""
        return self._run_git('log', '-1', '--format=%ae')
    
    def get_commit_date(self, iso: bool = True) -> Optional[str]:
        """Get commit date.
        
        Args:
            iso: If True, return ISO 8601 format
        """
        fmt = '%aI' if iso else '%ad'
        return self._run_git('log', '-1', f'--format={fmt}')
    
    def get_commit_message(self, full: bool = False) -> Optional[str]:
        """Get commit message.
        
        Args:
            full: If True, include full body; otherwise just subject
        """
        fmt = '%B' if full else '%s'
        return self._run_git('log', '-1', f'--format={fmt}')
    
    def get_tags(self) -> List[str]:
        """Get tags pointing to current commit."""
        tags = self._run_git('tag', '--points-at', 'HEAD')
        if tags:
            return tags.split('\n')
        return []
    
    def get_remote_url(self, remote: str = 'origin') -> Optional[str]:
        """Get remote URL.
        
        Args:
            remote: Remote name (default: origin)
        """
        return self._run_git('remote', 'get-url', remote)
    
    def get_info(self) -> Optional[GitInfo]:
        """Get comprehensive git information.
        
        Returns:
            GitInfo object or None if not in a git repo
        """
        if not self.is_git_repo():
            return None
        
        return GitInfo(
            commit_hash=self.get_commit_hash() or '',
            short_hash=self.get_commit_hash(short=True) or '',
            branch=self.get_branch_name() or 'HEAD',
            is_dirty=self.is_dirty(),
            author=self.get_author() or '',
            author_email=self.get_author_email() or '',
            commit_date=self.get_commit_date() or '',
            commit_message=self.get_commit_message() or '',
            tags=self.get_tags(),
            remote_url=self.get_remote_url(),
        )
    
    def get_metadata_for_receipt(self) -> Dict[str, Any]:
        """Get git metadata suitable for including in receipts.
        
        Returns:
            Dictionary with git metadata
        """
        info = self.get_info()
        if not info:
            return {}
        
        return {
            'git_commit': info.commit_hash,
            'git_commit_short': info.short_hash,
            'git_branch': info.branch,
            'git_dirty': info.is_dirty,
            'git_author': info.author,
            'git_date': info.commit_date,
            'git_tags': info.tags,
        }
    
    def get_diff_stats(self) -> Dict[str, int]:
        """Get diff statistics.
        
        Returns:
            Dictionary with files_changed, insertions, deletions
        """
        shortstat = self._run_git('diff', '--shortstat')
        if not shortstat:
            return {'files_changed': 0, 'insertions': 0, 'deletions': 0}
        
        import re
        files = insertions = deletions = 0
        
        m = re.search(r'(\d+) files? changed', shortstat)
        if m:
            files = int(m.group(1))
        m = re.search(r'(\d+) insertions?', shortstat)
        if m:
            insertions = int(m.group(1))
        m = re.search(r'(\d+) deletions?', shortstat)
        if m:
            deletions = int(m.group(1))
        
        return {
            'files_changed': files,
            'insertions': insertions,
            'deletions': deletions,
        }


# Module-level convenience functions
def get_git_info(repo_path: Optional[str] = None) -> Optional[GitInfo]:
    """Get git information for a repository."""
    return GitIntegration(repo_path).get_info()


def get_commit_hash(repo_path: Optional[str] = None, short: bool = False) -> Optional[str]:
    """Get current commit hash."""
    return GitIntegration(repo_path).get_commit_hash(short=short)


def get_branch_name(repo_path: Optional[str] = None) -> Optional[str]:
    """Get current branch name."""
    return GitIntegration(repo_path).get_branch_name()


def is_dirty(repo_path: Optional[str] = None) -> bool:
    """Check if repository has uncommitted changes."""
    return GitIntegration(repo_path).is_dirty()


def get_git_metadata(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """Get git metadata for receipts."""
    return GitIntegration(repo_path).get_metadata_for_receipt()
