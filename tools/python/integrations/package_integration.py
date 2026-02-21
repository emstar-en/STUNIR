"""STUNIR Package Manager Integration.

Provides integration with package managers:
- Cargo (Rust)
- pip/poetry (Python)
- Cabal/Stack (Haskell)
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class PackageInfo:
    """Package information."""
    name: str
    version: str
    dependencies: List[str]
    dev_dependencies: List[str]
    metadata: Dict[str, Any]


class BasePackageManager:
    """Base class for package manager integrations."""
    
    def __init__(self, project_path: Optional[str] = None):
        """Initialize package manager.
        
        Args:
            project_path: Path to project root
        """
        self.project_path = Path(project_path) if project_path else Path.cwd()
    
    def is_available(self) -> bool:
        """Check if package manager is available."""
        raise NotImplementedError
    
    def get_package_info(self) -> Optional[PackageInfo]:
        """Get package information."""
        raise NotImplementedError
    
    def install_dependencies(self) -> bool:
        """Install project dependencies."""
        raise NotImplementedError
    
    def build(self, release: bool = False) -> bool:
        """Build the project."""
        raise NotImplementedError


class CargoIntegration(BasePackageManager):
    """Cargo (Rust) package manager integration."""
    
    def is_available(self) -> bool:
        """Check if Cargo is available."""
        try:
            result = subprocess.run(
                ['cargo', '--version'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def has_cargo_toml(self) -> bool:
        """Check if Cargo.toml exists."""
        return (self.project_path / 'Cargo.toml').exists()
    
    def get_package_info(self) -> Optional[PackageInfo]:
        """Get package info from Cargo.toml."""
        cargo_toml = self.project_path / 'Cargo.toml'
        if not cargo_toml.exists():
            return None
        
        try:
            import tomllib
        except ImportError:
            try:
                import toml as tomllib
            except ImportError:
                # Fallback to basic parsing
                return self._parse_cargo_toml_basic(cargo_toml)
        
        try:
            with open(cargo_toml, 'rb') as f:
                data = tomllib.load(f)
            
            package = data.get('package', {})
            deps = list(data.get('dependencies', {}).keys())
            dev_deps = list(data.get('dev-dependencies', {}).keys())
            
            return PackageInfo(
                name=package.get('name', 'unknown'),
                version=package.get('version', '0.0.0'),
                dependencies=deps,
                dev_dependencies=dev_deps,
                metadata={
                    'edition': package.get('edition'),
                    'authors': package.get('authors', []),
                    'description': package.get('description'),
                },
            )
        except Exception:
            return None
    
    def _parse_cargo_toml_basic(self, cargo_toml: Path) -> Optional[PackageInfo]:
        """Basic Cargo.toml parser without external dependencies."""
        try:
            content = cargo_toml.read_text()
            name = version = 'unknown'
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('name = '):
                    name = line.split('=', 1)[1].strip().strip('"')
                elif line.startswith('version = '):
                    version = line.split('=', 1)[1].strip().strip('"')
            
            return PackageInfo(
                name=name,
                version=version,
                dependencies=[],
                dev_dependencies=[],
                metadata={},
            )
        except Exception:
            return None
    
    def install_dependencies(self) -> bool:
        """Run cargo fetch."""
        try:
            result = subprocess.run(
                ['cargo', 'fetch'],
                cwd=self.project_path,
                capture_output=True,
                timeout=300
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def build(self, release: bool = False) -> bool:
        """Build Rust project."""
        cmd = ['cargo', 'build']
        if release:
            cmd.append('--release')
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                timeout=600
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def test(self) -> bool:
        """Run cargo tests."""
        try:
            result = subprocess.run(
                ['cargo', 'test'],
                cwd=self.project_path,
                capture_output=True,
                timeout=600
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_build_config(self) -> Dict[str, Any]:
        """Get optimized build configuration."""
        return {
            'profile.release': {
                'lto': 'thin',
                'codegen-units': 1,
                'opt-level': 3,
            },
            'profile.dev': {
                'opt-level': 0,
                'debug': True,
                'incremental': True,
            },
        }


class PipIntegration(BasePackageManager):
    """pip/poetry Python package manager integration."""
    
    def is_available(self) -> bool:
        """Check if pip is available."""
        try:
            result = subprocess.run(
                ['pip', '--version'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def has_pyproject(self) -> bool:
        """Check if pyproject.toml exists."""
        return (self.project_path / 'pyproject.toml').exists()
    
    def has_setup_py(self) -> bool:
        """Check if setup.py exists."""
        return (self.project_path / 'setup.py').exists()
    
    def has_requirements(self) -> bool:
        """Check if requirements.txt exists."""
        return (self.project_path / 'requirements.txt').exists()
    
    def get_package_info(self) -> Optional[PackageInfo]:
        """Get package info from pyproject.toml or setup.py."""
        if self.has_pyproject():
            return self._parse_pyproject()
        elif self.has_setup_py():
            return self._parse_setup_py()
        return None
    
    def _parse_pyproject(self) -> Optional[PackageInfo]:
        """Parse pyproject.toml."""
        try:
            import tomllib
        except ImportError:
            try:
                import toml as tomllib
            except ImportError:
                return None
        
        try:
            with open(self.project_path / 'pyproject.toml', 'rb') as f:
                data = tomllib.load(f)
            
            project = data.get('project', {})
            poetry = data.get('tool', {}).get('poetry', {})
            
            # Try poetry format first, then PEP 621
            name = poetry.get('name') or project.get('name', 'unknown')
            version = poetry.get('version') or project.get('version', '0.0.0')
            
            deps = project.get('dependencies', [])
            if isinstance(deps, dict):
                deps = list(deps.keys())
            
            return PackageInfo(
                name=name,
                version=version,
                dependencies=deps,
                dev_dependencies=[],
                metadata={
                    'description': project.get('description'),
                    'authors': project.get('authors', []),
                    'python_requires': project.get('requires-python'),
                },
            )
        except Exception:
            return None
    
    def _parse_setup_py(self) -> Optional[PackageInfo]:
        """Basic setup.py parsing."""
        try:
            content = (self.project_path / 'setup.py').read_text()
            name = version = 'unknown'
            
            import re
            name_match = re.search(r'name=["\']([^"\']+)["\']', content)
            if name_match:
                name = name_match.group(1)
            version_match = re.search(r'version=["\']([^"\']+)["\']', content)
            if version_match:
                version = version_match.group(1)
            
            return PackageInfo(
                name=name,
                version=version,
                dependencies=[],
                dev_dependencies=[],
                metadata={},
            )
        except Exception:
            return None
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        if self.has_pyproject():
            cmd = ['pip', 'install', '-e', '.']
        elif self.has_requirements():
            cmd = ['pip', 'install', '-r', 'requirements.txt']
        else:
            return False
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                timeout=300
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def build(self, release: bool = False) -> bool:
        """Build Python package."""
        try:
            result = subprocess.run(
                ['python', '-m', 'build'],
                cwd=self.project_path,
                capture_output=True,
                timeout=300
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False


class CabalIntegration(BasePackageManager):
    """Cabal/Stack Haskell package manager integration."""
    
    def is_available(self) -> bool:
        """Check if cabal is available."""
        try:
            result = subprocess.run(
                ['cabal', '--version'],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def has_cabal_file(self) -> bool:
        """Check if a .cabal file exists."""
        return any(self.project_path.glob('*.cabal'))
    
    def has_stack_yaml(self) -> bool:
        """Check if stack.yaml exists."""
        return (self.project_path / 'stack.yaml').exists()
    
    def get_package_info(self) -> Optional[PackageInfo]:
        """Get package info from .cabal file."""
        cabal_files = list(self.project_path.glob('*.cabal'))
        if not cabal_files:
            return None
        
        try:
            content = cabal_files[0].read_text()
            name = version = 'unknown'
            deps = []
            
            for line in content.split('\n'):
                line_lower = line.lower().strip()
                if line_lower.startswith('name:'):
                    name = line.split(':', 1)[1].strip()
                elif line_lower.startswith('version:'):
                    version = line.split(':', 1)[1].strip()
                elif line_lower.startswith('build-depends:'):
                    dep_str = line.split(':', 1)[1].strip()
                    deps.extend([d.strip().split()[0] for d in dep_str.split(',') if d.strip()])
            
            return PackageInfo(
                name=name,
                version=version,
                dependencies=deps,
                dev_dependencies=[],
                metadata={},
            )
        except Exception:
            return None
    
    def install_dependencies(self) -> bool:
        """Install Haskell dependencies."""
        if self.has_stack_yaml():
            cmd = ['stack', 'setup']
        else:
            cmd = ['cabal', 'update']
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                timeout=600
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def build(self, release: bool = False) -> bool:
        """Build Haskell project."""
        if self.has_stack_yaml():
            cmd = ['stack', 'build']
            if release:
                cmd.append('--ghc-options=-O2')
        else:
            cmd = ['cabal', 'build']
            if release:
                cmd.extend(['-O2'])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_path,
                capture_output=True,
                timeout=600
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def get_build_config(self) -> Dict[str, Any]:
        """Get optimized build configuration."""
        return {
            'ghc-options': ['-O2', '-threaded', '-rtsopts'],
            'library-profiling': False,
            'executable-profiling': False,
        }


def detect_package_manager(project_path: Optional[str] = None) -> Dict[str, bool]:
    """Detect available package managers for a project.
    
    Args:
        project_path: Path to project
        
    Returns:
        Dictionary with package manager availability
    """
    cargo = CargoIntegration(project_path)
    pip = PipIntegration(project_path)
    cabal = CabalIntegration(project_path)
    
    return {
        'cargo': cargo.is_available() and cargo.has_cargo_toml(),
        'pip': pip.is_available() and (pip.has_pyproject() or pip.has_setup_py() or pip.has_requirements()),
        'cabal': cabal.is_available() and cabal.has_cabal_file(),
        'stack': cabal.has_stack_yaml(),
    }
