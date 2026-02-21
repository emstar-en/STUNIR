"""Configuration file loaders for STUNIR.

Supports YAML, JSON, TOML, and environment variables.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader(ABC):
    """Abstract base class for configuration loaders."""
    
    @abstractmethod
    def load(self, source: Any) -> Dict[str, Any]:
        """Load configuration from source.
        
        Args:
            source: Configuration source (path, dict, etc.)
        
        Returns:
            Configuration dictionary
        """
        pass


class JsonLoader(ConfigLoader):
    """JSON configuration loader."""
    
    def load(self, source: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        with open(source, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def loads(self, content: str) -> Dict[str, Any]:
        """Load configuration from JSON string."""
        return json.loads(content)


class YamlLoader(ConfigLoader):
    """YAML configuration loader."""
    
    def load(self, source: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(source, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
    
    def loads(self, content: str) -> Dict[str, Any]:
        """Load configuration from YAML string."""
        try:
            import yaml
            return yaml.safe_load(content) or {}
        except ImportError:
            raise ImportError("PyYAML is required for YAML config: pip install pyyaml")


class TomlLoader(ConfigLoader):
    """TOML configuration loader."""
    
    def load(self, source: str) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        try:
            # Python 3.11+ has tomllib built-in
            try:
                import tomllib
                with open(source, 'rb') as f:
                    return tomllib.load(f)
            except ImportError:
                import toml
                with open(source, 'r', encoding='utf-8') as f:
                    return toml.load(f)
        except ImportError:
            raise ImportError("TOML support requires: pip install toml")
    
    def loads(self, content: str) -> Dict[str, Any]:
        """Load configuration from TOML string."""
        try:
            try:
                import tomllib
                return tomllib.loads(content)
            except ImportError:
                import toml
                return toml.loads(content)
        except ImportError:
            raise ImportError("TOML support requires: pip install toml")


class EnvLoader(ConfigLoader):
    """Environment variable configuration loader.
    
    Converts environment variables to nested configuration:
    STUNIR_DATABASE_HOST -> {'database': {'host': value}}
    """
    
    def __init__(self, prefix: str = 'STUNIR_', delimiter: str = '_'):
        self.prefix = prefix
        self.delimiter = delimiter
    
    def load(self, source: Any = None) -> Dict[str, Any]:
        """Load configuration from environment variables.
        
        Args:
            source: Ignored (reads from os.environ)
        
        Returns:
            Nested configuration dictionary
        """
        result: Dict[str, Any] = {}
        
        for key, value in os.environ.items():
            if not key.startswith(self.prefix):
                continue
            
            # Remove prefix and convert to lowercase
            config_key = key[len(self.prefix):].lower()
            parts = config_key.split(self.delimiter)
            
            # Parse value type
            parsed_value = self._parse_value(value)
            
            # Build nested structure
            current = result
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = parsed_value
        
        return result
    
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # None
        if value.lower() in ('null', 'none', ''):
            return None
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON array or object
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String
        return value


class DictLoader(ConfigLoader):
    """Dictionary configuration loader (pass-through)."""
    
    def load(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """Return the source dictionary."""
        return source.copy()


class ChainLoader(ConfigLoader):
    """Chain multiple loaders with fallback."""
    
    def __init__(self, *loaders: ConfigLoader):
        self.loaders = loaders
    
    def load(self, source: Any) -> Dict[str, Any]:
        """Try loaders in order until one succeeds."""
        for loader in self.loaders:
            try:
                return loader.load(source)
            except (FileNotFoundError, json.JSONDecodeError, ImportError):
                continue
        return {}


def auto_load(path: str) -> Dict[str, Any]:
    """Auto-detect file format and load configuration.
    
    Args:
        path: Configuration file path
    
    Returns:
        Configuration dictionary
    """
    suffix = Path(path).suffix.lower()
    
    loaders = {
        '.json': JsonLoader,
        '.yaml': YamlLoader,
        '.yml': YamlLoader,
        '.toml': TomlLoader,
    }
    
    loader_class = loaders.get(suffix, JsonLoader)
    return loader_class().load(path)
