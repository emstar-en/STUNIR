"""Main configuration manager for STUNIR.

Provides a unified interface for configuration management with:
- Multiple source support (files, env vars, CLI args)
- Hierarchical merging
- Hot-reload support
- Type-safe access
"""

import os
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class ConfigSource:
    """Represents a configuration source."""
    name: str
    priority: int
    data: Dict[str, Any]
    path: Optional[str] = None
    mtime: Optional[float] = None


class Config:
    """Configuration container with dot-notation access.
    
    Example:
        config = Config({'database': {'host': 'localhost', 'port': 5432}})
        print(config.database.host)  # 'localhost'
        print(config['database.port'])  # 5432
    """
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if name not in self._data:
            raise AttributeError(f"Config has no attribute '{name}'")
        
        value = self._data[name]
        if isinstance(value, dict):
            return Config(value)
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get value by dot-notation key."""
        parts = key.split('.')
        value = self._data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        value = self[key]
        return value if value is not None else default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return deepcopy(self._data)
    
    def __repr__(self) -> str:
        return f"Config({self._data})"


class ConfigManager:
    """Central configuration manager.
    
    Features:
    - Multiple source support with priority-based merging
    - Environment variable override
    - Hot-reload for file changes
    - Validation support
    """
    
    def __init__(self, app_name: str = 'stunir'):
        self.app_name = app_name
        self._sources: List[ConfigSource] = []
        self._config: Optional[Config] = None
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[Config], None]] = []
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watch = threading.Event()
    
    def add_source(
        self,
        name: str,
        data: Dict[str, Any],
        priority: int = 50,
        path: Optional[str] = None,
    ) -> None:
        """Add a configuration source.
        
        Args:
            name: Source identifier
            data: Configuration data
            priority: Merge priority (higher = later merge)
            path: Optional file path for hot-reload
        """
        with self._lock:
            mtime = None
            if path and os.path.exists(path):
                mtime = os.path.getmtime(path)
            
            source = ConfigSource(
                name=name,
                priority=priority,
                data=data,
                path=path,
                mtime=mtime,
            )
            
            # Remove existing source with same name
            self._sources = [s for s in self._sources if s.name != name]
            self._sources.append(source)
            self._sources.sort(key=lambda s: s.priority)
            self._config = None  # Invalidate cache
    
    def load_file(
        self,
        path: str,
        priority: int = 50,
        required: bool = True,
    ) -> None:
        """Load configuration from a file.
        
        Args:
            path: File path (JSON, YAML, or TOML)
            priority: Merge priority
            required: Raise error if file not found
        """
        from .loaders import YamlLoader, JsonLoader, TomlLoader
        
        path_obj = Path(path)
        if not path_obj.exists():
            if required:
                raise FileNotFoundError(f"Config file not found: {path}")
            return
        
        suffix = path_obj.suffix.lower()
        if suffix in ('.yaml', '.yml'):
            loader = YamlLoader()
        elif suffix == '.toml':
            loader = TomlLoader()
        else:
            loader = JsonLoader()
        
        data = loader.load(path)
        self.add_source(path, data, priority=priority, path=path)
    
    def load_env(self, prefix: str = None, priority: int = 75) -> None:
        """Load configuration from environment variables.
        
        Args:
            prefix: Variable prefix (default: APP_NAME)
            priority: Merge priority (higher than files by default)
        """
        from .loaders import EnvLoader
        
        prefix = prefix or f"{self.app_name.upper()}_"
        loader = EnvLoader(prefix=prefix)
        data = loader.load()
        self.add_source('env', data, priority=priority)
    
    def load_cli(self, args: Dict[str, Any], priority: int = 100) -> None:
        """Load configuration from CLI arguments.
        
        Args:
            args: CLI argument dictionary
            priority: Merge priority (highest by default)
        """
        # Filter out None values
        data = {k: v for k, v in args.items() if v is not None}
        self.add_source('cli', data, priority=priority)
    
    def _merge_configs(self) -> Dict[str, Any]:
        """Merge all configuration sources."""
        result = {}
        for source in self._sources:
            self._deep_merge(result, source.data)
        return result
    
    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Recursively merge dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)
        return base
    
    def get_config(self) -> Config:
        """Get the merged configuration.
        
        Returns:
            Config object with all sources merged
        """
        with self._lock:
            if self._config is None:
                merged = self._merge_configs()
                self._config = Config(merged)
            return self._config
    
    def validate(self, schema: 'ConfigSchema') -> List[str]:
        """Validate configuration against a schema.
        
        Args:
            schema: Configuration schema
        
        Returns:
            List of validation errors (empty if valid)
        """
        from .validators import ConfigValidator
        validator = ConfigValidator(schema)
        return validator.validate(self.get_config().to_dict())
    
    def on_change(self, callback: Callable[[Config], None]) -> None:
        """Register a callback for configuration changes."""
        self._callbacks.append(callback)
    
    def start_watch(self, interval: float = 5.0) -> None:
        """Start watching for file changes.
        
        Args:
            interval: Check interval in seconds
        """
        if self._watch_thread is not None:
            return
        
        self._stop_watch.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True,
        )
        self._watch_thread.start()
    
    def stop_watch(self) -> None:
        """Stop watching for file changes."""
        self._stop_watch.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=5.0)
            self._watch_thread = None
    
    def _watch_loop(self, interval: float) -> None:
        """Background loop for watching file changes."""
        while not self._stop_watch.is_set():
            try:
                self._check_for_changes()
            except Exception:
                pass
            self._stop_watch.wait(interval)
    
    def _check_for_changes(self) -> None:
        """Check if any config files have changed."""
        changed = False
        
        with self._lock:
            for source in self._sources:
                if source.path and os.path.exists(source.path):
                    mtime = os.path.getmtime(source.path)
                    if source.mtime and mtime > source.mtime:
                        self.load_file(source.path, source.priority)
                        changed = True
        
        if changed:
            config = self.get_config()
            for callback in self._callbacks:
                try:
                    callback(config)
                except Exception:
                    pass


# Global config manager
_global_manager: Optional[ConfigManager] = None
_global_lock = threading.Lock()


def get_config() -> Config:
    """Get the global configuration.
    
    Returns:
        Global Config instance
    """
    global _global_manager
    with _global_lock:
        if _global_manager is None:
            _global_manager = ConfigManager()
        return _global_manager.get_config()


def load_config(
    files: Optional[List[str]] = None,
    env_prefix: Optional[str] = None,
    cli_args: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load configuration from multiple sources.
    
    Args:
        files: List of config file paths
        env_prefix: Environment variable prefix
        cli_args: CLI arguments dictionary
    
    Returns:
        Merged Config instance
    """
    global _global_manager
    with _global_lock:
        _global_manager = ConfigManager()
        
        if files:
            for i, path in enumerate(files):
                _global_manager.load_file(path, priority=50 + i, required=False)
        
        if env_prefix:
            _global_manager.load_env(prefix=env_prefix)
        
        if cli_args:
            _global_manager.load_cli(cli_args)
        
        return _global_manager.get_config()
