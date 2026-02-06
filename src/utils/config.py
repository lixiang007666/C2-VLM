"""
Configuration management for C2-VLM.
"""

from typing import Any, Dict
from omegaconf import OmegaConf


class Config:
    """Configuration wrapper with dot notation access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = OmegaConf.create(config_dict)
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return super().__getattribute__(name)
        return getattr(self._config, name)
    
    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            setattr(self._config, name, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        try:
            return OmegaConf.select(self._config, key, default=default)
        except:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return OmegaConf.to_container(self._config, resolve=True)
    
    def merge(self, other_config: Dict[str, Any]):
        """Merge with another configuration."""
        self._config = OmegaConf.merge(self._config, other_config)
    
    def save(self, filepath: str):
        """Save configuration to file."""
        OmegaConf.save(self._config, filepath)