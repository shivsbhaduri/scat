"""
Core Configuration Module

Handles loading configuration from config.yaml and managing model versions.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OpenRouterConfig:
    """OpenRouter API configuration."""
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1"


@dataclass
class ThresholdsConfig:
    """Classification confidence thresholds."""
    hybrid_confidence: float = 0.75
    llm_fallback: float = 0.60


@dataclass
class VersionConfig:
    """Version-specific paths and metadata."""
    version: str
    category_tree_path: Path
    hybrid_index_path: Path
    xgboost_model_path: Path
    label_encoder_path: Path
    feature_names_path: Path
    feature_config_path: Path          # NEW
    metadata_path: Path
    
    @classmethod
    def from_version(cls, version: str, base_path: Path = None):
        """Create VersionConfig from version string."""
        if base_path is None:
            base_path = Path("data/versions")
        
        version_path = base_path / version
        
        return cls(
            version=version,
            category_tree_path=version_path / "category_tree.json",
            hybrid_index_path=version_path / "hybrid_index",
            xgboost_model_path=version_path / "xgboost_model.json",
            label_encoder_path=version_path / "label_encoder.json",
            feature_names_path=version_path / "feature_names.json",
            feature_config_path=version_path / "feature_config.json",  # NEW
            metadata_path=version_path / "metadata.json"
        )
    
    def validate(self) -> bool:
        """Check if all required files exist for this version."""
        required_paths = [
            self.category_tree_path,
            self.xgboost_model_path,
            self.label_encoder_path,
            self.feature_names_path,
            self.feature_config_path       # NEW
        ]
        
        # Hybrid index is a directory
        if not self.hybrid_index_path.exists():
            return False
        
        for path in required_paths:
            if not path.exists():
                return False
        
        return True


class Config:
    """Application configuration manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load YAML config
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)
        
        # Parse configurations
        self.openrouter = self._parse_openrouter_config()
        self.thresholds = self._parse_thresholds_config()
        self.default_version = self._config.get('default_version', 'v1.0')
        
        # Cache for version configs
        self._version_cache: Dict[str, VersionConfig] = {}
    
    def _parse_openrouter_config(self) -> OpenRouterConfig:
        """Parse OpenRouter configuration."""
        openrouter_config = self._config.get('openrouter', {})
        
        api_key = openrouter_config.get('api_key')
        if not api_key:
            raise ValueError("OpenRouter API key not found in config")
        
        return OpenRouterConfig(
            api_key=api_key,
            model=openrouter_config.get('model', 'anthropic/claude-3.5-sonnet'),
            base_url=openrouter_config.get('base_url', 'https://openrouter.ai/api/v1')
        )
    
    def _parse_thresholds_config(self) -> ThresholdsConfig:
        """Parse classification thresholds."""
        thresholds = self._config.get('thresholds', {})
        
        return ThresholdsConfig(
            hybrid_confidence=thresholds.get('hybrid_confidence', 0.75),
            llm_fallback=thresholds.get('llm_fallback', 0.60)
        )
    
    def get_version_config(self, version: Optional[str] = None) -> VersionConfig:
        """
        Get version-specific configuration.
        
        Args:
            version: Version string (e.g., 'v1.0'). Uses default if not provided.
        
        Returns:
            VersionConfig for the specified version
        
        Raises:
            ValueError: If version files don't exist or are incomplete
        """
        if version is None:
            version = self.default_version
        
        # Check cache
        if version in self._version_cache:
            return self._version_cache[version]
        
        # Create version config
        version_config = VersionConfig.from_version(version)
        
        # Validate
        if not version_config.validate():
            raise ValueError(
                f"Version {version} is incomplete or not found. "
                f"Run training script first: python scripts/train_models.py --version {version}"
            )
        
        # Cache and return
        self._version_cache[version] = version_config
        return version_config
    
    def list_available_versions(self) -> list[str]:
        """List all available trained versions."""
        versions_path = Path("data/versions")
        
        if not versions_path.exists():
            return []
        
        available = []
        for version_dir in versions_path.iterdir():
            if version_dir.is_dir():
                version_config = VersionConfig.from_version(version_dir.name)
                if version_config.validate():
                    available.append(version_dir.name)
        
        return sorted(available)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get raw configuration value.
        
        Args:
            key: Configuration key (supports nested keys with dots, e.g., 'training.test_split')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value


# Global config instance (lazy loaded)
_config_instance: Optional[Config] = None


def get_config(config_path: str = "config/config.yaml") -> Config:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_path: Path to config file (only used on first call)
    
    Returns:
        Global Config instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_path)
    
    return _config_instance


def reload_config(config_path: str = "config/config.yaml") -> Config:
    """
    Force reload of configuration (useful for testing or config updates).
    
    Args:
        config_path: Path to config file
    
    Returns:
        New Config instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance