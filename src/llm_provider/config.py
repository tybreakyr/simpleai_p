"""
Configuration system for LLM provider factory.
"""

from typing import Dict, Any, Optional
import json

from .models import FactoryConfig, ProviderConfig
from .errors import ValidationError, InvalidConfigurationError


def validate_factory_config(config: FactoryConfig) -> None:
    """
    Validate a factory configuration.
    
    Args:
        config: Factory configuration to validate
        
    Raises:
        ValidationError: If configuration is invalid
    """
    field_errors: Dict[str, str] = {}
    
    # Validate default provider
    if not config.default_provider:
        field_errors["default_provider"] = "Default provider cannot be empty"
    
    # Validate provider configs
    if not config.provider_configs:
        field_errors["provider_configs"] = "At least one provider configuration is required"
    else:
        # Validate default provider exists
        if config.default_provider not in config.provider_configs:
            field_errors["default_provider"] = (
                f"Default provider '{config.default_provider}' must exist in provider_configs"
            )
        
        # Validate each provider config
        for name, provider_config in config.provider_configs.items():
            try:
                validate_provider_config(provider_config)
            except ValidationError as e:
                field_errors[f"provider_configs.{name}"] = str(e)
    
    if field_errors:
        raise ValidationError(
            message="Factory configuration validation failed",
            field_errors=field_errors
        )


def validate_provider_config(config: ProviderConfig) -> None:
    """
    Validate a provider configuration.
    
    Args:
        config: Provider configuration to validate
        
    Raises:
        ValidationError: If configuration is invalid
    """
    field_errors: Dict[str, str] = {}
    
    # Validate host
    if not config.host:
        field_errors["host"] = "Provider host cannot be empty"
    
    # Validate default model
    if not config.default_model:
        field_errors["default_model"] = "Provider default model cannot be empty"
    
    # Validate timeout
    if config.timeout <= 0:
        field_errors["timeout"] = f"Provider timeout must be positive, got {config.timeout}"
    
    # Validate retry attempts
    if config.retry_attempts < 0:
        field_errors["retry_attempts"] = (
            f"Provider retry attempts must be non-negative, got {config.retry_attempts}"
        )
    
    # Validate rate limit if specified
    if config.rate_limit is not None and config.rate_limit <= 0:
        field_errors["rate_limit"] = (
            f"Provider rate limit must be positive if specified, got {config.rate_limit}"
        )
    
    if field_errors:
        raise ValidationError(
            message="Provider configuration validation failed",
            field_errors=field_errors
        )


def load_factory_config_from_dict(config_dict: Dict[str, Any]) -> FactoryConfig:
    """
    Load factory configuration from a dictionary.
    
    Args:
        config_dict: Dictionary containing configuration
        
    Returns:
        FactoryConfig instance
        
    Raises:
        InvalidConfigurationError: If configuration is invalid
    """
    try:
        # Extract default provider
        default_provider = config_dict.get("default_provider", "")
        
        # Extract provider configs
        provider_configs_dict = config_dict.get("provider_configs", {})
        provider_configs: Dict[str, ProviderConfig] = {}
        
        for name, provider_dict in provider_configs_dict.items():
            provider_configs[name] = ProviderConfig(
                host=provider_dict.get("host", ""),
                default_model=provider_dict.get("default_model", ""),
                timeout=provider_dict.get("timeout", 30.0),
                retry_attempts=provider_dict.get("retry_attempts", 3),
                api_key=provider_dict.get("api_key"),
                rate_limit=provider_dict.get("rate_limit"),
                extra_settings=provider_dict.get("extra_settings", {})
            )
        
        # Extract optional fields
        model_preferences = config_dict.get("model_preferences", {})
        fallback_providers = config_dict.get("fallback_providers", [])
        
        config = FactoryConfig(
            default_provider=default_provider,
            provider_configs=provider_configs,
            model_preferences=model_preferences,
            fallback_providers=fallback_providers
        )
        
        # Validate configuration
        validate_factory_config(config)
        
        return config
    
    except (KeyError, ValueError, TypeError) as e:
        raise InvalidConfigurationError(
            f"Failed to load factory configuration: {str(e)}",
            cause=e
        ) from e


def load_factory_config_from_json(json_str: str) -> FactoryConfig:
    """
    Load factory configuration from JSON string.
    
    Args:
        json_str: JSON string containing configuration
        
    Returns:
        FactoryConfig instance
        
    Raises:
        InvalidConfigurationError: If configuration is invalid
    """
    try:
        config_dict = json.loads(json_str)
        return load_factory_config_from_dict(config_dict)
    except json.JSONDecodeError as e:
        raise InvalidConfigurationError(
            f"Failed to parse JSON configuration: {str(e)}",
            cause=e
        ) from e

