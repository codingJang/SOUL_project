"""
Wrapper module for direct import of configuration classes.

This module provides easy access to all configuration classes:
    from config import MacroSimEnvConfig, RLlibTrainConfig, SB3TrainConfig

Instead of having to use:
    from configs.environment_config import MacroSimEnvConfig
    from configs.rllib_train_config import RLlibTrainConfig
    from configs.sb3_train_config import SB3TrainConfig
"""

# Import all configuration classes
from configs.environment_config import MacroSimEnvConfig
from configs.rllib_train_config import RLlibTrainConfig  
from configs.sb3_train_config import SB3TrainConfig

# Re-export everything for convenience
__all__ = [
    'MacroSimEnvConfig',
    'RLlibTrainConfig',
    'SB3TrainConfig'
]

# Convenience function to create default configs
def get_default_configs():
    """Return a tuple of default configuration instances."""
    return (
        MacroSimEnvConfig(),
        RLlibTrainConfig(), 
        SB3TrainConfig()
    )

# Convenience function to validate all configs
def validate_configs(*configs):
    """Validate multiple configuration instances."""
    for config in configs:
        if hasattr(config, 'validate'):
            config.validate()
    print(f"✓ All {len(configs)} configurations validated successfully") 