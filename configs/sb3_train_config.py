"""
Stable Baselines3 training configuration for the macro-economic simulation.

This module contains the configuration parameters for PPO algorithm training
using Stable Baselines3, including hyperparameter optimization ranges,
resource allocation, and training settings.
"""

from dataclasses import dataclass
from typing import List
import platform


@dataclass
class SB3TrainConfig:
    """Configuration class for Stable Baselines3 PPO training."""
    
    # Platform detection
    @property
    def is_local(self) -> bool:
        """Detect if running on local machine (MacBook) vs cluster."""
        return "MacBookAir" in platform.node()
    
    # Hyperparameter optimization ranges
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    gamma_min: float = 0.9
    gamma_max: float = 0.9999
    clip_range_min: float = 0.1
    clip_range_max: float = 0.4
    batch_size_options: List[int] = None
    n_epochs_min: int = 3
    n_epochs_max: int = 10
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.batch_size_options is None:
            self.batch_size_options = [64, 128, 256, 512]
    
    # Environment configuration
    n_envs_local: int = 1
    n_envs_cluster: int = 4
    
    @property
    def n_envs(self) -> int:
        """Get number of environments based on platform."""
        return self.n_envs_local if self.is_local else self.n_envs_cluster
    
    # Training timesteps
    timesteps_local: int = 50000
    timesteps_cluster: int = 200000
    final_timesteps_local: int = 500000
    final_timesteps_cluster: int = 2000000
    
    @property
    def training_timesteps(self) -> int:
        """Get training timesteps for hyperparameter optimization."""
        return self.timesteps_local if self.is_local else self.timesteps_cluster
    
    @property
    def final_training_timesteps(self) -> int:
        """Get final training timesteps for best model."""
        return self.final_timesteps_local if self.is_local else self.final_timesteps_cluster
    
    # Optuna optimization configuration
    n_trials_local: int = 5
    n_trials_cluster: int = 20
    timeout_hours_local: float = 2.0
    timeout_hours_cluster: float = 4.0
    
    @property
    def n_trials(self) -> int:
        """Get number of trials based on platform."""
        return self.n_trials_local if self.is_local else self.n_trials_cluster
    
    @property
    def timeout_seconds(self) -> int:
        """Get timeout in seconds based on platform."""
        timeout_hours = self.timeout_hours_local if self.is_local else self.timeout_hours_cluster
        return int(timeout_hours * 60 * 60)
    
    # Model configuration
    policy_type: str = "MlpPolicy"
    verbose: int = 1
    verbose_optuna: int = 0
    
    # Logging configuration
    tensorboard_log_dir: str = "./tensorboard_logs/"
    model_save_name: str = "best_macro_sim_model"
    
    # Callback configuration
    metrics_log_frequency: int = 100  # Log metrics every N steps
    evaluation_episodes: int = 10  # Number of recent episodes for evaluation
    
    # Early stopping criteria (similar to RLlib)
    early_stop_interest_rate_threshold: float = 0.001
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.lr_min <= 0 or self.lr_max <= 0 or self.lr_min >= self.lr_max:
            raise ValueError("Learning rate bounds must be positive and lr_min < lr_max")
        if self.gamma_min <= 0 or self.gamma_max >= 1 or self.gamma_min >= self.gamma_max:
            raise ValueError("Gamma bounds must be in (0, 1) and gamma_min < gamma_max")
        if self.clip_range_min <= 0 or self.clip_range_max <= 0 or self.clip_range_min >= self.clip_range_max:
            raise ValueError("Clip range bounds must be positive and clip_range_min < clip_range_max")
        if not all(bs > 0 for bs in self.batch_size_options):
            raise ValueError("All batch sizes must be positive")
        if self.n_epochs_min <= 0 or self.n_epochs_max <= 0 or self.n_epochs_min >= self.n_epochs_max:
            raise ValueError("N epochs bounds must be positive and n_epochs_min < n_epochs_max")
        if self.timesteps_local <= 0 or self.timesteps_cluster <= 0:
            raise ValueError("Training timesteps must be positive")
        if self.final_timesteps_local <= 0 or self.final_timesteps_cluster <= 0:
            raise ValueError("Final training timesteps must be positive")
        if self.n_trials_local <= 0 or self.n_trials_cluster <= 0:
            raise ValueError("Number of trials must be positive")
        if self.timeout_hours_local <= 0 or self.timeout_hours_cluster <= 0:
            raise ValueError("Timeout hours must be positive")
        if self.metrics_log_frequency <= 0:
            raise ValueError("Metrics log frequency must be positive")
        if self.evaluation_episodes <= 0:
            raise ValueError("Evaluation episodes must be positive") 