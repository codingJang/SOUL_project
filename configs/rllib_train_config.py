"""
RLlib training configuration for the macro-economic simulation.

This module contains the configuration parameters for APPO algorithm training
using Ray RLlib, including hyperparameter tuning ranges, resource allocation,
and stopping criteria.
"""

from dataclasses import dataclass
from typing import Tuple, List
import platform


@dataclass
class RLlibTrainConfig:
    """Configuration class for RLlib APPO training."""
    
    # Platform detection
    @property
    def is_local(self) -> bool:
        """Detect if running on local machine (MacBook) vs cluster."""
        return "MacBookAir" in platform.node()
    
    # Resource allocation
    num_workers_local: int = 3
    num_workers_cluster: int = 84
    num_cpus_for_local_worker: int = 1
    
    @property
    def num_workers(self) -> int:
        """Get number of workers based on platform."""
        return self.num_workers_local if self.is_local else self.num_workers_cluster
    
    @property
    def num_learner_workers(self) -> int:
        """Get number of learner workers (same as rollout workers)."""
        return self.num_workers
    
    # Hyperparameter tuning ranges
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    gamma_min: float = 0.9
    gamma_max: float = 0.9999
    clip_param: float = 0.2
    train_batch_size: int = 512
    
    # Model configuration
    use_lstm: bool = True
    
    # Framework and debugging
    framework: str = "torch"
    log_level: str = "INFO"
    
    # Environment configuration
    clip_actions: bool = True
    recreate_failed_workers: bool = True
    restart_failed_sub_environments: bool = True
    
    # Stopping criteria
    max_timesteps: int = 10000000
    min_interest_rate_threshold: float = 0.001
    max_entropy_threshold: float = 100.0
    
    # Tune configuration
    num_samples: int = 20
    time_budget_hours: float = 4.0
    max_concurrent_trials_local: int = 1
    max_concurrent_trials_cluster: int = 4
    
    @property
    def time_budget_seconds(self) -> int:
        """Convert time budget from hours to seconds."""
        return int(self.time_budget_hours * 60 * 60)
    
    @property
    def max_concurrent_trials(self) -> int:
        """Get max concurrent trials based on platform."""
        return self.max_concurrent_trials_local if self.is_local else self.max_concurrent_trials_cluster
    
    # Checkpoint configuration
    checkpoint_frequency: int = 10
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.lr_min <= 0 or self.lr_max <= 0 or self.lr_min >= self.lr_max:
            raise ValueError("Learning rate bounds must be positive and lr_min < lr_max")
        if self.gamma_min <= 0 or self.gamma_max >= 1 or self.gamma_min >= self.gamma_max:
            raise ValueError("Gamma bounds must be in (0, 1) and gamma_min < gamma_max")
        if self.clip_param <= 0:
            raise ValueError("clip_param must be positive")
        if self.train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive")
        if self.max_timesteps <= 0:
            raise ValueError("max_timesteps must be positive")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.time_budget_hours <= 0:
            raise ValueError("time_budget_hours must be positive")
        if self.checkpoint_frequency <= 0:
            raise ValueError("checkpoint_frequency must be positive") 