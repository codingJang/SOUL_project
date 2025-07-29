"""
Environment configuration for the macro-economic simulation.

This module contains the MacroSimEnvConfig class with configuration parameters
for the MacroSimEnv environment, including economic simulation parameters, 
noise levels, and episode settings.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class MacroSimEnvConfig:
    """Configuration class for macro-economic simulation environment."""
    
    # Agent configuration
    num_agents: int = 7
    
    # Economic simulation parameters
    rho: float = 0.8  # Economic parameter for shock persistence
    demand_penalty: float = 1.0  # Penalty coefficient for demand dynamics
    ex_int_degree: float = 1.0  # Exchange rate interest degree
    delta: float = 0.01  # Affinity change rate
    shock_lvl: float = 0.0  # Global shock level
    
    # Noise standard deviations
    std_eta: float = 0.03  # Standard deviation for ETA noise
    std_given_demand: float = 0.1  # Standard deviation for given demand
    std_one_plus_shock: float = 0.1  # Standard deviation for one plus shock
    std_prev_price_lvl: float = 0.01  # Standard deviation for previous price level
    std_price_lvl: float = 0.01  # Standard deviation for price level
    std_prev_net_ex: float = 0.1  # Standard deviation for previous net exports
    std_net_ex: float = 0.1  # Standard deviation for net exports
    
    # Episode configuration
    max_episode_steps: int = 100  # Maximum steps per episode
    
    # Observation and action space configuration
    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Calculate observation space shape based on number of agents."""
        return ((self.num_agents + 4) * self.num_agents,)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive")
        if self.rho < 0 or self.rho >= 1:
            raise ValueError("rho must be in [0, 1)")
        if any(std < 0 for std in [self.std_eta, self.std_given_demand, 
                                   self.std_one_plus_shock, self.std_prev_price_lvl,
                                   self.std_price_lvl, self.std_prev_net_ex, self.std_net_ex]):
            raise ValueError("All standard deviations must be non-negative")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive") 