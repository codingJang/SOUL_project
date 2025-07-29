# Configuration Files

This directory contains configuration classes for the SOUL project's multi-agent economic simulation system. All configurable parameters have been moved from hardcoded values in the source files to structured dataclasses.

## Configuration Classes

### `MacroSimEnvConfig` (environment_config.py)
Contains parameters for the macro-economic simulation environment:
- **Agent Configuration**: Number of agents, episode length
- **Economic Parameters**: Economic dynamics coefficients (rho, demand_penalty, etc.)
- **Noise Parameters**: Standard deviations for various random components
- **Space Configuration**: Observation and action space dimensions

```python
from configs.environment_config import MacroSimEnvConfig

# Use default configuration
env_config = MacroSimEnvConfig()

# Or customize parameters
custom_config = MacroSimEnvConfig(
    num_agents=10,
    max_episode_steps=200,
    rho=0.9
)
```

### `RLlibTrainConfig` (rllib_train_config.py)
Contains parameters for training with Ray RLlib's APPO algorithm:
- **Resource Allocation**: Worker counts, CPU allocation
- **Hyperparameter Ranges**: Learning rate, gamma, clip parameters
- **Training Configuration**: Timestep limits, stopping criteria
- **Platform Detection**: Automatic local vs cluster resource allocation

```python
from configs.rllib_train_config import RLlibTrainConfig

train_config = RLlibTrainConfig()
print(f"Using {train_config.num_workers} workers")
print(f"Training for {train_config.max_timesteps} timesteps")
```

### `SB3TrainConfig` (sb3_train_config.py)
Contains parameters for training with Stable Baselines3's PPO:
- **Hyperparameter Optimization**: Optuna search ranges
- **Environment Configuration**: Number of parallel environments
- **Training Timesteps**: Optimization and final training durations
- **Platform Detection**: Automatic local vs cluster resource allocation

```python
from configs.sb3_train_config import SB3TrainConfig

train_config = SB3TrainConfig()
print(f"Running {train_config.n_trials} optimization trials")
print(f"Using {train_config.n_envs} parallel environments")
```

## Platform Detection

Both training configurations automatically detect whether you're running on a local machine (MacBook) or a cluster environment and adjust resource allocation accordingly:

- **Local (MacBook)**: Fewer workers, shorter training times, fewer trials
- **Cluster**: More workers, longer training times, more trials

## Running the Training Scripts

The proper way to run the training scripts is using `python -m` from the project root directory:

```bash
# Run RLlib training
python -m appo_rllib

# Run SB3 training  
python -m appo_sb3

# Run environment tests
python -m env
```

## Usage in Code

### Environment Creation
```python
from configs.environment_config import MacroSimEnvConfig
from env import MacroSimEnv

env_config = MacroSimEnvConfig(num_agents=5)
env = MacroSimEnv(config=env_config)
```

### RLlib Training
```python
from configs.rllib_train_config import RLlibTrainConfig
from configs.environment_config import MacroSimEnvConfig

train_config = RLlibTrainConfig()
env_config = MacroSimEnvConfig()

# Use in APPOConfig
config = APPOConfig().training(
    lr=tune.loguniform(train_config.lr_min, train_config.lr_max),
    gamma=tune.uniform(train_config.gamma_min, train_config.gamma_max)
)
```

### SB3 Training
```python
from configs.sb3_train_config import SB3TrainConfig
from configs.environment_config import MacroSimEnvConfig

train_config = SB3TrainConfig()
env_config = MacroSimEnvConfig()

# Use in hyperparameter optimization
learning_rate = trial.suggest_float('lr', train_config.lr_min, train_config.lr_max, log=True)
```

## Validation

All configuration classes include a `validate()` method that checks parameter ranges and constraints:

```python
config = MacroSimEnvConfig(num_agents=-1)  # Invalid
config.validate()  # Raises ValueError
```

## Customization

To modify default values, you can:

1. **Pass parameters to constructor**:
```python
config = MacroSimEnvConfig(num_agents=10, rho=0.9)
```

2. **Modify after creation**:
```python
config = MacroSimEnvConfig()
config.num_agents = 10
config.rho = 0.9
config.validate()  # Always validate after changes
```

3. **Create custom configuration files** by inheriting from the base classes and overriding defaults.
