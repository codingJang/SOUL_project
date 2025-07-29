import numpy as np
import platform
import optuna
from typing import Dict, Any
from .env import MacroSimPettingZooEnv, N
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from supersuit import pettingzoo_env_to_vec_env_v1
import torch
from configs.sb3_train_config import SB3TrainConfig
from configs.environment_config import MacroSimEnvConfig


class CustomMetricsCallback(BaseCallback):
    """
    Custom callback to track GDP and interest rate metrics similar to the RLlib version.
    """
    
    def __init__(self, config: SB3TrainConfig = None, verbose=0):
        super().__init__(verbose)
        self.config = config or SB3TrainConfig()
        self.agent_gdps = []
        self.agent_interest_rates = []
        
    def _on_step(self) -> bool:
        # Get the current environment info
        if hasattr(self.training_env, 'envs'):
            # For vectorized environments, get the first environment
            env = self.training_env.envs[0]
            if hasattr(env, 'env'):
                # Unwrap to get the actual environment
                actual_env = env.env
                if hasattr(actual_env, 'GDP') and hasattr(actual_env, 'one_plus_int_rate'):
                    # Track GDP and interest rates
                    gdps = actual_env.GDP
                    interest_rates = np.exp(actual_env.one_plus_int_rate) - 1
                    
                    self.agent_gdps.append(gdps)
                    self.agent_interest_rates.append(interest_rates)
                    
                    # Log metrics every N steps
                    if self.num_timesteps % self.config.metrics_log_frequency == 0:
                        avg_gdp = np.mean(gdps)
                        avg_interest_rate = np.mean(interest_rates)
                        max_interest_rate = np.max(interest_rates)
                        
                        self.logger.record("custom/avg_gdp", avg_gdp)
                        self.logger.record("custom/avg_interest_rate", avg_interest_rate)
                        self.logger.record("custom/max_interest_rate", max_interest_rate)
                        
                        # Check stopping criteria similar to RLlib version
                        if max_interest_rate <= self.config.early_stop_interest_rate_threshold:
                            if self.verbose > 0:
                                print(f"Stopping early: Max interest rate {max_interest_rate} <= {self.config.early_stop_interest_rate_threshold}")
                            return False
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        if len(self.agent_gdps) > 0:
            final_gdps = self.agent_gdps[-1]
            final_interest_rates = self.agent_interest_rates[-1]
            
            print("Final Training Metrics:")
            print(f"Final GDPs: {final_gdps}")
            print(f"Final Interest Rates: {final_interest_rates}")


def create_env(render_mode='array', config: MacroSimEnvConfig = None):
    """Create and wrap the PettingZoo environment for SB3."""
    def _make_env():
        env = MacroSimPettingZooEnv(render_mode=render_mode, config=config)
        return env
    return _make_env


def create_vectorized_env(n_envs=1, render_mode='array', config: MacroSimEnvConfig = None):
    """Create a vectorized environment using SuperSuit."""
    def make_env():
        env = MacroSimPettingZooEnv(render_mode=render_mode, config=config)
        # Convert PettingZoo parallel env to a vectorized environment
        env = pettingzoo_env_to_vec_env_v1(env)
        return env
    
    if n_envs == 1:
        return DummyVecEnv([make_env])
    else:
        return SubprocVecEnv([make_env for _ in range(n_envs)])


def optimize_hyperparameters(trial: optuna.trial.Trial, train_config: SB3TrainConfig, env_config: MacroSimEnvConfig) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', train_config.lr_min, train_config.lr_max, log=True)
    gamma = trial.suggest_float('gamma', train_config.gamma_min, train_config.gamma_max)
    clip_range = trial.suggest_float('clip_range', train_config.clip_range_min, train_config.clip_range_max)
    batch_size = trial.suggest_categorical('batch_size', train_config.batch_size_options)
    n_epochs = trial.suggest_int('n_epochs', train_config.n_epochs_min, train_config.n_epochs_max)
    
    # Create environment
    env = create_vectorized_env(n_envs=train_config.n_envs, config=env_config)
    
    # Create the model
    model = PPO(
        train_config.policy_type,
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        clip_range=clip_range,
        batch_size=batch_size,
        n_epochs=n_epochs,
        verbose=train_config.verbose_optuna,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        tensorboard_log=f"{train_config.tensorboard_log_dir}trial_{trial.number}/",
    )
    
    # Create callback
    callback = CustomMetricsCallback(config=train_config, verbose=1)
    
    try:
        # Train the model
        model.learn(
            total_timesteps=train_config.training_timesteps,
            callback=callback,
            progress_bar=False
        )
        
        # Evaluate the model - use mean reward as optimization target
        if len(callback.agent_gdps) > 0:
            # Use average GDP from last N episodes as the metric to optimize
            recent_gdps = callback.agent_gdps[-train_config.evaluation_episodes:]
            mean_gdp = np.mean([np.mean(gdp) for gdp in recent_gdps])
            return mean_gdp
        else:
            return 0.0
            
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return 0.0
    finally:
        env.close()


def train_best_model(best_params: Dict[str, Any], train_config: SB3TrainConfig, env_config: MacroSimEnvConfig):
    """Train a model with the best hyperparameters."""
    
    print(f"Training best model with parameters: {best_params}")
    print(f"Platform: {platform.node()}, Local: {train_config.is_local}, Environments: {train_config.n_envs}")
    
    # Create environment
    env = create_vectorized_env(n_envs=train_config.n_envs, config=env_config)
    
    # Create model with best parameters
    model = PPO(
        train_config.policy_type,
        env,
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma'],
        clip_range=best_params['clip_range'],
        batch_size=best_params['batch_size'],
        n_epochs=best_params['n_epochs'],
        verbose=train_config.verbose,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        tensorboard_log=f"{train_config.tensorboard_log_dir}best_model/",
    )
    
    # Create callback for metrics tracking
    callback = CustomMetricsCallback(config=train_config, verbose=1)
    
    # Train the model
    model.learn(
        total_timesteps=train_config.final_training_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the model
    model.save(train_config.model_save_name)
    print(f"Model saved as '{train_config.model_save_name}'")
    
    # Print final metrics
    if len(callback.agent_gdps) > 0:
        final_gdps = callback.agent_gdps[-1]
        final_interest_rates = callback.agent_interest_rates[-1]
        
        print("\nFinal Training Results:")
        for i in range(env_config.num_agents):
            print(f"Agent {i} - GDP: {final_gdps[i]:.4f}, Interest Rate: {final_interest_rates[i]:.4f}")
        
        print(f"\nAverage GDP: {np.mean(final_gdps):.4f}")
        print(f"Average Interest Rate: {np.mean(final_interest_rates):.4f}")
    
    env.close()
    return model


if __name__ == "__main__":
    print("Starting Stable Baselines3 Multi-Agent Economic Simulation")
    
    # Load configurations
    train_config = SB3TrainConfig()
    env_config = MacroSimEnvConfig()
    train_config.validate()
    env_config.validate()
    
    print(f"Platform: {platform.node()}")
    print(f"Running hyperparameter optimization with {train_config.n_trials} trials and {train_config.timeout_seconds/3600:.1f} hour timeout")
    
    # Create Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Optimize hyperparameters
    study.optimize(
        lambda trial: optimize_hyperparameters(trial, train_config, env_config),
        n_trials=train_config.n_trials,
        timeout=train_config.timeout_seconds,
        show_progress_bar=True
    )
    
    # Print best parameters
    print("\nBest hyperparameters:")
    print(study.best_params)
    print(f"Best value: {study.best_value}")
    
    # Train final model with best parameters
    best_model = train_best_model(study.best_params, train_config, env_config)
    
    print("Training completed successfully!")
