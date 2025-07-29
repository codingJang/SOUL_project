import ray
import numpy as np
import platform
import os
from env import *
from typing import Dict, Tuple
from ray import air, train, tune
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from configs.rllib_train_config import RLlibTrainConfig
from configs.environment_config import MacroSimEnvConfig

# Set environment variables to reduce Ray verbosity
os.environ["RAY_AIR_NEW_OUTPUT"] = "0"  # Disable new verbose output engine
os.environ["RAY_DEDUP_LOGS"] = "1"  # Deduplicate repeated log messages
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"  # Disable import warnings

class CombinedEnvCallbacks(DefaultCallbacks):
    def __init__(self, config: RLlibTrainConfig = None):
        super().__init__()
        self.config = config or RLlibTrainConfig()
        self.last_print_iteration = 0
    
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        # Create lists to store angles in
        for i in range(self.config.env_config.num_agents if hasattr(self.config, 'env_config') else N):
            episode.custom_metrics[f"agent_{i}_GDPs"] = []
            episode.custom_metrics[f"agent_{i}_interest_rates"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        for agent_id, collector in episode._agent_collectors.items():
            episode.custom_metrics[agent_id+"_GDPs"].append(np.sum(collector.buffers['rewards']))
            assert np.all(0.20 / (1 + np.exp(-np.array(collector.buffers['actions']))) >= 0), f"{0.20 / (1 + np.exp(-np.array(collector.buffers['actions'])))}"
            episode.custom_metrics[agent_id+"_interest_rates"].append(np.mean(0.20 / (1 + np.exp(-np.array(collector.buffers['actions'])))))

    def on_algorithm_init(self, *, algorithm, **kwargs):
        print("🚀 APPO algorithm initialized successfully!")
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Print progress every 5 iterations to show we're making progress
        iteration = result.get("training_iteration", 0)
        if iteration % 5 == 0 and iteration > self.last_print_iteration:
            self.last_print_iteration = iteration
            timesteps = result.get("timesteps_total", 0)
            episode_reward_mean = result.get("episode_reward_mean", 0)
            print(f"📊 Iteration {iteration}: {timesteps:,} timesteps, avg reward: {episode_reward_mean:.3f}")


def env_creator(env_config):
    return MacroSimRayRLlibEnv(render_mode='array', config=env_config.get('env_config'))


if __name__ == "__main__":
    print("🔧 Loading configurations...")
    # Load configurations
    train_config = RLlibTrainConfig()
    env_config = MacroSimEnvConfig()
    train_config.validate()
    env_config.validate()
    print("✅ Configurations loaded and validated")
    
    print("🔌 Initializing Ray...")
    ray.init(num_gpus=0, log_to_driver=True, logging_level='INFO')
    platform_name = platform.node()
    env_name = "macro_sim_rllib_env_v0"
    print("✅ Ray initialized successfully")
    
    print("🏗️ Setting up environment...")
    register_env(env_name, env_creator)
    temp_env = env_creator({'env_config': env_config})
    print("✅ Environment registered and tested")
    
    print("⚙️ Configuring APPO algorithm...")
    config = (
        APPOConfig()
        .training(
            lr=tune.loguniform(train_config.lr_min, train_config.lr_max), 
            gamma=tune.uniform(train_config.gamma_min, train_config.gamma_max), 
            clip_param=train_config.clip_param, 
            train_batch_size=train_config.train_batch_size
        )
        .environment(env=env_name, clip_actions=train_config.clip_actions, env_config={'env_config': env_config})                                                                                                                                               
        .rollouts(
            num_rollout_workers=train_config.num_workers, 
            recreate_failed_workers=train_config.recreate_failed_workers, 
            restart_failed_sub_environments=train_config.restart_failed_sub_environments
        )
        .framework(framework=train_config.framework)
        .resources(
            num_learner_workers=train_config.num_learner_workers, 
            num_cpus_for_local_worker=train_config.num_cpus_for_local_worker
        )
        .multi_agent(
            policies=temp_env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),  # all policies map to themselves (independent PPO learning)
        )
        .debugging(
            log_level=train_config.log_level
        )
        .callbacks(CombinedEnvCallbacks)
    )
    config.model['use_lstm'] = train_config.use_lstm
    print("✅ APPO configuration complete")
    
    def stop_fn(trial_id: str, result: dict) -> bool:
        bool_value_1 = result["timesteps_total"] >= train_config.max_timesteps
        bool_value_2 = any([result["custom_metrics"][f"agent_{i}_interest_rates_max"] <= train_config.min_interest_rate_threshold for i in range(env_config.num_agents)])
        bool_value_3 = any([result['info']['learner'][f'agent_{i}']['learner_stats']['entropy'] >= train_config.max_entropy_threshold for i in range(env_config.num_agents)])
        return bool_value_1 or bool_value_2 or bool_value_3
    
    print("🎯 Creating training tuner...")
    tuner = tune.Tuner(
        "APPO",
        run_config=air.RunConfig(
            storage_path=os.path.abspath("models"),
            checkpoint_config=train.CheckpointConfig(checkpoint_frequency=train_config.checkpoint_frequency),
            stop=stop_fn,
            verbose=1  # Show some progress but not too verbose
        ),
        tune_config=tune.TuneConfig(
            num_samples=train_config.num_samples, 
            time_budget_s=train_config.time_budget_seconds,
            max_concurrent_trials=train_config.max_concurrent_trials
        ),
        param_space=config.to_dict()
    )
    # there is only one trial involved.
    print("🚀 Starting APPO training...")
    print(f"📝 Training with {train_config.num_samples} samples, max {train_config.time_budget_hours} hours")
    print("⏱️ Training progress will be shown every 5 iterations...")
    
    result = tuner.fit().get_best_result()

    print("\nTraining completed!")
    custom_metrics = result.metrics["custom_metrics"]
    print("Final custom metrics:")
    print(custom_metrics)
