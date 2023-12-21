import platform

import ray
from ray import air, train, tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
ray.rllib.models.ModelCatalog

from economics_env import *

from typing import Dict, Tuple
import os



class EconomicsEnvCallbacks(DefaultCallbacks):
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
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        # assert episode.length == 0, (
        #     "ERROR: `on_episode_start()` callback should be called right "
        #     f"after env reset!, episode.length was: {episode.length}"
        # )
        # Create lists to store angles in
        for i in range(N):
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
        # for agent_id, collector in episode._agent_collectors.items():
        #     print(collector.buffers['actions'])

        # for i in range(N):
        #     agent_GDP = episode._agent_reward_history[f'agent_{i}']
        #     episode._agent_collectors.items()
        #     # agent_interest_rate = episode._agent_action_history[f'agent_{i}']
        #     episode.user_data[f"agent_{i}_GDPs"].append(agent_GDP)

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
        # print(episode.user_data["agent_1_rewards"])
        # episode.custom_metrics["agent_1_rewards"] = np.sum(episode.user_data["agent_1_rewards"])
        # raise TypeError(f"What's wrong? type = {episode.__dict__.keys()}")
        for agent_id, collector in episode._agent_collectors.items():
            episode.custom_metrics[agent_id+"_GDPs"].append(np.sum(collector.buffers['rewards']))
            assert np.all(0.20 / (1 + np.exp(-np.array(collector.buffers['actions']))) >= 0), f"{0.20 / (1 + np.exp(-np.array(collector.buffers['actions'])))}"
            episode.custom_metrics[agent_id+"_interest_rates"].append(np.mean(0.20 / (1 + np.exp(-np.array(collector.buffers['actions'])))))

def env_creator(args):
    env = EconomicsEnv()
    # env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.init(num_gpus=0)
    # ray.init(local_mode=True)
    platform_name = platform.node()
    env_name = "economics_environment"
    env = env_creator({})
    register_env(env_name, lambda config: ParallelPettingZooEnv(env))
    config = (
        APPOConfig()
        .training(lr=tune.loguniform(1e-5, 1e-3), gamma=tune.uniform(0.9, 0.9999), clip_param=0.2, train_batch_size=512)
        .environment(env=env_name, clip_actions=True)                                                                                                                                               
        .rollouts(num_rollout_workers=7 if platform.node()=="jang-yejun-ui-MacBookAir.local" else 84, recreate_failed_workers=True, restart_failed_sub_environments=True)
        .framework(framework="torch")
        .resources(num_learner_workers=7 if platform.node()=="jang-yejun-ui-MacBookAir.local" else 84, num_cpus_for_local_worker=1)
        .multi_agent(
            # policies={
            #     "agent_0": (None, obs_space, act_space, {}),
            #     "agent_1": (None, obs_space, act_space, {}),
            #     "agent_2": (None, obs_space, act_space, {})
            # },
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),  # all policies map to themselves (independent PPO learning)
        )
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
        .callbacks(EconomicsEnvCallbacks)
        # .rollouts(enable_connectors=False)
        # .reporting(keep_per_episode_custom_metrics=True)
    )
    config.model['use_lstm'] = True
        
    
    """
    tune.run(
        "SAC",
        name="SAC",
        stop={"timesteps_total": 1000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
    """
    def stop_fn(trial_id: str, result: dict) -> bool:
        # raise NotImplementedError(f"trial_id: {trial_id}, result.keys(): {result.keys()}")
        bool_value_1 = result["timesteps_total"] >= 10000000
        bool_value_2 = any([result["custom_metrics"][f"agent_{i}_interest_rates_max"] <= 0.001 for i in range(N)])
        bool_value_3 = any([result['info']['learner'][f'agent_{i}']['learner_stats']['entropy'] >= 2.0 for i in range(N)])
        return bool_value_1 or bool_value_2 or bool_value_3
    
    tuner = tune.Tuner(
        "APPO",
        run_config=air.RunConfig(
            checkpoint_config=train.CheckpointConfig(checkpoint_frequency=10),
            stop=stop_fn
        ),
        tune_config=tune.TuneConfig(num_samples=-1, time_budget_s=4*60*60),
        param_space=config.to_dict()
    )
    # there is only one trial involved.
    result = tuner.fit().get_best_result()

    custom_metrics = result.metrics["custom_metrics"]
    print(custom_metrics)

    # import ray.rllib.env.wrappers.pettingzoo_env