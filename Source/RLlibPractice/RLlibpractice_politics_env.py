"""Uses Ray's RLlib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""


import ray
from ray import air, tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from politics_environment import *

from typing import Dict, Tuple
import os



class MyCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        episode.custom_metrics["pole_angle"] = 1.0
        print("AHHHHH")


def env_creator(args):
    env = PoliticsEnv()
    # env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    # ray.init()
    ray.init(local_mode=True)
    env_name = "politics_environment"
    env = env_creator({})
    register_env(env_name, lambda config: ParallelPettingZooEnv(env))

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=7, rollout_fragment_length='auto')
        # .training(gamma=0.9, lr=0.01)
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .multi_agent(
            # policies={
            #     "agent_0": (None, obs_space, act_space, {}),
            #     "agent_1": (None, obs_space, act_space, {}),
            #     "agent_2": (None, obs_space, act_space, {})
            # },
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
        .callbacks(MyCallbacks)
        # .rollouts(enable_connectors=False)
        .reporting(keep_per_episode_custom_metrics=True)
    )
        
    
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
    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={
                "training_iteration": 10,
            },
        ),
        param_space=config,
    )
    # there is only one trial involved.
    result = tuner.fit().get_best_result()

    custom_metrics = result.metrics["custom_metrics"]
    print(custom_metrics)