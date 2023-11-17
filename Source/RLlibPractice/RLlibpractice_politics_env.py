"""Uses Ray's RLlib to train agents to play Pistonball.

Author: Rohan (https://github.com/Rohan138)
"""

import os

import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from politics_environment import *


def env_creator(args):
    env = PoliticsEnv()
    # env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "politics_environment"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    config = (
        SACConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=7, rollout_fragment_length=128)
        .training(gamma=0.9, lr=0.01)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .multi_agent(
            policies={
                "agent_0": (None, obs_space, act_space, {}),
                "agent_1": (None, obs_space, act_space, {}),
                "agent_2": (None, obs_space, act_space, {})
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    tune.run(
        "SAC",
        name="SAC",
        stop={"timesteps_total": 100000},
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )