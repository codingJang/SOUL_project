"""Example of using RLlib's debug callbacks.

Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.

We then use `keep_per_episode_custom_metrics` to keep the per-episode values
of our custom metrics and do our own summarization of them.
"""

from typing import Dict, Tuple
import argparse
import gymnasium as gym
import numpy as np
import random
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from gymnasium.spaces import Box, MultiBinary, Discrete
from ray.rllib.policy.policy import PolicySpec


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

import numpy as np
obs_space = Box(
    low=np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38]), 
    high=np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38]), 
    shape=(4,), dtype=np.float32
    )
act_space = Discrete(2)

def gen_policy(i):

        if bool(os.environ.get("RLLIB_ENABLE_RL_MODULE", False)):
            # just change the gammas between the two policies.
            # changing the module is not a critical part of this example.
            # the important part is that the policies are different.
            config = {
                "gamma": random.choice([0.95, 0.99]),
            }
        else:
            config = PPOConfig.overrides(
                model={
                    "custom_model": ["model1", "model2"][i % 2],
                },
                gamma=random.choice([0.95, 0.99]),
            )
        return PolicySpec(config=config)

if __name__ == "__main__":
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        return pol_id
    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {f"policy_{i}": gen_policy(i) for i in range(3)}
    policy_ids = list(policies.keys())

    config = (
        PPOConfig()
        .environment(MultiAgentCartPole, env_config={"num_agents": 3})
        .rollouts(num_rollout_workers=7, rollout_fragment_length='auto')
        .framework("torch")
        .callbacks(MyCallbacks)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        # .rollouts(enable_connectors=True) # not essential
        .reporting(keep_per_episode_custom_metrics=True)
    )

    # ray.init(local_mode=True) # not essential
    ray.init()
    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={
                "training_iteration": 1000,
            },
        ),
        param_space=config,
    )
    # there is only one trial involved.
    result = tuner.fit().get_best_result()

    # Verify episode-related custom metrics are there.
    custom_metrics = result.metrics["custom_metrics"]
    print(custom_metrics)
    