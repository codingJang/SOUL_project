import argparse

import ray
from typing import Dict, Tuple
from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from pettingzoo.sisl import waterworld_v4
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy

# Based on code from github.com/parametersharingmadrl/parametersharingmadrl

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
        raise AssertionError("Tracing back AHHHHH")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-gpus",
    type=int,
    default=0,
    help="Number of GPUs to use for training.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: Only one episode will be "
    "sampled.",
)

if __name__ == "__main__":
    context = ray.init()
    print(context.dashboard_url)
    args = parser.parse_args()

    def env_creator(args):
        return ParallelPettingZooEnv(waterworld_v4.parallel_env())

    env = env_creator({})
    register_env("waterworld", env_creator)

    config = (
        PPOConfig()
        .environment("waterworld")
        .resources(num_gpus=args.num_gpus)
        .rollouts(num_rollout_workers=7)
        .callbacks(MyCallbacks)
        .multi_agent(
            policies=env.get_agent_ids(),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
    )

    if args.as_test:
        # Only a compilation test of running waterworld / independent learning.
        stop = {"training_iteration": 1}
    else:
        stop = {"episodes_total": 60000}

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop=stop,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
        param_space=config,
    ).fit()
