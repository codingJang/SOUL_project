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
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from politics_environment import *

from typing import Dict, Tuple
import os



class MyCallbacks(DefaultCallbacks):
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
        episode.user_data["agent_1_rewards"] = []
        episode.hist_data["agent_1_rewards"] = []

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
        agent_1_reward = episode._agent_reward_history['agent_1']
        episode.user_data["agent_1_rewards"].append(agent_1_reward)

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
        episode.custom_metrics["agent_1_rewards"] = np.mean(episode.user_data["agent_1_rewards"])
        print(">>", episode.agent_rewards)

def env_creator(args):
    env = PoliticsEnv()
    # env = ss.frame_stack_v1(env, 3)
    return env


if __name__ == "__main__":
    ray.init()
    # ray.init(local_mode=True)
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
        # .reporting(keep_per_episode_custom_metrics=True)
    )
        
    config['gamma'] = ray.tune.grid_search([0.8, 0.9])
    config['lr'] = ray.tune.grid_search([1e-5, 5e-5, 1e-4])
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
                "training_iteration": 1000,
            },
        ),
        param_space=config,
    )
    # there is only one trial involved.
    result = tuner.fit().get_best_result()

    custom_metrics = result.metrics["custom_metrics"]
    print(custom_metrics)

    # import ray.rllib.env.wrappers.pettingzoo_env