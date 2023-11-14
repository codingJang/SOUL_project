import functools
from copy import copy

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None):
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = PoliticsEnvironment(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class PoliticsEnvironment(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "international_environment_v0"
    }

    possible_agents = [f"agent_{i}" for i in range(3)]
    agents = copy(possible_agents)

    def __init__(self, render_mode=None):
        super().__init__()
        self.delta = 0.05
        self.render_mode = render_mode
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=0, high=2, shape=(self.num_agents, self.num_agents))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Dict(
            {
                "invite":MultiBinary(self.num_agents-1),
                "accept":MultiBinary(self.num_agents-1)
            }
        )
    
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        print(self.affinity if len(self.agents) == self.num_agents else "Game over")
    
    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.t = 0
        self.affinity = np.eye(self.num_agents)
        observations = {agent:self.affinity for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, infos
    
    def step(self, actions):
        invites = []
        accepts = []
        for i, (agent, action) in enumerate(actions.items()):
            invite = np.insert(action["invite"], i, False)
            accept = np.insert(action["accept"], i, False)
            invites.append(invite)
            accepts.append(accept)
        invites = np.vstack(invites)
        accepts = np.vstack(accepts)
        delta_affinity = self.delta * 0.5 * (accepts.T * invites + invites.T * accepts)
        # print(self.affinity)
        # print(delta_affinity)
        self.affinity += delta_affinity
        observations = {agent:self.affinity for agent in self.agents}
        rewards = np.random.normal(size=self.num_agents)
        rewards[0] += 2
        rewards = self.affinity @ rewards
        rewards = dict(zip(self.agents, list(rewards)))
        # print(rewards)
        terminations = {agent:False for agent in self.agents}
        truncations = {agent:False for agent in self.agents}
        infos = {agent:{} for agent in self.agents}

        if self.t >= 100:
            truncations = {self.agents[i]:True for i in range(self.num_agents)}
        
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
            
        return observations, rewards, terminations, truncations, infos
    


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test

    env = PoliticsEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)