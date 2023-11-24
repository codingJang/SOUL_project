import functools
from copy import copy

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary, Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from ray.rllib.utils.numpy import one_hot


N = 3
delta = 0.01
var = 1
obs_space = Box(low=0, high=1, shape=(N * N, ))
act_space = Box(low=np.full(2*N, -np.inf), high=np.full(2*N, np.inf), shape=(2*N,))


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
    env = PoliticsEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class PoliticsEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "politics_environment_v0"
    }

    possible_agents = [f"agent_{i}" for i in range(N)]
    agents = copy(possible_agents)
    observation_spaces = {f'agent_{i}':obs_space for i in range(N)}
    action_spaces = {f'agent_{i}':act_space for i in range(N)}

    def __init__(self, render_mode=None):
        super().__init__()
        self.delta = delta
        self.render_mode = render_mode
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return act_space
    
    def get_agent_ids(self):
        return self.agents

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
        self.affinity = np.eye(self.num_agents, dtype=np.float32)
        observations = {agent:self.affinity.flatten() for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        invites = []
        accepts = []
        for i, (agent, action) in enumerate(actions.items()):
            invite_pref = action[:self.num_agents]
            accept_pref = action[self.num_agents:]
            softmax = lambda x : np.exp(x) / np.sum(np.exp(x), axis=0)
            sigmoid = lambda x : 1 / (1 + np.exp(x))
            invite_prob = softmax(invite_pref)
            accept_prob = sigmoid(accept_pref)
            invite_choice = np.random.choice(self.num_agents, p=invite_prob)
            invite = one_hot(invite_choice, depth=self.num_agents)
            accept = np.random.uniform(size=self.num_agents) < accept_prob
            invite[i] = 0
            accept[i] = 0
            invites.append(invite)
            accepts.append(accept)
        invites = np.vstack(invites)
        accepts = np.vstack(accepts)
        delta_affinity = self.delta * 0.5 * (accepts.T * invites + invites.T * accepts)
        # print(self.affinity)
        # print(delta_affinity)
        self.affinity += delta_affinity
        observations = {agent:self.affinity.flatten() for agent in self.agents}
        rewards = np.random.normal(size=self.num_agents, scale=var)
        rewards[0] += 1
        rewards = self.affinity @ rewards
        rewards = dict(zip(self.agents, list(rewards)))
        # print(rewards)
        terminations = {agent:False for agent in self.agents}
        truncations = {agent:False for agent in self.agents}
        infos = {agent:{} for agent in self.agents}

        self.t += 1

        if self.t >= 100:
            # print("Game does end")
            truncations = {agent:True for agent in self.agents}
            terminations = {agent:True for agent in self.agents}
        
        if all(terminations.values()) or all(truncations.values()):
            self.agents = []
            
        return observations, rewards, terminations, truncations, infos
    


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test, render_test

    env = PoliticsEnv()
    # parallel_api_test(env, num_cycles=1_000)
    render_test(PoliticsEnv)
