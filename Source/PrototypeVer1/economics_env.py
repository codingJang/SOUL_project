import functools
from copy import copy

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary, Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from ray.rllib.utils.numpy import one_hot
import supersuit as ss


N = 3
obs_space = Box(low=-np.inf, high=np.inf, shape=(4 * N,))
act_space = Box(low=-np.inf, high=np.inf, shape=(1,))


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
    env = EconomicsEnv(render_mode=render_mode)
    env = parallel_to_aec(env)
    env = ss.clip_actions_v0(env)
    return env


class EconomicsEnv(ParallelEnv):
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
        self.render_mode = render_mode
        self.rho = 0.8
        self.STD_ETA = 0.03
        self.std_gd = 0.1
        self.std_ops = 0.1
        self.std_ppl = 0.1
        self.std_pl = 0.1
        self.std_pne = 0.1
        self.std_ne = 0.1
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return act_space
    
    def get_agent_ids(self):
        return self.agents

    def render(self, mode='human'):
        if mode == 'human':
            self._render_human_readable()
        elif mode == 'array':
            return self._render_array()
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def _render_human_readable(self):
        print("Current timestep:", self.t)
        print("Agents:", self.agents)
        print("INT_RATE:", np.exp(self.one_plus_int_rate)-1)
        print("GDP:", self.GDP)

        print("\nObservables:")
        observable_vars = ['dem_after_shock', 'prev_price_lvl', 'price_lvl']
        for var in observable_vars:
            print(f"{var.upper()}: {np.exp(getattr(self, var))}")
        print("PREV_NET_EX:", self.PREV_NET_EX)

        print("\nHidden Variables:")
        print("ETA:", self.ETA)
        hidden_vars = ['one_plus_shock', 'given_demand', 'dem_after_shock', 'total_demand', 'one_plus_inf_rate']
        for var in hidden_vars:
            print(f"{var.upper()}: {np.exp(getattr(self, var))}")
        
        print("\nInter-agent Variables:")
        print("Nominal Exchange Rate:\n", np.exp(self.nom_exchange_rate))
        print("Real Exchange Rate:\n", np.exp(self.real_exchange_rate))
        print("Trade Coefficients:\n", self.TRADE_COEFF)

        print("\nEconomic Indicators:")
        print("Exports:", self.EX)
        print("Imports:", self.IM)
        print("Net Exports:", self.NET_EX)
        print()
        print()

    def _render_array(self):
        # Concatenate all relevant arrays into a single numpy array for more technical analysis
        state_arrays = [self.given_demand, self.one_plus_shock, self.one_plus_inf_rate, self.price_lvl, 
                        self.dem_after_shock, self.nom_exchange_rate, self.real_exchange_rate, 
                        self.TRADE_COEFF, self.EX, self.IM, self.NET_EX, self.price_lvl]
        return np.concatenate([arr.flatten() for arr in state_arrays])
    
    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.t = 0
        # all small letter variables denote logarithmic variables
        # all capital letter variables are non-logarithmetic variables
        # ex: self.one_plus_inf_rate = ln(1 + (INFLATION_RATE))
        self.given_demand = self.std_gd * np.random.randn(self.num_agents)
        self.one_plus_shock = self.std_ops * np.random.randn(self.num_agents)
        self.dem_after_shock = self.given_demand + self.one_plus_shock
        self.prev_price_lvl = self.std_ppl * np.random.randn(self.num_agents)
        self.price_lvl = self.std_pl * np.random.randn(self.num_agents)
        self.PREV_NET_EX = np.exp(self.std_pne * np.random.randn(self.num_agents))
        self.NET_EX = np.exp(self.std_ne * np.random.randn(self.num_agents))
        self.observation = np.vstack((self.dem_after_shock, self.prev_price_lvl, self.price_lvl, self.PREV_NET_EX))
        observations = {agent:self.observation.flatten().astype(np.float32) for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        actions = [action for (agent, action) in actions.items()]
        self.one_plus_int_rate = 0.20 / (1 + np.exp(-np.array(actions).squeeze()))
        # print(f"self.one_plus_int_rate={self.one_plus_int_rate}")
        self.t += 1

        self.total_demand = self.dem_after_shock - self.one_plus_int_rate
        # print(f"-------self.total_demand------: {self.total_demand}")
        price_diff = self.price_lvl.reshape(-1, 1) - self.price_lvl.reshape(1, -1)
        int_rate_diff = self.one_plus_int_rate.reshape(-1, 1) - self.one_plus_int_rate.reshape(1, -1)
        self.nom_exchange_rate = price_diff - 7 * int_rate_diff
        self.real_exchange_rate = - 7 * int_rate_diff

        NUM = np.exp(self.real_exchange_rate).T
        DEN = np.sum(NUM, axis=0, keepdims=True)
        self.TRADE_COEFF = NUM / DEN
        TEMP = np.copy(self.TRADE_COEFF)
        np.fill_diagonal(TEMP, 0)
        self.EX = TEMP.T @ np.exp(self.total_demand)

        NUM_STAR = np.exp(self.real_exchange_rate)
        DEN_STAR = np.sum(NUM_STAR, axis=1, keepdims=True)
        self.TRADE_COEFF_STAR = NUM_STAR / DEN_STAR
        TEMP_STAR = np.copy(self.TRADE_COEFF_STAR)
        np.fill_diagonal(TEMP_STAR, 0)
        self.EX_STAR = TEMP_STAR @ np.exp(self.total_demand)

        assert (self.EX == self.EX_STAR).all(), f"self.EX=={self.EX} self.EX_STAR={self.EX_STAR}"

        self.IM = np.sum(TEMP, axis=1) * np.exp(self.total_demand)
        self.PREV_NET_EX = self.NET_EX
        self.NET_EX = self.EX - self.IM
        self.prev_price_lvl = self.price_lvl
        # print(f"1 + self.NET_EX/np.exp(self.total_demand): {1 + self.NET_EX/np.exp(self.total_demand)}")
        self.price_lvl = np.log(np.exp(self.price_lvl)) + np.log(np.maximum(1e-10, 1 + self.NET_EX/np.exp(self.total_demand))) - self.one_plus_int_rate
        self.price_lvl = (self.price_lvl - np.mean(self.price_lvl)) / np.std(self.price_lvl)
        self.one_plus_inf_rate = self.price_lvl - self.prev_price_lvl
        self.given_demand = self.given_demand - self.one_plus_inf_rate
        self.ETA = self.STD_ETA * np.random.randn(self.num_agents)
        self.one_plus_shock = np.log(np.maximum(1e-10, 1+(self.rho * (np.exp(self.one_plus_shock)-1) + self.ETA)))
        self.dem_after_shock = self.given_demand + self.one_plus_shock
        
        self.observation = np.vstack((self.dem_after_shock, self.prev_price_lvl, self.price_lvl, self.PREV_NET_EX))
        observations = {agent:self.observation.flatten().astype(np.float32) for agent in self.agents}

        self.GDP = np.exp(self.total_demand) + self.NET_EX
        rewards = dict(zip(self.agents, list(self.GDP)))
        # print(rewards)
        terminations = {agent:False for agent in self.agents}
        truncations = {agent:False for agent in self.agents}
        infos = {agent:{} for agent in self.agents}

        if self.t >= 100:
            # print("Game does end")
            truncations = {agent:True for agent in self.agents}
            terminations = {agent:True for agent in self.agents}
        
        if all(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        # a = np.array([1,2,3]) / np.array([0, 1, 2])
        # self.render(mode='human')

        return observations, rewards, terminations, truncations, infos


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test, render_test

    np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
    # np.seterr
    env = EconomicsEnv()
    parallel_api_test(env, num_cycles=1_000)
    # render_test(EconomicsEnv)
