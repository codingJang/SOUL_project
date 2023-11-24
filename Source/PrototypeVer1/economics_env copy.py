import functools
from copy import copy

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict, MultiBinary, Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from ray.rllib.utils.numpy import one_hot


N = 3
obs_space = Box(low=0, high=1, shape=(5 * N,))
act_space = Box(low=np.full(1, -np.inf), high=np.full(1, np.inf), shape=(1,))


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
        self.stdev_eta = 0.03
    
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
        print("\nState Variables:")
        # given demand: nan, one_plus_inf_rate: nan, price_lvl: nan, dem_after_shock: nan
        state_variables = ['given_demand', 'one_plus_shock', 'one_plus_inf_rate', 'price_lvl', 'dem_after_shock']
        for var in state_variables:
            print(f"{var}: {getattr(self, var)}")
        print()
        print("Inter-agent Variables:")
        print("Nominal Exchange Rate:\n", self.nom_exchange_rate)
        print("Real Exchange Rate:\n", self.real_exchange_rate)
        print("Trade Coefficients:\n", self.trade_coeff)
        print("\nEconomic Indicators:")
        print("Exports:", self.ex)
        print("Imports:", self.im)
        print("Net Exports:", self.net_ex)
        print("Price Level:", self.price_lvl)
        print()

    def _render_array(self):
        # Concatenate all relevant arrays into a single numpy array for more technical analysis
        state_arrays = [self.given_demand, self.one_plus_shock, self.one_plus_inf_rate, self.price_lvl, self.dem_after_shock, self.nom_exchange_rate, self.real_exchange_rate, self.trade_coeff, self.ex, self.im, self.net_ex, self.price_lvl]
        return np.concatenate([arr.flatten() for arr in state_arrays])
    
    def close(self):
        pass

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.t = 0
        # all small letter variables denote logarithmic variables
        # all capital letter variables are non-logarithmetic variables
        # ex: self.one_plus_inf_rate = ln(1 + (INFLATION_RATE))
        self.given_demand = np.random.randn(self.num_agents)
        self.one_plus_shock = np.zeros(self.num_agents)
        # self.one_plus_inf_rate = np.random.randn(self.num_agents)
        self.price_lvl = np.random.randn(self.num_agents)
        self.dem_after_shock = self.one_plus_shock + self.given_demand
        self.states = np.vstack((self.given_demand, self.one_plus_shock, self.one_plus_inf_rate, self.price_lvl, self.dem_after_shock))
        observations = {agent:self.states.flatten() for agent in self.agents}
        infos = {agent:{} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # print(f"actions: ", actions)
        actions = [action for (agent, action) in actions.items()]
        # print(f"-------actions------: {actions}")
        self.one_plus_int_rate = np.array(actions).squeeze()
        # print(f"-------self.one_plus_int_rate------: {self.one_plus_int_rate}")
        self.total_demand = self.dem_after_shock - self.one_plus_int_rate
        print(f"-------self.total_demand------: {self.total_demand}")
        price_diff = self.price_lvl.reshape(-1, 1) - self.price_lvl.reshape(1, -1)
        int_rate_diff = self.one_plus_int_rate.reshape(1, -1) - self.one_plus_int_rate.reshape(-1, 1)
        # print(f"price_diff: {price_diff.shape}")
        # print(f"int_rate_diff: {int_rate_diff.shape}")
        self.nom_exchange_rate = price_diff + int_rate_diff
        # nom_exchange_rate = np.zeros(shape=(self.num_agents, self.num_agents))
        # for i in range(self.num_agents):
        #     for j in range(self.num_agents):
        #         nom_exchange_rate[i, j] = self.price_lev[i]-self.price_lev[j]+one_plus_int_rate[j]-one_plus_int_rate[i]
        self.real_exchange_rate = int_rate_diff
        # real_exchange_rate = np.zeros(shape=(self.num_agents, self.num_agents))
        # for i in range(self.num_agents):
        #     for j in range(self.num_agents):
        #         nom_exchange_rate[i, j] = self.price_lev[i]-self.price_lev[j]+one_plus_int_rate[j]-one_plus_int_rate[i]
        num = np.exp(self.real_exchange_rate)
        den = np.sum(num, axis=0, keepdims=True)
        self.trade_coeff = num / den
        # trade_coeff = np.zeros(shape=(self.num_agents, self.num_agents))
        # for i in range(self.num_agents):
        #     num = np.exp(real_exchange_rate[:,i])
        #     den = np.sum(num)
        #     trade_coeff[i, :] = num / den
        # spec_demand = trade_coeff @ total_demand
        temp = np.copy(self.trade_coeff)
        np.fill_diagonal(temp, 0)
        self.ex = temp @ np.exp(self.total_demand)
        self.im = np.sum(temp, axis=1) * np.exp(self.total_demand)
        self.net_ex = self.ex - self.im
        prev_price_lvl = self.price_lvl

        # self.render(mode="human")
        self.t += 1

        self.price_lvl = np.log(np.exp(self.price_lvl) * (1 + self.net_ex/np.exp(self.total_demand)) * np.exp(-self.one_plus_int_rate))
        self.one_plus_inf_rate = self.price_lvl - prev_price_lvl
        eta = self.stdev_eta * np.random.randn(self.num_agents)
        self.one_plus_shock = np.log(np.maximum(1e-10, 1+(self.rho * (np.exp(self.one_plus_shock)-1) + eta)))
        self.given_demand = self.given_demand - self.one_plus_inf_rate
        self.dem_after_shock = self.given_demand + self.one_plus_shock
        
        self.states = np.vstack((self.given_demand, self.one_plus_shock, self.one_plus_inf_rate, self.price_lvl, self.dem_after_shock))
        observations = {agent:self.states.flatten() for i, agent in enumerate(self.agents)}

        rewards = np.exp(self.total_demand) + self.net_ex
        rewards = dict(zip(self.agents, list(rewards)))
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

        return observations, rewards, terminations, truncations, infos
    


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test, render_test

    # np.seterr(divide='raise')
    # np.seterr
    env = EconomicsEnv()
    parallel_api_test(env, num_cycles=1_000)
    # render_test(EconomicsEnv)
