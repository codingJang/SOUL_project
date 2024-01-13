import ray
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.numpy import one_hot
import numpy as np
from gymnasium.spaces import Dict, Box
from copy import copy


N = 7
obs_space = Box(low=-np.inf, high=np.inf, shape=((N + 4) * N,))
act_space = Dict({'eco':Box(low=-np.inf, high=np.inf, shape=(1,)), 'pol':Box(low=-np.inf, high=np.inf, shape=(2 * N,))})


class CombinedEnv(MultiAgentEnv):
    metadata = {
        "render_modes": ['human', 'array'],
        "name": "politics_environment_v0"
    }
    possible_agents = [f"agent_{i}" for i in range(N)]
    observation_spaces = {f'agent_{i}':obs_space for i in range(N)}
    action_spaces = {f'agent_{i}':act_space for i in range(N)}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.agents = copy(self.possible_agents)
        self.rho = 0.8
        self.STD_ETA = 0.03
        self.std_gd = 0.1
        self.std_ops = 0.1
        self.std_ppl = 0.1
        self.std_pl = 0.1
        self.std_pne = 0.1
        self.std_ne = 0.1
        self.ex_int_degree = 1
        self.demand_penalty = 1
        self.delta = 0.01 

    def render(self, mode='human'):
        if mode == 'human':
            return self._render_human()
        elif mode == 'array':
            return self._render_array()
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def _render_human(self):
        if self.t == 0:
            print("Current timestep:", self.t)

            print("\nObservables:")
            observable_vars = ['dem_after_shock', 'prev_price_lvl', 'price_lvl']
            for var in observable_vars:
                print(f"{var.upper()}: {np.exp(getattr(self, var))}")
            print("PREV_NET_EX:", self.PREV_NET_EX)
            print()
            print()

            return None

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
        
        print("\nAffinity Matrix:")
        print(self.affinity)
        print()
        print()

        render_res = {
            'ts': self.t,
            'agents': self.agents,
            'interest_rates': np.exp(self.one_plus_int_rate)-1,
            'gdp': self.GDP,
            'dem_after_shock': self.dem_after_shock, 
            'delta_price_lvl': self.price_lvl - self.prev_price_lvl, 
            'price_lvl': self.price_lvl,
            'affinity': self.affinity,
            'delta_affinity': self.delta_affinity
        }
        return render_res


    def _render_array(self):
        # Concatenate all relevant arrays into a single numpy array for more technical analysis
        state_arrays = [self.given_demand, self.one_plus_shock, self.one_plus_inf_rate, self.price_lvl, 
                        self.dem_after_shock, self.nom_exchange_rate, self.real_exchange_rate, 
                        self.TRADE_COEFF, self.EX, self.IM, self.NET_EX, self.price_lvl]
        return np.concatenate([arr.flatten() for arr in state_arrays])
    
    def close(self):
        pass

    def reset(self, *, seed=None, options=None):
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
        self.eco_observation = np.vstack((self.dem_after_shock, self.prev_price_lvl, self.price_lvl, self.PREV_NET_EX)).T
        self.affinity = np.eye(self.num_agents)
        self.delta_affinity = self.affinity
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = np.hstack((self.eco_observation, self.affinity))
            observations[agent] = np.roll(observations[agent], i, axis=0)
            observations[agent] = observations[agent].flatten().astype(np.float32)
        infos = {agent:{} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        
        eco_actions = [actions[agent]['eco'] for agent in self.agents if 'eco' in actions[agent]]
        pol_actions = [actions[agent]['pol'] for agent in self.agents if 'pol' in actions[agent]]

        self.one_plus_int_rate = 0.20 / (1 + np.exp(-np.array(eco_actions).squeeze()))
        self.total_demand = self.dem_after_shock - self.one_plus_int_rate
        price_diff = self.price_lvl.reshape(-1, 1) - self.price_lvl.reshape(1, -1)
        int_rate_diff = self.one_plus_int_rate.reshape(-1, 1) - self.one_plus_int_rate.reshape(1, -1)
        self.nom_exchange_rate = price_diff - self.ex_int_degree * int_rate_diff
        self.real_exchange_rate = - self.ex_int_degree * int_rate_diff
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
        self.price_lvl = np.log(np.exp(self.price_lvl)) + np.log(np.maximum(1e-10, 1 + self.NET_EX/np.exp(self.total_demand))) - self.one_plus_int_rate
        self.price_lvl = (self.price_lvl - np.mean(self.price_lvl)) / np.std(self.price_lvl)
        self.one_plus_inf_rate = self.price_lvl - self.prev_price_lvl
        self.given_demand = self.given_demand - self.demand_penalty * self.one_plus_inf_rate
        self.ETA = self.STD_ETA * np.random.randn(self.num_agents)
        self.one_plus_shock = np.log(np.maximum(1e-10, 1+(self.rho * (np.exp(self.one_plus_shock)-1) + self.ETA)))
        self.dem_after_shock = self.given_demand + self.one_plus_shock
        self.eco_observation = np.vstack((self.dem_after_shock, self.prev_price_lvl, self.price_lvl, self.PREV_NET_EX)).T
        self.GDP = np.exp(self.total_demand) + self.NET_EX
        
        invites = []
        accepts = []
        softmax = lambda x : np.exp(x) / np.sum(np.exp(x), axis=0)
        sigmoid = lambda x : 1 / (1 + np.exp(x))
        for i, pol_action in enumerate(pol_actions):
            invite_pref = pol_action[:self.num_agents]
            accept_pref = pol_action[self.num_agents:]
            invite_prob = softmax(invite_pref)
            accept_prob = sigmoid(accept_pref)
            invite_choice = np.random.choice(self.num_agents, p=invite_prob)
            invite = one_hot(invite_choice, depth=self.num_agents)
            print(accept_prob)
            accept = np.random.uniform(size=self.num_agents) < accept_prob
            invite[i] = 0
            accept[i] = 0
            invites.append(invite)
            accepts.append(accept)
        invites = np.vstack(invites)
        accepts = np.vstack(accepts)
        delta_affinity = self.delta * 0.5 * (accepts.T * invites + invites.T * accepts)
        self.delta_affinity = delta_affinity
        self.affinity += delta_affinity

        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = np.hstack((self.eco_observation, self.affinity))
            observations[agent] = np.roll(observations[agent], i, axis=0)
            observations[agent] = observations[agent].flatten().astype(np.float32)

        print(self.affinity)
        print(self.GDP)
        rewards = self.affinity @ self.GDP
        rewards = dict(zip(self.agents, list(rewards)))
        terminations = {agent:False for agent in self.agents}
        truncations = {agent:False for agent in self.agents}
        infos = {agent:{} for agent in self.agents}

        self.t += 1
        self.render(mode=self.render_mode)

        if self.t >= 100:
            truncations = {agent:True for agent in self.agents}
            terminations = {agent:True for agent in self.agents}
        
        if all(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos


if __name__ == "__main__":
    from pettingzoo.test import parallel_api_test, render_test

    np.seterr(all='raise', divide='raise', over='raise', under='raise', invalid='raise')
    # np.seterr
    my_env = CombinedEnv(render_mode='human')
    parallel_api_test(my_env, num_cycles=1_000)
