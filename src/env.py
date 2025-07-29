from ray.rllib.env import MultiAgentEnv
from pettingzoo.utils.env import ParallelEnv
import gymnasium as gym
from gymnasium.spaces import Dict, Box
import numpy as np
from copy import copy
from configs.environment_config import MacroSimEnvConfig

# Load default configuration
ENV_CONFIG = MacroSimEnvConfig()
N = ENV_CONFIG.num_agents
obs_space = Box(low=-np.inf, high=np.inf, shape=ENV_CONFIG.observation_space_shape)
act_space = Dict({'eco': Box(low=-np.inf, high=np.inf, shape=(1,)), 'pol': Box(low=-np.inf, high=np.inf, shape=(2 * N,))})


class MacroSimEnv(gym.Env):
    """
    Base class for macro-economic simulation environments.
    
    This environment simulates a macro-economic system with N agents that make
    both economic decisions (interest rates) and political decisions (diplomatic actions).
    Each agent observes economic indicators and makes decisions that affect the entire system.
    """
    
    metadata = {
        "render_modes": ['human', 'array'],
        "name": "macro_sim_env_v0"
    }

    def __init__(self, render_mode=None, config: MacroSimEnvConfig = None):
        super().__init__()
        self.config = config or MacroSimEnvConfig()
        self.config.validate()
        
        self._num_agents = self.config.num_agents
        self.render_mode = render_mode
        
        # Economic simulation parameters from config
        self.rho = self.config.rho
        self.STD_ETA = self.config.std_eta
        self.std_gd = self.config.std_given_demand
        self.std_ops = self.config.std_one_plus_shock
        self.std_ppl = self.config.std_prev_price_lvl
        self.std_pl = self.config.std_price_lvl
        self.std_pne = self.config.std_prev_net_ex
        self.std_ne = self.config.std_net_ex
        self.shock_lvl = self.config.shock_lvl
        self.ex_int_degree = self.config.ex_int_degree
        self.demand_penalty = self.config.demand_penalty
        self.delta = self.config.delta
        
        # Initialize timestep
        self.t = 0

    @property
    def num_agents(self):
        """Return the number of agents (read-only property)."""
        return self._num_agents

    def _render_human(self):
        """Render environment state in human-readable format."""
        if self.t == 0:
            print("Current timestep:", self.t)
            print("\nObservables:")
            observable_vars = ['dem_after_shock', 'prev_price_lvl', 'price_lvl']
            for var in observable_vars:
                print(f"{var.upper()}: {np.exp(getattr(self, var))}")
            print("PREV_NET_EX:", self.PREV_NET_EX)
            print()
            print()
            return

        print("Current timestep:", self.t)
        agent_ids = getattr(self, '_agent_ids', getattr(self, 'agents', []))
        print("Agents:", agent_ids)
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
            'agents': agent_ids,
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

    def render(self, mode=None):
        """Render the environment based on the render_mode."""
        render_mode = mode or self.render_mode
        if render_mode == 'human':
            return self._render_human()
        elif render_mode == 'array':
            return self._render_array()
        else:
            raise NotImplementedError(f"Render mode {render_mode} not supported")
    
    def close(self):
        """Close the environment."""
        pass

    def _reset_economic_state(self, seed=None):
        """Reset the economic state variables."""
        if seed is not None:
            np.random.seed(seed)
            
        self.t = 0
        # all small letter variables denote logarithmic variables
        # all capital letter variables are non-logarithmetic variables
        # ex: self.one_plus_inf_rate = ln(1 + (INFLATION_RATE))
        self.given_demand = self.std_gd * np.random.randn(self._num_agents)
        self.one_plus_shock = self.std_ops * np.random.randn(self._num_agents)
        self.dem_after_shock = self.given_demand + self.one_plus_shock
        self.prev_price_lvl = self.std_ppl * np.random.randn(self._num_agents)
        self.price_lvl = self.std_pl * np.random.randn(self._num_agents)
        self.PREV_NET_EX = np.exp(self.std_pne * np.random.randn(self._num_agents))
        self.NET_EX = np.exp(self.std_ne * np.random.randn(self._num_agents))
        self.eco_observation = np.vstack((self.dem_after_shock, self.prev_price_lvl, self.price_lvl, self.PREV_NET_EX)).T
        self.affinity = np.eye(self._num_agents)
        self.delta_affinity = self.affinity

    def _step_economic_dynamics(self, eco_actions):
        """Execute economic dynamics for one step."""
        # Economic dynamics
        self.one_plus_int_rate = 0.20 / (1 + np.exp(-np.array(eco_actions).squeeze()))
        self.total_demand = self.dem_after_shock - self.one_plus_int_rate
        price_diff = self.price_lvl.reshape(-1, 1) - self.price_lvl.reshape(1, -1)
        int_rate_diff = self.one_plus_int_rate.reshape(-1, 1) - self.one_plus_int_rate.reshape(1, -1)
        self.nom_exchange_rate = price_diff - self.ex_int_degree * int_rate_diff
        self.real_exchange_rate = - self.ex_int_degree * int_rate_diff
        
        # Trade dynamics
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
        
        # Price level updates
        self.prev_price_lvl = self.price_lvl
        self.price_lvl = np.log(np.exp(self.price_lvl)) - self.one_plus_int_rate
        self.price_lvl = (self.price_lvl - np.mean(self.price_lvl)) / np.std(self.price_lvl)
        self.one_plus_inf_rate = self.price_lvl - self.prev_price_lvl
        self.given_demand = self.given_demand - self.demand_penalty * self.one_plus_inf_rate
        self.ETA = self.STD_ETA * np.random.randn(self.num_agents)
        self.one_plus_shock = np.log(np.maximum(1e-10, 1+(self.rho * (np.exp(self.one_plus_shock)-1) + self.ETA)))
        self.dem_after_shock = self.given_demand + self.shock_lvl * self.one_plus_shock
        self.eco_observation = np.vstack((self.dem_after_shock, self.prev_price_lvl, self.price_lvl, self.PREV_NET_EX)).T
        self.GDP = np.exp(self.total_demand) + self.NET_EX

    def _step_political_dynamics(self, pol_actions):
        """Execute political dynamics for one step."""
        invites = []
        accepts = []
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        
        for i, pol_action in enumerate(pol_actions):
            invite_pref = pol_action[:self._num_agents]
            accept_pref = pol_action[self._num_agents:]
            invite_prob = softmax(invite_pref)
            accept_prob = sigmoid(accept_pref)
            invite_choice = np.random.choice(self._num_agents, p=invite_prob)
            invite = np.zeros(self._num_agents)
            invite[invite_choice] = 1
            accept = np.random.uniform(size=self._num_agents) < accept_prob
            invite[i] = 0
            accept[i] = 0
            invites.append(invite)
            accepts.append(accept)
        
        invites = np.vstack(invites)
        accepts = np.vstack(accepts)
        delta_affinity = self.delta * 0.5 * (accepts.T * invites + invites.T * accepts)
        self.delta_affinity = delta_affinity
        self.affinity += delta_affinity

    def _build_observation(self, agent_idx):
        """Build observation for a specific agent."""
        observation = np.hstack((self.eco_observation, self.affinity))
        observation = np.roll(observation, agent_idx, axis=0)
        return observation.flatten().astype(np.float32)

    def _calculate_rewards(self):
        """Calculate rewards for all agents."""
        return self.affinity @ self.GDP


class MacroSimRayRLlibEnv(MacroSimEnv, MultiAgentEnv):
    metadata = {
        "render_modes": ['human', 'array'],
        "name": "macro_sim_rllib_env_v0"
    }

    observation_space = Dict({f'agent_{i}': obs_space for i in range(N)})
    action_space = Dict({f'agent_{i}': act_space for i in range(N)})
    _agent_ids = [f'agent_{i}' for i in range(N)]

    def __init__(self, render_mode=None, config: MacroSimEnvConfig = None):
        # Initialize both parent classes
        MacroSimEnv.__init__(self, render_mode, config)
        MultiAgentEnv.__init__(self)

    def render(self, mode='array'):
        if mode == 'human':
            return self._render_human()
        elif mode == 'array':
            return self._render_array()
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")

    def reset(self, *, seed=None, options=None):
        self._reset_economic_state(seed)
        observations = {}
        for i, agent in enumerate(self._agent_ids):
            observations[agent] = self._build_observation(i)
        infos = {agent:{} for agent in self._agent_ids}
        return observations, infos

    def step(self, actions):
        eco_actions = [actions[agent]['eco'] for agent in self._agent_ids if 'eco' in actions[agent]]
        pol_actions = [actions[agent]['pol'] for agent in self._agent_ids if 'pol' in actions[agent]]

        # Execute dynamics using base class methods
        self._step_economic_dynamics(eco_actions)
        self._step_political_dynamics(pol_actions)

        observations = {}
        for i, agent in enumerate(self._agent_ids):
            observations[agent] = self._build_observation(i)

        rewards = self._calculate_rewards()
        rewards = dict(zip(self._agent_ids, list(rewards)))
        terminateds = {agent:False for agent in self._agent_ids}
        truncateds = {agent:False for agent in self._agent_ids}
        infos = {agent:{} for agent in self._agent_ids}

        self.t += 1
        self.render(mode=self.render_mode)

        if self.t >= self.config.max_episode_steps:
            truncateds = {agent:True for agent in self._agent_ids}
            terminateds = {agent:True for agent in self._agent_ids}
        
        terminateds['__all__'] = all(terminateds.values())
        truncateds['__all__'] = all(truncateds.values())

        return observations, rewards, terminateds, truncateds, infos


class MacroSimPettingZooEnv(MacroSimEnv, ParallelEnv):
    """
    PettingZoo Parallel Environment for multi-agent economic simulation.
    
    This environment simulates a macro-economic system with N agents that make
    both economic decisions (interest rates) and political decisions (diplomatic actions).
    Each agent observes economic indicators and makes decisions that affect the entire system.
    """
    
    metadata = {
        "render_modes": ['human', 'array'],
        "name": "macro_sim_pettingzoo_env_v0"
    }

    def __init__(self, render_mode=None, config: MacroSimEnvConfig = None):
        # Initialize both parent classes
        MacroSimEnv.__init__(self, render_mode, config)
        ParallelEnv.__init__(self)
        
        self.possible_agents = [f"agent_{i}" for i in range(self._num_agents)]
        self.agents = copy(self.possible_agents)
        
        # PettingZoo required attributes
        self.observation_spaces = {agent: obs_space for agent in self.possible_agents}
        self.action_spaces = {agent: act_space for agent in self.possible_agents}

    def observation_space(self, agent):
        """Return the observation space for the given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for the given agent."""
        return self.action_spaces[agent]

    def render(self):
        """Render the environment based on the render_mode."""
        return super().render()

    def state(self):
        """
        Return the global state of the environment.
        
        This provides a centralized view for algorithms that require global information.
        """
        return self._render_array()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            options: Optional dictionary of options
            
        Returns:
            observations: Dict of initial observations for each agent
            infos: Dict of info dictionaries for each agent
        """
        self._reset_economic_state(seed)
        # Reset all agents
        self.agents = copy(self.possible_agents)
        
        # Build observations for each agent
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._build_observation(i)
        
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """
        Execute one step of the environment.
        
        Args:
            actions: Dict mapping agent names to their actions
            
        Returns:
            observations: Dict of observations for each agent
            rewards: Dict of rewards for each agent  
            terminations: Dict of termination flags for each agent
            truncations: Dict of truncation flags for each agent
            infos: Dict of info dictionaries for each agent
        """
        # Extract economic and political actions
        eco_actions = [actions[agent]['eco'] for agent in self.agents if 'eco' in actions[agent]]
        pol_actions = [actions[agent]['pol'] for agent in self.agents if 'pol' in actions[agent]]

        # Execute dynamics using base class methods
        self._step_economic_dynamics(eco_actions)
        self._step_political_dynamics(pol_actions)  # PettingZoo uses standard implementation

        # Build observations
        observations = {}
        for i, agent in enumerate(self.agents):
            observations[agent] = self._build_observation(i)

        # Calculate rewards (affinity-weighted GDP)
        rewards = self._calculate_rewards()
        rewards = dict(zip(self.agents, list(rewards)))
        
        # Check for episode termination
        self.t += 1
        episode_over = self.t >= self.config.max_episode_steps
        
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: episode_over for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Render if specified
        if self.render_mode:
            self.render()

        return observations, rewards, terminations, truncations, infos


def env(**kwargs):
    """Factory function to create a MacroSimEnv instance."""
    return MacroSimPettingZooEnv(**kwargs)


def parallel_env(**kwargs):
    """Factory function to create a MacroSimEnv instance (PettingZoo convention)."""
    return MacroSimPettingZooEnv(**kwargs) 