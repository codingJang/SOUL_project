from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import air
from ray import tune

config = DQNConfig()
config = config.training(
    n_step=tune.grid_search([1, 2, 4, 8]),
    noisy=tune.grid_search([False, True]),
    num_atoms=tune.grid_search([1, 2, 4, 8]),
    double_q=tune.grid_search([False, True]),
    dueling=tune.grid_search([False, True]),
    )
config = config.environment(env="ALE/MsPacman-v5") 
config = config.rollouts(num_rollout_workers=7)
tune.Tuner(  
    "DQN",
    run_config=air.RunConfig(stop={"episode_reward_mean":1000, "training_iteration":100}),
    param_space=config.to_dict()
).fit()