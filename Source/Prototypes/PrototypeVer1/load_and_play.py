from economics_env import *
import os
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

def get_human_action(env):
    """
    Get the action from the human player.
    Displays the current observation and prompts for a float value between 0 and 20.
    """
    env.render(mode='human')
    action = None
    while True:
        try:
            response = float(input("Enter your action (a float value between 0 and 20): "))
            if 0 <= response <= 20:
                action = response
                break
            else:
                print("Invalid input. Please enter a float between 0 and 20.")
        except ValueError:
            print("Invalid input. Please enter a valid float.")
    # compute the inverse of the sigmoid function
    # 1/(1+exp(-y))=x, x+x*exp(-y)=1, x*exp(-y)=1-x, exp(-y)=(1-x)/x, -y=log((1-x)/x), y=log(x/(1-x))
    # values range from 0 to 20, so we need to scale them to range from 0 to 1
    # y=log((x/20)/((1-x)/20))=log(x/(20-x))
    action = np.log(np.clip(action/(20-action), 1e-5, 1-1e-5))
    return action

def env_creator(args):
    env = EconomicsEnv()
    # env = ss.frame_stack_v1(env, 3)
    return env

# Initialize the PettingZoo environment
economics_env = env_creator({})
env_name = "economics_environment"
register_env(env_name, lambda config: ParallelPettingZooEnv(economics_env))

# Reset the environment at the beginning of the game
observations, infos = economics_env.reset()

# Load trained policies for AI agents
num_agents = N  # Assuming N is defined in economics_env
policies = {}
my_experiment_name = "APPO_2023-11-27_21-47-18"
my_trial_name = "APPO_economics_environment_89a3a_00001_1_gamma=0.9523,lr=0.0001_2023-11-27_21-47-19"
checkpoint_folder_name = "checkpoint_000002"

for i in range(1, num_agents):
    checkpoint_path = os.path.expanduser(f"~/ray_results/{my_experiment_name}/{my_trial_name}/{checkpoint_folder_name}/policies/agent_{i}")
    policies[f'agent_{i}'] = Policy.from_checkpoint(checkpoint_path)

# Game loop
done = {f'agent_{i}': False for i in range(num_agents)}
while not all(done.values()):
    # Get human player's action for agent_0
    human_action = get_human_action(economics_env)

    # Get actions for AI agents
    actions = {f'agent_{i}': policies[f'agent_{i}'].compute_single_action(observations[f'agent_{i}'])[0] for i in range(1, num_agents)}

    # Add human action to actions
    actions['agent_0'] = np.array([human_action], dtype=np.float32)

    # Step the environment
    observations, rewards, terminateds, truncateds, infos = economics_env.step(actions)
    done = terminateds

    # Visualize game stats
    economics_env.render(mode='human')

# Close the environment
economics_env.close()
