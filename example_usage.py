#!/usr/bin/env python3
"""
Example script demonstrating the new direct import capabilities.

This script shows how you can now import environment and configuration
classes directly without navigating the package structure.
"""

# Direct imports - much cleaner!
import env
from env import MacroSimEnv, MacroSimPettingZooEnv
from config import MacroSimEnvConfig, RLlibTrainConfig, SB3TrainConfig

def main():
    print("🚀 SOUL Project - Direct Import Demo")
    print("=" * 50)
    
    # Create custom configurations
    print("Creating configurations...")
    env_config = MacroSimEnvConfig(
        num_agents=5,
        max_episode_steps=50,
        rho=0.85
    )
    
    rllib_config = RLlibTrainConfig()
    sb3_config = SB3TrainConfig()
    
    # Validate all configs
    env_config.validate()
    rllib_config.validate() 
    sb3_config.validate()
    print("✓ All configurations validated")
    
    # Create environments
    print("\nCreating environments...")
    
    # Base environment
    base_env = MacroSimEnv(config=env_config)
    print(f"✓ MacroSimEnv: {base_env.num_agents} agents, rho={base_env.rho}")
    
    # PettingZoo environment for SB3
    pz_env = MacroSimPettingZooEnv(config=env_config)
    print(f"✓ PettingZooEnv: {pz_env.num_agents} agents")
    
    # Show platform detection
    print(f"\nPlatform Detection:")
    print(f"  Local machine: {rllib_config.is_local}")
    print(f"  RLlib workers: {rllib_config.num_workers}")
    print(f"  SB3 environments: {sb3_config.n_envs}")
    
    # Reset and step example
    print(f"\nEnvironment Demo:")
    obs, info = pz_env.reset()
    print(f"  Reset successful: {len(obs)} agent observations")
    print(f"  Observation shape: {list(obs.values())[0].shape}")
    
    # Clean up
    pz_env.close()
    base_env.close()
    
    print("\n🎉 Direct import demo completed successfully!")
    print("\nYou can now use these simple imports in your code:")
    print("  import env")
    print("  from env import MacroSimEnv")
    print("  from config import MacroSimEnvConfig")

if __name__ == "__main__":
    main() 