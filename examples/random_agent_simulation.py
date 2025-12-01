"""
Example: Random Agent Simulation

This script demonstrates running a simulation with a random agent
in the Grid World environment.
"""

import numpy as np
from rl_env.envs.grid_world import GridWorldEnv
from rl_env.agents.random_agent import RandomAgent
from rl_env.simulation import Simulation


def main():
    """Run random agent simulation."""
    
    # Create environment with custom configuration
    print("Creating Grid World environment...")
    env = GridWorldEnv(
        width=8,                    # Grid width
        height=8,                   # Grid height
        initial_pos=np.array([0, 0]),  # Agent starts at top-left
        goal_pos=np.array([7, 7]),     # Goal at bottom-right
        obstacles=[                  # Obstacle positions
            [2, 2],
            [2, 3],
            [3, 2],
            [5, 5],
            [5, 6],
            [6, 5]
        ],
        render_mode="human"
    )
    
    # Create random agent
    print("Creating random agent...")
    agent = RandomAgent(env.action_space, seed=42)
    
    # Create simulation
    print("Setting up simulation...")
    sim = Simulation(
        env=env,
        agent=agent,
        max_steps=200,  # Maximum steps per episode
        render=True,    # Enable visualization
        verbose=True    # Print episode information
    )
    
    # Run simulation for multiple episodes
    print("\n" + "="*50)
    print("Starting Random Agent Simulation")
    print("="*50)
    
    stats = sim.run_episodes(num_episodes=3, seed=42)
    
    # Cleanup
    env.close()
    
    print("\nSimulation completed!")


if __name__ == "__main__":
    main()

