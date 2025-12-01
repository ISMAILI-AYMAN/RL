"""
Q-Learning Simulation

This script demonstrates Q-Learning algorithm for learning optimal policy
in the Grid World environment.
"""

import numpy as np
from rl_env.envs.grid_world import GridWorldEnv
from rl_env.agents.q_learning_agent import QLearningAgent
from rl_env.simulation import Simulation


def print_q_table_stats(agent: QLearningAgent, env: GridWorldEnv):
    """Print Q-table statistics."""
    q_table = agent.get_q_table()
    
    print("\n" + "="*60)
    print("Q-Table Statistics")
    print("="*60)
    print(f"Q-table shape: {q_table.shape}")
    print(f"Max Q-value: {np.max(q_table):.2f}")
    print(f"Min Q-value: {np.min(q_table):.2f}")
    print(f"Mean Q-value: {np.mean(q_table):.2f}")
    print(f"Non-zero entries: {np.count_nonzero(q_table)} / {q_table.size}")
    print(f"Current epsilon: {agent.epsilon:.4f}")
    print("="*60)


def visualize_policy(agent: QLearningAgent, env: GridWorldEnv):
    """Visualize learned policy by showing best action for each state."""
    width = env.width
    height = env.height
    
    # Action symbols
    action_symbols = {
        0: '↑',  # Up
        1: '→',  # Right
        2: '↓',  # Down
        3: '←'   # Left
    }
    
    print("\n" + "="*60)
    print("Learned Policy (Best Action for each state)")
    print("="*60)
    
    for row in range(height):
        line = ""
        for col in range(width):
            # Check if obstacle
            is_obstacle = False
            for obs in env.obstacles:
                if obs[0] == row and obs[1] == col:
                    is_obstacle = True
                    break
            
            if is_obstacle:
                line += " X "
            elif row == env.goal_pos[0] and col == env.goal_pos[1]:
                line += " G "
            else:
                # Get best action for this state
                state_idx = row * agent.grid_width + col
                q_values = agent.q_table[state_idx]
                best_action = np.argmax(q_values)
                line += f" {action_symbols[best_action]} "
        print(line)
    print("="*60)


def main():
    """Run Q-Learning simulation."""
    
    print("="*60)
    print("Q-Learning Simulation")
    print("="*60)
    
    # Create environment
    print("\nCreating Grid World environment...")
    env = GridWorldEnv(
        width=8,
        height=8,
        initial_pos=np.array([0, 0]),
        goal_pos=np.array([7, 7]),
        obstacles=[
            [2, 2],
            [2, 3],
            [3, 2],
            [5, 5],
            [5, 6],
            [6, 5]
        ],
        render_mode="human"
    )
    
    # Create Q-Learning agent
    print("Creating Q-Learning agent...")
    agent = QLearningAgent(
        action_space=env.action_space,
        grid_width=env.width,
        grid_height=env.height,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.9,  # Start with high exploration
        epsilon_decay=0.995,
        epsilon_min=0.01,
        seed=42
    )
    
    # Training phase
    print("\n" + "="*60)
    print("Training Phase")
    print("="*60)
    
    sim_train = Simulation(
        env=env,
        agent=agent,
        max_steps=200,
        render=False,  # Don't render during training
        verbose=False   # Less verbose during training
    )
    
    num_training_episodes = 100
    print(f"\nTraining for {num_training_episodes} episodes...")
    
    training_stats = sim_train.run_episodes(num_episodes=num_training_episodes, seed=42)
    
    print(f"\nTraining completed!")
    print(f"Mean reward: {training_stats['mean_reward']:.2f}")
    print(f"Best reward: {training_stats['max_reward']:.2f}")
    
    # Print Q-table statistics
    print_q_table_stats(agent, env)
    
    # Visualize learned policy
    visualize_policy(agent, env)
    
    # Testing phase (with low epsilon for exploitation)
    print("\n" + "="*60)
    print("Testing Phase (Exploitation)")
    print("="*60)
    
    agent.set_epsilon(0.0)  # No exploration during testing
    
    sim_test = Simulation(
        env=env,
        agent=agent,
        max_steps=200,
        render=True,   # Render during testing
        verbose=True   # Verbose during testing
    )
    
    num_test_episodes = 5
    print(f"\nTesting for {num_test_episodes} episodes with epsilon=0.0...")
    
    test_stats = sim_test.run_episodes(num_episodes=num_test_episodes, seed=123)
    
    print(f"\nTesting completed!")
    print(f"Mean reward: {test_stats['mean_reward']:.2f}")
    print(f"Best reward: {test_stats['max_reward']:.2f}")
    print(f"Mean episode length: {test_stats['mean_length']:.2f}")
    
    # Cleanup
    env.close()
    
    print("\nQ-Learning Simulation completed!")


if __name__ == "__main__":
    main()

