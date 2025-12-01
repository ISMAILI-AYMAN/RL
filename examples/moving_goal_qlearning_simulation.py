"""
Moving Goal Q-Learning Simulation

This script demonstrates Q-Learning in a dynamic environment where the goal
position changes after each episode. The agent must learn to adapt to new goal locations.
"""

import numpy as np
import time
from rl_env.envs.grid_world import GridWorldEnv
from rl_env.agents.q_learning_agent import QLearningAgent
from rl_env.simulation import Simulation


def generate_goal_positions(width: int, height: int, num_episodes: int, 
                           obstacles: list, initial_pos: np.ndarray) -> list:
    """
    Generate a sequence of goal positions that avoid obstacles and initial position.
    
    Args:
        width: Grid width
        height: Grid height
        num_episodes: Number of episodes (goal positions)
        obstacles: List of obstacle positions
        initial_pos: Initial agent position
        
    Returns:
        List of goal positions
    """
    goal_positions = []
    occupied_positions = set()
    
    # Add obstacles and initial position to occupied set
    for obs in obstacles:
        occupied_positions.add((obs[0], obs[1]))
    occupied_positions.add((initial_pos[0], initial_pos[1]))
    
    # Generate random goal positions
    np.random.seed(42)
    for _ in range(num_episodes):
        while True:
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            if (row, col) not in occupied_positions:
                goal_positions.append([row, col])
                break
    
    return goal_positions


def main():
    """Run moving goal Q-Learning simulation."""
    
    print("="*60)
    print("Moving Goal Q-Learning Simulation")
    print("="*60)
    print("Q-Learning agent adapts to goal position changes!")
    print("="*60)
    
    # Create environment
    print("\nCreating Grid World environment...")
    env = GridWorldEnv(
        width=8,
        height=8,
        initial_pos=np.array([0, 0]),
        goal_pos=np.array([7, 7]),  # Initial goal
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
        epsilon_decay=0.98,  # Decay epsilon faster for moving goal
        epsilon_min=0.05,
        seed=42
    )
    
    # Generate goal positions for multiple episodes
    num_episodes = 10
    # Get initial position (agent_pos might be None before reset)
    initial_pos = env._initial_pos if hasattr(env, '_initial_pos') else np.array([0, 0])
    goal_positions = generate_goal_positions(
        env.width, env.height, num_episodes,
        env.get_obstacles(), initial_pos
    )
    
    print(f"\nGenerated {num_episodes} goal positions:")
    for i, goal in enumerate(goal_positions):
        print(f"  Episode {i+1}: Goal at [{goal[0]}, {goal[1]}]")
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    episode_success = []
    
    # Run episodes with moving goal
    print("\n" + "="*60)
    print("Starting Q-Learning with Moving Goal")
    print("="*60)
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        print(f"Epsilon: {agent.epsilon:.3f}")
        
        # Set new goal position
        new_goal = goal_positions[episode]
        print(f"Moving goal to position [{new_goal[0]}, {new_goal[1]}]")
        env.set_goal_position(new_goal[0], new_goal[1])
        
        # Reset environment
        obs, info = env.reset(seed=42 if episode == 0 else None)
        agent.reset()  # This also decays epsilon
        
        # Render initial state
        env.render()
        time.sleep(0.3)
        
        # Run episode
        episode_reward = 0.0
        episode_length = 0
        done = False
        max_steps = 200
        
        while not done and episode_length < max_steps:
            # Select action
            action = agent.select_action(obs, info)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update agent (Q-learning)
            agent.update(obs, action, reward, next_obs, terminated, truncated, info)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Render (less frequently for speed)
            if episode_length % 2 == 0:  # Render every 2 steps
                env.render()
                time.sleep(0.02)
            
            obs = next_obs
        
        # Check if goal was reached
        reached_goal = np.array_equal(env.get_agent_position(), env.get_goal_position())
        episode_success.append(reached_goal)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if reached_goal:
            print(f"✓ Episode {episode + 1} SUCCESS: Reached goal!")
            print(f"  Reward: {episode_reward:.2f}, Steps: {episode_length}")
        else:
            print(f"✗ Episode {episode + 1} FAILED: Did not reach goal")
            print(f"  Reward: {episode_reward:.2f}, Steps: {episode_length}")
            print(f"  Final position: {env.get_agent_position()}")
            print(f"  Goal position: {env.get_goal_position()}")
        
        # Pause between episodes
        time.sleep(0.5)
    
    # Final statistics
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total episodes: {num_episodes}")
    print(f"Successful episodes: {sum(episode_success)}/{num_episodes}")
    print(f"Success rate: {100*sum(episode_success)/num_episodes:.1f}%")
    print(f"\nMean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    
    # Show learning progress
    print("\nLearning Progress:")
    first_half = episode_success[:num_episodes//2]
    second_half = episode_success[num_episodes//2:]
    print(f"  First half success rate: {100*sum(first_half)/len(first_half):.1f}%")
    print(f"  Second half success rate: {100*sum(second_half)/len(second_half):.1f}%")
    
    print("\nGoal positions visited:")
    for i, goal in enumerate(goal_positions):
        status = "✓" if episode_success[i] else "✗"
        print(f"  Episode {i+1}: [{goal[0]}, {goal[1]}] {status}")
    
    print("="*60)
    
    # Keep window open
    print("\nSimulation completed!")
    print("Close the matplotlib window to exit.")
    
    # Cleanup
    input("\nPress Enter to close...")
    env.close()


if __name__ == "__main__":
    main()

