"""
Markov Reward Process (MRP) Simulation

This script demonstrates computing the value function for a Markov Reward Process.
In an MRP, we compute the expected cumulative reward from each state.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rl_env.envs.grid_world import GridWorldEnv


def compute_value_function(env: GridWorldEnv, discount_factor: float = 0.99, 
                          tolerance: float = 1e-6, max_iterations: int = 1000):
    """
    Compute value function using iterative policy evaluation (for MRP).
    
    V(s) = R(s) + gamma * sum(P(s'|s) * V(s'))
    
    For deterministic transitions in grid world, this simplifies to:
    V(s) = R(s) + gamma * V(s')
    
    Args:
        env: Grid World environment
        discount_factor: Discount factor (gamma)
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
        
    Returns:
        value_function: Dictionary mapping (row, col) -> value
    """
    width = env.width
    height = env.height
    
    # Initialize value function
    value_function = np.zeros((height, width))
    
    # Get reward function
    def get_reward(row, col):
        """Get reward for being in a state."""
        pos = np.array([row, col], dtype=np.int32)
        
        # Check if goal
        if np.array_equal(pos, env.goal_pos):
            return 10.0
        
        # Check if obstacle (shouldn't happen in valid states, but handle it)
        for obs in env.obstacles:
            if np.array_equal(pos, obs):
                return -1.0
        
        # Default step reward
        return -0.1
    
    # Iterative value function computation
    for iteration in range(max_iterations):
        new_value_function = np.zeros((height, width))
        max_change = 0.0
        
        for row in range(height):
            for col in range(width):
                # Skip obstacles (they are terminal/blocking states)
                is_obstacle = False
                for obs in env.obstacles:
                    if obs[0] == row and obs[1] == col:
                        is_obstacle = True
                        break
                
                if is_obstacle:
                    new_value_function[row, col] = -1.0  # Obstacle value
                    continue
                
                # Check if goal (terminal state)
                if row == env.goal_pos[0] and col == env.goal_pos[1]:
                    new_value_function[row, col] = get_reward(row, col)
                    continue
                
                # Compute value for this state
                # For each possible action, compute expected value
                action_values = []
                
                # Action mapping: 0=Up, 1=Right, 2=Down, 3=Left
                actions = [
                    (-1, 0),   # Up
                    (0, 1),    # Right
                    (1, 0),    # Down
                    (0, -1)    # Left
                ]
                
                for dr, dc in actions:
                    next_row = np.clip(row + dr, 0, height - 1)
                    next_col = np.clip(col + dc, 0, width - 1)
                    
                    # Check obstacle collision
                    hit_obstacle = False
                    for obs in env.obstacles:
                        if obs[0] == next_row and obs[1] == next_col:
                            hit_obstacle = True
                            break
                    
                    if hit_obstacle:
                        # Stay in current state if hit obstacle
                        next_row, next_col = row, col
                    
                    # Compute value
                    reward = get_reward(row, col)
                    next_value = value_function[next_row, next_col]
                    action_value = reward + discount_factor * next_value
                    action_values.append(action_value)
                
                # For MRP, we assume uniform random policy (all actions equally likely)
                new_value_function[row, col] = np.mean(action_values)
                max_change = max(max_change, abs(new_value_function[row, col] - value_function[row, col]))
        
        value_function = new_value_function.copy()
        
        # Check convergence
        if max_change < tolerance:
            print(f"Value function converged after {iteration + 1} iterations")
            break
    
    return value_function


def visualize_value_function(value_function: np.ndarray, env: GridWorldEnv):
    """Visualize value function using matplotlib heatmap."""
    height, width = value_function.shape
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare data for visualization (mask obstacles)
    display_values = value_function.copy()
    for obs in env.obstacles:
        display_values[obs[0], obs[1]] = np.nan
    
    # Heatmap visualization
    im = ax1.imshow(display_values, cmap='RdYlGn', aspect='auto', 
                    interpolation='nearest', vmin=np.nanmin(value_function), 
                    vmax=np.nanmax(value_function))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Value (Expected Cumulative Reward)', rotation=270, labelpad=20)
    
    # Add grid
    ax1.set_xticks(range(width))
    ax1.set_yticks(range(height))
    ax1.grid(True, color='black', linewidth=0.5, alpha=0.3)
    ax1.set_title('Value Function Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add text annotations for values
    for row in range(height):
        for col in range(width):
            # Check if obstacle
            is_obstacle = False
            for obs in env.obstacles:
                if obs[0] == row and obs[1] == col:
                    is_obstacle = True
                    break
            
            if is_obstacle:
                ax1.text(col, row, 'X', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='black')
            elif row == env.goal_pos[0] and col == env.goal_pos[1]:
                ax1.text(col, row, 'G', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='white')
            else:
                val = value_function[row, col]
                text_color = 'white' if val < np.nanmean(value_function) else 'black'
                ax1.text(col, row, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')
    
    # Text-based visualization
    ax2.axis('off')
    ax2.set_title('Value Function (Text Format)', fontsize=14, fontweight='bold', pad=20)
    
    text_content = "Value Function (Expected Cumulative Reward)\n" + "="*60 + "\n\n"
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
                line += "  XXXX  "
            elif row == env.goal_pos[0] and col == env.goal_pos[1]:
                line += "  GOAL  "
            else:
                line += f" {value_function[row, col]:6.2f} "
        text_content += line + "\n"
    text_content += "="*60
    
    ax2.text(0.05, 0.95, text_content, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def print_value_function(value_function: np.ndarray, env: GridWorldEnv):
    """Print value function in a readable format."""
    height, width = value_function.shape
    
    print("\n" + "="*60)
    print("Value Function (Expected Cumulative Reward from each state)")
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
                line += "  XXXX  "
            elif row == env.goal_pos[0] and col == env.goal_pos[1]:
                line += f"  GOAL  "
            else:
                line += f" {value_function[row, col]:6.2f} "
        print(line)
    print("="*60)


def simulate_agent_movement(env: GridWorldEnv, value_function: np.ndarray, 
                            num_episodes: int = 3, max_steps: int = 200):
    """
    Simulate agent movement through the environment with value function visualization.
    
    Args:
        env: Grid World environment
        value_function: Computed value function
        num_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
    """
    import time
    from rl_env.agents.random_agent import RandomAgent
    
    # Create random agent (uniform random policy for MRP)
    agent = RandomAgent(env.action_space, seed=42)
    
    # Create figure for live visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prepare value function data (mask obstacles)
    display_values = value_function.copy()
    for obs in env.obstacles:
        display_values[obs[0], obs[1]] = np.nan
    
    # Initialize heatmap
    im = ax1.imshow(display_values, cmap='RdYlGn', aspect='auto', 
                    interpolation='nearest', vmin=np.nanmin(value_function), 
                    vmax=np.nanmax(value_function))
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Value (Expected Cumulative Reward)', rotation=270, labelpad=20)
    
    ax1.set_xticks(range(env.width))
    ax1.set_yticks(range(env.height))
    ax1.grid(True, color='black', linewidth=0.5, alpha=0.3)
    ax1.set_title('Value Function with Agent Movement', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add text annotations for values
    text_annotations = []
    for row in range(env.height):
        row_texts = []
        for col in range(env.width):
            is_obstacle = False
            for obs in env.obstacles:
                if obs[0] == row and obs[1] == col:
                    is_obstacle = True
                    break
            
            if is_obstacle:
                text = ax1.text(col, row, 'X', ha='center', va='center',
                              fontsize=10, fontweight='bold', color='black')
            elif row == env.goal_pos[0] and col == env.goal_pos[1]:
                text = ax1.text(col, row, 'G', ha='center', va='center',
                              fontsize=12, fontweight='bold', color='white')
            else:
                val = value_function[row, col]
                text_color = 'white' if val < np.nanmean(value_function) else 'black'
                text = ax1.text(col, row, f'{val:.1f}', ha='center', va='center',
                              fontsize=8, color=text_color, fontweight='bold')
            row_texts.append(text)
        text_annotations.append(row_texts)
    
    # Agent visualization
    agent_circle = None
    agent_text = None
    
    # Statistics plot
    ax2.set_title('Episode Statistics', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    episode_rewards = []
    episode_lengths = []
    cumulative_reward = 0.0
    
    # Run episodes
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 if episode == 0 else None)
        agent.reset()
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Remove old agent visualization
        if agent_circle is not None:
            agent_circle.remove()
        if agent_text is not None:
            agent_text.remove()
        
        # Draw agent at start
        agent_y, agent_x = obs
        agent_circle = patches.Circle(
            (agent_x, agent_y), 0.3,
            color='blue', ec='darkblue', linewidth=3, zorder=5
        )
        ax1.add_patch(agent_circle)
        agent_text = ax1.text(agent_x, agent_y, 'A', ha='center', va='center',
                             fontsize=14, fontweight='bold', color='white', zorder=6)
        
        plt.draw()
        plt.pause(0.5)
        
        step = 0
        while not done and step < max_steps:
            # Select action
            action = agent.select_action(obs, info)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Update agent visualization
            agent_y, agent_x = next_obs
            agent_circle.center = (agent_x, agent_y)
            agent_text.set_position((agent_x, agent_y))
            
            # Update statistics display
            cumulative_reward += reward
            stats_text = f"""
Episode {episode + 1}/{num_episodes}
Step: {step + 1}
Current Position: [{agent_y}, {agent_x}]
Value at position: {value_function[agent_y, agent_x]:.2f}
Episode Reward: {episode_reward:.2f}
Episode Length: {episode_length}
Cumulative Reward: {cumulative_reward:.2f}
            """
            ax2.clear()
            ax2.axis('off')
            ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes,
                    fontsize=12, verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.draw()
            plt.pause(0.1)  # Animation speed
            
            obs = next_obs
            step += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Pause at end of episode
        plt.pause(1.0)
        
        if done:
            print(f"Episode {episode + 1} completed: reward={episode_reward:.2f}, length={episode_length}")
        else:
            print(f"Episode {episode + 1} truncated: reward={episode_reward:.2f}, length={episode_length}")
    
    # Final statistics
    final_stats = f"""
SIMULATION COMPLETE

Episodes: {num_episodes}
Mean Reward: {np.mean(episode_rewards):.2f}
Mean Length: {np.mean(episode_lengths):.2f}
Total Reward: {cumulative_reward:.2f}
    """
    ax2.clear()
    ax2.axis('off')
    ax2.text(0.1, 0.5, final_stats, transform=ax2.transAxes,
            fontsize=12, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.draw()
    plt.show()


def main():
    """Run Markov Reward Process simulation."""
    
    print("="*60)
    print("Markov Reward Process (MRP) Simulation")
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
        render_mode=None  # We'll handle rendering ourselves
    )
    
    # Compute value function
    print("\nComputing value function...")
    value_function = compute_value_function(env, discount_factor=0.99)
    
    # Print results
    print_value_function(value_function, env)
    
    # Show some statistics
    print("\nStatistics:")
    print(f"Max value: {np.max(value_function):.2f}")
    print(f"Min value: {np.min(value_function):.2f}")
    print(f"Mean value: {np.mean(value_function):.2f}")
    print(f"\nValue at start position [0, 0]: {value_function[0, 0]:.2f}")
    print(f"Value at goal position [7, 7]: {value_function[7, 7]:.2f}")
    
    # Simulate agent movement with visualization
    print("\n" + "="*60)
    print("Starting Agent Movement Simulation")
    print("="*60)
    print("Agent will move through the environment following a random policy.")
    print("The value function is shown as a heatmap in the background.")
    
    simulate_agent_movement(env, value_function, num_episodes=3, max_steps=200)
    
    print("\nMRP Simulation completed!")
    print("Close the matplotlib window to exit.")


if __name__ == "__main__":
    main()

