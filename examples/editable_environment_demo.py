"""
Editable Environment Demo

This script demonstrates how to edit the environment directly via Python:
- Move agent position
- Add/remove obstacles
- Change goal position
"""

import numpy as np
import time
from rl_env.envs.grid_world import GridWorldEnv


def main():
    """Demonstrate editable environment features."""
    
    print("="*60)
    print("Editable Environment Demo")
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
            [3, 2]
        ],
        render_mode="human"
    )
    
    # Initial render
    env.render()
    print("\nInitial environment state:")
    print(f"Agent position: {env.get_agent_position()}")
    print(f"Goal position: {env.get_goal_position()}")
    print(f"Obstacles: {env.get_obstacles()}")
    
    time.sleep(2)
    
    # Example 1: Move agent
    print("\n" + "="*60)
    print("Example 1: Moving agent to position [3, 3]")
    print("="*60)
    if env.set_agent_position(3, 3):
        print("Agent moved successfully!")
        env.render()
        time.sleep(2)
    else:
        print("Failed to move agent")
    
    # Example 2: Add obstacles
    print("\n" + "="*60)
    print("Example 2: Adding obstacles")
    print("="*60)
    new_obstacles = [[4, 4], [4, 5], [5, 4]]
    for row, col in new_obstacles:
        if env.add_obstacle(row, col):
            print(f"Added obstacle at [{row}, {col}]")
        else:
            print(f"Failed to add obstacle at [{row}, {col}]")
    env.render()
    time.sleep(2)
    
    # Example 3: Remove obstacle
    print("\n" + "="*60)
    print("Example 3: Removing obstacle at [2, 2]")
    print("="*60)
    if env.remove_obstacle(2, 2):
        print("Obstacle removed successfully!")
        env.render()
        time.sleep(2)
    else:
        print("Failed to remove obstacle")
    
    # Example 4: Change goal position
    print("\n" + "="*60)
    print("Example 4: Moving goal to position [1, 1]")
    print("="*60)
    if env.set_goal_position(1, 1):
        print("Goal moved successfully!")
        env.render()
        time.sleep(2)
    else:
        print("Failed to move goal")
    
    # Example 5: Move agent to new goal
    print("\n" + "="*60)
    print("Example 5: Moving agent to new goal position")
    print("="*60)
    if env.set_agent_position(1, 1):
        print("Agent reached goal!")
        env.render()
        time.sleep(2)
    
    # Example 6: Get grid info
    print("\n" + "="*60)
    print("Example 6: Getting complete grid information")
    print("="*60)
    grid_info = env.get_grid_info()
    print(f"Grid dimensions: {grid_info['width']}x{grid_info['height']}")
    print(f"Agent position: {grid_info['agent_position']}")
    print(f"Goal position: {grid_info['goal_position']}")
    print(f"Number of obstacles: {grid_info['num_obstacles']}")
    print(f"Obstacle positions: {grid_info['obstacles']}")
    
    # Example 7: Clear all obstacles
    print("\n" + "="*60)
    print("Example 7: Clearing all obstacles")
    print("="*60)
    env.clear_obstacles()
    print("All obstacles cleared!")
    env.render()
    time.sleep(2)
    
    # Example 8: Interactive editing loop
    print("\n" + "="*60)
    print("Example 8: Interactive editing")
    print("="*60)
    print("You can now edit the environment programmatically:")
    print("  - env.set_agent_position(row, col)")
    print("  - env.set_goal_position(row, col)")
    print("  - env.add_obstacle(row, col)")
    print("  - env.remove_obstacle(row, col)")
    print("  - env.clear_obstacles()")
    print("  - env.get_grid_info()")
    
    # Demonstrate some edits
    print("\nPerforming some edits...")
    env.set_agent_position(0, 0)
    env.set_goal_position(7, 7)
    env.add_obstacle(3, 3)
    env.add_obstacle(4, 4)
    env.add_obstacle(5, 5)
    env.render()
    
    print("\nFinal state:")
    print(env.get_grid_info())
    
    # Keep window open
    print("\nEnvironment is ready for interactive editing!")
    print("Close the matplotlib window when done.")
    
    # Cleanup
    input("\nPress Enter to close...")
    env.close()
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()

