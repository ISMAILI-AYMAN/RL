"""
Grid World Environment
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rl_env.core.env import Env
from rl_env.core.spaces import Discrete, Box


class GridWorldEnv(Env):
    """
    Configurable Grid World Environment
    
    Actions: 0=Up, 1=Right, 2=Down, 3=Left
    Goal: Reach the target position
    
    Args:
        width: Width of the grid (default: 5)
        height: Height of the grid (default: 5, or width if only size specified)
        initial_pos: Initial agent position [row, col] (default: [0, 0])
        goal_pos: Goal position [row, col] (default: [height-1, width-1])
        obstacles: List of obstacle positions [[row1, col1], [row2, col2], ...] (default: [])
        render_mode: Rendering mode ("human", "rgb_array", or None)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        width: int = 5,
        height: Optional[int] = None,
        initial_pos: Optional[np.ndarray] = None,
        goal_pos: Optional[np.ndarray] = None,
        obstacles: Optional[list] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        # Grid dimensions
        self.width = width
        self.height = height if height is not None else width
        
        # Validate dimensions
        if self.width < 1 or self.height < 1:
            raise ValueError("Grid dimensions must be at least 1x1")
        
        self.render_mode = render_mode
        
        # Action space: 4 directions
        self.action_space = Discrete(4)
        
        # Observation space: agent position (row, col)
        self.observation_space = Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([self.height-1, self.width-1], dtype=np.int32),
            shape=(2,), dtype=np.int32
        )
        
        # Store initial configuration
        self._initial_pos = initial_pos if initial_pos is not None else np.array([0, 0], dtype=np.int32)
        self._goal_pos = goal_pos if goal_pos is not None else np.array([self.height-1, self.width-1], dtype=np.int32)
        self._obstacles = obstacles if obstacles is not None else []
        
        # Validate positions
        self._validate_positions()
        
        # Current state
        self.agent_pos = None
        self.goal_pos = self._goal_pos.copy()
        self.obstacles = [np.array(obs, dtype=np.int32) for obs in self._obstacles]
        self._np_random = None
        
        # Matplotlib figure and axes for rendering
        self._fig = None
        self._ax = None
        self._agent_patch = None
        self._agent_text = None
        self._goal_patch = None
        self._goal_text = None
        self._obstacle_patches = []
    
    def _validate_positions(self):
        """Validate that all positions are within grid bounds and don't overlap."""
        # Check initial position
        if not (0 <= self._initial_pos[0] < self.height and 0 <= self._initial_pos[1] < self.width):
            raise ValueError(f"Initial position {self._initial_pos} is out of bounds")
        
        # Check goal position
        if not (0 <= self._goal_pos[0] < self.height and 0 <= self._goal_pos[1] < self.width):
            raise ValueError(f"Goal position {self._goal_pos} is out of bounds")
        
        # Check obstacles
        for i, obs in enumerate(self._obstacles):
            obs_pos = np.array(obs, dtype=np.int32)
            if not (0 <= obs_pos[0] < self.height and 0 <= obs_pos[1] < self.width):
                raise ValueError(f"Obstacle {i} at {obs_pos} is out of bounds")
        
        # Check for overlaps
        all_positions = [tuple(self._initial_pos), tuple(self._goal_pos)]
        for obs in self._obstacles:
            obs_tuple = tuple(obs)
            if obs_tuple in all_positions:
                raise ValueError(f"Obstacle at {obs} overlaps with agent or goal")
            all_positions.append(obs_tuple)
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Optional dictionary that may contain:
                - 'initial_pos': Override initial position [row, col]
                - 'goal_pos': Override goal position [row, col]
                - 'obstacles': Override obstacles list
        """
        if seed is not None:
            np.random.seed(seed)
            self._np_random = np.random.RandomState(seed)
        
        # Use options to override positions if provided
        if options is not None:
            if 'initial_pos' in options:
                self._initial_pos = np.array(options['initial_pos'], dtype=np.int32)
            if 'goal_pos' in options:
                self._goal_pos = np.array(options['goal_pos'], dtype=np.int32)
            if 'obstacles' in options:
                self._obstacles = options['obstacles']
            self._validate_positions()
        
        # Reset to initial position
        self.agent_pos = self._initial_pos.copy()
        self.goal_pos = self._goal_pos.copy()
        self.obstacles = [np.array(obs, dtype=np.int32) for obs in self._obstacles]
        
        # Clear and redraw rendering patches if figure exists
        if self._fig is not None:
            # Remove old obstacle patches
            for patch in self._obstacle_patches:
                patch.remove()
            self._obstacle_patches = []
            
            # Redraw obstacles
            for obs in self.obstacles:
                obs_y, obs_x = obs
                obs_patch = patches.Rectangle(
                    (obs_x - 0.4, obs_y - 0.4), 0.8, 0.8,
                    color='red', ec='darkred', linewidth=2, zorder=2
                )
                self._ax.add_patch(obs_patch)
                self._obstacle_patches.append(obs_patch)
            
            # Update goal position if it changed
            if self._goal_patch is not None:
                self._goal_patch.remove()
            if self._goal_text is not None:
                self._goal_text.remove()
            
            goal_y, goal_x = self.goal_pos
            self._goal_patch = patches.Circle(
                (goal_x, goal_y), 0.3,
                color='green', ec='darkgreen', linewidth=2, zorder=3
            )
            self._ax.add_patch(self._goal_patch)
            self._goal_text = self._ax.text(goal_x, goal_y, 'G', ha='center', va='center',
                         fontsize=14, fontweight='bold', color='white', zorder=4)
        
        observation = self.agent_pos.copy()
        info = {
            "distance_to_goal": float(np.linalg.norm(self.agent_pos - self.goal_pos))
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step in the environment."""
        # Action mapping: 0=Up, 1=Right, 2=Down, 3=Left
        action_map = {
            0: np.array([-1, 0], dtype=np.int32),  # Up (decrease row)
            1: np.array([0, 1], dtype=np.int32),   # Right (increase col)
            2: np.array([1, 0], dtype=np.int32),  # Down (increase row)
            3: np.array([0, -1], dtype=np.int32)  # Left (decrease col)
        }
        
        # Calculate new position
        new_pos = self.agent_pos + action_map[action]
        
        # Keep agent within bounds
        new_pos[0] = np.clip(new_pos[0], 0, self.height - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.width - 1)
        
        # Check for obstacle collision
        obstacle_collision = False
        for obs in self.obstacles:
            if np.array_equal(new_pos, obs):
                obstacle_collision = True
                break
        
        # If no obstacle collision, move agent
        if not obstacle_collision:
            self.agent_pos = new_pos
        
        # Calculate reward
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10.0
            terminated = True
        elif obstacle_collision:
            reward = -1.0  # Penalty for hitting obstacle
            terminated = False
        else:
            reward = -0.1  # Small negative reward for each step
            terminated = False
        
        truncated = False
        observation = self.agent_pos.copy()
        info = {
            "distance_to_goal": float(np.linalg.norm(self.agent_pos - self.goal_pos)),
            "obstacle_collision": obstacle_collision
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment using matplotlib."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            # Fallback to text rendering if no render mode specified
            if self.agent_pos is not None:
                grid = np.zeros((self.height, self.width), dtype=str)
                grid[:] = "."
                # Mark obstacles
                for obs in self.obstacles:
                    grid[obs[0], obs[1]] = "X"
                # Mark agent and goal
                grid[self.agent_pos[0], self.agent_pos[1]] = "A"
                grid[self.goal_pos[0], self.goal_pos[1]] = "G"
                print("\n".join([" ".join(row) for row in grid]))
                print()
    
    def _render_human(self):
        """Render environment in interactive matplotlib window."""
        if self._fig is None:
            # Initialize figure on first render
            self._fig, self._ax = plt.subplots(figsize=(max(8, self.width), max(8, self.height)))
            self._ax.set_xlim(-0.5, self.width - 0.5)
            self._ax.set_ylim(-0.5, self.height - 0.5)
            self._ax.set_aspect('equal')
            self._ax.set_xticks(range(self.width))
            self._ax.set_yticks(range(self.height))
            self._ax.grid(True, color='gray', linewidth=1.5, alpha=0.5)
            self._ax.set_title('Grid World Environment', fontsize=16, fontweight='bold')
            self._ax.invert_yaxis()  # Invert y-axis to match array indexing
            
            # Draw obstacles
            for obs in self.obstacles:
                obs_y, obs_x = obs
                obs_patch = patches.Rectangle(
                    (obs_x - 0.4, obs_y - 0.4), 0.8, 0.8,
                    color='red', ec='darkred', linewidth=2, zorder=2
                )
                self._ax.add_patch(obs_patch)
                self._obstacle_patches.append(obs_patch)
            
            # Draw goal position (green circle)
            goal_y, goal_x = self.goal_pos
            self._goal_patch = patches.Circle(
                (goal_x, goal_y), 0.3, 
                color='green', ec='darkgreen', linewidth=2, zorder=3
            )
            self._ax.add_patch(self._goal_patch)
            self._goal_text = self._ax.text(goal_x, goal_y, 'G', ha='center', va='center', 
                         fontsize=14, fontweight='bold', color='white', zorder=4)
        else:
            # Ensure obstacles are always visible (in case they were cleared)
            if len(self._obstacle_patches) != len(self.obstacles):
                # Remove any existing obstacle patches
                for patch in self._obstacle_patches:
                    patch.remove()
                self._obstacle_patches = []
                
                # Redraw all obstacles
                for obs in self.obstacles:
                    obs_y, obs_x = obs
                    obs_patch = patches.Rectangle(
                        (obs_x - 0.4, obs_y - 0.4), 0.8, 0.8,
                        color='red', ec='darkred', linewidth=2, zorder=2
                    )
                    self._ax.add_patch(obs_patch)
                    self._obstacle_patches.append(obs_patch)
        
        # Update agent position
        if self.agent_pos is not None:
            agent_y, agent_x = self.agent_pos
            
            # Remove old agent patch and text if they exist
            if self._agent_patch is not None:
                self._agent_patch.remove()
            if self._agent_text is not None:
                self._agent_text.remove()
            
            # Draw agent (blue circle)
            self._agent_patch = patches.Circle(
                (agent_x, agent_y), 0.3,
                color='blue', ec='darkblue', linewidth=2, zorder=3
            )
            self._ax.add_patch(self._agent_patch)
            self._agent_text = self._ax.text(agent_x, agent_y, 'A', ha='center', va='center',
                         fontsize=14, fontweight='bold', color='white', zorder=4)
        
        # Update display
        plt.draw()
        plt.pause(0.01)  # Small pause to allow GUI to update
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array."""
        fig, ax = plt.subplots(figsize=(max(8, self.width), max(8, self.height)))
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, color='gray', linewidth=1.5, alpha=0.5)
        ax.set_title('Grid World Environment', fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        
        # Draw obstacles
        for obs in self.obstacles:
            obs_y, obs_x = obs
            obs_rect = patches.Rectangle(
                (obs_x - 0.4, obs_y - 0.4), 0.8, 0.8,
                color='red', ec='darkred', linewidth=2, zorder=2
            )
            ax.add_patch(obs_rect)
        
        # Draw goal
        goal_y, goal_x = self.goal_pos
        goal_circle = patches.Circle(
            (goal_x, goal_y), 0.3,
            color='green', ec='darkgreen', linewidth=2, zorder=3
        )
        ax.add_patch(goal_circle)
        ax.text(goal_x, goal_y, 'G', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white', zorder=4)
        
        # Draw agent
        if self.agent_pos is not None:
            agent_y, agent_x = self.agent_pos
            agent_circle = patches.Circle(
                (agent_x, agent_y), 0.3,
                color='blue', ec='darkblue', linewidth=2, zorder=3
            )
            ax.add_patch(agent_circle)
            ax.text(agent_x, agent_y, 'A', ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white', zorder=4)
        
        # Convert to RGB array
        fig.canvas.draw()
        # Get the RGBA buffer
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        plt.close(fig)
        
        return buf
    
    def set_agent_position(self, row: int, col: int) -> bool:
        """
        Set agent position directly.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            True if position was set successfully, False if invalid
        """
        pos = np.array([row, col], dtype=np.int32)
        
        # Validate position
        if not (0 <= row < self.height and 0 <= col < self.width):
            print(f"Error: Position [{row}, {col}] is out of bounds")
            return False
        
        # Check if position is an obstacle
        for obs in self.obstacles:
            if obs[0] == row and obs[1] == col:
                print(f"Error: Position [{row}, {col}] is an obstacle")
                return False
        
        # Check if position is goal
        if self.goal_pos[0] == row and self.goal_pos[1] == col:
            print(f"Warning: Position [{row}, {col}] is the goal position")
        
        self.agent_pos = pos.copy()
        
        # Update rendering if active
        if self.render_mode == "human" and self._fig is not None:
            self._render_human()
        
        return True
    
    def set_goal_position(self, row: int, col: int) -> bool:
        """
        Set goal position directly.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            True if position was set successfully, False if invalid
        """
        pos = np.array([row, col], dtype=np.int32)
        
        # Validate position
        if not (0 <= row < self.height and 0 <= col < self.width):
            print(f"Error: Position [{row}, {col}] is out of bounds")
            return False
        
        # Check if position is an obstacle
        for obs in self.obstacles:
            if obs[0] == row and obs[1] == col:
                print(f"Error: Position [{row}, {col}] is an obstacle")
                return False
        
        # Check if position is agent
        if self.agent_pos is not None and self.agent_pos[0] == row and self.agent_pos[1] == col:
            print(f"Warning: Position [{row}, {col}] is the agent position")
        
        self.goal_pos = pos.copy()
        self._goal_pos = pos.copy()
        
        # Update rendering if active
        if self.render_mode == "human" and self._fig is not None:
            # Remove old goal
            if self._goal_patch is not None:
                self._goal_patch.remove()
            if self._goal_text is not None:
                self._goal_text.remove()
            
            # Draw new goal
            goal_y, goal_x = self.goal_pos
            self._goal_patch = patches.Circle(
                (goal_x, goal_y), 0.3,
                color='green', ec='darkgreen', linewidth=2, zorder=3
            )
            self._ax.add_patch(self._goal_patch)
            self._goal_text = self._ax.text(goal_x, goal_y, 'G', ha='center', va='center',
                         fontsize=14, fontweight='bold', color='white', zorder=4)
            plt.draw()
            plt.pause(0.01)
        
        return True
    
    def add_obstacle(self, row: int, col: int) -> bool:
        """
        Add an obstacle at the specified position.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            True if obstacle was added successfully, False if invalid
        """
        pos = np.array([row, col], dtype=np.int32)
        
        # Validate position
        if not (0 <= row < self.height and 0 <= col < self.width):
            print(f"Error: Position [{row}, {col}] is out of bounds")
            return False
        
        # Check if position already has an obstacle
        for obs in self.obstacles:
            if obs[0] == row and obs[1] == col:
                print(f"Warning: Obstacle already exists at [{row}, {col}]")
                return False
        
        # Check if position is agent
        if self.agent_pos is not None and self.agent_pos[0] == row and self.agent_pos[1] == col:
            print(f"Error: Cannot place obstacle at agent position [{row}, {col}]")
            return False
        
        # Check if position is goal
        if self.goal_pos[0] == row and self.goal_pos[1] == col:
            print(f"Error: Cannot place obstacle at goal position [{row}, {col}]")
            return False
        
        # Add obstacle
        self.obstacles.append(pos.copy())
        self._obstacles.append([row, col])
        
        # Update rendering if active
        if self.render_mode == "human" and self._fig is not None:
            obs_y, obs_x = row, col
            obs_patch = patches.Rectangle(
                (obs_x - 0.4, obs_y - 0.4), 0.8, 0.8,
                color='red', ec='darkred', linewidth=2, zorder=2
            )
            self._ax.add_patch(obs_patch)
            self._obstacle_patches.append(obs_patch)
            plt.draw()
            plt.pause(0.01)
        
        return True
    
    def remove_obstacle(self, row: int, col: int) -> bool:
        """
        Remove an obstacle at the specified position.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            True if obstacle was removed successfully, False if not found
        """
        # Find and remove obstacle
        for i, obs in enumerate(self.obstacles):
            if obs[0] == row and obs[1] == col:
                self.obstacles.pop(i)
                self._obstacles.pop(i)
                
                # Update rendering if active
                if self.render_mode == "human" and self._fig is not None:
                    # Remove patch from rendering
                    if i < len(self._obstacle_patches):
                        self._obstacle_patches[i].remove()
                        self._obstacle_patches.pop(i)
                    plt.draw()
                    plt.pause(0.01)
                
                return True
        
        print(f"Warning: No obstacle found at [{row}, {col}]")
        return False
    
    def clear_obstacles(self):
        """Remove all obstacles from the environment."""
        self.obstacles = []
        self._obstacles = []
        
        # Update rendering if active
        if self.render_mode == "human" and self._fig is not None:
            for patch in self._obstacle_patches:
                patch.remove()
            self._obstacle_patches = []
            plt.draw()
            plt.pause(0.01)
    
    def get_agent_position(self) -> Optional[np.ndarray]:
        """Get current agent position."""
        return self.agent_pos.copy() if self.agent_pos is not None else None
    
    def get_goal_position(self) -> np.ndarray:
        """Get current goal position."""
        return self.goal_pos.copy()
    
    def get_obstacles(self) -> list:
        """Get list of obstacle positions."""
        return [obs.copy() for obs in self.obstacles]
    
    def get_grid_info(self) -> Dict:
        """
        Get complete grid information.
        
        Returns:
            Dictionary with grid dimensions, agent, goal, and obstacles
        """
        return {
            "width": self.width,
            "height": self.height,
            "agent_position": self.get_agent_position(),
            "goal_position": self.get_goal_position(),
            "obstacles": self.get_obstacles(),
            "num_obstacles": len(self.obstacles)
        }
    
    def close(self):
        """Clean up resources."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
            self._agent_patch = None
            self._agent_text = None
            self._goal_patch = None
            self._goal_text = None
            self._obstacle_patches = []

