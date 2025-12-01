"""
Simulation module for running RL episodes
"""

from typing import Optional, Dict, List, Callable, Any
import numpy as np
from rl_env.core.env import Env
from rl_env.agents.base import Agent


class Simulation:
    """
    Simulation runner for RL environments and agents.
    """
    
    def __init__(
        self,
        env: Env,
        agent: Agent,
        max_steps: Optional[int] = None,
        render: bool = False,
        verbose: bool = True
    ):
        """
        Initialize simulation.
        
        Args:
            env: Environment to simulate
            agent: Agent to use
            max_steps: Maximum steps per episode (None for no limit)
            render: Whether to render the environment
            verbose: Whether to print episode information
        """
        self.env = env
        self.agent = agent
        self.max_steps = max_steps
        self.render = render
        self.verbose = verbose
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
    
    def run_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Args:
            seed: Random seed for episode
            
        Returns:
            Dictionary with episode statistics
        """
        # Reset environment and agent
        observation, info = self.env.reset(seed=seed)
        self.agent.reset(seed=seed)
        
        # Initialize episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        done = False
        
        # Run episode
        while not done:
            # Select action
            action = self.agent.select_action(observation, info)
            
            # Step environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Update agent (for learning agents)
            self.agent.update(
                observation, action, reward, next_observation, 
                terminated, truncated, info
            )
            
            # Update statistics
            self.current_episode_reward += reward
            self.current_episode_length += 1
            
            # Render if requested
            if self.render and self.env.render_mode == "human":
                self.env.render()
            
            # Check termination conditions
            done = terminated or truncated
            if self.max_steps is not None and self.current_episode_length >= self.max_steps:
                done = True
            
            # Update observation
            observation = next_observation
        
        # Store episode statistics
        episode_stats = {
            "reward": self.current_episode_reward,
            "length": self.current_episode_length,
            "terminated": terminated,
            "truncated": truncated
        }
        
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        
        if self.verbose:
            print(f"Episode finished: reward={self.current_episode_reward:.2f}, "
                  f"length={self.current_episode_length}")
        
        return episode_stats
    
    def run_episodes(self, num_episodes: int, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            seed: Random seed for first episode (subsequent episodes use None)
            
        Returns:
            Dictionary with overall statistics
        """
        self.episode_rewards = []
        self.episode_lengths = []
        
        for episode in range(num_episodes):
            if self.verbose:
                print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            episode_seed = seed if episode == 0 else None
            self.run_episode(seed=episode_seed)
        
        # Calculate statistics
        stats = {
            "num_episodes": num_episodes,
            "mean_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "mean_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths
        }
        
        if self.verbose:
            print("\n" + "="*50)
            print("SIMULATION SUMMARY")
            print("="*50)
            print(f"Episodes: {num_episodes}")
            print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"Reward Range: [{stats['min_reward']:.2f}, {stats['max_reward']:.2f}]")
            print(f"Mean Length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
            print("="*50)
        
        return stats
    
    def reset_statistics(self):
        """Reset simulation statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

