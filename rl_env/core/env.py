"""
Base Environment Class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class Env(ABC):
    """
    Abstract base class for all environments.
    Standard RL environment interface.
    """
    
    metadata = {"render_modes": [], "render_fps": 30}
    reward_range = (-float("inf"), float("inf"))
    spec = None
    
    def __init__(self):
        self.action_space = None
        self.observation_space = None
        self.render_mode = None
    
    @abstractmethod
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Dictionary with additional information
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action: Action to take
            
        Returns:
            observation: Agent's observation of the current environment
            reward: Reward for taking the action
            terminated: Whether the episode has ended (terminal state)
            truncated: Whether the episode was truncated (time limit)
            info: Dictionary with additional information
        """
        pass
    
    def render(self):
        """Render the environment."""
        raise NotImplementedError
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        pass

