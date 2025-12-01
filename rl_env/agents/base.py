"""
Base Agent Class
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Agent(ABC):
    """
    Abstract base class for all agents.
    """
    
    def __init__(self, action_space):
        """
        Initialize agent.
        
        Args:
            action_space: The action space of the environment
        """
        self.action_space = action_space
    
    @abstractmethod
    def select_action(self, observation: Any, info: Optional[Dict] = None) -> Any:
        """
        Select an action given the current observation.
        
        Args:
            observation: Current observation from the environment
            info: Additional information dictionary
            
        Returns:
            action: Selected action
        """
        pass
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the agent's internal state.
        
        Args:
            seed: Random seed for reproducibility
        """
        pass
    
    def update(self, observation: Any, action: Any, reward: float, 
               next_observation: Any, terminated: bool, truncated: bool, 
               info: Optional[Dict] = None):
        """
        Update agent based on experience (for learning agents).
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        pass

