"""
Random Agent - selects actions randomly
"""

import numpy as np
from typing import Any, Dict, Optional
from rl_env.agents.base import Agent


class RandomAgent(Agent):
    """
    Agent that selects actions randomly from the action space.
    """
    
    def __init__(self, action_space, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            action_space: The action space of the environment
            seed: Random seed for reproducibility
        """
        super().__init__(action_space)
        self._np_random = np.random.RandomState(seed)
    
    def select_action(self, observation: Any, info: Optional[Dict] = None) -> Any:
        """
        Select a random action.
        
        Args:
            observation: Current observation (not used for random agent)
            info: Additional information (not used)
            
        Returns:
            action: Randomly selected action
        """
        return self.action_space.sample()
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the agent's random state.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

