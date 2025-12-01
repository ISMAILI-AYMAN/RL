"""
Discrete Actions Wrapper
"""

import numpy as np
from rl_env.wrappers.base import ActionWrapper
from rl_env.core.spaces import Discrete


class DiscreteActions(ActionWrapper):
    """Convert continuous actions to discrete actions."""
    
    def __init__(self, env, n_actions: int = 4):
        super().__init__(env)
        self.n_actions = n_actions
        self.action_space = Discrete(n_actions)
    
    def action(self, action: int) -> np.ndarray:
        """Convert discrete action to continuous action."""
        # Map discrete action to continuous action vector
        # Example: 4 discrete actions -> 2D continuous actions
        action_map = {
            0: np.array([-1.0, 0.0]),   # Left
            1: np.array([1.0, 0.0]),    # Right
            2: np.array([0.0, -1.0]),   # Down
            3: np.array([0.0, 1.0]),    # Up
        }
        return action_map.get(action, np.array([0.0, 0.0]))

