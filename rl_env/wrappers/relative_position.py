"""
Relative Position Wrapper
"""

import numpy as np
from rl_env.wrappers.base import ObservationWrapper
from rl_env.core.spaces import Box


class RelativePosition(ObservationWrapper):
    """Convert absolute positions to relative positions."""
    
    def __init__(self, env, reference_point: np.ndarray = None):
        super().__init__(env)
        self.reference_point = reference_point
        
        # Update observation space if needed
        if isinstance(env.observation_space, Box):
            self.observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype
            )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Convert absolute position to relative position."""
        if self.reference_point is None:
            # Use origin as reference
            self.reference_point = np.zeros_like(observation)
        
        return observation - self.reference_point
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Reset reference point if needed
        if self.reference_point is None:
            self.reference_point = np.zeros_like(obs)
        return obs, info

