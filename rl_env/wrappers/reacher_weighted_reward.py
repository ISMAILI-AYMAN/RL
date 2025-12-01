"""
Reacher Weighted Reward Wrapper
"""

import numpy as np
from rl_env.wrappers.base import RewardWrapper


class ReacherWeightedReward(RewardWrapper):
    """
    Weighted reward wrapper for reacher-like tasks.
    Combines distance reward with action penalty.
    """
    
    def __init__(
        self, 
        env, 
        distance_weight: float = 1.0,
        action_weight: float = 0.01,
        target_threshold: float = 0.05
    ):
        super().__init__(env)
        self.distance_weight = distance_weight
        self.action_weight = action_weight
        self.target_threshold = target_threshold
        self.last_action = None
        self.target_position = None
    
    def reward(self, reward: float) -> float:
        """Calculate weighted reward."""
        # Get current observation (assumed to be position)
        # This is a simplified version - adjust based on your environment
        
        # Distance-based reward (negative distance)
        distance_reward = 0.0
        if hasattr(self.env, 'agent_pos') and hasattr(self.env, 'goal_pos'):
            distance = np.linalg.norm(self.env.agent_pos - self.env.goal_pos)
            distance_reward = -distance * self.distance_weight
            
            # Bonus for reaching target
            if distance < self.target_threshold:
                distance_reward += 10.0
        
        # Action penalty (encourage smaller actions)
        action_penalty = 0.0
        if self.last_action is not None:
            if isinstance(self.last_action, np.ndarray):
                action_penalty = -np.linalg.norm(self.last_action) * self.action_weight
            elif isinstance(self.last_action, (int, float)):
                action_penalty = -abs(self.last_action) * self.action_weight
        
        return reward + distance_reward + action_penalty
    
    def step(self, action):
        self.last_action = action
        return super().step(action)

