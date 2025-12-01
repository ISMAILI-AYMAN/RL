"""
Clip Reward Wrapper
"""

from rl_env.wrappers.base import RewardWrapper


class ClipReward(RewardWrapper):
    """Clip rewards to a specified range."""
    
    def __init__(self, env, min_reward: float = -1.0, max_reward: float = 1.0):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
    
    def reward(self, reward: float) -> float:
        """Clip reward to [min_reward, max_reward]."""
        return max(self.min_reward, min(self.max_reward, reward))

