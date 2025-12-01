"""
Base wrapper classes
"""

from typing import Any, Dict, Optional, Tuple
from rl_env.core.env import Env


class Wrapper(Env):
    """Base wrapper class for environments."""
    
    def __init__(self, env: Env):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.render_mode = env.render_mode
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        return self.env.reset(seed=seed, options=options)
    
    def step(self, action: Any):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped environment."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)


class ObservationWrapper(Wrapper):
    """Wrapper that transforms observations."""
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info
    
    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info
    
    def observation(self, observation: Any) -> Any:
        """Transform observation. Override in subclasses."""
        return observation


class RewardWrapper(Wrapper):
    """Wrapper that transforms rewards."""
    
    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, self.reward(reward), terminated, truncated, info
    
    def reward(self, reward: float) -> float:
        """Transform reward. Override in subclasses."""
        return reward


class ActionWrapper(Wrapper):
    """Wrapper that transforms actions."""
    
    def step(self, action: Any):
        return self.env.step(self.action(action))
    
    def action(self, action: Any) -> Any:
        """Transform action. Override in subclasses."""
        return action

