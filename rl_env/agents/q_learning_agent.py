"""
Q-Learning Agent
"""

import numpy as np
from typing import Any, Dict, Optional
from rl_env.agents.base import Agent


class QLearningAgent(Agent):
    """
    Q-Learning agent using epsilon-greedy exploration.
    """
    
    def __init__(
        self,
        action_space,
        grid_width: int = 100,
        grid_height: int = 100,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            action_space: The action space of the environment
            grid_width: Maximum grid width (for state indexing)
            grid_height: Maximum grid height (for state indexing)
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate per episode
            epsilon_min: Minimum epsilon value
            seed: Random seed
        """
        super().__init__(action_space)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.state_space_size = grid_width * grid_height
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self._np_random = np.random.RandomState(seed)
        
        # Q-table: state -> action -> Q-value
        # For grid world, we use (row * width + col) as state index
        self.q_table = np.zeros((self.state_space_size, action_space.n))
        
        # Track last state and action for Q-learning update
        self.last_state = None
        self.last_action = None
    
    def _state_to_index(self, observation: Any) -> int:
        """
        Convert observation to state index.
        For grid world, we use (row * width + col) as state index.
        """
        if isinstance(observation, np.ndarray) and len(observation) == 2:
            row, col = int(observation[0]), int(observation[1])
            # Ensure within bounds
            row = np.clip(row, 0, self.grid_height - 1)
            col = np.clip(col, 0, self.grid_width - 1)
            return row * self.grid_width + col
        # Fallback for non-array observations
        return hash(tuple(observation)) % self.state_space_size
    
    def select_action(self, observation: Any, info: Optional[Dict] = None) -> Any:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            info: Additional information
            
        Returns:
            action: Selected action
        """
        state_idx = self._state_to_index(observation)
        
        # Epsilon-greedy action selection
        if self._np_random.random() < self.epsilon:
            # Explore: random action
            action = self.action_space.sample()
        else:
            # Exploit: best action according to Q-table
            q_values = self.q_table[state_idx]
            # If multiple actions have same Q-value, choose randomly among them
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = self._np_random.choice(best_actions)
        
        # Store for Q-learning update
        self.last_state = state_idx
        self.last_action = action
        
        return action
    
    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        terminated: bool,
        truncated: bool,
        info: Optional[Dict] = None
    ):
        """
        Update Q-table using Q-learning algorithm.
        
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        """
        if self.last_state is None:
            return
        
        state_idx = self.last_state
        next_state_idx = self._state_to_index(next_observation)
        
        # Current Q-value
        current_q = self.q_table[state_idx, self.last_action]
        
        # Next state max Q-value
        if terminated or truncated:
            next_max_q = 0.0
        else:
            next_max_q = np.max(self.q_table[next_state_idx])
        
        # Q-learning update
        target = reward + self.discount_factor * next_max_q
        td_error = target - current_q
        self.q_table[state_idx, self.last_action] += self.learning_rate * td_error
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset agent state (but keep Q-table).
        Decay epsilon for exploration schedule.
        """
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.last_state = None
        self.last_action = None
    
    def get_q_table(self) -> np.ndarray:
        """Get current Q-table."""
        return self.q_table.copy()
    
    def set_epsilon(self, epsilon: float):
        """Set epsilon value."""
        self.epsilon = np.clip(epsilon, self.epsilon_min, 1.0)

