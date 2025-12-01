"""
Space classes for action and observation spaces
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class Space(ABC):
    """Abstract base class for action and observation spaces."""
    
    def __init__(self, shape: Optional[Tuple] = None, dtype: Any = None):
        self.shape = shape
        self.dtype = dtype
    
    @abstractmethod
    def sample(self) -> Any:
        """Randomly sample an element from this space."""
        pass
    
    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if x is a valid element of this space."""
        pass
    
    def __contains__(self, x: Any) -> bool:
        return self.contains(x)


class Discrete(Space):
    """Discrete space: {0, 1, 2, ..., n-1}"""
    
    def __init__(self, n: int, seed: Optional[int] = None):
        super().__init__(shape=(), dtype=int)
        self.n = n
        self._np_random = np.random.RandomState(seed)
    
    def sample(self) -> int:
        return self._np_random.randint(self.n)
    
    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n
    
    def __repr__(self):
        return f"Discrete({self.n})"


class Box(Space):
    """Box space: continuous values in [low, high]"""
    
    def __init__(
        self,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: Optional[Tuple] = None,
        dtype: type = np.float32,
        seed: Optional[int] = None
    ):
        if shape is None:
            shape = np.shape(low)
        
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape)
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape)
        self.dtype = dtype
        self._np_random = np.random.RandomState(seed)
        
        super().__init__(shape=shape, dtype=dtype)
    
    def sample(self) -> np.ndarray:
        return self._np_random.uniform(
            low=self.low, high=self.high, size=self.shape
        ).astype(self.dtype)
    
    def contains(self, x: Any) -> bool:
        if isinstance(x, np.ndarray):
            return (
                x.shape == self.shape
                and np.all(x >= self.low)
                and np.all(x <= self.high)
            )
        return False
    
    def __repr__(self):
        return f"Box({self.low}, {self.high}, {self.shape}, {self.dtype})"


class MultiDiscrete(Space):
    """Multi-discrete space: multiple discrete spaces"""
    
    def __init__(self, nvec: Union[List[int], np.ndarray], seed: Optional[int] = None):
        nvec = np.asarray(nvec, dtype=np.int32)
        self.nvec = nvec
        super().__init__(shape=nvec.shape, dtype=np.int32)
        self._np_random = np.random.RandomState(seed)
    
    def sample(self) -> np.ndarray:
        return (self._np_random.random(self.shape) * self.nvec).astype(self.dtype)
    
    def contains(self, x: Any) -> bool:
        if isinstance(x, np.ndarray):
            return (
                x.shape == self.shape
                and np.all(x >= 0)
                and np.all(x < self.nvec)
            )
        return False
    
    def __repr__(self):
        return f"MultiDiscrete({self.nvec})"


class Dict(Space):
    """Dictionary space: mapping of keys to spaces"""
    
    def __init__(self, spaces: Dict[str, Space], seed: Optional[int] = None):
        self.spaces = spaces
        self._np_random = np.random.RandomState(seed)
        super().__init__(shape=None, dtype=dict)
    
    def sample(self) -> Dict[str, Any]:
        return {key: space.sample() for key, space in self.spaces.items()}
    
    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(self.spaces[key].contains(x[key]) for key in x.keys())
    
    def __repr__(self):
        return f"Dict({list(self.spaces.keys())})"


class Tuple(Space):
    """Tuple space: ordered collection of spaces"""
    
    def __init__(self, spaces: Tuple[Space, ...], seed: Optional[int] = None):
        self.spaces = spaces
        self._np_random = np.random.RandomState(seed)
        super().__init__(shape=None, dtype=tuple)
    
    def sample(self) -> Tuple[Any, ...]:
        return tuple(space.sample() for space in self.spaces)
    
    def contains(self, x: Any) -> bool:
        if not isinstance(x, (tuple, list)):
            return False
        if len(x) != len(self.spaces):
            return False
        return all(space.contains(item) for space, item in zip(self.spaces, x))
    
    def __repr__(self):
        return f"Tuple({len(self.spaces)} spaces)"

