from collections.abc import Callable
from math import isfinite
from numbers import Integral

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DemandGenerator = Callable[[int, np.random.Generator], int]


def _default_demand(day: int, rng: np.random.Generator) -> int:
    return int(rng.poisson(8 if day % 7 >= 5 else 5))


