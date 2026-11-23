from collections.abc import Sequence

import numpy as np


def random_policy(
    observation: Sequence[float], *, n_actions: int, rng: np.random.Generator
) -> int:
    """Pick an action index uniformly at random, ignoring the observation."""
    return int(rng.integers(n_actions))
