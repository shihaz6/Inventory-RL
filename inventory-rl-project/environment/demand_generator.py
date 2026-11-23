
"""
Demand Generator Module
-----------------------
Generates daily customer demand for the Inventory
Reinforcement Learning environment.

Demand is generated using a Poisson distribution.

Weekdays : Average demand = 5
Weekends : Average demand = 8
"""

from numbers import Integral

import numpy as np


def generate_demand(day: int, rng: np.random.Generator) -> int:
    """
    Generate daily customer demand.

    Parameters
    ----------
    day : int
        Current simulation day (starting from 0).

    rng : np.random.Generator
        Random number generator provided by the
        Gymnasium inventory environment.

    Returns
    -------
    int
        Generated daily customer demand.

    Raises
    ------
    ValueError
        If the simulation day is invalid.

    TypeError
        If the random generator is invalid.
    """

    # Validate simulation day
    if not isinstance(day, Integral) or isinstance(day, bool) or day < 0:
        raise ValueError("day must be a nonnegative integer")

    # Validate random number generator
    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be an instance of np.random.Generator"
        )

    # Determine average demand based on the day of the week
    if day % 7 >= 5:
        average_demand = 8  # Saturday and Sunday
    else:
        average_demand = 5  # Monday to Friday

    # Generate demand using Poisson distribution
    demand = rng.poisson(lam=average_demand)

    return int(demand)


# Default demand generator used by the inventory environment
_default_demand = generate_demand
