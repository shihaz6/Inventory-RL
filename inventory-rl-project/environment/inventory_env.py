from collections.abc import Callable, Sequence
from math import isfinite
from numbers import Integral

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DemandGenerator = Callable[[int, np.random.Generator], int]
ORDER_QUANTITIES = (0, 5, 10, 20, 30, 50)


def _default_demand(day: int, rng: np.random.Generator) -> int:
    return int(rng.poisson(8 if day % 7 >= 5 else 5))


class InventoryEnv(gym.Env):
    def __init__(
        self,
        *,
        initial_stock: int = 20,
        max_order: int = 20,
        lead_time: int = 2,
        episode_days: int = 60,
        holding_cost: float = 1.0,
        ordering_cost: float = 2.0,
        shortage_cost: float = 5.0,
        demand_generator: DemandGenerator = _default_demand,
        order_quantities: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        for name, value, minimum in (
            ("initial_stock", initial_stock, 0),
            ("max_order", max_order, 0),
            ("lead_time", lead_time, 1),
            ("episode_days", episode_days, 1),
        ):
            if not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, value in (
            ("holding_cost", holding_cost),
            ("ordering_cost", ordering_cost),
            ("shortage_cost", shortage_cost),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

        self.initial_stock = int(initial_stock)
        self.lead_time = int(lead_time)
        self.episode_days = int(episode_days)
        self.holding_cost = float(holding_cost)
        self.ordering_cost = float(ordering_cost)
        self.shortage_cost = float(shortage_cost)
        self.demand_generator = demand_generator

        max_order = int(max_order)
        self.order_quantities = (
            tuple(range(max_order + 1))
            if order_quantities is None
            else tuple(order_quantities)
        )
        if (
            not self.order_quantities
            or self.order_quantities[0] != 0
            or any(not isinstance(quantity, Integral) for quantity in self.order_quantities)
            or any(a >= b for a, b in zip(self.order_quantities, self.order_quantities[1:]))
        ):
            raise ValueError("order_quantities must be increasing integers starting at zero")
        max_order = int(self.order_quantities[-1])
        self.action_space = spaces.Discrete(len(self.order_quantities))
        max_stock = self.initial_stock + max_order * self.episode_days
        upper_bounds = [max_stock, *([max_order] * self.lead_time), 6]
        self.observation_space = spaces.Box(
            low=np.zeros(len(upper_bounds), dtype=np.float32),
            high=np.asarray(upper_bounds, dtype=np.float32),
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        deliveries = [
            self._pending_orders.get(self._day + offset, 0)
            for offset in range(self.lead_time)
        ]
        return np.asarray(
            [self._stock, *deliveries, self._day % 7], dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._stock = self.initial_stock
        self._day = 0
        self._pending_orders: dict[int, int] = {}
        return self._observation(), {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid order quantity: {action!r}")

        day = self._day
        order_quantity = int(self.order_quantities[int(action)])
        demand = self.demand_generator(day, self.np_random)
        if not isinstance(demand, Integral) or demand < 0:
            raise ValueError("Demand must be a nonnegative integer")
        demand = int(demand)

        received_quantity = self._pending_orders.pop(day, 0)
        self._stock += received_quantity
        if order_quantity:
            self._pending_orders[day + self.lead_time] = order_quantity

        sales = min(self._stock, demand)
        self._stock -= sales
        unmet_demand = demand - sales
        holding = self.holding_cost * self._stock
        ordering = self.ordering_cost * order_quantity
        shortage = self.shortage_cost * unmet_demand
        total_cost = holding + ordering + shortage

        self._day += 1
        info = {
            "day": day,
            "demand": demand,
            "sales": sales,
            "unmet_demand": unmet_demand,
            "ending_stock": self._stock,
            "order_quantity": order_quantity,
            "received_quantity": received_quantity,
            "holding_cost": holding,
            "ordering_cost": ordering,
            "shortage_cost": shortage,
            "total_cost": total_cost,
        }
        truncated = self._day >= self.episode_days
        return self._observation(), -total_cost, False, truncated, info
