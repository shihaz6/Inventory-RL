# Inventory-RL

Reinforcement-learning inventory control project. This document defines the interface that agents, baselines, and evaluation code target.

## Requirements

The pinned dependencies require Python 3.12.

```bash
python -m pip install -r requirements.txt
```

## Environment contract

`InventoryEnv` (`environment/inventory_env.py`) is a Gymnasium `gym.Env`. Its constructor takes keyword-only arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `initial_stock` | `20` | On-hand units at the start of an episode. |
| `max_order` | `20` | Maximum units that can be ordered in one day. |
| `lead_time` | `2` | Days between placing an order and receiving it. |
| `episode_days` | `60` | Number of days in one episode. |
| `holding_cost` | `1.0` | Cost per unit remaining in stock at the end of a day. |
| `ordering_cost` | `2.0` | Cost per unit ordered. |
| `shortage_cost` | `5.0` | Cost per unit of unmet demand. |
| `demand_generator` | `_default_demand` | Callable `(day, rng) -> int`. The default draws Poisson(5) on weekdays and Poisson(8) on days 5 and 6. |

### Observation

`spaces.Box(shape=(lead_time + 2,), low=0, dtype=np.float32)`, where:

| Position | Meaning |
| --- | --- |
| `0` | On-hand stock after the previous day's demand (initial stock at reset). |
| `1 + i` | Quantity scheduled to arrive in `i` steps, for `i = 0 .. lead_time - 1`. Position `1` is the next step's delivery. |
| `lead_time + 1` | Day of the week, `day % 7` (`0`–`6`). |

The upper bound is `[initial_stock + max_order * episode_days, max_order, ..., max_order, 6]`. With the defaults the observation has shape `(4,)`: `[stock, next_delivery, following_delivery, day_of_week]`.

### Action

`spaces.Discrete(max_order + 1)`.

An action is the integer number of units to order on the current day, from `0` to `max_order` inclusive. Orders arrive `lead_time` days later, before that day's demand is served. An action outside the space raises `ValueError`.

### Step semantics

For day `d` (zero-based), `step(action)` performs these operations in order:

1. Validate `action`.
2. Draw demand: `demand_generator(d, self.np_random)`.
3. Receive the delivery scheduled for day `d`.
4. Schedule `action` units for delivery on day `d + lead_time` (if non-zero).
5. Fulfil demand: `sales = min(stock, demand)` and `unmet_demand = demand - sales`.
6. Compute costs:
   - holding: `holding_cost * ending_stock`
   - ordering: `ordering_cost * order_quantity`
   - shortage: `shortage_cost * unmet_demand`
   - total: sum of the three costs
7. Advance the day and return.

The reward is the negative of that day's total cost.

### Info dictionary

Each `step` returns an info dict with:

| Key | Type | Meaning |
| --- | --- | --- |
| `day` | `int` | Zero-based day index of the step (`0` to `episode_days - 1`). |
| `demand` | `int` | Units demanded that day. |
| `sales` | `int` | Units fulfilled (`min(stock, demand)`). |
| `unmet_demand` | `int` | Demand that could not be fulfilled. |
| `ending_stock` | `int` | On-hand stock after fulfilling demand. |
| `order_quantity` | `int` | Units ordered that day. |
| `received_quantity` | `int` | Units delivered that day. |
| `holding_cost` | `float` | Holding cost for the day. |
| `ordering_cost` | `float` | Ordering cost for the day. |
| `shortage_cost` | `float` | Shortage cost for the day. |
| `total_cost` | `float` | Sum of the three daily costs. |

A list of these dicts is the input to `evaluation.metrics.compute_kpis`.

### Reset and episode end

`reset(seed=None, options=None)` returns `(observation, {})`. It restores `initial_stock`, sets the day to `0`, and clears pending orders; `seed` controls demand sampling.

`terminated` is always `False`. `truncated` becomes `True` when the episode reaches `episode_days` steps (day 60 with the default configuration).

### Example

```python
from environment.inventory_env import InventoryEnv

env = InventoryEnv()
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(5)
```
