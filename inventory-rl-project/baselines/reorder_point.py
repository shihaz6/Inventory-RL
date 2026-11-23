from collections.abc import Sequence

from environment.inventory_env import ORDER_QUANTITIES


def inventory_position(observation: Sequence[float]) -> int:
    return sum(int(quantity) for quantity in observation[:-1])


def reorder_point_policy(
    observation: Sequence[float], *, reorder_point: int, order_quantity: int
) -> int:
    if order_quantity not in ORDER_QUANTITIES[1:]:
        raise ValueError(f"order_quantity must be one of {ORDER_QUANTITIES[1:]}")
    if inventory_position(observation) <= reorder_point:
        return ORDER_QUANTITIES.index(order_quantity)
    return 0
