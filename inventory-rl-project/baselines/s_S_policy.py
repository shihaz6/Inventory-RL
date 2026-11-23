from collections.abc import Sequence

from environment.inventory_env import ORDER_QUANTITIES

from .reorder_point import inventory_position


def s_S_policy(observation: Sequence[float], *, s: int, S: int) -> int:
    if not 0 <= s < S:
        raise ValueError("expected 0 <= s < S")
    position = inventory_position(observation)
    if position > s:
        return 0

    desired_quantity = S - position
    return min(
        range(1, len(ORDER_QUANTITIES)),
        key=lambda action: abs(ORDER_QUANTITIES[action] - desired_quantity),
    )
