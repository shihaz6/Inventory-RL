import unittest
from functools import partial

from baselines.reorder_point import reorder_point_policy
from baselines.s_S_policy import s_S_policy
from environment.inventory_env import ORDER_QUANTITIES, InventoryEnv


class TestBaselines(unittest.TestCase):
    def test_reorder_point_counts_pending_stock_but_not_weekday(self):
        policy = partial(reorder_point_policy, reorder_point=10, order_quantity=30)
        self.assertEqual(policy([3, 2, 5, 6]), 4)
        self.assertEqual(policy([4, 2, 5, 0]), 0)

    def test_reorder_point_rejects_unsupported_order_quantity(self):
        with self.assertRaises(ValueError):
            reorder_point_policy([0, 0, 0, 0], reorder_point=10, order_quantity=15)

    def test_order_up_to_uses_nearest_quantity_with_smaller_tie(self):
        policy = partial(s_S_policy, s=10, S=25)
        self.assertEqual(policy([5, 0, 5, 6]), 2)
        self.assertEqual(policy([5, 0, 6, 0]), 0)
        self.assertEqual(s_S_policy([0, 0, 0, 0], s=10, S=75), 5)
        self.assertEqual(s_S_policy([10, 0, 0, 0], s=10, S=11), 1)

    def test_both_policies_can_drive_mapped_environment(self):
        policies = (
            partial(reorder_point_policy, reorder_point=10, order_quantity=30),
            partial(s_S_policy, s=10, S=30),
        )
        for policy in policies:
            with self.subTest(policy=policy):
                env = InventoryEnv(
                    initial_stock=0,
                    order_quantities=ORDER_QUANTITIES,
                    demand_generator=lambda day, rng: 0,
                )
                observation, _ = env.reset()
                self.assertEqual(env.action_space.n, 6)
                action = policy(observation)
                self.assertEqual(action, 4)
                observation, _, _, _, info = env.step(action)
                self.assertEqual(info["order_quantity"], 30)
                self.assertEqual(policy(observation), 0)
