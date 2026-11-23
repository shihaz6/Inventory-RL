import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.inventory_env import InventoryEnv


class TestInventoryEnv(unittest.TestCase):
    def test_gymnasium_api(self):
        check_env(InventoryEnv(), skip_render_check=True)

    def test_delivery_timing_and_costs(self):
        env = InventoryEnv(
            initial_stock=5,
            lead_time=2,
            episode_days=3,
            demand_generator=lambda day, rng: (7, 1, 1)[day],
        )
        observation, _ = env.reset()
        np.testing.assert_array_equal(observation, [5, 0, 0, 0])

        observation, reward, terminated, truncated, info = env.step(4)
        np.testing.assert_array_equal(observation, [0, 0, 4, 1])
        self.assertEqual(info["received_quantity"], 0)
        self.assertEqual(info["sales"], 5)
        self.assertEqual(info["unmet_demand"], 2)
        self.assertEqual(info["holding_cost"], 0)
        self.assertEqual(info["ordering_cost"], 8)
        self.assertEqual(info["shortage_cost"], 10)
        self.assertEqual(info["total_cost"], 18)
        self.assertEqual(reward, -18)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        observation, _, _, truncated, info = env.step(0)
        np.testing.assert_array_equal(observation, [0, 4, 0, 2])
        self.assertEqual(info["received_quantity"], 0)
        self.assertFalse(truncated)

        observation, reward, terminated, truncated, info = env.step(0)
        np.testing.assert_array_equal(observation, [3, 0, 0, 3])
        self.assertEqual(info["received_quantity"], 4)
        self.assertEqual(info["holding_cost"], 3)
        self.assertEqual(info["ordering_cost"], 0)
        self.assertEqual(info["shortage_cost"], 0)
        self.assertEqual(reward, -3)
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_seed_reproduces_demand_and_reset_clears_orders(self):
        env = InventoryEnv()

        def run_episode():
            observation, _ = env.reset(seed=42)
            np.testing.assert_array_equal(observation, [20, 0, 0, 0])
            return [env.step(4)[-1] for _ in range(7)]

        self.assertEqual(run_episode(), run_episode())

    def test_invalid_action_is_rejected(self):
        env = InventoryEnv(max_order=4)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(5)
