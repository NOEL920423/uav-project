"""Regression tests for the minimal city task and BC/PPO contracts."""

import unittest

import numpy as np
import torch

from uav_ml.envs import FixedHeightCityCore
from uav_ml.models import LatentActorCritic, LatentBcPolicy


class FixedHeightCityTest(unittest.TestCase):
    def test_reset_is_seed_deterministic(self) -> None:
        first = FixedHeightCityCore()
        second = FixedHeightCityCore()
        first_observation = first.reset(123)
        second_observation = second.reset(123)
        self.assertTrue(np.array_equal(first_observation["state"], second_observation["state"]))
        self.assertEqual(first.buildings, second.buildings)
        self.assertEqual(first.path_world, second.path_world)

    def test_astar_expert_finishes_closed_loop(self) -> None:
        for seed in range(12):
            env = FixedHeightCityCore()
            observation = env.reset(seed)
            self.assertEqual(observation["state"].shape, (8,))
            while True:
                action = env.expert_action()
                self.assertEqual(action.shape, (3,))
                _, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(np.isfinite(reward))
                if terminated or truncated:
                    break
            self.assertTrue(info["success"], msg=f"A* failed seed {seed}")

    def test_unreachable_first_city_is_deterministically_regenerated(self) -> None:
        first = FixedHeightCityCore()
        second = FixedHeightCityCore()
        first.reset(40375)
        second.reset(40375)
        self.assertGreater(first.city_generation_attempt, 1)
        self.assertEqual(first.city_generation_attempt, second.city_generation_attempt)
        self.assertEqual(first.buildings, second.buildings)
        self.assertEqual(first.path_world, second.path_world)

    def test_reward_is_exact_sum_of_reported_terms(self) -> None:
        env = FixedHeightCityCore()
        env.reset(9)
        _, reward, _, _, info = env.step(env.expert_action())
        self.assertAlmostEqual(reward, sum(info["reward_terms"].values()))

    def test_bc_and_ppo_share_actor_contract(self) -> None:
        bc = LatentBcPolicy()
        ppo = LatentActorCritic()
        ppo.actor.load_state_dict(bc.state_dict())
        observation = torch.randn(4, 72)
        self.assertTrue(torch.equal(bc(observation), ppo.deterministic_action(observation)))
        distribution = ppo.distribution(observation)
        self.assertEqual(distribution.sample().shape, (4, 3))
        self.assertEqual(ppo.value(observation).shape, (4,))


if __name__ == "__main__":
    unittest.main()
