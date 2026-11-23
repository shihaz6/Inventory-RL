"""Grid search and locked benchmark for the classical inventory baselines.

Protocol
--------
* Tuning seeds (``TUNING_SEEDS``) are used only to choose parameters.
* Test seeds (``TEST_SEEDS``) are disjoint and used only to score the chosen
  parameters. The benchmark score is the mean total cost on the test seeds.
* Every policy is run on the same seeds, so demand sequences are shared
  (common random numbers) and comparisons are paired.

Costs are minimised; the RL reward is ``-total_cost``, so a DQN or PPO policy
beats the benchmark when its mean episode reward is above ``-benchmark_cost``.
"""

from __future__ import annotations

import csv
import json
from functools import partial
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from baselines.random_policy import random_policy
from baselines.reorder_point import reorder_point_policy
from baselines.s_S_policy import s_S_policy
from environment.inventory_env import ORDER_QUANTITIES, InventoryEnv
from evaluation.metrics import compute_kpis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

ENV_KWARGS = dict(
    order_quantities=ORDER_QUANTITIES,
    initial_stock=20,
    lead_time=2,
    episode_days=60,
    holding_cost=1.0,
    ordering_cost=2.0,
    shortage_cost=5.0,
)

TUNING_SEEDS = tuple(range(50))
TEST_SEEDS = tuple(range(1000, 1100))

REORDER_POINTS = tuple(range(0, 101, 5))
REORDER_QUANTITIES = ORDER_QUANTITIES[1:]
S_LEVELS = tuple(range(0, 151, 5))
S_REORDER_LEVELS = tuple(range(0, 101, 5))

# A policy factory takes the episode seed and returns ``policy(observation)``.
PolicyFactory = Callable[[int], Callable[[Sequence[float]], int]]


def run_episode(policy: Callable[[Sequence[float]], int], seed: int) -> list[dict]:
    """Run one full episode and return the per-step info dicts."""
    env = InventoryEnv(**ENV_KWARGS)
    observation, _ = env.reset(seed=seed)
    infos: list[dict] = []
    while True:
        observation, _, terminated, truncated, info = env.step(policy(observation))
        infos.append(info)
        if terminated or truncated:
            break
    env.close()
    return infos


def evaluate_policy(make_policy: PolicyFactory, seeds: Sequence[int]) -> dict:
    """Run the policy on each seed and summarise total cost and KPIs."""
    costs: list[float] = []
    kpi_rows: list[dict] = []
    for seed in seeds:
        infos = run_episode(make_policy(seed), seed)
        kpis = compute_kpis(infos)
        costs.append(kpis.total_cost)
        kpi_rows.append(kpis.as_dict())

    costs_arr = np.asarray(costs)
    summary = {
        "mean_cost": float(costs_arr.mean()),
        "std_cost": float(costs_arr.std(ddof=1)) if len(costs) > 1 else 0.0,
        "stderr_cost": float(costs_arr.std(ddof=1) / np.sqrt(len(costs)))
        if len(costs) > 1
        else 0.0,
        "n_episodes": len(costs),
    }
    for key in kpi_rows[0]:
        summary[f"mean_{key}"] = float(np.mean([row[key] for row in kpi_rows]))
    return summary


def grid_search_reorder_point(seeds: Sequence[int] = TUNING_SEEDS) -> list[dict]:
    """Evaluate every (R, Q) pair on the tuning seeds."""
    rows = []
    for reorder_point in REORDER_POINTS:
        for order_quantity in REORDER_QUANTITIES:
            def make(seed, R=reorder_point, Q=order_quantity):
                return partial(reorder_point_policy, reorder_point=R, order_quantity=Q)

            summary = evaluate_policy(make, seeds)
            rows.append({"reorder_point": reorder_point, "order_quantity": order_quantity, **summary})
    return rows


def grid_search_s_S(seeds: Sequence[int] = TUNING_SEEDS) -> list[dict]:
    """Evaluate every (s, S) pair with s < S on the tuning seeds."""
    rows = []
    for s in S_REORDER_LEVELS:
        for S in S_LEVELS:
            if s >= S:
                continue

            def make(seed, s=s, S=S):
                return partial(s_S_policy, s=s, S=S)

            summary = evaluate_policy(make, seeds)
            rows.append({"s": s, "S": S, **summary})
    return rows


def best_row(rows: Sequence[dict], keys: Sequence[str]) -> dict:
    """Lowest mean cost; ties go to the smaller parameter values."""
    return min(rows, key=lambda row: (row["mean_cost"], *(row[k] for k in keys)))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(tables_dir: Path | str = TABLES_DIR) -> dict:
    """Run both grid searches, score the winners on the test seeds, and lock the result.

    Writes the grid results as CSV and the locked benchmark as
    ``baseline_benchmark.json`` in ``tables_dir``.
    """
    tables_dir = Path(tables_dir)

    rp_rows = grid_search_reorder_point()
    rp_best = best_row(rp_rows, ("reorder_point", "order_quantity"))

    ss_rows = grid_search_s_S()
    ss_best = best_row(ss_rows, ("s", "S"))

    rp_test = evaluate_policy(
        lambda seed: partial(
            reorder_point_policy,
            reorder_point=rp_best["reorder_point"],
            order_quantity=rp_best["order_quantity"],
        ),
        TEST_SEEDS,
    )
    ss_test = evaluate_policy(
        lambda seed: partial(s_S_policy, s=ss_best["s"], S=ss_best["S"]), TEST_SEEDS
    )
    random_test = evaluate_policy(
        lambda seed: partial(
            random_policy,
            n_actions=len(ORDER_QUANTITIES),
            rng=np.random.default_rng(seed),
        ),
        TEST_SEEDS,
    )

    candidates = {
        "reorder_point": rp_test,
        "s_S": ss_test,
    }
    winner = min(candidates, key=lambda name: candidates[name]["mean_cost"])

    benchmark = {
        "protocol": {
            "env_kwargs": {k: list(v) if isinstance(v, tuple) else v for k, v in ENV_KWARGS.items()},
            "tuning_seeds": [TUNING_SEEDS[0], TUNING_SEEDS[-1]],
            "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
            "score": "mean total cost on test seeds (lower is better)",
        },
        "reorder_point": {
            "params": {
                "reorder_point": rp_best["reorder_point"],
                "order_quantity": rp_best["order_quantity"],
            },
            "tuning": {k: rp_best[k] for k in ("mean_cost", "stderr_cost")},
            "test": rp_test,
        },
        "s_S": {
            "params": {"s": ss_best["s"], "S": ss_best["S"]},
            "tuning": {k: ss_best[k] for k in ("mean_cost", "stderr_cost")},
            "test": ss_test,
        },
        "random": {"test": random_test},
        "benchmark": {
            "policy": winner,
            "mean_cost": candidates[winner]["mean_cost"],
            "stderr_cost": candidates[winner]["stderr_cost"],
            "mean_reward": -candidates[winner]["mean_cost"],
        },
    }

    _write_csv(tables_dir / "baseline_grid_reorder_point.csv", rp_rows)
    _write_csv(tables_dir / "baseline_grid_s_S.csv", ss_rows)
    tables_dir.mkdir(parents=True, exist_ok=True)
    with (tables_dir / "baseline_benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, indent=2)

    return benchmark
