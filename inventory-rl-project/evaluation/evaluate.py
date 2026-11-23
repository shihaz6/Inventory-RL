"""Policy evaluation for the inventory environment.

Runs any policy (baseline or trained agent) for a number of episodes, computes
the KPIs from :mod:`evaluation.metrics` for each episode, and aggregates them.

A *policy* can be any of:

* a callable ``policy(observation) -> int``
* an object with ``act(observation) -> int``
* an object with ``predict(observation, deterministic=True) -> (action, state)``
  (Stable-Baselines3 style)

Policies may optionally define ``reset()``, which is called at the start of
every episode.

When several policies are compared, episode ``i`` uses the same environment
seed for every policy (common random numbers), so differences in cost reflect
the policy rather than demand luck.

Example
-------
>>> from environment.inventory_env import InventoryEnv
>>> from evaluation.evaluate import compare_policies, format_report
>>> results = compare_policies(
...     {"base-stock": BaseStock(30)}, env_factory=InventoryEnv, n_episodes=100
... )
>>> print(format_report(results))

Command line (from the project root)::

    python -m evaluation.evaluate --episodes 100 --seed 0
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.metrics import KPIs, compute_kpis

__all__ = [
    "EvaluationResult",
    "run_episode",
    "evaluate_policy",
    "compare_policies",
    "format_report",
]

_KPI_FIELDS = tuple(KPIs.__dataclass_fields__)


@dataclass(frozen=True)
class EvaluationResult:
    """Per-episode KPIs for one policy, with summary statistics."""

    episodes: tuple[KPIs, ...]
    total_reward: tuple[float, ...]

    def values(self, kpi: str) -> list[float]:
        """Return one KPI's value for every episode."""
        if kpi not in _KPI_FIELDS:
            raise KeyError(f"Unknown KPI {kpi!r}; choose from {_KPI_FIELDS}.")
        return [getattr(k, kpi) for k in self.episodes]

    def mean(self, kpi: str) -> float:
        return statistics.fmean(self.values(kpi))

    def std(self, kpi: str) -> float:
        """Sample standard deviation across episodes (0.0 for one episode)."""
        values = self.values(kpi)
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def stderr(self, kpi: str) -> float:
        """Standard error of the mean."""
        return self.std(kpi) / len(self.episodes) ** 0.5

    def summary(self) -> dict[str, dict[str, float]]:
        """``{kpi: {"mean", "std", "stderr"}}`` plus the mean episode reward."""
        out = {
            kpi: {"mean": self.mean(kpi), "std": self.std(kpi), "stderr": self.stderr(kpi)}
            for kpi in _KPI_FIELDS
        }
        rewards = list(self.total_reward)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        out["total_reward"] = {
            "mean": statistics.fmean(rewards),
            "std": std,
            "stderr": std / len(rewards) ** 0.5,
        }
        return out


def _as_action_fn(policy: Any) -> Callable[[np.ndarray], int]:
    """Adapt the supported policy shapes to ``observation -> action``."""
    if hasattr(policy, "act"):
        return policy.act
    if hasattr(policy, "predict"):
        return lambda obs: policy.predict(obs, deterministic=True)[0]
    if callable(policy):
        return policy
    raise TypeError(
        "policy must be callable or define act(obs) or predict(obs, deterministic)."
    )


def run_episode(env, policy: Any, *, seed: int | None = None) -> tuple[list[dict], float]:
    """Play one full episode; return the per-step infos and total reward."""
    act = _as_action_fn(policy)
    if hasattr(policy, "reset") and callable(policy.reset):
        policy.reset()

    observation, _ = env.reset(seed=seed)
    infos: list[dict] = []
    total_reward = 0.0
    done = False
    while not done:
        action = int(np.asarray(act(observation)).item())
        observation, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        total_reward += float(reward)
        done = terminated or truncated
    return infos, total_reward


def evaluate_policy(
    policy: Any,
    env_factory: Callable[[], Any],
    *,
    n_episodes: int = 100,
    seed: int = 0,
) -> EvaluationResult:
    """Evaluate ``policy`` over ``n_episodes`` episodes.

    Episode ``i`` is reset with seed ``seed + i``.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1.")

    env = env_factory()
    try:
        kpis, rewards = [], []
        for i in range(n_episodes):
            infos, total_reward = run_episode(env, policy, seed=seed + i)
            kpis.append(compute_kpis(infos))
            rewards.append(total_reward)
    finally:
        env.close()
    return EvaluationResult(episodes=tuple(kpis), total_reward=tuple(rewards))


def compare_policies(
    policies: Mapping[str, Any],
    env_factory: Callable[[], Any],
    *,
    n_episodes: int = 100,
    seed: int = 0,
) -> dict[str, EvaluationResult]:
    """Evaluate several named policies on identical episode seeds."""
    if not policies:
        raise ValueError("policies must not be empty.")
    return {
        name: evaluate_policy(policy, env_factory, n_episodes=n_episodes, seed=seed)
        for name, policy in policies.items()
    }


def format_report(results: Mapping[str, EvaluationResult]) -> str:
    """Render a plain-text table of mean ± std per policy, best cost first."""
    columns = [
        ("total_cost", "Total cost", "{:.1f}"),
        ("holding_cost", "Holding", "{:.1f}"),
        ("ordering_cost", "Ordering", "{:.1f}"),
        ("shortage_cost", "Shortage", "{:.1f}"),
        ("stockout_rate", "Stockout", "{:.3f}"),
        ("service_level", "Service", "{:.3f}"),
    ]
    rows = []
    for name, result in sorted(results.items(), key=lambda kv: kv[1].mean("total_cost")):
        cells = [name]
        for kpi, _, fmt in columns:
            cells.append(f"{fmt.format(result.mean(kpi))} ± {fmt.format(result.std(kpi))}")
        rows.append(cells)

    header = ["Policy", *(title for _, title, _ in columns)]
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()
    n = len(next(iter(results.values())).episodes)
    return "\n".join(
        [f"Mean ± std over {n} episodes", line(header), line(["-" * w for w in widths]),
         *map(line, rows)]
    )


# --------------------------------------------------------------------------
# Command line: quick sanity comparison using two simple reference policies.
# --------------------------------------------------------------------------
class _RandomPolicy:
    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self._n = n_actions
        self._rng = np.random.default_rng(seed)

    def act(self, observation) -> int:
        return int(self._rng.integers(self._n))


class _BaseStockPolicy:
    """Order up to ``level`` units of inventory position (stock + pipeline)."""

    def __init__(self, level: int, max_order: int) -> None:
        self._level, self._max_order = level, max_order

    def act(self, observation) -> int:
        position = float(np.sum(observation[:-1]))  # stock + pending deliveries
        return int(min(self._max_order, max(0, round(self._level - position))))


def _main(argv: Sequence[str] | None = None) -> None:
    from environment.inventory_env import InventoryEnv

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-stock", type=int, default=30,
                        help="order-up-to level for the reference policy")
    parser.add_argument("--json", metavar="PATH",
                        help="also write summary statistics to this JSON file")
    args = parser.parse_args(argv)

    n_actions = InventoryEnv().action_space.n
    policies = {
        "random": _RandomPolicy(n_actions, seed=args.seed),
        f"base-stock({args.base_stock})": _BaseStockPolicy(args.base_stock, n_actions - 1),
    }
    results = compare_policies(
        policies, InventoryEnv, n_episodes=args.episodes, seed=args.seed
    )
    print(format_report(results))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({k: v.summary() for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    _main()
