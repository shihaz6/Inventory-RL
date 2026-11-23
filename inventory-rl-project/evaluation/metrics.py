"""KPI computation for inventory-control experiments.

Consumes the per-step ``info`` dicts logged by the environment (the same dicts
collected while running the random agent in
``notebooks/02_environment_testing.ipynb``) and aggregates them into the KPIs
reported in the project proposal.

Every step info dict must contain:

* ``holding_cost``, ``ordering_cost``, ``shortage_cost`` (``stockout_cost`` is
  also accepted) -- costs incurred during that step.
* ``demand`` -- units requested during that step.
* either fulfilled units (``fulfilled_demand``, ``units_sold``, ``sales``,
  ``demand_met``) or unmet units (``unmet_demand``, ``lost_sales``,
  ``stockout_units``, ``lost_demand``).

Definitions
-----------
Total Cost
    Holding + ordering + shortage cost over all steps.
Stockout Rate
    Fraction of steps in which at least one unit of demand was unmet.
Service Level
    Fill rate: total fulfilled units / total demand (1.0 when there is no
    demand at all).

Example
-------
>>> from evaluation.metrics import compute_kpis
>>> kpis = compute_kpis(step_infos)
>>> kpis.total_cost, kpis.service_level
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

__all__ = ["KPIs", "compute_kpis"]

_FULFILLED_KEYS = ("fulfilled_demand", "units_sold", "sales", "demand_met")
_UNMET_KEYS = ("unmet_demand", "lost_sales", "stockout_units", "lost_demand")


@dataclass(frozen=True)
class KPIs:
    """Aggregated performance indicators for an inventory policy."""

    total_cost: float
    holding_cost: float
    ordering_cost: float
    shortage_cost: float
    stockout_rate: float
    service_level: float

    def as_dict(self) -> dict[str, float]:
        """Return the KPIs as a plain dictionary (handy for DataFrames)."""
        return asdict(self)


def _value(info: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    """Return the first present, non-null key from ``info`` as a float."""
    for key in keys:
        if key in info and info[key] is not None:
            return float(info[key])
    return None


def compute_kpis(step_infos: Sequence[Mapping[str, Any]]) -> KPIs:
    """Aggregate per-step info dicts into the project KPIs.

    Parameters
    ----------
    step_infos:
        One info dict per environment step, in chronological order.

    Raises
    ------
    ValueError
        If ``step_infos`` is empty.
    KeyError
        If a step is missing a required cost, demand, or fulfilled/unmet field.
    """
    if not step_infos:
        raise ValueError("step_infos must contain at least one step info dict.")

    holding_cost = 0.0
    ordering_cost = 0.0
    shortage_cost = 0.0
    total_demand = 0.0
    total_fulfilled = 0.0
    stockout_steps = 0

    for step, info in enumerate(step_infos):
        holding = _value(info, ("holding_cost",))
        ordering = _value(info, ("ordering_cost", "order_cost"))
        shortage = _value(info, ("shortage_cost", "stockout_cost"))
        demand = _value(info, ("demand",))

        missing = [
            name
            for name, value in (
                ("holding_cost", holding),
                ("ordering_cost", ordering),
                ("shortage_cost", shortage),
                ("demand", demand),
            )
            if value is None
        ]
        if missing:
            raise KeyError(
                f"Step {step} is missing required info field(s): {', '.join(missing)}."
            )

        fulfilled = _value(info, _FULFILLED_KEYS)
        if fulfilled is None:
            unmet = _value(info, _UNMET_KEYS)
            if unmet is None:
                raise KeyError(
                    f"Step {step} needs fulfilled units ({', '.join(_FULFILLED_KEYS)}) "
                    f"or unmet units ({', '.join(_UNMET_KEYS)})."
                )
            fulfilled = demand - unmet

        fulfilled = max(0.0, fulfilled)

        holding_cost += holding
        ordering_cost += ordering
        shortage_cost += shortage
        total_demand += demand
        total_fulfilled += fulfilled
        if fulfilled < demand:
            stockout_steps += 1

    return KPIs(
        total_cost=holding_cost + ordering_cost + shortage_cost,
        holding_cost=holding_cost,
        ordering_cost=ordering_cost,
        shortage_cost=shortage_cost,
        stockout_rate=stockout_steps / len(step_infos),
        service_level=total_fulfilled / total_demand if total_demand > 0 else 1.0,
    )
