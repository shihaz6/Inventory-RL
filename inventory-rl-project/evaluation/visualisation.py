"""Plots for inventory-policy evaluation results.

Works on the objects produced by :mod:`evaluation.evaluate`:

* :func:`plot_cost_breakdown`    -- mean holding / ordering / shortage cost per
  policy as stacked horizontal bars (lowest total cost on top).
* :func:`plot_cost_distribution` -- every episode's total cost per policy, with
  the mean marked, to show variability and not just the average.
* :func:`plot_episode_trajectory` -- stock, orders and demand over one episode,
  from the step ``info`` dicts returned by ``run_episode``.
* :func:`save_report`            -- write all of the above to a directory.

Colours come from a validated colour-blind-safe categorical palette, assigned
in a fixed order: a policy keeps its colour for as long as it keeps its position
in the mapping you pass in, never according to its rank. Every function accepts
``theme="light"`` or ``theme="dark"`` and returns the matplotlib ``Figure``.

Example
-------
>>> results = compare_policies(policies, InventoryEnv, n_episodes=100)
>>> fig = plot_cost_breakdown(results)
>>> fig.savefig("cost_breakdown.png", dpi=200)

Command line (from the project root)::

    python -m evaluation.visualisation --episodes 100 --out reports
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from evaluation.evaluate import EvaluationResult

__all__ = [
    "plot_cost_breakdown",
    "plot_cost_distribution",
    "plot_episode_trajectory",
    "save_report",
]

# Categorical slots in fixed order (light, dark). Only the first eight exist;
# beyond that, fold policies together or plot them in separate figures.
_SERIES = (
    ("#2a78d6", "#3987e5"),  # blue
    ("#eb6834", "#d95926"),  # orange
    ("#1baf7a", "#199e70"),  # aqua
    ("#eda100", "#c98500"),  # yellow
    ("#e87ba4", "#d55181"),  # magenta
    ("#008300", "#008300"),  # green
    ("#6250d6", "#9085e9"),  # violet
    ("#e34948", "#e66767"),  # red
)

_THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e", grid="#e6e5e1"),
    "dark": dict(surface="#1a1a19", text="#f0efec", muted="#c3c2b7", grid="#333331"),
}

# Cost components always use the same three slots.
_COMPONENTS = (
    ("holding_cost", "Holding"),
    ("ordering_cost", "Ordering"),
    ("shortage_cost", "Shortage"),
)


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "evaluation.visualisation needs matplotlib (pip install matplotlib)."
        ) from exc
    return plt


def _style(theme: str) -> dict:
    if theme not in _THEMES:
        raise ValueError(f"theme must be one of {sorted(_THEMES)}.")
    return _THEMES[theme]


def _colors(n: int, theme: str) -> list[str]:
    if n > len(_SERIES):
        raise ValueError(
            f"At most {len(_SERIES)} series can be coloured distinctly; got {n}. "
            "Plot fewer policies per figure."
        )
    index = 0 if theme == "light" else 1
    return [_SERIES[i][index] for i in range(n)]


def _figure(theme: str, width: float, height: float, **subplot_kw):
    plt = _pyplot()
    s = _style(theme)
    fig, axes = plt.subplots(figsize=(width, height), facecolor=s["surface"], **subplot_kw)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(s["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(s["grid"])
        ax.tick_params(colors=s["muted"], length=0, labelsize=9)
    return fig, axes


def _titles(fig, ax, title: str, subtitle: str | None, s: dict) -> None:
    ax.set_title(title, loc="left", color=s["text"], fontsize=12, fontweight="bold",
                 pad=24 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=s["muted"], fontsize=9,
                va="bottom")


def plot_cost_breakdown(
    results: Mapping[str, EvaluationResult], *, theme: str = "light"
):
    """Stacked horizontal bars of mean holding/ordering/shortage cost."""
    s = _style(theme)
    comp_colors = _colors(len(_COMPONENTS), theme)
    names = sorted(results, key=lambda n: results[n].mean("total_cost"), reverse=True)
    n_episodes = len(next(iter(results.values())).episodes)

    fig, ax = _figure(theme, 8, 1.4 + 0.6 * len(names))
    ypos = np.arange(len(names))
    left = np.zeros(len(names))
    for (kpi, label), color in zip(_COMPONENTS, comp_colors):
        widths = np.array([results[n].mean(kpi) for n in names])
        # Surface-coloured edge = the gap that separates touching segments.
        ax.barh(ypos, widths, left=left, height=0.5, color=color, label=label,
                edgecolor=s["surface"], linewidth=2)
        left += widths
    totals = left
    top = max(totals.max(), 1.0)
    for y, total in zip(ypos, totals):
        ax.text(total + top * 0.01, y, f"{total:,.0f}", va="center", color=s["text"],
                fontsize=9)

    ax.set_yticks(ypos, names)
    ax.tick_params(axis="y", labelcolor=s["text"], labelsize=10)
    ax.set_xlim(0, top * 1.1)
    ax.xaxis.grid(True, color=s["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean cost per episode", color=s["muted"], fontsize=9)
    legend = ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False,
                       fontsize=9, handlelength=1, handleheight=1)
    for text in legend.get_texts():
        text.set_color(s["muted"])
    _titles(fig, ax, "Cost breakdown by policy", f"Mean over {n_episodes} episodes", s)
    fig.tight_layout()
    return fig


def plot_cost_distribution(
    results: Mapping[str, EvaluationResult], *, theme: str = "light"
):
    """Per-episode total cost for every policy, with the mean marked."""
    s = _style(theme)
    names = list(results)  # insertion order fixes each policy's colour
    colors = dict(zip(names, _colors(len(names), theme)))
    order = sorted(names, key=lambda n: results[n].mean("total_cost"), reverse=True)
    n_episodes = len(results[order[0]].episodes)
    rng = np.random.default_rng(0)  # jitter only; fixed so figures are reproducible

    fig, ax = _figure(theme, 8, 1.4 + 0.7 * len(order))
    for y, name in enumerate(order):
        values = np.asarray(results[name].values("total_cost"))
        ax.scatter(values, y + rng.uniform(-0.18, 0.18, len(values)), s=16,
                   color=colors[name], alpha=0.45, linewidths=0, zorder=2)
        mean = values.mean()
        ax.scatter([mean], [y], s=90, color=colors[name], edgecolors=s["surface"],
                   linewidths=2, zorder=3)
        ax.annotate(f"mean {mean:,.0f}", (mean, y), xytext=(0, 20),
                    textcoords="offset points", ha="center", color=s["text"], fontsize=9)

    ax.set_yticks(range(len(order)), order)
    ax.tick_params(axis="y", labelcolor=s["text"], labelsize=10)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.xaxis.grid(True, color=s["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Total cost per episode", color=s["muted"], fontsize=9)
    _titles(fig, ax, "Episode cost distribution",
            f"Each dot is one of {n_episodes} episodes; the large dot is the mean", s)
    fig.tight_layout()
    return fig


def plot_episode_trajectory(
    step_infos: Sequence[Mapping], *, title: str = "Episode trajectory",
    theme: str = "light",
):
    """Ending stock, order quantity and demand per day for one episode."""
    if not step_infos:
        raise ValueError("step_infos must not be empty.")
    s = _style(theme)
    c_stock, c_order, c_demand = _colors(3, theme)
    days = np.arange(len(step_infos))
    series = (
        ("Ending stock (units)", "ending_stock", c_stock, "line"),
        ("Order quantity (units)", "order_quantity", c_order, "bar"),
        ("Demand (units)", "demand", c_demand, "bar"),
    )
    missing = [k for _, k, _, _ in series if k not in step_infos[0]]
    if missing:
        raise KeyError(f"step infos are missing field(s): {', '.join(missing)}.")

    fig, axes = _figure(theme, 9, 6.2, nrows=3, sharex=True)
    for ax, (label, key, color, kind) in zip(axes, series):
        values = np.array([info[key] for info in step_infos], dtype=float)
        if kind == "line":
            ax.plot(days, values, color=color, linewidth=2, solid_capstyle="round")
            ax.fill_between(days, values, color=color, alpha=0.1, linewidth=0)
            ax.scatter([days[-1]], [values[-1]], s=40, color=color,
                       edgecolors=s["surface"], linewidths=2, zorder=3)
        else:
            ax.bar(days, values, width=0.6, color=color)
        ax.set_title(label, loc="left", color=s["muted"], fontsize=9, pad=4)
        ax.set_ylim(bottom=0)
        ax.yaxis.grid(True, color=s["grid"], linewidth=1)
        ax.set_axisbelow(True)
    stockouts = [d for d, info in zip(days, step_infos) if info.get("unmet_demand", 0) > 0]
    if stockouts:
        axes[0].scatter(stockouts, np.zeros(len(stockouts)), marker="x", s=30,
                        color=s["text"], linewidths=1.5, zorder=4, clip_on=False,
                        label="Stockout day")
        legend = axes[0].legend(loc="upper right", frameon=False, fontsize=9)
        for text in legend.get_texts():
            text.set_color(s["muted"])
    axes[-1].set_xlabel("Day", color=s["muted"], fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=s["text"], fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_report(
    results: Mapping[str, EvaluationResult],
    out_dir: str | Path,
    *,
    theme: str = "light",
    fmt: str = "png",
    dpi: int = 200,
) -> list[Path]:
    """Save the comparison figures into ``out_dir``; return the written paths."""
    plt = _pyplot()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "cost_breakdown": plot_cost_breakdown(results, theme=theme),
        "cost_distribution": plot_cost_distribution(results, theme=theme),
    }
    paths = []
    for name, fig in figures.items():
        path = out / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return paths


def _main(argv: Sequence[str] | None = None) -> None:
    from environment.inventory_env import InventoryEnv
    from evaluation.evaluate import (
        _BaseStockPolicy, _RandomPolicy, compare_policies, run_episode,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="reports", help="output directory")
    parser.add_argument("--theme", choices=sorted(_THEMES), default="light")
    args = parser.parse_args(argv)

    n_actions = InventoryEnv().action_space.n
    policies = {
        "random": _RandomPolicy(n_actions, seed=args.seed),
        "base-stock(30)": _BaseStockPolicy(30, n_actions - 1),
    }
    results = compare_policies(policies, InventoryEnv, n_episodes=args.episodes,
                               seed=args.seed)
    paths = save_report(results, args.out, theme=args.theme)

    infos, _ = run_episode(InventoryEnv(), policies["base-stock(30)"], seed=args.seed)
    fig = plot_episode_trajectory(infos, title="Episode trajectory: base-stock(30)",
                                  theme=args.theme)
    path = Path(args.out) / "episode_trajectory.png"
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    paths.append(path)
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    _main()
