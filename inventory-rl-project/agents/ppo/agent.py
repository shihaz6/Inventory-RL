"""PPO agent for the inventory environment.

Wraps Stable-Baselines3 ``PPO("MlpPolicy", env)`` with the same evaluation
setup as ``agents/dqn/agent.py`` so training curves can be compared
head-to-head:

- ``EvalCallback``: evaluates on a separate environment and saves the best
  weights to ``models/ppo/best_model.zip``.
- ``CheckpointCallback``: periodic checkpoints in ``models/ppo/checkpoints/``.
- Google Drive mirroring when running in Colab, to
  ``MyDrive/inventory-rl/models/ppo/``.
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from agents.dqn.agent import COLAB_DRIVE_ROOT, MirrorToDriveCallback, in_colab
from environment.inventory_env import InventoryEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models" / "ppo"
COLAB_DRIVE_DIR = COLAB_DRIVE_ROOT / "inventory-rl" / "models" / "ppo"

DEFAULT_PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
)


def build_ppo_agent(env, seed: int | None = None, **ppo_kwargs) -> PPO:
    """Create ``PPO("MlpPolicy", env)`` with the project defaults.

    Keyword arguments override ``DEFAULT_PPO_KWARGS``.
    """
    kwargs = {**DEFAULT_PPO_KWARGS, **ppo_kwargs}
    return PPO("MlpPolicy", env, seed=seed, verbose=0, **kwargs)


def train_ppo(
    total_timesteps: int = 200_000,
    env_kwargs: dict | None = None,
    ppo_kwargs: dict | None = None,
    seed: int = 42,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 20,
    checkpoint_freq: int = 20_000,
    models_dir: Path | str = MODELS_DIR,
    drive_dir: Path | str | None = "auto",
    verbose: int = 1,
):
    """Train PPO on ``InventoryEnv`` and save the best and periodic weights.

    The arguments mirror ``agents.dqn.agent.train_dqn`` so the two runs use
    the same evaluation schedule and environment settings.

    Parameters
    ----------
    env_kwargs : passed to ``InventoryEnv(**env_kwargs)``.
    ppo_kwargs : overrides for ``DEFAULT_PPO_KWARGS``.
    eval_freq : evaluate (and sync to Drive) every this many environment steps.
    drive_dir : ``"auto"`` uses ``COLAB_DRIVE_DIR`` inside Colab and disables
        mirroring elsewhere; ``None`` disables it; a path mirrors to that path.

    Returns
    -------
    (model, best_model_path, final_model_path)
    """
    models_dir = Path(models_dir)
    checkpoint_dir = models_dir / "checkpoints"
    log_dir = models_dir / "logs"
    for path in (models_dir, checkpoint_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    if drive_dir == "auto":
        drive_dir = COLAB_DRIVE_DIR if in_colab() else None

    train_env = Monitor(InventoryEnv(**(env_kwargs or {})))
    eval_env = Monitor(InventoryEnv(**(env_kwargs or {})))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="ppo",
    )
    callbacks = [eval_callback, checkpoint_callback]

    mirror = None
    if drive_dir is not None:
        mirror = MirrorToDriveCallback(models_dir, Path(drive_dir), freq=eval_freq)
        callbacks.append(mirror)

    model = build_ppo_agent(train_env, seed=seed, **(ppo_kwargs or {}))
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    final_path = models_dir / "final_model.zip"
    model.save(final_path.with_suffix(""))
    if mirror is not None:
        mirror.sync()

    best_path = models_dir / "best_model.zip"
    return model, best_path, final_path
