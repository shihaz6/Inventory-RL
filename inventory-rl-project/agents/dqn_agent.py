"""DQN agent for the inventory environment.

Wraps Stable-Baselines3 ``DQN("MlpPolicy", env)`` with:

- ``EvalCallback``: evaluates on a separate environment and saves the best
  weights to ``models/dqn/best_model.zip``.
- ``CheckpointCallback``: periodic checkpoints in ``models/dqn/checkpoints/``.
- Google Drive mirroring when running in Colab: the ``.zip`` files in
  ``models/dqn/`` are copied to ``MyDrive/inventory-rl/models/dqn/`` as they
  are written.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from environment.inventory_env import InventoryEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "dqn"
COLAB_DRIVE_ROOT = Path("/content/drive/MyDrive")
COLAB_DRIVE_DIR = COLAB_DRIVE_ROOT / "inventory-rl" / "models" / "dqn"

DEFAULT_DQN_KWARGS = dict(
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
)


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def build_dqn_agent(env, seed: int | None = None, **dqn_kwargs) -> DQN:
    """Create ``DQN("MlpPolicy", env)`` with the project defaults.

    Keyword arguments override ``DEFAULT_DQN_KWARGS``.
    """
    kwargs = {**DEFAULT_DQN_KWARGS, **dqn_kwargs}
    return DQN("MlpPolicy", env, seed=seed, verbose=0, **kwargs)


class MirrorToDriveCallback(BaseCallback):
    """Copy ``.zip``/``.npz`` files from a local folder to Google Drive.

    A file is copied again only when its modification time changes. Skips
    silently (with a warning) if Drive is not mounted.
    """

    def __init__(self, local_dir: Path, drive_dir: Path, freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.local_dir = Path(local_dir)
        self.drive_dir = Path(drive_dir)
        self.freq = max(1, int(freq))
        self._copied_mtimes: dict[Path, float] = {}
        self._warned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            self.sync()
        return True

    def _on_training_end(self) -> None:
        self.sync()

    def sync(self) -> None:
        if not COLAB_DRIVE_ROOT.exists():
            if not self._warned:
                print("Google Drive is not mounted; skipping Drive copy.")
                self._warned = True
            return
        for pattern in ("*.zip", "*.npz"):
            for src in self.local_dir.rglob(pattern):
                mtime = src.stat().st_mtime
                if self._copied_mtimes.get(src) == mtime:
                    continue
                dst = self.drive_dir / src.relative_to(self.local_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                self._copied_mtimes[src] = mtime


def train_dqn(
    total_timesteps: int = 200_000,
    env_kwargs: dict | None = None,
    dqn_kwargs: dict | None = None,
    seed: int = 42,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 20,
    checkpoint_freq: int = 20_000,
    models_dir: Path | str = MODELS_DIR,
    drive_dir: Path | str | None = "auto",
    verbose: int = 1,
):
    """Train DQN on ``InventoryEnv`` and save the best and periodic weights.

    Parameters
    ----------
    env_kwargs : passed to ``InventoryEnv(**env_kwargs)``.
    dqn_kwargs : overrides for ``DEFAULT_DQN_KWARGS``.
    eval_freq : evaluate (and sync to Drive) every this many environment steps.
    drive_dir : ``"auto"`` uses ``COLAB_DRIVE_DIR`` inside Colab and disables
        mirroring elsewhere; ``None`` disables it; a path mirrors to that path.

    Returns
    -------
    (model, best_model_path, final_model_path)
        ``best_model_path`` is the EvalCallback best checkpoint; the final
        weights are saved to ``final_model.zip`` in ``models_dir``.
    """
    models_dir = Path(models_dir)
    best_dir = models_dir
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
        best_model_save_path=str(best_dir),
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
        name_prefix="dqn",
    )
    callbacks = [eval_callback, checkpoint_callback]

    mirror = None
    if drive_dir is not None:
        mirror = MirrorToDriveCallback(models_dir, Path(drive_dir), freq=eval_freq)
        callbacks.append(mirror)

    model = build_dqn_agent(train_env, seed=seed, **(dqn_kwargs or {}))
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=False,
    )

    final_path = models_dir / "final_model.zip"
    model.save(final_path.with_suffix(""))
    if mirror is not None:
        mirror.sync()

    best_path = best_dir / "best_model.zip"
    return model, best_path, final_path
