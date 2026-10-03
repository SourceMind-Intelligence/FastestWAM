"""Per-trial recordings for diagnostic LIBERO runs, off unless OPENWAM_LIBERO_RECORD_DIR is set.

Each trial is saved as ``<dir>/<suite>/task<NN>/trial<NN>.npz`` with:

- the agent-view and wrist frames, upright and at half size, every
  OPENWAM_LIBERO_RECORD_EVERY steps (default 2) plus the last one;
- the 7-D action sent at every step;
- the end-effector position and gripper opening after every step.

Recording only reads the observations the client already has, so the trial's
rollout and result are the same with it on or off.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

RECORD_DIR_ENV = "OPENWAM_LIBERO_RECORD_DIR"
RECORD_EVERY_ENV = "OPENWAM_LIBERO_RECORD_EVERY"
CAMERA_KEYS = {"agentview": "agentview_image", "wrist": "robot0_eye_in_hand_image"}


def upright_half(image) -> np.ndarray:
    """LIBERO renders upside down: rotate by 180 degrees, then keep every second pixel."""
    image = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(image[::-1, ::-1][::2, ::2])


def _vector(obs: dict, key: str, size: int) -> np.ndarray:
    value = obs.get(key)
    if value is None:
        return np.full(size, np.nan, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


class EpisodeRecorder:
    """Collects one trial's frames, actions and end-effector states, then saves them as one ``.npz``."""

    def __init__(self, path: Path, every: int = 2):
        if every < 1:
            raise ValueError(f"{RECORD_EVERY_ENV} must be at least 1, got {every}")
        self.path = Path(path)
        self.every = int(every)
        self._cameras: list[str] | None = None
        self._frames: dict[str, list[np.ndarray]] = {}
        self._frame_steps: list[int] = []
        self._actions: list[np.ndarray] = []
        self._eef_pos: list[np.ndarray] = []
        self._gripper: list[np.ndarray] = []
        self._last_obs: dict | None = None

    @classmethod
    def from_env(cls, suite: str, task_id: int, trial: int) -> "EpisodeRecorder | None":
        root = os.environ.get(RECORD_DIR_ENV, "").strip()
        if not root:
            return None
        every = int(os.environ.get(RECORD_EVERY_ENV, "2"))
        return cls(Path(root) / suite / f"task{task_id:02d}" / f"trial{trial:02d}.npz", every)

    @property
    def steps(self) -> int:
        return len(self._actions)

    def observe(self, obs: dict, action=None) -> None:
        """Record ``obs``; ``action`` is the one that led to it (``None`` for the first observation)."""
        if self._cameras is None:
            self._cameras = [name for name, key in CAMERA_KEYS.items() if obs.get(key) is not None]
            self._frames = {name: [] for name in self._cameras}
        if action is not None:
            self._actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self._eef_pos.append(_vector(obs, "robot0_eef_pos", 3))
        self._gripper.append(_vector(obs, "robot0_gripper_qpos", 2))
        self._last_obs = obs
        if self.steps % self.every == 0:
            self._add_frames(obs)

    def _add_frames(self, obs: dict) -> None:
        for name in self._cameras or []:
            self._frames[name].append(upright_half(obs[CAMERA_KEYS[name]]))
        self._frame_steps.append(self.steps)

    def save(self, **meta) -> Path:
        """Write the trial, adding the last frame if the sampling skipped it; ``meta`` is stored as-is."""
        if self._last_obs is not None and (not self._frame_steps or self._frame_steps[-1] != self.steps):
            self._add_frames(self._last_obs)
        arrays = {
            "frame_steps": np.asarray(self._frame_steps, dtype=np.int32),
            "actions": np.asarray(self._actions, dtype=np.float32).reshape(len(self._actions), -1),
            "eef_pos": np.asarray(self._eef_pos, dtype=np.float32).reshape(len(self._eef_pos), -1),
            "gripper_qpos": np.asarray(self._gripper, dtype=np.float32).reshape(len(self._gripper), -1),
        }
        for name, frames in self._frames.items():
            arrays[name] = np.stack(frames) if frames else np.zeros((0, 0, 0, 3), dtype=np.uint8)
        for key, value in meta.items():
            arrays[key] = np.asarray(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path, **arrays)
        return self.path
