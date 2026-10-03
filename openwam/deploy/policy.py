"""WAM policy facade: one obs→action entry point over the two executors.

``WAMPolicy`` is the seam between the server (which hands it preprocessed
observations) and the execution mechanism (which schedules engine calls):

- sync mode (default): :class:`SyncInferenceExecutor` — blocking
  buffer-and-replan with a bounded execution horizon.
- async mode: :class:`AsyncInferenceExecutor` — double-buffered background
  inference overlapping generation with execution.

The executor is chosen once at construction from the normalized async
config; per-step dispatch is plain delegation.
"""

from collections import deque

import numpy as np
from omegaconf import OmegaConf

from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.executors import (
    AsyncInferenceExecutor,
    SyncInferenceExecutor,
    normalize_execution_config,
)


def _cfg_select(cfg, path: str, default):
    """Read a dotted key from an OmegaConf config or a plain namespace/dict."""
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, path, default=default)
    node = cfg
    for key in path.split("."):
        node = node.get(key) if isinstance(node, dict) else getattr(node, key, None)
        if node is None:
            return default
    return node


class FrameHistory:
    """Past frames for a history (memory) checkpoint, sampled the way training reads them.

    The training reader (``openwam/dataloader/robodojo.py``) conditions the window
    starting at frame ``t`` on frames ``max(t - k * stride, 0)`` for
    ``k = num_frames .. 1``, oldest first, with the oldest slot pinned to frame 0
    when ``include_first_frame`` is set. Deploy reproduces that with one
    :meth:`push` per environment step: before ``k * stride`` steps have passed a
    slot holds the episode's first frame (the reader's clamp), and the pinned slot
    keeps the first frame after the bounded buffer has dropped it.
    """

    def __init__(self, num_frames: int, stride: int, include_first_frame: bool = False):
        self.num_frames = int(num_frames)
        self.stride = int(stride)
        self.include_first_frame = bool(include_first_frame)
        self._frames = deque(maxlen=self.num_frames * self.stride + 1)
        self._first_frame = None

    @classmethod
    def from_cfg(cls, cfg) -> "FrameHistory":
        """The contract a checkpoint was trained with (``dataloader.history_*``); 0 frames = off."""
        return cls(
            int(_cfg_select(cfg, "dataloader.history_num_frames", 0) or 0),
            int(_cfg_select(cfg, "dataloader.history_stride", 25) or 25),
            bool(_cfg_select(cfg, "dataloader.history_include_first_frame", False)),
        )

    def push(self, img) -> list:
        """Record this step's frame and return the past ones, oldest first."""
        if self._first_frame is None:
            self._first_frame = img
        self._frames.append(img)
        newest = len(self._frames) - 1
        frames = [self._frames[max(newest - k * self.stride, 0)] for k in range(self.num_frames, 0, -1)]
        if self.include_first_frame:
            frames[0] = self._first_frame
        return frames

    def clear(self):
        """Forget the episode (call between episodes)."""
        self._frames.clear()
        self._first_frame = None


class WAMPolicy:
    """Unified policy facade over the sync / async execution mechanisms.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Root config, retained for policy-level consumers.
        execution_config: ExecutionConfig-like. ``inference_horizon`` applies
            to both modes; ``inference_delay_steps`` applies only to async.
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, execution_config=None):
        self.cfg = cfg
        self.engine = engine

        # History (memory) contract comes from the CKPT's dataloader config, so
        # deploy samples past frames exactly as training did; 0 frames = off.
        self._history = FrameHistory.from_cfg(cfg)

        self._execution_config = normalize_execution_config(execution_config)
        self._async = self._execution_config.enabled
        if self._async:
            self._executor = AsyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
                inference_delay_steps=self._execution_config.inference_delay_steps,
            )
        else:
            self._executor = SyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
            )

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given (already preprocessed) observation.

        The final legality projection for two-point command dims
        (``architecture.binary_command_dims``, from the CKPT's dataloader.binary_action_dims) runs
        HERE — after all executor arithmetic. The normalizer already emits exact ±1 for those dims,
        and this final boundary also protects engines or checkpoints that emit
        values between the two legal commands. Threshold 0.5 preserves the
        downstream command contract for the WS server and direct consumers.
        """
        return self._project_binary_dims(self._executor.predict_action(self._build_conditions(obs)))

    def _project_binary_dims(self, action) -> np.ndarray:
        """Final legality projection for two-point command dims (see predict_action).

        Works on a single action or any stack of them (the last axis is the action width).
        """
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if dims:
            action = np.array(action)
            for d in dims:
                if d >= action.shape[-1]:
                    raise ValueError(
                        f"binary_command_dims includes {d} but the action is {action.shape[-1]}-D; "
                        "the ckpt config and the served action width disagree."
                    )
                action[..., d] = np.where(action[..., d] > 0.5, 1.0, -1.0)
        return action

    def reset(self):
        """Clear executor state and frame history between episodes."""
        self._executor.reset()
        self._history.clear()

    def shutdown(self):
        """Release executor resources (background threads in async mode)."""
        self._executor.shutdown()

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from the current observation.

        Populates the engine-facing fields (``first_frame_image``,
        ``prompt``) from the server-preprocessed observation so the
        pipeline receives images without any further client-side work.
        """
        conditions = {
            "observation": obs,
        }
        img = obs.get("image")
        if img is not None:
            # Single first frame — pipeline expects list[PIL.Image]
            conditions["first_frame_image"] = [img]
            if self._history.num_frames:
                conditions["history_images"] = self._history.push(img)
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio"] = obs["state"]
        return conditions
