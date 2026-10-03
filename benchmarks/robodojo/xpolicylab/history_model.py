"""XPolicyLab adapter for OpenWAM checkpoints trained with history (memory) frames.

XPolicyLab's OpenWAM adapter (``policy/OpenWAM/model.py``, used at fa431ec) conditions every
action chunk on the current frame only, and the OpenWAM copy it vendors has no history code. A
checkpoint trained with ``dataloader.history_num_frames`` (for R4: the episode's first frame plus
the frames 6, 4 and 2 s back) would be scored without the memory it was trained to use.

This adapter subclasses that one and changes two things:

* History. Each parallel env keeps an ``openwam.deploy.policy.FrameHistory`` of composed
  canvases. XPolicyLab's batch loop calls ``update_obs_batch`` once per environment step for the
  envs still running (once before each chunk and after every non-final action), so one push per
  call samples past frames exactly as training and the WebSocket server do. ``reset``, sent before
  each batch of episodes, clears every env. One env step is 0.04 s (collect_freq 25), the rate of
  the training data, so ``history_stride`` 50 is 2 s here too.
* Generation. This repository's engine has no ``generate_batch``, so a chunk is generated env by
  env through ``engine.generate``, the path the WebSocket server serves. The stock adapter's
  batch bookkeeping, binary-dim projection, replan cut and pose conversion run unchanged.

Camera mapping, pose frames, the prompt template and the forced inference settings (dit_cache,
compile and video decoding off, sync) are the stock adapter's. ``openwam_root`` must be a
FastestWAM checkout. A checkpoint without history runs here exactly as through the stock
adapter, one env at a time, which is how the two code paths are compared.

Installed by ``stage_xpolicylab.sh`` as ``policy/OpenWAM/model.py`` of a container-local copy of
XPolicyLab, next to the stock file renamed ``openwam_stock_model.py``.
"""

from __future__ import annotations

import numpy as np

from .openwam_stock_model import Model as OpenWAMModel


class GenerateEachEngine:
    """``generate_batch`` over an engine that generates one sample per call.

    Sample ``i`` gets exactly what ``engine.generate(conditions_list[i])`` returns: same
    conditions, same seed, nothing shared between samples.
    """

    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def generate_batch(self, conditions_list: list) -> dict:
        if not conditions_list:
            raise ValueError("generate_batch requires a non-empty conditions list.")
        actions = [np.asarray(self.engine.generate(conditions)["actions"]) for conditions in conditions_list]
        return {"video": None, "actions": np.stack(actions)}


class Model(OpenWAMModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        self._histories: dict = {}
        self._new_history = None
        if self._engine is None:
            # allow_dummy_policy: protocol debugging only, nothing to condition.
            return
        try:
            from openwam.deploy.policy import FrameHistory
        except ImportError as exc:
            raise RuntimeError(
                f"openwam_root {self.openwam_root} has no openwam.deploy.policy.FrameHistory; "
                "point it at a FastestWAM checkout."
            ) from exc
        cfg = self._wam_policy.cfg
        spec = FrameHistory.from_cfg(cfg)
        if spec.num_frames:
            self._new_history = lambda: FrameHistory.from_cfg(cfg)
        self._engine = GenerateEachEngine(self._engine)
        print(
            f"[OpenWAMHistory] openwam_root={self.openwam_root} | history_num_frames={spec.num_frames} "
            f"history_stride={spec.stride} history_include_first_frame={spec.include_first_frame} | "
            "one engine.generate call per env"
        )

    def update_obs_batch(self, obs_list):
        super().update_obs_batch(obs_list)
        if self._new_history is None:
            return
        for env_idx in self._order:
            history = self._histories.get(env_idx)
            if history is None:
                history = self._histories[env_idx] = self._new_history()
            payload = self._batch[env_idx]
            payload["history_images"] = history.push(self._preprocessor.preprocess(dict(payload))["image"])

    def _conditions(self, payload: dict) -> dict:
        payload = dict(payload)
        history = payload.pop("history_images", None)
        conditions = super()._conditions(payload)
        if history is not None:
            conditions["history_images"] = history
        return conditions

    def reset(self):
        super().reset()
        self._histories = {}
