#!/usr/bin/env python3
"""Record the two policy camera views while running the unchanged LIBERO-Plus client."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


client = Path(os.environ["WAM_PLUS_SINGLE_EVAL"]).resolve()
sys.path.insert(0, str(client.parent))
spec = importlib.util.spec_from_file_location("wam_plus_single_eval", client)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

output = Path(os.environ["WAM_VIDEO_OUT"]).resolve()
output.parent.mkdir(parents=True, exist_ok=True)


class RecordingPolicy(module.OpenWAMLiberoPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._writer = None
        self._frames = 0

    def reset(self):
        super().reset()
        if self._writer is not None:
            self._writer.close()
        self._writer = imageio.get_writer(str(output), fps=10, codec="libx264", quality=6)
        self._frames = 0

    def act(self, obs: dict, prompt: str) -> np.ndarray:
        # The canonical policy rotates both images by 180 degrees before inference.
        head = np.rot90(np.asarray(obs["agentview_image"]), 2)
        wrist = np.rot90(np.asarray(obs["robot0_eye_in_hand_image"]), 2)
        frame = np.ascontiguousarray(np.concatenate((head, wrist), axis=1)).astype(np.uint8)
        self._writer.append_data(frame)
        self._frames += 1
        return super().act(obs, prompt)

    def close(self):
        try:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            print(f"[video] frames={self._frames} path={output}", flush=True)
        finally:
            super().close()


module.OpenWAMLiberoPolicy = RecordingPolicy
raise SystemExit(module.main())
