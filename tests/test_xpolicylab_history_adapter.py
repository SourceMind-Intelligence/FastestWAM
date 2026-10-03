"""The XPolicyLab history adapter (benchmarks/robodojo/xpolicylab/history_model.py) and its synthetic driver.

XPolicyLab is not installed here, so the stock adapter it subclasses is replaced by a stand-in with the
same protocol (stored payloads per env, conditions per payload, one generate_batch per chunk). The
frames each chunk is conditioned on are checked against the training reader's rule
(openwam/dataloader/robodojo.py: frames max(t - k * stride, 0), oldest slot pinned to frame 0).
"""

from __future__ import annotations

import importlib
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from openwam.deploy.obs_preprocess import ObsPreprocessor
from openwam.deploy.policy import FrameHistory, WAMPolicy

ROOT = Path(__file__).resolve().parents[1]
XPL = ROOT / "benchmarks" / "robodojo" / "xpolicylab"

STOCK_STANDIN = '''
import numpy as np


class Model:
    """Stand-in for XPolicyLab policy/OpenWAM/model.py: the protocol history_model.py builds on."""

    FAKES = {}

    def __init__(self, model_cfg):
        self.openwam_root = "fastestwam"
        self.allow_dummy_policy = bool(model_cfg.get("allow_dummy_policy"))
        self.replan_steps = model_cfg.get("replan_steps")
        self._batch, self._order = {}, []
        self._engine = self._wam_policy = self._preprocessor = None
        if not self.allow_dummy_policy:
            self._engine = self.FAKES["engine"]
            self._wam_policy = self.FAKES["policy"]
            self._preprocessor = self.FAKES["preprocessor"]

    def _encode_obs(self, obs):
        view = obs["image"]
        images = {"head_camera": view, "left_wrist_camera": view, "right_wrist_camera": view}
        return {"images": images, "prompt": obs["prompt"], "state": [0.0] * 20}

    def _conditions(self, payload):
        obs = self._preprocessor.preprocess(dict(payload))
        cond = {"observation": obs, "first_frame_image": [obs["image"]], "prompt": obs["prompt"]}
        if obs.get("state") is not None:
            cond["proprio"] = obs["state"]
        return cond

    def update_obs_batch(self, obs_list):
        self._batch, self._order = {}, []
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            self._batch[env_idx] = self._encode_obs(obs)
            self._order.append(env_idx)

    def get_action_batch(self, env_idx_list):
        conditions = [self._conditions(self._batch[e]) for e in env_idx_list]
        actions = np.asarray(self._engine.generate_batch(conditions)["actions"])
        actions = self._wam_policy._project_binary_dims(actions)
        n = actions.shape[1] if self.replan_steps is None else min(self.replan_steps, actions.shape[1])
        return [list(actions[b, :n]) for b in range(actions.shape[0])]

    def reset(self):
        self._batch, self._order = {}, []
'''


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    """history_model.py installed as <pkg>.model beside the stand-in, as stage_xpolicylab.sh installs it."""
    pkg = tmp_path / "xpl_policy_openwam"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "openwam_stock_model.py").write_text(STOCK_STANDIN)
    shutil.copy(XPL / "history_model.py", pkg / "model.py")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [n for n in sys.modules if n.startswith(pkg.name)]:
        monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module(f"{pkg.name}.model")
    yield module
    for name in [n for n in sys.modules if n.startswith(pkg.name)]:
        sys.modules.pop(name, None)


class _Engine:
    """Generates one chunk per call and records the conditions it was given."""

    def __init__(self, chunk=8):
        self.chunk = chunk
        self.calls = []

    def generate(self, conditions):
        self.calls.append(conditions)
        return {"video": None, "actions": np.full((self.chunk, 20), float(len(self.calls)))}


def _cfg(num=0, stride=4, first=True):
    dataloader = types.SimpleNamespace(history_num_frames=num, history_stride=stride, history_include_first_frame=first)
    return types.SimpleNamespace(dataloader=dataloader)


def _build(module, cfg, replan_steps=5, chunk=8):
    engine = _Engine(chunk)
    stock = sys.modules[module.__name__.rsplit(".", 1)[0] + ".openwam_stock_model"].Model
    stock.FAKES = {
        "engine": engine,
        "policy": WAMPolicy(engine, cfg),
        "preprocessor": ObsPreprocessor(
            multiview=True,
            camera_layout=["cam_head", "cam_left_wrist", "cam_right_wrist"],
            img_height=384,
            img_width=320,
        ),
    }
    return module.Model({"replan_steps": replan_steps}), engine


def _view(env, step):
    """A solid view that names its env and step in the composed canvas's top-left pixel."""
    return Image.new("RGB", (8, 6), (step % 256, step // 256, 40 * (env + 1)))


def _ident(canvas):
    r, g, b = canvas.getpixel((0, 0))
    return b // 40 - 1, r + 256 * g


def _episode(model, ends):
    """XPolicyLab's eval_one_episode_batch over envs that start together; env e stops after ends[e] steps."""
    steps = [0] * len(ends)

    def running():
        return [e for e, end in enumerate(ends) if steps[e] < end]

    def obs(envs):
        return [{"image": _view(e, steps[e]), "prompt": "stack the bowls", "env_idx": e} for e in envs]

    model.reset()
    while running():
        envs = running()
        model.update_obs_batch(obs(envs))
        actions = model.get_action_batch(envs)
        size = len(actions[0])
        for k in range(size):
            for e in envs:
                steps[e] += 1
            if not running() or k + 1 == size:
                break
            keep = [i for i, e in enumerate(envs) if e in running()]
            actions = [actions[i] for i in keep]
            envs = [envs[i] for i in keep]
            model.update_obs_batch(obs(envs))


def _training_rule(t, num, stride, first):
    ids = [max(t - k * stride, 0) for k in range(num, 0, -1)]
    if first:
        ids[0] = 0
    return ids


@pytest.mark.parametrize("first", [True, False])
def test_each_env_is_conditioned_on_the_frames_training_samples(adapter, first):
    model, engine = _build(adapter, _cfg(num=3, stride=4, first=first))
    _episode(model, ends=[23, 23, 9])
    starts = {}
    for cond in engine.calls:
        env, t = _ident(cond["first_frame_image"][0])
        history = [_ident(img) for img in cond["history_images"]]
        assert [e for e, _ in history] == [env] * 3
        assert [s for _, s in history] == _training_rule(t, 3, 4, first)
        starts.setdefault(env, []).append(t)
    # Chunks of 5 (replan_steps) from step 0; env 2 leaves after 9 steps, the others run on.
    assert starts == {0: [0, 5, 10, 15, 20], 1: [0, 5, 10, 15, 20], 2: [0, 5]}


def test_reset_starts_every_env_over(adapter):
    model, engine = _build(adapter, _cfg(num=2, stride=3, first=False))
    _episode(model, ends=[12, 12])
    first_episode = len(engine.calls)
    _episode(model, ends=[7, 7])
    for cond in engine.calls[first_episode:]:
        _, t = _ident(cond["first_frame_image"][0])
        assert [s for _, s in map(_ident, cond["history_images"])] == _training_rule(t, 2, 3, False)


def test_without_history_the_adapter_sends_none_and_composes_only_at_chunk_starts(adapter, monkeypatch):
    model, engine = _build(adapter, _cfg(num=0))
    composed = []
    real = model._preprocessor.preprocess
    monkeypatch.setattr(model._preprocessor, "preprocess", lambda obs: composed.append(1) or real(obs))
    _episode(model, ends=[12, 12])
    assert all("history_images" not in cond for cond in engine.calls)
    assert len(composed) == len(engine.calls) == 6


def test_generate_each_engine_keeps_the_order_and_passes_attributes_through(adapter):
    engine = _Engine(chunk=4)
    engine.cfg = "cfg"
    batch = adapter.GenerateEachEngine(engine)
    out = batch.generate_batch([{"i": 0}, {"i": 1}, {"i": 2}])
    assert out["actions"].shape == (3, 4, 20)
    assert [float(a[0, 0]) for a in out["actions"]] == [1.0, 2.0, 3.0]
    assert [c["i"] for c in engine.calls] == [0, 1, 2]
    assert batch.cfg == "cfg"
    with pytest.raises(ValueError):
        batch.generate_batch([])


def test_the_engine_is_wrapped_and_a_dummy_policy_is_left_alone(adapter):
    model, engine = _build(adapter, _cfg(num=3))
    assert isinstance(model._engine, adapter.GenerateEachEngine) and model._engine.engine is engine
    dummy = adapter.Model({"allow_dummy_policy": True})
    assert dummy._engine is None
    dummy.update_obs_batch([{"image": _view(0, 0), "prompt": "p", "env_idx": 0}])
    assert "history_images" not in dummy._batch[0]


def test_frame_history_from_cfg_defaults_to_off():
    history = FrameHistory.from_cfg(types.SimpleNamespace())
    assert (history.num_frames, history.stride, history.include_first_frame) == (0, 25, False)
    assert FrameHistory.from_cfg(_cfg(4, 50, True)).push("f0") == ["f0"] * 4


def test_binary_projection_works_on_a_stack_of_chunks():
    arch = types.SimpleNamespace(binary_command_dims=(1,))
    policy = WAMPolicy(types.SimpleNamespace(architecture=arch), types.SimpleNamespace())
    stack = np.zeros((2, 3, 4))
    stack[0, :, 1] = 0.7
    out = policy._project_binary_dims(stack)
    assert out.shape == (2, 3, 4)
    assert (out[0, :, 1] == 1.0).all() and (out[1, :, 1] == -1.0).all()
    assert (out[..., 0] == 0.0).all()


def _load_driver():
    spec = importlib.util.spec_from_file_location("adapter_actions", XPL / "adapter_actions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_driver_runs_an_episode_and_compares_runs(adapter, tmp_path, monkeypatch):
    driver = _load_driver()
    _build(adapter, _cfg(num=2, stride=3))

    def observation(env, step, prompt):
        return {"image": _view(env, step), "prompt": prompt, "env_idx": env}

    def as_array(chunk):
        return np.stack([np.full(16, float(a[0])) for a in chunk])

    monkeypatch.setattr(driver, "observation", observation)
    monkeypatch.setattr(driver, "as_array", as_array)
    outs = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.npz"
        driver.main(["--module", adapter.__name__, "--ckpt-dir", "ck", "--openwam-root", "src", "--envs", "2",
                     "--steps", "12", "--device", "cpu", "--out", str(out)])
        outs.append(str(out))
    data = np.load(outs[0])
    # The driver serves full chunks (replan_steps null, as deploy.yml): 8 actions, so chunks start at 0 and 8,
    # and every one of the 12 steps is observed once.
    assert data["actions"].shape == (2, 2, 8, 16)
    assert list(data["call_steps"]) == [0, 8]
    assert len(data["update_s"]) == 12
    report = driver.compare(*outs)
    assert report["chunks"] == 2 and report["envs"] == 2 and report["steps_per_chunk"] == 8
    # Each run's engine counts its own calls, so the second run's actions are offset by the first run's 6.
    assert not report["identical"]


def test_the_synthetic_views_move_and_differ_by_env():
    driver = _load_driver()
    a, b, c = driver.frame(0, 0, 0), driver.frame(0, 30, 0), driver.frame(1, 0, 0)
    assert a.shape == (480, 640, 3) and a.dtype == np.uint8
    assert not np.array_equal(a, b) and not np.array_equal(a, c)
    obs = driver.observation(3, 7, "Stack the bowls.")
    assert obs["env_idx"] == 3 and set(obs["vision"]) == set(driver.CAMERAS)
    assert np.isclose(np.linalg.norm(obs["state"]["left_ee_pose"][3:]), 1.0, atol=1e-6)
