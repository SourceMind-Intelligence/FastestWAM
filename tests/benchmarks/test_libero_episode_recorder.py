from __future__ import annotations

import importlib.util
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from benchmarks.libero.episode_recorder import RECORD_DIR_ENV, RECORD_EVERY_ENV, EpisodeRecorder, upright_half

REPO_ROOT = Path(__file__).resolve().parents[2]


def _obs(step: int, cameras: bool = True) -> dict:
    obs = {"robot0_eef_pos": np.array([0.1 * step, 0.0, 0.5]), "robot0_gripper_qpos": np.array([0.04, -0.04])}
    if cameras:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[1, 1] = step  # lands at [-1, -1] once rotated and halved
        obs["agentview_image"] = frame
        obs["robot0_eye_in_hand_image"] = frame + 1
    return obs


def test_upright_half_rotates_then_halves():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    assert np.array_equal(upright_half(image), image[::-1, ::-1][::2, ::2])
    assert upright_half(image).shape == (2, 2, 3)


def test_recorder_is_off_without_the_env(monkeypatch):
    monkeypatch.delenv(RECORD_DIR_ENV, raising=False)
    assert EpisodeRecorder.from_env("libero_goal", 9, 3) is None


def test_recorder_keeps_every_nth_frame_plus_the_last_and_every_action(monkeypatch, tmp_path):
    monkeypatch.setenv(RECORD_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(RECORD_EVERY_ENV, "2")
    recorder = EpisodeRecorder.from_env("libero_10", 4, 7)
    recorder.observe(_obs(0))
    for step in range(1, 6):
        recorder.observe(_obs(step), action=np.full(7, step, dtype=np.float32))
    path = recorder.save(success=False, policy_steps=5, instruction="put both moka pots on the stove")

    assert path == tmp_path / "libero_10" / "task04" / "trial07.npz"
    with np.load(path) as data:
        assert data["frame_steps"].tolist() == [0, 2, 4, 5]
        assert data["agentview"].shape == (4, 4, 4, 3)
        assert data["agentview"][:, -1, -1, 0].tolist() == [0, 2, 4, 5]
        assert np.array_equal(data["wrist"], data["agentview"] + 1)
        assert data["actions"].shape == (5, 7)
        assert data["actions"][:, 0].tolist() == [1, 2, 3, 4, 5]
        assert data["eef_pos"].shape == (6, 3)
        assert data["gripper_qpos"].shape == (6, 2)
        assert not bool(data["success"]) and int(data["policy_steps"]) == 5
        assert str(data["instruction"]) == "put both moka pots on the stove"


def test_recorder_without_cameras_still_saves_the_states(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "trial00.npz", every=1)
    recorder.observe(_obs(0, cameras=False))
    recorder.observe(_obs(1, cameras=False), action=np.zeros(7))
    with np.load(recorder.save()) as data:
        assert "agentview" not in data.files
        assert data["eef_pos"].shape == (2, 3)


def test_recorder_rejects_a_zero_stride(tmp_path):
    with pytest.raises(ValueError, match=RECORD_EVERY_ENV):
        EpisodeRecorder(tmp_path / "trial00.npz", every=0)


def _load_summary():
    path = REPO_ROOT / "scripts" / "diagnostics" / "l2_diag_summary_20261003.py"
    spec = importlib.util.spec_from_file_location("l2_diag_summary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diag_summary_motion_stats_count_reversals_switches_and_stillness():
    summary = _load_summary()
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 0] = [0.5, -0.5, 0.5, 0.5]  # two reversals among three pairs
    actions[:, 6] = [-1, 1, 1, -1]  # two gripper switches
    eef_pos = np.zeros((5, 3))
    eef_pos[:, 0] = [0.0, 0.01, 0.01, 0.02, 0.02]  # still on two of four steps

    stats = summary.motion_stats(actions, eef_pos)
    assert stats["steps"] == 4
    assert stats["reversal_share"] == pytest.approx(2 / 3)
    assert stats["gripper_switches"] == 2
    assert stats["still_share"] == pytest.approx(0.5)
    assert stats["path_per_100_steps_m"] == pytest.approx(0.02 / 4 * 100)


def test_diag_summary_writes_a_valid_png(tmp_path):
    summary = _load_summary()
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    image[1, 2] = [255, 0, 0]
    path = tmp_path / "sheet.png"
    summary.write_png(path, image)

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (5, 3)
    idat = data.index(b"IDAT")
    (length,) = struct.unpack(">I", data[idat - 4 : idat])
    rows = zlib.decompress(data[idat + 4 : idat + 4 + length])
    assert rows[1 + 5 * 3 + 1 + 2 * 3 : 1 + 5 * 3 + 1 + 3 * 3] == bytes([255, 0, 0])
