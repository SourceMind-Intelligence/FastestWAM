"""One-time same-GPU A/C server-only action-latency probe on h100-box.

This uses fixed synthetic two-view input and resets before each measured action.
It does not measure simulator or transport latency, nor deployed p95 latency.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path('/root/tengfei/FastestWAM')
OUT = Path('/root/evan/Fastest-WAM-evan/outputs/libero-object03-video-20260927')
PYTHON = Path('/root/tengfei/envs/openwam/bin/python')
CKPT_ROOT = REPO / 'outputs/experiments/libero_sf_suite_2026-09-25_00-52-16'
PORT = 8981

sys.path.insert(0, str(REPO))
from benchmarks.utils.client import (  # noqa: E402
    build_payload,
    encode_numpy_b64,
    resize_for_lshape_slot,
)
from benchmarks.utils.transport import WSPolicyClient  # noqa: E402


def fixed_payload() -> dict:
    rng = np.random.default_rng(42)
    head = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    return build_payload(
        head=encode_numpy_b64(resize_for_lshape_slot(head, 'head_camera')),
        left_wrist=encode_numpy_b64(resize_for_lshape_slot(wrist, 'left_wrist_camera')),
        prompt='pick up the object and place it in the target',
        state=[0.0] * 10,
    )


def run_one(label: str, ckpt_dir: Path, denoise_steps: int, payload: dict) -> dict:
    log_path = OUT / f'{label}_server_latency.log'
    command = [
        str(PYTHON), str(REPO / 'scripts/deploy.py'),
        '--ckpt-dir', str(ckpt_dir),
        '--ckpt-name', 'checkpoint_step_42730.safetensors',
        '--device', 'cuda:0', '--host', '127.0.0.1', '--port', str(PORT),
        '--denoise-steps', str(denoise_steps), '--denoise-mode', 'sync',
        '--inference-mode', 'sync', '--inference-horizon', '10',
        '--compile-enabled', 'false',
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', PYTHONUNBUFFERED='1')
    with log_path.open('w') as log:
        server = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        client = WSPolicyClient(f'ws://127.0.0.1:{PORT}', timeout=300)
        try:
            for _ in range(240):
                if server.poll() is not None:
                    raise RuntimeError(f'{label} server exited early; see {log_path}')
                try:
                    if client.ping().get('type') == 'pong':
                        break
                except Exception:
                    client.close()
                time.sleep(5)
            else:
                raise TimeoutError(f'{label} server did not become ready')

            server_ms: list[float] = []
            round_trip_ms: list[float] = []
            for i in range(35):
                assert client.reset().get('type') == 'reset_ack'
                start = time.perf_counter()
                answer = client.predict_once(payload)
                elapsed = (time.perf_counter() - start) * 1000
                assert answer.get('type') == 'action', answer
                assert len(answer['action']) == 10, answer
                if i >= 5:
                    server_ms.append(float(answer['latency_ms']))
                    round_trip_ms.append(elapsed)
            return {
                'checkpoint': str(ckpt_dir / 'checkpoint_step_42730.safetensors'),
                'denoise_steps_flag': denoise_steps,
                'gpu': 3,
                'warmup_requests': 5,
                'measured_requests': 30,
                'server_latency_ms': server_ms,
                'round_trip_ms': round_trip_ms,
                'server_median_ms': statistics.median(server_ms),
                'server_p95_ms': sorted(server_ms)[28],
                'round_trip_median_ms': statistics.median(round_trip_ms),
                'round_trip_p95_ms': sorted(round_trip_ms)[28],
            }
        finally:
            client.close()
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = fixed_payload()
    results = {
        'description': 'Server-only same-GPU fixed synthetic two-view first-action latency',
        'caveat': 'Not simulator throughput or deployed p95. A FM-2 and C MIP execute different action paths.',
        'input': 'Fixed seeded RGB head/wrist PNG, 10-zero state, fixed prompt; reset before each action',
        'runs': {},
    }
    candidates = (
        ('A', CKPT_ROOT / 'A_fm/2026-09-25_00-54-12', 2),
        ('C', CKPT_ROOT / 'C_mip_mixed/2026-09-25_00-54-11', 10),
        ('A_repeat', CKPT_ROOT / 'A_fm/2026-09-25_00-54-12', 2),
    )
    for label, ckpt_dir, denoise in candidates:
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Timing {label}', flush=True)
        results['runs'][label] = run_one(label, ckpt_dir, denoise, payload)
        (OUT / 'paired_server_latency.json').write_text(json.dumps(results, indent=2) + '\n')
    print({key: value['server_median_ms'] for key, value in results['runs'].items()}, flush=True)


if __name__ == '__main__':
    main()
