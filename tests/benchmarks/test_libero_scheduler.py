import concurrent.futures
import json
import queue
import threading
from types import SimpleNamespace

import pytest

import benchmarks.libero.scheduler as scheduler
from benchmarks.libero.scheduler import (
    ProcessRegistry,
    QueuedTask,
    ReplicaSlot,
    TaskJob,
    TrialRun,
    _dynamic_worker,
    _render_device_for_slot,
    _run_output_dir,
    _unfinished_jobs,
    _wait_for_dynamic_queue,
)


def test_unfinished_jobs_preserves_only_missing_jobs(tmp_path):
    jobs = [TaskJob("libero_goal", task_id) for task_id in range(3)]
    trial_run = TrialRun(trial_start=0, num_trials=1)
    completed = jobs[1]
    attempt_dir = _run_output_dir(tmp_path, completed, trial_run) / "attempt"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "results.json").write_text(
        json.dumps(
            {
                "suite": completed.suite,
                "task_id": completed.task_id,
                "trial_start": 0,
                "trial_stop": 1,
                "trials": [{"trial": 0}],
            }
        ),
        encoding="utf-8",
    )

    assert _unfinished_jobs(tmp_path, jobs, trial_run) == [jobs[0], jobs[2]]


def test_unfinished_jobs_ignores_malformed_results(tmp_path):
    job = TaskJob("libero_object", 7)
    trial_run = TrialRun(trial_start=5, num_trials=2)
    attempt_dir = _run_output_dir(tmp_path, job, trial_run) / "attempt"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "results.json").write_text("{}", encoding="utf-8")

    assert _unfinished_jobs(tmp_path, [job], trial_run) == [job]


def test_dynamic_worker_requeues_failure_behind_pending_requests(monkeypatch, tmp_path):
    first = TaskJob("libero_object", 7)
    second = TaskJob("libero_10", 9)
    calls = []

    def fake_run_client(*args, **kwargs):
        calls.append((kwargs["job"], kwargs["attempt"], kwargs["slot"]))
        return not (kwargs["job"] == first and kwargs["attempt"] == 1)

    monkeypatch.setattr(scheduler, "_run_client", fake_run_client)
    args = SimpleNamespace(
        client_max_attempts=2,
        client_retry_delay=0,
        base_port=8920,
        client_start_stagger=0,
        worker_max_consecutive_failures=3,
        worker_recovery_delay=0,
    )
    slot = ReplicaSlot(gpu=0, gpu_slot=0, replica=0, port=8920)
    trial_run = TrialRun(trial_start=0, num_trials=1)
    work_queue = queue.Queue()
    work_queue.put(QueuedTask(first))
    work_queue.put(QueuedTask(second))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _dynamic_worker,
            args,
            slot=slot,
            work_queue=work_queue,
            trial_run=trial_run,
            output_dir=tmp_path,
            registry=ProcessRegistry(),
            stop_event=threading.Event(),
        )
        work_queue.join()
        work_queue.put(None)
        assert future.result(timeout=2) == []

    assert [(job, attempt) for job, attempt, _ in calls] == [
        (first, 1),
        (second, 1),
        (first, 2),
    ]


def test_dynamic_worker_recovers_after_consecutive_failures(monkeypatch, tmp_path, capsys):
    job = TaskJob("libero_spatial", 8)
    calls = []

    def fake_run_client(*args, **kwargs):
        calls.append(kwargs["attempt"])
        return kwargs["attempt"] == 4

    monkeypatch.setattr(scheduler, "_run_client", fake_run_client)
    args = SimpleNamespace(
        client_max_attempts=4,
        client_retry_delay=0,
        base_port=8920,
        client_start_stagger=0,
        worker_max_consecutive_failures=3,
        worker_recovery_delay=0,
    )
    work_queue = queue.Queue()
    work_queue.put(QueuedTask(job))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _dynamic_worker,
            args,
            slot=ReplicaSlot(gpu=4, gpu_slot=4, replica=0, port=8928),
            work_queue=work_queue,
            trial_run=TrialRun(trial_start=0, num_trials=1),
            output_dir=tmp_path,
            registry=ProcessRegistry(),
            stop_event=threading.Event(),
        )
        work_queue.join()
        work_queue.put(None)
        assert future.result(timeout=2) == []

    assert calls == [1, 2, 3, 4]
    output = capsys.readouterr().out
    assert "[cooldown] gpu=4 replica=0" in output
    assert "[recovered] gpu=4 replica=0" in output


def test_render_devices_can_be_decoupled_from_policy_gpus():
    args = SimpleNamespace(gpus=list(range(8)), render_gpus=[0, 1, 2, 3])

    assert _render_device_for_slot(args, ReplicaSlot(0, 0, 0, 8920)) == 0
    assert _render_device_for_slot(args, ReplicaSlot(3, 3, 1, 8927)) == 3
    assert _render_device_for_slot(args, ReplicaSlot(4, 4, 0, 8928)) == 0
    assert _render_device_for_slot(args, ReplicaSlot(7, 7, 1, 8935)) == 3


def test_wait_for_dynamic_queue_fails_if_all_workers_retire_with_pending_work():
    work_queue = queue.Queue()
    work_queue.put(QueuedTask(TaskJob("libero_goal", 4)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(lambda: [])]
        with pytest.raises(RuntimeError, match="1 queued request.*still pending"):
            _wait_for_dynamic_queue(work_queue, futures, poll_interval=0.01)

    # The failure path drains abandoned queue entries and leaves Queue.join()
    # usable, rather than leaking an unfinished-task count.
    work_queue.join()


def test_diagnostic_protocol_is_explicit_and_preserves_physics(tmp_path):
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(Path('benchmarks/libero/policy_config.yml').read_text())
    path = tmp_path / 'policy.yml'
    path.write_text(yaml.safe_dump(cfg))
    args = SimpleNamespace(policy_config=path, seed=43, smoke=False,
                           num_trials=10, trial_start=0)
    with pytest.raises(ValueError, match='pinned protocol violation'):
        scheduler._validate_protocol(args)
    args.diagnostic_protocol = True
    scheduler._validate_protocol(args)
    cfg['settle_steps'] = 0
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='settling'):
        scheduler._validate_protocol(args)


def test_diagnostic_results_cannot_resume_as_canonical(tmp_path):
    args = scheduler._build_parser().parse_args([])
    args.policy_config = __import__('pathlib').Path('benchmarks/libero/policy_config.yml')
    kwargs = dict(jobs=[TaskJob('libero_goal', 0)],
                  trial_run=TrialRun(0, 50), mujoco_version='3.3.2')
    canonical = scheduler._resume_signature(args, **kwargs)
    args.diagnostic_protocol = True
    diagnostic = scheduler._resume_signature(args, **kwargs)
    assert diagnostic['protocol_version'] == canonical['protocol_version'] + '-diagnostic'
    assert diagnostic != canonical


def test_sigterm_cleans_up_managed_processes(monkeypatch, tmp_path):
    import os
    import signal
    import subprocess
    import sys

    children = []
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(scheduler, '_preflight', lambda args: None)
    monkeypatch.setattr(scheduler, '_mujoco_version', lambda _: scheduler.REQUIRED_MUJOCO_VERSION)
    monkeypatch.setattr(scheduler, '_enumerate_task_counts', lambda suites, **kw: {s: 1 for s in suites})
    monkeypatch.setattr(scheduler, '_port_is_open', lambda *args: False)

    def start_servers(args, *, slots, output_dir, registry):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        children.append(child)
        registry.add(child)
        return []

    def worker(*args, stop_event, **kwargs):
        stop_event.wait(5)
        return []

    monkeypatch.setattr(scheduler, '_start_servers', start_servers)
    monkeypatch.setattr(scheduler, '_dynamic_worker', worker)
    monkeypatch.setattr(scheduler, '_wait_for_dynamic_queue', lambda *args: os.kill(os.getpid(), signal.SIGTERM))
    try:
        result = scheduler.main(['--output-dir', str(tmp_path/'eval'), '--gpus', '0',
                                 '--replicas-per-gpu', '1', '--suites', 'spatial', '--smoke'])
        assert result == 130
        assert children and children[0].poll() is not None
        assert signal.getsignal(signal.SIGTERM) == previous
        summary = json.loads((tmp_path/'eval/summary.json').read_text())
        assert summary['overall']['trials'] == 0
        assert summary['overall']['tasks_complete'] == 0
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        signal.signal(signal.SIGTERM, previous)
