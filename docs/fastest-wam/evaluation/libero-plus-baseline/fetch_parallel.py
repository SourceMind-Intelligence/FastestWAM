#!/usr/bin/env python3
"""Resume a pinned Hugging Face LFS file using verified parallel byte ranges."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import quote

import requests

from fetch_artifacts import SOURCES


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path)
    parser.add_argument("--group", required=True, choices=SOURCES)
    parser.add_argument("--file", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--proxy-workers", type=int, default=0)
    parser.add_argument("--proxy", default="")
    args = parser.parse_args()
    if args.workers < 1 or args.proxy_workers < 0 or args.proxy_workers > args.workers:
        parser.error("invalid worker counts")

    repo_id, repo_type, metadata_file = SOURCES[args.group]
    info = json.loads((Path(__file__).resolve().parent / "metadata" / metadata_file).read_text())
    entry = next((x for x in info["siblings"] if x["rfilename"] == args.file), None)
    if entry is None or "lfs" not in entry:
        parser.error("file is absent from the pinned metadata or has no SHA-256")
    size, expected = entry["size"], entry["lfs"]["sha256"]
    prefix = "datasets/" if repo_type == "dataset" else ""
    suffix = f"/{prefix}{repo_id}/resolve/{info['sha']}/{quote(args.file)}"
    mirror_url = "https://hf-mirror.com" + suffix
    hub_url = "https://huggingface.co" + suffix
    output = args.dest / args.group / args.file
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".part")
    state_path = output.with_name(output.name + ".state.json")
    ok_path = output.with_name(output.name + ".sha256.ok")
    if output.exists() and output.stat().st_size == size and ok_path.exists():
        if ok_path.read_text().strip() == expected:
            print(f"already verified: {output}", flush=True)
            return

    chunk = (size + args.workers - 1) // args.workers
    completed = {str(i): 0 for i in range(args.workers)}
    if state_path.exists():
        completed.update(json.loads(state_path.read_text()))
    if not partial.exists():
        with partial.open("wb") as stream:
            stream.truncate(size)
    elif partial.stat().st_size != size:
        raise RuntimeError("partial file has an unexpected size")
    fd = os.open(partial, os.O_RDWR)
    lock = threading.Lock()

    def save_state():
        tmp = state_path.with_name(state_path.name + ".tmp")
        with lock:
            tmp.write_text(json.dumps(completed, sort_keys=True))
            os.replace(tmp, state_path)

    def worker(index):
        start = index * chunk
        end = min(size, (index + 1) * chunk) - 1
        if start > end:
            return
        proxy_worker = index >= args.workers - args.proxy_workers
        url = hub_url if proxy_worker else mirror_url
        session = requests.Session()
        session.trust_env = False
        if proxy_worker:
            session.proxies = {"http": args.proxy, "https": args.proxy}
        errors = 0
        while start + completed[str(index)] <= end:
            position = start + completed[str(index)]
            try:
                with session.get(
                    url,
                    headers={"Range": f"bytes={position}-{end}"},
                    stream=True,
                    timeout=(20, 90),
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise RuntimeError(f"expected HTTP 206, got {response.status_code}")
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {position}-"):
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        if position + len(block) - 1 > end:
                            raise RuntimeError("server sent bytes outside requested range")
                        os.pwrite(fd, block, position)
                        position += len(block)
                        with lock:
                            completed[str(index)] = position - start
                errors = 0
            except Exception as exc:
                errors += 1
                print(f"worker {index} retry {errors}: {exc}", flush=True)
                if errors >= 10:
                    raise
                time.sleep(min(5 * errors, 30))

    print(f"fetching {repo_id}/{args.file} ({size} bytes, {args.workers} workers)", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, index) for index in range(args.workers)]
        last = 0
        try:
            while not all(future.done() for future in futures):
                time.sleep(15)
                save_state()
                current = sum(completed.values())
                print(f"{current/1e9:.2f}/{size/1e9:.2f} GB, {(current-last)/15/1e6:.1f} MB/s", flush=True)
                last = current
            for future in futures:
                future.result()
        finally:
            save_state()
            os.close(fd)

    actual = sha256(partial)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: {actual} != {expected}")
    os.replace(partial, output)
    ok_path.write_text(actual + "\n")
    state_path.unlink(missing_ok=True)
    print(f"verified: {output}", flush=True)


if __name__ == "__main__":
    main()
