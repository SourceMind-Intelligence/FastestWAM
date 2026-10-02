"""Resumable ranged download of the pinned public Fast-WAM LIBERO checkpoint."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import os
import time
import urllib.request

URL = "https://huggingface.co/yuanty/fastwam/resolve/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705/libero_uncond_2cam224.pt"
SIZE = 12041735140
SHA256 = "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579"
BASE = Path("/root/evan/Fastest-WAM-evan/assets/fastwam_official")
PARTS = BASE / "direct_parts"
OUTPUT = BASE / "libero_uncond_2cam224.direct.pt"
CHUNK = 32 * 1024 * 1024
WORKERS = 12

with urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=30) as response:
    actual_size = int(response.headers["Content-Length"])
if actual_size != SIZE:
    raise RuntimeError(f"Unexpected checkpoint size: {actual_size}")
PARTS.mkdir(parents=True, exist_ok=True)
count = (SIZE + CHUNK - 1) // CHUNK
print(f"size={SIZE} parts={count} workers={WORKERS}", flush=True)


def part_path(index):
    return PARTS / f"part_{index:04d}"


def fetch(index):
    start = index * CHUNK
    end = min(SIZE - 1, start + CHUNK - 1)
    expected = end - start + 1
    path = part_path(index)
    if path.exists() and path.stat().st_size == expected:
        return index, "cached"
    tmp = path.with_suffix(".tmp")
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                got_range = response.headers.get("Content-Range", "")
                if response.status != 206 or got_range != f"bytes {start}-{end}/{SIZE}":
                    raise RuntimeError(f"Unexpected HTTP {response.status}, range {got_range}")
                written = 0
                with tmp.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)
            if written != expected:
                raise RuntimeError(f"Part {index}: received {written}, expected {expected}")
            os.replace(tmp, path)
            return index, "downloaded"
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            print(f"retry part={index} attempt={attempt}: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))

start_time = time.monotonic()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = [pool.submit(fetch, i) for i in range(count)]
    done = 0
    for future in as_completed(futures):
        index, state = future.result()
        done += 1
        if done % 8 == 0 or done == count:
            print(f"parts_done={done}/{count} last={index}:{state} elapsed_s={int(time.monotonic()-start_time)}", flush=True)

if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
tmp_output = OUTPUT.with_suffix(".assembling")
digest = hashlib.sha256()
written_total = 0
with tmp_output.open("wb") as output:
    for index in range(count):
        with part_path(index).open("rb") as part:
            while True:
                block = part.read(4 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                written_total += len(block)
        if (index + 1) % 32 == 0:
            print(f"assembled_parts={index+1}/{count}", flush=True)
actual_hash = digest.hexdigest()
print(f"assembled_bytes={written_total} sha256={actual_hash}", flush=True)
if written_total != SIZE or actual_hash != SHA256:
    raise RuntimeError("Checkpoint size or SHA256 mismatch; preserving parts for diagnosis")
os.replace(tmp_output, OUTPUT)
print(f"verified_checkpoint={OUTPUT}", flush=True)
