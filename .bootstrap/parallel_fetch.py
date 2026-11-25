#!/usr/bin/env python3
"""Parallel, resumable chunk downloader for slow links. Stdlib only."""
import json
import os
import sys
import threading
import urllib.request

ART = "/workspace/.bootstrap/artifacts"
CHUNKS = 8
RETRIES = 300
os.makedirs(ART, exist_ok=True)

WHEELS = [
    ("numpy", "1.26.4"),
    ("scipy", "1.11.4"),
    ("scikit-learn", "1.3.2"),
    ("joblib", "1.3.2"),
    ("threadpoolctl", "3.2.0"),
    ("pytest", "8.3.4"),
    ("iniconfig", "2.0.0"),
    ("pluggy", "1.5.0"),
]
EXTRA = [
    (
        "libpython3.11-dev_3.11.2-6+deb12u8_arm64.deb",
        "http://deb.debian.org/debian/pool/main/p/python3.11/"
        "libpython3.11-dev_3.11.2-6+deb12u8_arm64.deb",
    ),
]


def resolve(pkg, version):
    # Version-pinned endpoint: a few KB instead of the multi-MB full JSON.
    with urllib.request.urlopen(
        f"https://pypi.org/pypi/{pkg}/{version}/json", timeout=60
    ) as r:
        meta = json.load(r)
    files = meta["urls"]
    whl = [f for f in files if f["packagetype"] == "bdist_wheel"]
    pick = [f for f in whl if "cp311" in f["filename"] and "aarch64" in f["filename"]]
    if not pick:
        pick = [f for f in whl if "py3-none-any" in f["filename"]]
    return pick[0]["url"], pick[0]["size"]


def fetch_range(url, start, end, path, errors):
    for _ in range(RETRIES):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= end - start + 1:
            return
        req = urllib.request.Request(url)
        req.add_header("Range", f"bytes={start + have}-{end}")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(
                path, "ab"
            ) as fh:
                while True:
                    buf = resp.read(1 << 15)
                    if not buf:
                        break
                    fh.write(buf)
            if os.path.getsize(path) >= end - start + 1:
                return
        except Exception as exc:
            errors.append(str(exc))


def parallel_download(url, size, dest):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print("skip", os.path.basename(dest), flush=True)
        return
    part_dir = dest + ".parts"
    os.makedirs(part_dir, exist_ok=True)
    step = size // CHUNKS
    bounds = []
    for i in range(CHUNKS):
        start = i * step
        end = size - 1 if i == CHUNKS - 1 else (start + step - 1)
        bounds.append((start, end, os.path.join(part_dir, f"part{i}")))

    threads = []
    errors = []
    for start, end, path in bounds:
        t = threading.Thread(
            target=fetch_range, args=(url, start, end, path, errors)
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    with open(dest, "wb") as out:
        for _, _, path in bounds:
            with open(path, "rb") as fh:
                while True:
                    buf = fh.read(1 << 18)
                    if not buf:
                        break
                    out.write(buf)
    if os.path.getsize(dest) != size:
        raise RuntimeError(f"bad size for {dest}")
    for _, _, path in bounds:
        os.remove(path)
    os.rmdir(part_dir)
    print("OK", os.path.basename(dest), flush=True)


def main():
    jobs = []
    for pkg, ver in WHEELS:
        url, size = resolve(pkg, ver)
        jobs.append((url, size, os.path.join(ART, url.rsplit("/", 1)[-1])))
    for name, url in EXTRA:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            size = int(resp.headers["Content-Length"])
        jobs.append((url, size, os.path.join(ART, name)))

    # Download files sequentially, but chunk each across parallel connections.
    for url, size, dest in jobs:
        parallel_download(url, size, dest)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
