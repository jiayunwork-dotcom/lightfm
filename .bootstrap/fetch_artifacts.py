#!/usr/bin/env python3
"""Resumable, retrying downloader for the wheels and dev headers needed to
build and test LightFM on this offline-ish arm64 box. Stdlib only."""
import json
import os
import sys
import time
import urllib.request

DEST = "/workspace/.bootstrap/artifacts"
os.makedirs(DEST, exist_ok=True)

WHEELS = [
    ("numpy", "1.26.4"),
    ("scipy", "1.11.4"),
    ("scikit-learn", "1.3.2"),
    ("joblib", None),
    ("threadpoolctl", None),
    ("pytest", "8.3.4"),
    ("iniconfig", None),
    ("pluggy", None),
]

EXTRA_URLS = [
    (
        "libpython3.11-dev_3.11.2-6+deb12u8_arm64.deb",
        "http://deb.debian.org/debian/pool/main/p/python3.11/"
        "libpython3.11-dev_3.11.2-6+deb12u8_arm64.deb",
    ),
]


def pick_url(pkg, version):
    api = f"https://pypi.org/pypi/{pkg}/json"
    with urllib.request.urlopen(api, timeout=60) as resp:
        meta = json.load(resp)
    if version is None:
        version = meta["info"]["version"]
    files = meta["releases"][version]
    whls = [f for f in files if f["packagetype"] == "bdist_wheel"]
    preferred = [
        f
        for f in whls
        if "cp311" in f["filename"] and "aarch64" in f["filename"]
    ]
    if preferred:
        chosen = preferred[0]
    else:
        pure = [f for f in whls if "py3-none-any" in f["filename"]]
        chosen = pure[0]
    return chosen["url"], chosen["size"]


def download(url, dest, expected=None, attempts=200):
    tmp = dest + ".part"
    for attempt in range(attempts):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if expected and have >= expected:
            break
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(
                tmp, "ab"
            ) as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception as exc:  # network hiccups are expected here
            print(
                f"[{os.path.basename(dest)}] attempt {attempt + 1} stalled: {exc}",
                flush=True,
            )
            time.sleep(2)
            continue
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expected and have != expected:
        raise RuntimeError(
            f"incomplete download {dest}: {have}/{expected} bytes"
        )
    os.rename(tmp, dest)
    print(f"OK {os.path.basename(dest)} ({have} bytes)", flush=True)


def main():
    for pkg, version in WHEELS:
        url, size = pick_url(pkg, version)
        name = url.rsplit("/", 1)[-1]
        download(url, os.path.join(DEST, name), size)
    for name, url in EXTRA_URLS:
        download(url, os.path.join(DEST, name))
    print("ALL DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
