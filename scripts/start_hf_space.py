from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path


CHUNK_SIZE = 8 * 1024 * 1024
LOG_EVERY_BYTES = 512 * 1024 * 1024


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def download_if_needed(url_env: str, path_env: str, default_path: str) -> None:
    url = os.environ.get(url_env)
    path = env_path(path_env, default_path)
    if path.exists():
        print(f"{path_env} already exists at {path}")
        return
    if not url:
        print(f"{path_env} is missing at {path}; set {url_env} to download it on startup")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {url_env} to {path}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "harmonic-trends-hf-space/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        next_log = 0
        with tmp_path.open("wb") as file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_log:
                    if total:
                        percent = downloaded / total * 100
                        print(f"{path.name}: {downloaded / 1e9:.2f} GB / {total / 1e9:.2f} GB ({percent:.1f}%)")
                    else:
                        print(f"{path.name}: {downloaded / 1e9:.2f} GB downloaded")
                    next_log = downloaded + LOG_EVERY_BYTES

    tmp_path.replace(path)
    print(f"Finished downloading {path}")


def main() -> None:
    download_if_needed(
        "HARMONIC_DB_URL",
        "HARMONIC_DB_PATH",
        "/data/harmonic_trends.duckdb",
    )
    download_if_needed(
        "SPOTIFY_CACHE_URL",
        "SPOTIFY_CACHE_PATH",
        "/data/spotify_metadata_cache.sqlite",
    )
    os.execv(sys.executable, [sys.executable, "app.py"])


if __name__ == "__main__":
    main()
