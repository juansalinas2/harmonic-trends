from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


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


def dataset_file_if_needed(
    *,
    repo_env: str,
    filename_env: str,
    path_env: str,
    default_path: str,
    required: bool,
) -> None:
    path = env_path(path_env, default_path)
    if path.exists():
        print(f"{path_env} already exists at {path}")
        return

    repo_id = os.environ.get(repo_env)
    filename = os.environ.get(filename_env)
    if not repo_id or not filename:
        if required:
            print(f"{path_env} is missing at {path}; set {repo_env} and {filename_env} to download it from a Dataset")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {filename} from dataset {repo_id} to {path}")
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(path.parent),
            token=os.environ.get("HF_TOKEN") or None,
        )
    )
    if downloaded_path.resolve() != path.resolve():
        tmp_path = path.with_suffix(path.suffix + ".part")
        shutil.move(str(downloaded_path), tmp_path)
        tmp_path.replace(path)
    print(f"Ready: {path}")


def main() -> None:
    dataset_file_if_needed(
        repo_env="HARMONIC_DB_REPO_ID",
        filename_env="HARMONIC_DB_FILENAME",
        path_env="HARMONIC_DB_PATH",
        default_path="/data/harmonic_trends.duckdb",
        required=True,
    )
    dataset_file_if_needed(
        repo_env="SPOTIFY_CACHE_REPO_ID",
        filename_env="SPOTIFY_CACHE_FILENAME",
        path_env="SPOTIFY_CACHE_PATH",
        default_path="/data/spotify_metadata_cache.sqlite",
        required=False,
    )
    dataset_file_if_needed(
        repo_env="SONGS_MASTER_REPO_ID",
        filename_env="SONGS_MASTER_FILENAME",
        path_env="SONGS_MASTER_PARQUET_PATH",
        default_path="/data/songs_master.parquet",
        required=False,
    )
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
