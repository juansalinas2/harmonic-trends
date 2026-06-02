"""Upload runtime data artifacts for the Harmonic Trends app to a HF Dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "processed" / "harmonic_trends_app.duckdb"
DEFAULT_DB_FILENAME = "harmonic_trends_app.duckdb"
DEFAULT_FILES = [
    (
        ROOT / "data" / "processed" / "spotify_metadata_cache.sqlite",
        "spotify_metadata_cache.sqlite",
    ),
    (ROOT / "data" / "processed" / "songs_master.parquet", "songs_master.parquet"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload app data to a private HF Dataset.")
    parser.add_argument(
        "--repo-id",
        default="juansalinas2/harmonic-trends-data",
        help="Hugging Face Dataset repo id.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Create the Dataset as public. Default is private.",
    )
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="Upload only the app DuckDB artifact.",
    )
    parser.add_argument(
        "--sidecars-only",
        action="store_true",
        help="Upload only spotify_metadata_cache.sqlite and songs_master.parquet.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB,
        help="DuckDB artifact to upload for the app.",
    )
    parser.add_argument(
        "--db-filename",
        default=DEFAULT_DB_FILENAME,
        help="Filename to use for the app DuckDB in the Dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )

    if args.db_only and args.sidecars_only:
        raise SystemExit("Choose either --db-only or --sidecars-only, not both.")
    if args.db_only:
        files = [(args.db_path, args.db_filename)]
    elif args.sidecars_only:
        files = DEFAULT_FILES
    else:
        files = [(args.db_path, args.db_filename), *DEFAULT_FILES]
    for local_path, path_in_repo in files:
        if not local_path.exists():
            print(f"Skipping missing file: {local_path}")
            continue
        size_gb = local_path.stat().st_size / 1e9
        print(f"Uploading {path_in_repo} ({size_gb:.2f} GB)")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            commit_message=f"Upload {path_in_repo}",
        )

    print(f"Dataset ready: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
