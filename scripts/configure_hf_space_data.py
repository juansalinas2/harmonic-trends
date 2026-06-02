"""Create the HF Dataset and configure the Space to read it."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure HF Dataset-backed app storage.")
    parser.add_argument("--space-id", default="juansalinas2/harmonic-trends-explorer")
    parser.add_argument("--dataset-id", default="juansalinas2/harmonic-trends-data")
    parser.add_argument("--db-filename", default="harmonic_trends_app.duckdb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.dataset_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.add_space_secret(
        repo_id=args.space_id,
        key="HF_TOKEN",
        value=token,
        description="Read token for the private harmonic runtime Dataset.",
    )
    variables = {
        "HARMONIC_DB_REPO_ID": args.dataset_id,
        "HARMONIC_DB_FILENAME": args.db_filename,
        "SPOTIFY_CACHE_REPO_ID": args.dataset_id,
        "SPOTIFY_CACHE_FILENAME": "spotify_metadata_cache.sqlite",
        "SONGS_MASTER_REPO_ID": args.dataset_id,
        "SONGS_MASTER_FILENAME": "songs_master.parquet",
    }
    for key, value in variables.items():
        api.add_space_variable(repo_id=args.space_id, key=key, value=value)

    print(f"Dataset configured: https://huggingface.co/datasets/{args.dataset_id}")
    print(f"Space configured: https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
