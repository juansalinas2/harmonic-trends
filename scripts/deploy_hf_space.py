"""Deploy the static Harmonic Trends Space to Hugging Face.

Usage:
    HF_TOKEN=... python3 scripts/deploy_hf_space.py --repo-id USER/harmonic-trends

The repo-id should not include the "spaces/" prefix. The script creates the
Space if needed, then uploads the contents of spaces/harmonic-trends.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "spaces" / "harmonic-trends"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the Harmonic Trends static Space.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Space repo id, for example 'username/harmonic-trends'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private if it does not already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required. Create a token at https://huggingface.co/settings/tokens "
            "and run: HF_TOKEN=... python3 scripts/deploy_hf_space.py --repo-id USER/harmonic-trends"
        )

    if not SPACE_DIR.exists():
        raise SystemExit(f"Space directory not found: {SPACE_DIR}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy Harmonic Trends static Space",
    )

    print(f"Deployed: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
