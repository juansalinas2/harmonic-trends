"""Deploy the DuckDB-backed harmonic explorer as a Hugging Face Docker Space.

Usage:
    HF_TOKEN=... python3 scripts/deploy_hf_app_space.py --repo-id USER/harmonic-trends-explorer

The repo-id should not include the "spaces/" prefix. This uploads only the
runtime app files; it intentionally does not upload the local data directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "Missing deployment dependency. Install it with: "
        "python3 -m pip install huggingface_hub"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy the harmonic explorer Docker Space.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face Space repo id, for example 'username/harmonic-trends-explorer'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private if it does not already exist.",
    )
    return parser.parse_args()


def build_space_bundle(bundle_dir: Path) -> None:
    required_files = [
        ("Dockerfile", "Dockerfile"),
        ("requirements-app.txt", "requirements-app.txt"),
        ("app.py", "app.py"),
        (".dockerignore", ".dockerignore"),
        ("README.hf.md", "README.md"),
        ("scripts/start_hf_space.py", "scripts/start_hf_space.py"),
    ]
    for source, destination in required_files:
        source_path = ROOT / source
        if not source_path.exists():
            raise SystemExit(f"Required file is missing: {source_path}")
        destination_path = bundle_dir / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)

    shutil.copytree(ROOT / "static", bundle_dir / "static")


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required. Create a token at https://huggingface.co/settings/tokens "
            "and run: HF_TOKEN=... python3 scripts/deploy_hf_app_space.py "
            "--repo-id USER/harmonic-trends-explorer"
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp)
        build_space_bundle(bundle_dir)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=str(bundle_dir),
            commit_message="Deploy harmonic explorer Docker Space",
        )

    print(f"Deployed app files: https://huggingface.co/spaces/{args.repo_id}")
    print("Database is not uploaded. Configure /data/harmonic_trends.duckdb or HARMONIC_DB_URL in the Space.")


if __name__ == "__main__":
    main()
