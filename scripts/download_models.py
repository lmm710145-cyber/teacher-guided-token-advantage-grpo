"""Download unquantized Qwen3 bases; record exact resolved Hugging Face revisions."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="models")
    p.add_argument("--models", nargs="+", default=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
    p.add_argument(
        "--revision",
        default="main",
        help="Use a commit hash to reproduce an earlier download",
    )
    a = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    for model in a.models:
        revision = HfApi().model_info(model, revision=a.revision).sha
        dest = Path(a.output) / model.split("/")[-1]
        snapshot_download(
            model,
            revision=revision,
            local_dir=dest,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"],
        )
        (dest / "download_manifest.json").write_text(
            json.dumps(dict(repo_id=model, revision=revision), indent=2)
        )


if __name__ == "__main__":
    main()
