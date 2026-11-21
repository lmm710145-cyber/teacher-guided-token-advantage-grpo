"""Rebuild frozen splits from public demonstrations, import a project, or tokenize SFT."""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.protocol import format_valid, normalize_messages, tokenize_trajectory

DATASET = "AgentGym/AgentTraj-L"
REVISION = "cb518f5359f0b635a6b8c0c5cbf0469939a927c7"


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Refusing to overwrite different data: {path}")
    path.write_text(text)


def audit_sft(row, benchmark):
    ms = normalize_messages(row["messages"])
    assert len(ms) >= 4 and ms[-1]["role"] == "assistant", row["item_id"]
    for i, m in enumerate(ms):
        assert m["role"] == ("user" if i % 2 == 0 else "assistant"), (row["item_id"], i)
        assert type(row["messages"][i]["loss"]) is bool and m["loss"] == (
            i >= 3 and i % 2 == 1
        )
        assert m["content"].strip()
        if m["loss"]:
            assert format_valid(benchmark, m["content"]), (row["item_id"], i, "format")


def validate(directory, benchmark):
    root = Path(directory) / benchmark
    sets = {}
    for split in ("train", "dev", "test"):
        rows = read(root / f"rl_{split}.jsonl")
        assert rows
        sets[split] = {r["item_id"] for r in rows}
        assert len(sets[split]) == len(rows)
        for r in rows:
            assert isinstance(r["environment_index"], int)
            if benchmark == "alfworld":
                assert r["game_file"].endswith("game.tw-pddl")
    assert not (
        sets["train"] & sets["dev"]
        or sets["train"] & sets["test"]
        or sets["dev"] & sets["test"]
    )
    hashes = set()
    for split in ("train", "dev"):
        rows = read(root / f"sft_{split}.jsonl")
        assert len({r["item_id"] for r in rows}) == len(rows)
        for row in rows:
            assert row["item_id"] in sets[split]
            audit_sft(row, benchmark)
            h = hashlib.sha256(
                json.dumps(row["messages"], sort_keys=True).encode()
            ).hexdigest()
            assert h not in hashes, "Duplicate SFT conversation"
            hashes.add(h)
    print(
        json.dumps(
            dict(
                benchmark=benchmark,
                rl={k: len(v) for k, v in sets.items()},
                sft_total=len(hashes),
            )
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode", choices=["download", "import-project", "validate", "tokenize"]
    )
    p.add_argument("--source-project")
    p.add_argument(
        "--raw-dir", help="Use a local AgentTraj-L snapshot instead of downloading it"
    )
    p.add_argument("--directory", default=str(ROOT / "data/processed"))
    p.add_argument("--benchmark", choices=["alfworld", "webshop"])
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--model")
    a = p.parse_args()
    benchmarks = [a.benchmark] if a.benchmark else ["alfworld", "webshop"]
    if a.mode == "tokenize":
        assert a.benchmark and a.input and a.output and a.model
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            a.model, local_files_only=True, use_fast=True
        )
        rows = read(a.input)
        for r in rows:
            audit_sft(r, a.benchmark)
        # No silent truncation: overlong rows fail with their task ID.
        write(a.output, [tokenize_trajectory(tok, r, 4096) for r in rows])
        return
    if a.mode == "download":
        from huggingface_hub import hf_hub_download

        manifest = json.loads((ROOT / "configs/data_manifest.json").read_text())
        protocol = json.loads((ROOT / "configs/input_protocol.json").read_text())[
            "benchmarks"
        ]
        for b in benchmarks:
            src = (
                str(Path(a.raw_dir) / f"{b}_train.json")
                if a.raw_dir
                else hf_hub_download(
                    DATASET, f"{b}_train.json", repo_type="dataset", revision=REVISION
                )
            )
            raw = {r["item_id"]: r for r in json.loads(Path(src).read_text())}
            for split in ("train", "dev", "test"):
                rows = manifest[b]["rl_" + split]
                if b == "webshop":
                    catalog = json.loads(
                        (ROOT / "data/assets/webshop_catalog.json").read_text()
                    )
                    rows = [catalog[r["environment_index"]] for r in rows]
                    assert [r["item_id"] for r in rows] == [
                        r["item_id"] for r in manifest[b]["rl_" + split]
                    ]
                    assert all(
                        r["goal_family_key"] == m["goal_family_key"]
                        for r, m in zip(rows, manifest[b]["rl_" + split])
                    )
                write(Path(a.directory) / b / f"rl_{split}.jsonl", rows)
            for split in ("train", "dev"):
                rows = []
                for ref in manifest[b]["sft_" + split]:
                    ms = normalize_messages(raw[ref["source_id"]]["conversations"])
                    ms[0]["content"] = protocol[b]["prompt"]
                    ms[1]["content"] = protocol[b]["acknowledgement"]
                    h = hashlib.sha256(
                        json.dumps(ms, ensure_ascii=False, sort_keys=True).encode()
                    ).hexdigest()
                    assert h == ref["messages_sha256"], (
                        ref["item_id"],
                        "Public source differs from frozen manifest",
                    )
                    rows.append(dict(item_id=ref["item_id"], messages=ms))
                write(Path(a.directory) / b / f"sft_{split}.jsonl", rows)
    elif a.mode == "import-project":
        assert a.source_project
        source = Path(a.source_project).resolve()
        for b in benchmarks:
            for name in ("sft_train", "sft_dev", "rl_train", "rl_dev", "rl_test"):
                rows = read(source / f"data/processed/current/{b}/{name}.jsonl")
                if name.startswith("sft"):
                    rows = [
                        dict(item_id=r["item_id"], messages=r["messages"]) for r in rows
                    ]
                elif b == "alfworld":
                    for r in rows:
                        r["game_file"] = str(Path(r["game_file"]).relative_to(source))
                write(Path(a.directory) / b / f"{name}.jsonl", rows)
        (ROOT / "data/assets").mkdir(parents=True, exist_ok=True)
        for name in ("webshop_catalog.json", "webshop_product_prices.json"):
            shutil.copyfile(source / "reports" / name, ROOT / "data/assets" / name)
    for b in benchmarks:
        validate(a.directory, b)


if __name__ == "__main__":
    main()
