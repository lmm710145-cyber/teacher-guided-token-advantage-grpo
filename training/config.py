"""JSON configuration shared by launchers; all relative paths use repository root."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("grpo", "opd", "hybrid", "tokenadv")


def path(value):
    return str((ROOT / Path(value).expanduser()).resolve())


def load(filename, overrides=()):
    c = json.loads(Path(filename).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in c:
            raise ValueError(f"Unknown configuration key: {key}")
        try:
            c[key] = json.loads(value)
        except json.JSONDecodeError:
            c[key] = value
    if c["benchmark"] not in ("alfworld", "webshop"):
        raise ValueError("Only ALFWorld and WebShop are supported")
    for key in (
        "student_model",
        "teacher_model",
        "parent_checkpoint",
        "teacher_checkpoint",
        "train_data",
        "dev_data",
        "sft_data",
        "inference_python",
        "bench_python",
        "train_python",
    ):
        if c.get(key):
            c[key] = path(c[key])
    if c.get("method") in METHODS:
        validate(c)
    return c


def validate(c):
    if c["sampling"] != dict(temperature=1.0, top_p=1.0, top_k=-1):
        raise ValueError(
            "On-policy estimators require the untruncated T=1 distribution"
        )
    if c["enable_prefix_caching"]:
        raise ValueError("Prefix caching is disabled for this pinned vLLM/LoRA runtime")
    assert c["cutoff_length"] == 4096 and c["group_size"] == 4
    assert c["lora_r"] == 16 and c["lora_alpha"] == 32
    assert c["max_concurrent_trajectories"] >= 4
    assert c["batch_size"] > 0 and c["batch_size"] % 4 == 0
    assert c["training_micro_batch"] > 0 and c["scoring_micro_batch"] > 0
    assert c["max_steps"] > 0 and c["learning_rate"] > 0
    assert c["optimizer_steps_per_batch"] == 1
    assert c["kl_beta"] >= 0 and c["opd_weight"] >= 0 and c["token_lambda"] >= 0
    assert 1 <= c["teacher_top_k"] <= 151936
    assert c["teacher_temperature"] == c["student_baseline_temperature"] == 1.0
    assert c["max_steps"] in c["eval_steps"]


def fingerprint(c):
    """Bind preflight/resumption to code, data, configuration and input adapters."""
    files = [
        p
        for d in ("training", "scripts", "configs")
        for p in (ROOT / d).rglob("*")
        if p.suffix in (".py", ".json")
    ]
    assets = [Path(c[k]) for k in ("train_data", "dev_data", "parent_checkpoint")]
    if c["method"] != "grpo":
        assets.append(Path(c["teacher_checkpoint"]))
    files += [p / "adapter_model.safetensors" if p.is_dir() else p for p in assets]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return hashlib.sha256(
        json.dumps(dict(config=c, files=hashes), sort_keys=True).encode()
    ).hexdigest()
