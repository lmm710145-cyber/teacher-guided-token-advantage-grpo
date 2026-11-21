"""CPU-only native environment reset/step checks; no model weights loaded."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.env_clients import Bridge, Client


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["alfworld", "webshop"], required=True)
    p.add_argument("--tasks")
    a = p.parse_args()
    source = (
        Path(a.tasks)
        if a.tasks
        else ROOT / f"data/processed/{a.benchmark}/rl_train.jsonl"
    )
    tasks = [json.loads(l) for l in source.read_text().splitlines()]
    bridge = Bridge(a.benchmark, sys.stderr)
    try:
        for slot, task in enumerate(tasks[:2]):
            client = Client(a.benchmark, bridge, slot, task)
            client.reset(task["environment_index"])
            assert client.observe().strip()
            if a.benchmark == "alfworld":
                action = client.inner.info["available_actions"][0]
            else:
                assert task["instruction_text"] in client.observe(), (
                    "Catalog/session mismatch"
                )
                action = "search[shirt]"
            reply = client.step(f"Thought: Inspect the environment.\nAction: {action}")
            assert reply.state.strip() and not client.last["parse_error"]
        print(
            json.dumps(
                dict(benchmark=a.benchmark, passed=True, checked=min(2, len(tasks)))
            )
        )
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
