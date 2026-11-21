"""Greedy, sampled pass@1 or success-stopped pass@5 on an explicitly supplied split."""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.config import load, path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tasks", help="Override dev file, e.g. rl_test.jsonl")
    p.add_argument("--mode", choices=["greedy", "pass1", "pass5"], default="greedy")
    p.add_argument("--limit", type=int)
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    c = load(a.config, a.set)
    if a.tasks:
        c["dev_data"] = path(a.tasks)
    os.environ["AGENT_BENCH_PYTHON"] = c["bench_python"]
    from scripts.sampler import Sampler

    out = Path(path(a.output))
    out.mkdir(parents=True, exist_ok=True)
    # Do not silently reuse an evaluation from a different adapter/configuration.
    request = dict(
        adapter=path(a.adapter), version=1, step=0, mode=a.mode, output=str(out)
    )
    if a.limit:
        request["limit"] = a.limit
    manifest = dict(config=c, request=request)
    if (out / "request.json").exists():
        assert json.loads((out / "request.json").read_text()) == manifest
        if (out / "summary.json").exists():
            print((out / "summary.json").read_text())
            return
    (out / "request.json").write_text(json.dumps(manifest, indent=2))
    sampler = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            sampler = Sampler(c, out)
            result = sampler.evaluate(request)
        print(json.dumps(result, indent=2))
    finally:
        if sampler:
            sampler.close()


if __name__ == "__main__":
    main()
