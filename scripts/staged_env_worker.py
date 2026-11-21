"""Native WebShop worker; JSON lines on stdout, diagnostics on stderr."""

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Worker:
    def __init__(self):
        os.chdir(ROOT / "benchmarks/AgentGym/agentenv-webshop/webshop")
        from scripts.runtime_factories import make_webshop

        self.base = make_webshop()
        self.envs = {}
        catalog = json.loads((ROOT / "data/assets/webshop_catalog.json").read_text())
        assert len(self.base.server.goals) == len(catalog)
        assert all(
            g["instruction_text"] == r["instruction_text"]
            for g, r in zip(self.base.server.goals, catalog)
        )

    def call(self, r):
        op = r["op"]
        slot = r.get("slot", 0)
        if op == "close":
            for env in self.envs.values():
                env.close()
            self.envs.clear()
            self.base.close()
            return dict(closed=True)
        if op == "reset":
            if slot not in self.envs:
                from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

                self.envs[slot] = WebAgentTextEnv(
                    observation_mode="text",
                    server=self.base.server,
                    num_products=1000,
                    human_goals=False,
                )
            ob, _ = self.envs[slot].reset(session=r["task"]["environment_index"])
            return dict(
                observation=ob, score=0.0, reward=0.0, done=False, success=False
            )
        if op == "step":
            ob, score, done, _ = self.envs[slot].step(r["action"])
            return dict(
                observation=ob,
                score=float(score),
                reward=float(score),
                done=bool(done),
                success=bool(score >= 1),
            )
        raise ValueError(op)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["webshop"], required=True)
    p.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        worker = Worker()
    for line in sys.stdin:
        try:
            r = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = worker.call(r)
        except Exception:
            result = dict(error=traceback.format_exc())
        print(json.dumps(result), flush=True)
        if r.get("op") == "close":
            break


if __name__ == "__main__":
    main()
