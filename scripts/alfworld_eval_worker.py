"""JSON-lines bridge to the pinned ALF PDDL runtime (bench Python)."""

import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/alfworld"))
os.environ["ALFWORLD_DATA"] = str(ROOT / "data/environments/alfworld-0.3.3")
os.environ.setdefault("TMPDIR", str(ROOT / "cache/alfworld_online_tmp"))
Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
with contextlib.redirect_stdout(sys.stderr):
    from textworld import EnvInfos
    from textworld.envs.pddl.pddl import PddlEnv
    from alfworld.agents.environment.alfred_tw_env import AlfredInfos, AlfredDemangler

envs = {}
states = {}


def execute(r):
    slot = r.get("slot", 0)
    if r["op"] == "reset":
        if slot in envs:
            envs.pop(slot).close()
        env = AlfredInfos(
            AlfredDemangler(
                PddlEnv(EnvInfos(won=True, admissible_commands=True)), shuffle=False
            )
        )
        env.load(r["game_file"])
        state = env.reset()
        envs[slot] = env
        states[slot] = state
        return dict(
            observation="\n".join(state.feedback.split("\n\n")[1:]),
            available_actions=list(state.admissible_commands),
            done=False,
            won=False,
            reward=0.0,
        )
    if r["op"] == "step":
        allowed = list(states[slot].admissible_commands)
        state, _, done = envs[slot].step(r["action"])
        states[slot] = state
        observation = state.feedback
        if observation.startswith("You arrive at loc "):
            observation = observation[observation.find(". ") + 2 :]
        return dict(
            observation=observation,
            available_actions=list(state.admissible_commands),
            done=bool(done),
            won=bool(state.won),
            reward=float(state.won),
            admissible=r["action"] in allowed,
        )
    if r["op"] == "close":
        for e in envs.values():
            e.close()
        envs.clear()
        states.clear()
        return {"closed": True}
    raise ValueError(r["op"])


for line in sys.stdin:
    try:
        r = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(r)
    except Exception:
        result = {"error": traceback.format_exc()}
    print(json.dumps(result), flush=True)
    if r.get("op") == "close":
        break
