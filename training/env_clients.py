"""Native process clients; parser matches pinned AgentGym, not the stricter format metric."""

import ast, collections, contextlib, io, json, os, re, subprocess, threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def source_function(path, name, class_name):
    tree = ast.parse(path.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = {
        "re": re,
        "ActionWithTought": collections.namedtuple(
            "ActionWithTought", "thought action"
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
            str(path),
            "exec",
        ),
        ns,
    )
    return ns[name]


def PARSE(text):
    global _parser
    if "_parser" not in globals():
        path = ROOT / "benchmarks/AgentGym/agentenv/agentenv/controller/utils.py"
        _parser = source_function(path, "parse_react", "BaseAdapter")
    return _parser(text)


class AlfBridge:
    def __init__(self, log):
        self.log = log
        self.p = subprocess.Popen(
            [
                os.environ.get(
                    "AGENT_BENCH_PYTHON", str(ROOT / "envs/bench/bin/python")
                ),
                "-u",
                str(ROOT / "scripts/alfworld_eval_worker.py"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )

    def call(self, r):
        self.p.stdin.write(json.dumps(r) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"Environment worker exited: {self.p.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def close(self):
        if self.p.poll() is None:
            try:
                self.call({"op": "close"})
                self.p.wait(timeout=30)
            except (BrokenPipeError, RuntimeError, subprocess.TimeoutExpired):
                self.p.kill()
                self.p.wait()


class AlfClient:
    def __init__(self, bridge, slot, task):
        self.bridge = bridge
        self.slot = slot
        self.task = task

    def reset(self, index):
        self.info = self.bridge.call(
            dict(
                op="reset", slot=self.slot, game_file=str(ROOT / self.task["game_file"])
            )
        )

    def observe(self):
        return (
            self.info["observation"]
            + "\nAVAILABLE ACTIONS: "
            + ",".join(self.info["available_actions"])
        )

    def step(self, text):
        if text.endswith("</s>"):
            text = text[:-5]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                action = PARSE(text).action
        except Exception:
            self.last = dict(
                action=None, admissible=False, parse_error=True, won=False, done=False
            )
            return SimpleNamespace(
                state="Invalid Action.\n\n" + self.observe(), reward=0.0, done=False
            )
        self.info = self.bridge.call(dict(op="step", slot=self.slot, action=action))
        self.last = dict(
            action=action,
            admissible=self.info["admissible"],
            parse_error=False,
            won=self.info["won"],
            done=self.info["done"],
        )
        return SimpleNamespace(
            state=self.info["observation"],
            reward=self.info["reward"],
            done=self.info["done"],
        )


class StagedBridge:
    def __init__(self, benchmark, log):
        python = os.environ.get(
            "AGENT_BENCH_PYTHON", str(ROOT / "envs/bench/bin/python")
        )
        self.p = subprocess.Popen(
            [
                str(python),
                "-u",
                str(ROOT / "scripts/staged_env_worker.py"),
                "--benchmark",
                benchmark,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )
        self.lock = threading.Lock()

    def call(self, r):
        with self.lock:
            self.p.stdin.write(json.dumps(r) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"Environment worker exited: {self.p.poll()}")
            result = json.loads(line)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result

    def close(self):
        if self.p.poll() is None:
            try:
                self.call({"op": "close"})
                self.p.wait(timeout=30)
            except (BrokenPipeError, RuntimeError, subprocess.TimeoutExpired):
                self.p.kill()
                self.p.wait()


class StagedClient:
    def __init__(self, bridge, slot, task):
        self.bridge = bridge
        self.slot = slot
        self.task = task
        self.last = {}

    def reset(self, index):
        self.info = self.bridge.call(dict(op="reset", slot=self.slot, task=self.task))

    def observe(self):
        return self.info["observation"]

    def step(self, text):
        if text.endswith("</s>"):
            text = text[:-5]
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                action = PARSE(text).action
        except Exception:
            self.last = dict(
                action=None,
                parse_error=True,
                score=self.info["score"],
                success=self.info["success"],
            )
            return SimpleNamespace(
                state="Invalid Action.\n\n" + self.observe(),
                reward=self.info["reward"],
                done=False,
            )
        self.info = self.bridge.call(dict(op="step", slot=self.slot, action=action))
        self.last = dict(
            action=action,
            parse_error=False,
            score=self.info["score"],
            success=self.info["success"],
        )
        return SimpleNamespace(
            state=self.info["observation"],
            reward=self.info["reward"],
            done=self.info["done"],
        )


class Bridge:
    def __init__(self, benchmark, log):
        self.lock = threading.Lock()
        self.inner = (
            AlfBridge(log) if benchmark == "alfworld" else StagedBridge(benchmark, log)
        )

    def call(self, request):
        with self.lock:
            return self.inner.call(request)

    def close(self):
        with self.lock:
            self.inner.close()


class Client:
    def __init__(self, benchmark, bridge, slot, task):
        self.benchmark = benchmark
        self.inner = (AlfClient if benchmark == "alfworld" else StagedClient)(
            bridge, slot, task
        )

    def reset(self, index):
        return self.inner.reset(index)

    def observe(self):
        return self.inner.observe()

    def step(self, text):
        return self.inner.step(text)

    @property
    def info(self):
        value = self.inner.info
        if self.benchmark != "alfworld":
            return value
        return dict(
            observation=value["observation"],
            score=float(value["won"]),
            reward=value["reward"],
            success=bool(value["won"]),
            done=value["done"],
        )

    @property
    def last(self):
        value = self.inner.last
        out = dict(
            action=value["action"],
            parse_error=value["parse_error"],
            score=self.info["score"],
            success=self.info["success"],
        )
        if "admissible" in value:
            out["admissible"] = value["admissible"]
        return out
