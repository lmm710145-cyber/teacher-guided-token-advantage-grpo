import json, os, queue, threading, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Worker:
    def __init__(self, cpath, run):
        self.log = (run / "sampler.log").open("a")
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "HF_HUB_OFFLINE": "1",
        }
        env["AGENT_BENCH_PYTHON"] = json.loads(Path(cpath).read_text())["bench_python"]
        self.p = subprocess.Popen(
            [
                json.loads(Path(cpath).read_text())["inference_python"],
                "-u",
                str(ROOT / "scripts/sampler.py"),
                "--config",
                str(cpath),
                "--run",
                str(run),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.messages = queue.Queue()

        def reader():
            for line in self.p.stdout:
                self.messages.put(line)
            self.messages.put(None)

        threading.Thread(target=reader, daemon=True).start()
        self.read()

    def read(self):
        deadline = time.time() + 3600
        while time.time() < deadline:
            try:
                line = self.messages.get(timeout=30)
            except queue.Empty:
                if self.p.poll() is not None:
                    raise RuntimeError("sampler exited " + str(self.p.returncode))
                continue
            if not line:
                raise RuntimeError("sampler pipe closed; inspect sampler.log")
            if not line.startswith("__GRPO_RPC__"):
                self.log.write(line)
                self.log.flush()
                continue
            r = json.loads(line[len("__GRPO_RPC__") :])
            if "error" in r:
                raise RuntimeError(r["error"])
            return r.get("result", r)
        raise TimeoutError("sampler RPC one hour timeout")

    def call(self, op, **kw):
        self.p.stdin.write(json.dumps(dict(op=op, **kw)) + "\n")
        self.p.stdin.flush()
        return self.read()

    def close(self):
        if self.p.poll() is None:
            try:
                self.call("close")
                self.p.wait(timeout=30)
            except Exception:
                self.p.kill()
                self.p.wait()
        self.log.close()
