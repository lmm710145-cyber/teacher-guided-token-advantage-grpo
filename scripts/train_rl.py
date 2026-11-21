"""Single-GPU synchronous OPD/GRPO runner with mandatory numerical preflight."""

import argparse
import fcntl
import json
import random
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.config import load, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["preflight", "train", "both"])
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    c = load(a.config, a.set)
    run = (ROOT / a.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from transformers import AutoTokenizer
    from training import runtime as rt
    from training.worker import Worker
    from training.probe_validation import parity

    torch.set_num_threads(4)
    random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    identity = fingerprint(c)
    config_path = run / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == c, (
            "Run directory has another configuration"
        )
    rt.write_json(config_path, c)
    if c["method"] != "grpo":
        s = AutoTokenizer.from_pretrained(c["student_model"], local_files_only=True)
        t = AutoTokenizer.from_pretrained(c["teacher_model"], local_files_only=True)
        assert s.get_vocab() == t.get_vocab(), (
            "Teacher and student vocabulary must be identical"
        )
        assert s.backend_tokenizer.to_str() == t.backend_tokenizer.to_str(), (
            "Tokenizer pipelines must match"
        )
        del s, t
    worker = None

    def status(stage, **kw):
        result = dict(stage=stage, updated_at=time.time(), **kw)
        rt.write_json(run / "status.json", result)
        print(json.dumps(result), flush=True)

    def signals(fs, student, reference, teacher):
        if c["method"] == "opd":
            return rt.score(teacher, fs, c["scoring_micro_batch"])[0]
        refs = rt.score(reference, fs, c["scoring_micro_batch"])[0]
        if c["method"] == "hybrid":
            lps, _ = rt.score(teacher, fs, c["scoring_micro_batch"])
            for f, lp in zip(fs, lps):
                f["teacher_logprobs"] = lp
        elif c["method"] == "tokenadv":
            support, _ = rt.score_topk(teacher, fs, c, c["scoring_micro_batch"])
            rt.freeze_advantages(student, fs, support, c, c["scoring_micro_batch"])
        return refs

    def update(student, opt, fs, refs, step):
        return getattr(rt, c["method"] + "_update")(
            student, opt, fs, refs, c, step, c["training_micro_batch"]
        )

    def restore(student, weights):
        with torch.no_grad():
            for name, param in student.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
        rt.release()

    try:
        pre = run / "preflight.json"
        if a.mode in ("preflight", "both") and not pre.exists():
            status("preflight_sampler")
            worker = Worker(config_path, run)
            folder = run / "preflight"
            folder.mkdir(exist_ok=True)
            bp = folder / "batch.json"
            worker.call(
                "collect",
                adapter=c["parent_checkpoint"],
                version=1,
                step=0,
                cursor=0,
                output=str(bp),
            )
            batch = json.loads(bp.read_text())
            fs = rt.features(batch["episodes"])
            prompt = batch["episodes"][0]["trace"][0]["prompt_ids"]
            before = worker.call(
                "token_probe",
                adapter=c["parent_checkpoint"],
                version=2,
                prompt_ids=prompt,
            )
            worker.call("sleep")
            student = rt.load_model(c)
            reference = rt.load_reference(c) if c["method"] != "opd" else None
            teacher = rt.load_model(c, True) if c["method"] != "grpo" else None
            weights = {
                n: p.detach().cpu().clone()
                for n, p in student.named_parameters()
                if p.requires_grad
            }
            checks = dict(before=parity(student, before, rt.score))
            status("preflight_gradient")
            refs = signals(fs, student, reference, teacher)
            opt = rt.optimizer(student, c)
            checks["update"] = update(student, opt, fs, refs, 1)
            # Long, dense generated-token stress tests the documented full context budget.
            restore(student, weights)
            long = []
            for j in range(c["training_micro_batch"]):
                ids = (
                    [151644, 77091, 198]
                    + [1000 + (i + j) % 1000 for i in range(4092)]
                    + [151645]
                )
                long.append(
                    dict(
                        input_ids=ids,
                        target_positions=list(range(3, 4096)),
                        behavior_logprobs=[0.0] * 4093,
                        item_id=f"stress-{j}",
                        attempt=1,
                        trajectory_tokens=4093,
                        advantage=(-1.0) ** j,
                    )
                )
            old, _ = rt.score(student, long, c["scoring_micro_batch"])
            for f, lp in zip(long, old):
                f["behavior_logprobs"] = lp.tolist()
            refs = signals(long, student, reference, teacher)
            opt = rt.optimizer(student, c)
            checks["full_context_stress"] = update(student, opt, long, refs, 1)
            budget = (
                torch.cuda.get_device_properties(0).total_memory
                / 2**30
                * c["preflight_memory_fraction"]
            )
            assert (
                max(
                    checks["update"]["peak_reserved_gib"],
                    checks["full_context_stress"]["peak_reserved_gib"],
                )
                < budget
            )
            restore(student, weights)
            refs = signals(fs, student, reference, teacher)
            opt = rt.optimizer(student, c)
            update(student, opt, fs, refs, 1)
            target = folder / "updated-adapter"
            student.save_pretrained(target)
            after = worker.call(
                "token_probe", adapter=str(target), version=3, prompt_ids=prompt
            )
            worker.call("sleep")
            checks["after"] = parity(student, after, rt.score)
            assert before["adapter_sha256"] != after["adapter_sha256"]
            back = worker.call(
                "token_probe",
                adapter=c["parent_checkpoint"],
                version=4,
                prompt_ids=prompt,
            )
            worker.call("sleep")
            restore(student, weights)
            checks["restored"] = parity(student, back, rt.score)
            assert back["adapter_sha256"] == before["adapter_sha256"]
            rt.write_json(pre, dict(passed=True, fingerprint=identity, checks=checks))
            worker.close()
            worker = None
            del student, reference, teacher, opt, weights, refs, fs, long
            rt.release()
            status("preflight_passed")
        assert (
            pre.exists() and json.loads(pre.read_text())["fingerprint"] == identity
        ), "Missing or stale preflight; use a new run directory"
        assert json.loads(pre.read_text())["passed"]
        if a.mode == "preflight":
            return
        # A complete marker is written only after adapter and optimizer/RNG state are durable.
        candidates = [
            d
            for d in (run / "checkpoints").glob("step-*")
            if (d / "complete.json").exists()
        ]
        current = Path(c["parent_checkpoint"])
        step = 0
        cursor = 0
        if candidates:
            current = max(candidates, key=lambda d: int(d.name.split("-")[-1]))
            state = json.loads((current / "state.json").read_text())
            assert state["fingerprint"] == identity
            step, cursor = state["step"], state["cursor"]
        status("starting_sampler", step=step)
        worker = Worker(config_path, run)

        def evaluate(step):
            folder = run / f"dev-step-{step}"
            if (folder / "summary.json").exists():
                return
            status("dev_evaluation", step=step)
            worker.call(
                "evaluate",
                adapter=str(current),
                version=10000 + step,
                step=step,
                mode="greedy",
                output=str(folder),
            )

        if step in c["eval_steps"]:
            evaluate(step)
        worker.call("sleep")
        student = rt.load_model({**c, "parent_checkpoint": str(current)})
        reference = rt.load_reference(c) if c["method"] != "opd" else None
        teacher = rt.load_model(c, True) if c["method"] != "grpo" else None
        opt = rt.optimizer(student, c)
        if candidates:
            state = torch.load(
                current / "training_state.pt", map_location="cpu", weights_only=False
            )
            opt.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            random.setstate(state["python_rng"])
        while step < c["max_steps"]:
            step += 1
            folder = run / "batches" / f"update-{step}"
            folder.mkdir(parents=True, exist_ok=True)
            bp = folder / "batch.json"
            status("collecting", step=step, total=c["max_steps"], cursor=cursor)
            if not bp.exists():
                worker.call(
                    "collect",
                    adapter=str(current),
                    version=100 + step,
                    step=step,
                    cursor=cursor,
                    output=str(bp),
                )
            batch = json.loads(bp.read_text())
            assert batch["adapter_sha256"] == rt.digest(
                current / "adapter_model.safetensors"
            )
            assert (
                batch["start_cursor"] == cursor
                and len(batch["episodes"]) == c["batch_size"]
            )
            worker.call("sleep")
            fs = rt.features(batch["episodes"])
            status("scoring", step=step)
            refs = signals(fs, student, reference, teacher)
            if c["method"] == "tokenadv":
                torch.save(
                    [
                        {
                            k: f[k]
                            for k in (
                                "item_id",
                                "attempt",
                                "target_positions",
                                "token_reward",
                                "token_baseline",
                                "token_advantage",
                                "combined_advantage",
                            )
                        }
                        for f in fs
                    ],
                    folder / "token_advantages.pt",
                )
            status("updating", step=step)
            stats = update(student, opt, fs, refs, step)
            rt.write_json(folder / "update.json", stats)
            rt.write_json(folder / "behavior.json", rt.audit(batch["episodes"]))
            current = run / "checkpoints" / f"step-{step}"
            cursor = batch["cursor"]
            rt.save(
                student,
                opt,
                current,
                dict(step=step, cursor=cursor, fingerprint=identity, config=c),
            )
            with (run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(dict(stats, cursor=cursor)) + "\n")
            if step in c["eval_steps"]:
                evaluate(step)
        rt.write_json(
            run / "complete.json",
            dict(complete=True, steps=step, finished_at=time.time()),
        )
        status("complete", step=step)
    except BaseException:
        status("failed", error=traceback.format_exc())
        raise
    finally:
        if worker:
            worker.close()


if __name__ == "__main__":
    main()
