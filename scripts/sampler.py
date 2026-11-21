"""Persistent single-GPU sampler; JSONL control, file-based trajectory exchange."""

import argparse, collections, concurrent.futures, contextlib, hashlib, json, os, pathlib, random, sys, time, traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.grpo_core import write_json, group_advantages, pack_episode, digest
from training.env_clients import Bridge, Client
from training.rollout import RolloutSession
from training.protocol import format_valid
from training.passk import wilson


def stagnant(ts):
    norm = lambda x: " ".join(str(x).lower().split())
    for p in range(1, 5):
        if len(ts) < 3 * p:
            continue
        z = ts[-3 * p :]
        pairs = [(norm(t.get("action", "")), norm(t["observation"])) for t in z]
        if (
            pairs[:p] * 3 == pairs
            and len({t.get("score") for t in z}) == 1
            and not any(a.startswith("wait") for a, o in pairs)
        ):
            return True
    return False


class Sampler:
    def __init__(self, config, run):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.c = config
        self.run = run
        self.sleeping = False
        self.lora_id = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            ROOT / config["student_model"], local_files_only=True
        )
        self.llm = LLM(
            model=str(ROOT / config["student_model"]),
            dtype="bfloat16",
            generation_config="vllm",
            gpu_memory_utilization=config["vllm_memory_utilization"],
            max_model_len=config["cutoff_length"],
            max_num_seqs=config["max_concurrent_trajectories"],
            max_num_batched_tokens=config["max_num_batched_tokens"],
            enable_prefix_caching=config.get("enable_prefix_caching", False),
            enable_chunked_prefill=True,
            enable_sleep_mode=True,
            enable_lora=True,
            max_lora_rank=config["lora_r"],
            max_loras=1,
            max_cpu_loras=1,
            seed=config["seed"],
            enforce_eager=config.get("enforce_eager", False),
        )
        self.log = (run / "environment.log").open("a")
        self.bridges = [
            Bridge(config["benchmark"], self.log)
            for _ in range(config["environment_workers"])
        ]
        self.train = [
            json.loads(s)
            for s in (ROOT / config["train_data"]).read_text().splitlines()
        ]
        self.dev = [
            json.loads(s) for s in (ROOT / config["dev_data"]).read_text().splitlines()
        ]

    def activate(self, path, version):
        from vllm.lora.request import LoRARequest

        if self.sleeping:
            self.llm.wake_up()
            self.sleeping = False
        if self.lora_id is not None:
            self.llm.llm_engine.remove_lora(self.lora_id)
        self.llm.reset_prefix_cache()
        self.lora_id = version
        self.lora = LoRARequest(f"policy_{version}", version, str(ROOT / path))
        self.llm.llm_engine.add_lora(self.lora)

    def sleep(self):
        if not self.sleeping:
            self.llm.sleep(level=1)
            self.sleeping = True
        import torch

        free, total = torch.cuda.mem_get_info()
        return dict(sleeping=True, gpu_used_bytes=total - free)

    def probe(self, request):
        from vllm import SamplingParams

        self.activate(request["adapter"], request["version"])
        length = self.c["cutoff_length"] - 256
        ids = [151644, 77091, 198] + [1000 + (i % 1000) for i in range(length - 3)]
        out = self.llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(temperature=0, max_tokens=16),
            lora_request=self.lora,
            use_tqdm=False,
        )
        return dict(
            prompt_tokens=length,
            output_tokens=len(out[0].outputs[0].token_ids),
            **self.sleep(),
        )

    def seed(self, task, attempt, turn, step, evaluation):
        # Eval seeds match between checkpoints. Training seeds also include update.
        key = (
            f"{self.c['seed']}|{self.c['benchmark']}|{task['item_id']}|{attempt}|{turn}"
        )
        if not evaluation:
            key += f"|update={step}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (
            2**31 - 1
        )

    def prepare_environment_wave(self, trajectory_count):
        # Pinned TextWorld PddlEnv.close() inherits a no-op. Each reset loads
        # another deleted-but-mapped ~34 MiB planner library, whose disk blocks
        # remain allocated until process exit. Recycle only between complete
        # waves; never replace a live trajectory or alter its observations.
        if self.c["benchmark"] != "alfworld":
            return
        previous = getattr(self, "alf_resets_since_restart", 0)
        if previous + trajectory_count > 64:
            old_pids = [b.inner.p.pid for b in self.bridges]
            for bridge in self.bridges:
                bridge.close()
            self.bridges = [
                Bridge("alfworld", self.log)
                for _ in range(self.c["environment_workers"])
            ]
            event = dict(
                unix_time=time.time(),
                previous_resets=previous,
                old_pids=old_pids,
                new_pids=[b.inner.p.pid for b in self.bridges],
                reason="release_deleted_planner_library_mappings",
            )
            with (self.run / "environment_restarts.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            previous = 0
        self.alf_resets_since_restart = previous + trajectory_count

    def episodes(self, jobs, step, evaluation=False, greedy=False):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        c = self.c
        active = {}
        requests = {}
        completed = []
        started = time.monotonic()
        last_activity = started
        assert len(jobs) <= c["max_concurrent_trajectories"]
        self.prepare_environment_wave(len(jobs))

        def reset(pair):
            slot, (task, attempt) = pair
            client = Client(
                c["benchmark"], self.bridges[slot % len(self.bridges)], slot, task
            )
            session = RolloutSession(
                c["benchmark"],
                client,
                self.tokenizer,
                "dev" if evaluation else "online",
            )
            session.reset(task["environment_index"])
            return slot, dict(
                task=task,
                attempt=attempt,
                client=client,
                session=session,
                trace=[],
                initial_observation=client.observe(),
            )

        def finish(slot, reason):
            a = active.pop(slot)
            info = a["client"].info
            e = dict(
                item_id=a["task"]["item_id"],
                task_type=a["task"].get(
                    "task_type", a["task"].get("task_name", a["task"].get("category"))
                ),
                attempt=a["attempt"],
                initial_observation=a["initial_observation"],
                trace=a["trace"],
                success=bool(info["success"]),
                reward=float(info["reward"]),
                score=float(info["score"]),
                termination=reason,
            )
            e["turns"] = len(e["trace"])
            e["generated_tokens"] = sum(len(t["output_ids"]) for t in e["trace"])
            if e["trace"]:
                pack_episode(
                    e
                )  # Verify exact prefix/target alignment before accepting any data.
            completed.append(e)

        def enqueue(slot):
            a = active[slot]
            ids = a["session"].model_input()["input_ids"]
            if len(ids) + c["max_new_tokens"] > c["cutoff_length"]:
                finish(slot, "context_limit")
                return
            turn = len(a["trace"]) + 1
            rid = f"{slot}:{turn}"
            seed = self.seed(a["task"], a["attempt"], turn, step, evaluation)
            params = SamplingParams(
                temperature=0
                if greedy
                else (
                    c["dev_sampling"]["temperature"]
                    if evaluation
                    else c["sampling"]["temperature"]
                ),
                top_p=(
                    c["dev_sampling"]["top_p"] if evaluation else c["sampling"]["top_p"]
                )
                if not greedy
                else 1.0,
                top_k=c["dev_sampling"]["top_k"]
                if evaluation
                else c["sampling"]["top_k"],
                max_tokens=c["max_new_tokens"],
                stop_token_ids=[151645],
                skip_special_tokens=False,
                seed=seed,
                logprobs=0,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )
            a["prompt_ids"] = ids
            requests[rid] = slot
            self.llm.llm_engine.add_request(
                rid, {"prompt_token_ids": ids}, params, lora_request=self.lora
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for slot, a in pool.map(reset, enumerate(jobs)):
                active[slot] = a
        for slot in list(active):
            enqueue(slot)
        while self.llm.llm_engine.has_unfinished_requests():
            for response in self.llm.llm_engine.step():
                if not response.finished:
                    continue
                slot = requests.pop(response.request_id)
                a = active[slot]
                out = response.outputs[0]
                ids = list(out.token_ids)
                assert ids and len(ids) == len(out.logprobs)
                logps = [
                    float(lp[token].logprob) for token, lp in zip(ids, out.logprobs)
                ]
                display = ids[: ids.index(151645)] if 151645 in ids else ids
                text = self.tokenizer.decode(display, skip_special_tokens=False)
                encoded = self.tokenizer(
                    text, add_special_tokens=False, return_offsets_mapping=True
                )
                action_start = text.rfind("Action:")
                action_mask = [False] * len(ids)
                if action_start >= 0 and encoded["input_ids"] == display:
                    action_mask = [
                        end > action_start + len("Action:")
                        for _, end in encoded["offset_mapping"]
                    ] + [False] * (len(ids) - len(display))
                result = a["session"].step(text)
                a["trace"].append(
                    dict(
                        prompt_ids=a["prompt_ids"],
                        output_ids=ids,
                        behavior_logprobs=logps,
                        action_token_mask=action_mask,
                        output=text,
                        observation=result.state,
                        format_valid=format_valid(c["benchmark"], text),
                        stopped=out.finish_reason == "stop",
                        reward=float(result.reward),
                        **a["client"].last,
                    )
                )
                if result.done or a["client"].info["success"]:
                    finish(slot, "environment_done")
                # Pure OPD retains looping and unsuccessful trajectories through the same turn budget.
                elif len(a["trace"]) >= c["max_turns"]:
                    finish(slot, "turn_limit")
                else:
                    enqueue(slot)
            if time.monotonic() - last_activity >= 30:
                write_json(
                    self.run / "rollout_activity.json",
                    dict(
                        step=step,
                        evaluation=evaluation,
                        active_trajectories=len(active),
                        finished_trajectories=len(completed),
                        batch_trajectories=len(jobs),
                        completed_turns=sum(len(a["trace"]) for a in active.values())
                        + sum(e["turns"] for e in completed),
                        seconds=time.monotonic() - started,
                    ),
                )
                last_activity = time.monotonic()
        assert not active and not requests
        return completed

    def collect(self, request):
        c = self.c
        self.activate(request["adapter"], request["version"])
        started = time.monotonic()
        gsize = 1 if c["method"] == "opd" else c["group_size"]
        target = c["batch_size"] // gsize
        limit = (
            c["max_candidate_groups_per_update"] if c["method"] == "grpo" else target
        )
        cursor = request.get("cursor", 0)
        accepted = []
        all_episodes = []
        summaries = []
        orders = {}
        seen = set()
        if len(self.train) < target:
            raise ValueError("Training split has fewer tasks than one update requires")
        while len(accepted) < target and len(summaries) < limit:
            if time.monotonic() - started > c["max_collection_seconds"]:
                break
            n = min(
                target - len(accepted),
                c["max_concurrent_trajectories"] // gsize,
                limit - len(summaries),
                len(self.train) - len(seen),
            )
            if n == 0:
                break
            tasks = []
            while len(tasks) < n:
                epoch, i = divmod(cursor, len(self.train))
                cursor += 1
                if epoch not in orders:
                    order = sorted(self.train, key=lambda t: t["item_id"])
                    random.Random(c["seed"] + epoch).shuffle(order)
                    orders[epoch] = order
                task = orders[epoch][i]
                if task["item_id"] in seen:
                    continue
                seen.add(task["item_id"])
                tasks.append(task)
            # Each wave uses unique tasks; datasets must be at least as large as concurrency.
            assert len({t["item_id"] for t in tasks}) == len(tasks)
            episodes = self.episodes(
                [(t, a) for t in tasks for a in range(1, gsize + 1)], request["step"]
            )
            for task in tasks:
                es = sorted(
                    [e for e in episodes if e["item_id"] == task["item_id"]],
                    key=lambda e: e["attempt"],
                )
                assert (
                    len(es) == gsize
                    and len({e["initial_observation"] for e in es}) == 1
                )
                rewards = [e["reward"] for e in es]
                adv = group_advantages(rewards) if gsize > 1 else [0.0]
                nonconstant = max(rewards) != min(rewards)
                keep = all(e["trace"] for e in es) and (
                    c["method"] != "grpo"
                    or adv is not None
                    or not c["filter_constant_groups"]
                )
                summaries.append(
                    dict(
                        item_id=task["item_id"],
                        rewards=rewards,
                        accepted=keep,
                        nonconstant=nonconstant,
                    )
                )
                for e, v in zip(es, adv or [0.0] * gsize):
                    e["advantage"] = v
                if keep:
                    accepted.append(es)
                all_episodes.extend(es)
            write_json(
                self.run / "collection_progress.json",
                dict(
                    step=request["step"],
                    accepted_groups=len(accepted),
                    required_groups=target,
                    candidate_groups=len(summaries),
                    cursor=cursor,
                    seconds=time.monotonic() - started,
                ),
            )
        result = dict(
            step=request["step"],
            start_cursor=request.get("cursor", 0),
            cursor=cursor,
            adapter=request["adapter"],
            adapter_sha256=digest(
                ROOT / request["adapter"] / "adapter_model.safetensors"
            ),
            episodes=[e for es in accepted for e in es],
            candidate_episodes=all_episodes,
            groups=summaries,
            accepted_groups=len(accepted),
            candidate_groups=len(summaries),
            sampled_trajectories=len(all_episodes),
            seconds=time.monotonic() - started,
            nonconstant_groups=sum(g["nonconstant"] for g in summaries),
            zero_variance_groups=sum(not g["nonconstant"] for g in summaries),
        )
        write_json(request["output"], result)
        if len(accepted) != target:
            raise RuntimeError(
                f"Collection budget exhausted: {len(accepted)}/{target} groups. No partial optimizer update performed."
            )
        return dict(path=request["output"], seconds=result["seconds"])

    def token_probe(self, request):
        from vllm import SamplingParams

        self.activate(request["adapter"], request["version"])
        ids = request["prompt_ids"]
        outputs = []
        for repeat in range(2):
            out = self.llm.generate(
                [{"prompt_token_ids": ids}],
                SamplingParams(
                    temperature=0, max_tokens=16, logprobs=5, skip_special_tokens=False
                ),
                lora_request=self.lora,
                use_tqdm=False,
            )[0].outputs[0]
            outputs.append(
                dict(
                    ids=list(out.token_ids),
                    logprobs=[
                        float(lp[t].logprob)
                        for t, lp in zip(out.token_ids, out.logprobs)
                    ],
                    top_logprobs=[
                        {str(k): float(v.logprob) for k, v in lp.items()}
                        for lp in out.logprobs
                    ],
                )
            )
        return dict(
            prompt_ids=ids,
            outputs=outputs,
            adapter_sha256=digest(
                ROOT / request["adapter"] / "adapter_model.safetensors"
            ),
            version=request["version"],
        )

    def evaluate(self, request):
        c = self.c
        self.activate(request["adapter"], request["version"])
        folder = pathlib.Path(request["output"])
        folder.mkdir(parents=True, exist_ok=True)
        mode = request["mode"]
        k = 5 if mode == "pass5" else 1
        tasks = self.dev[: request.get("limit", len(self.dev))]
        records = {}
        attempts = []
        start = time.monotonic()
        for attempt in range(1, k + 1):
            pending = [
                t for t in tasks if not records.get(t["item_id"], {}).get("success")
            ]
            for offset in range(0, len(pending), c["max_concurrent_trajectories"]):
                group = pending[offset : offset + c["max_concurrent_trajectories"]]
                episodes = self.episodes(
                    [(t, attempt) for t in group],
                    request["step"],
                    True,
                    mode == "greedy",
                )
                for e in episodes:
                    write_json(
                        folder / "attempts" / e["item_id"] / f"attempt-{attempt}.json",
                        e,
                    )
                    attempts.append(e)
                    r = records.setdefault(
                        e["item_id"],
                        dict(
                            item_id=e["item_id"],
                            success=False,
                            first_success_attempt=None,
                            attempts=0,
                        ),
                    )
                    r["attempts"] += 1
                    if e["success"]:
                        r.update(success=True, first_success_attempt=attempt)
                write_json(
                    folder / "progress.json",
                    dict(
                        mode=mode,
                        step=request["step"],
                        attempt=attempt,
                        attempts=len(attempts),
                        seen_tasks=len(records),
                        tasks=len(tasks),
                        successes=sum(r["success"] for r in records.values()),
                        seconds=time.monotonic() - start,
                    ),
                )
        turns = [t for e in attempts for t in e["trace"]]
        wins = sum(r["success"] for r in records.values())
        result = dict(
            complete=True,
            benchmark=c["benchmark"],
            step=request["step"],
            mode=mode,
            tasks=len(tasks),
            successes=wins,
            success_rate=wins / len(tasks),
            pass_at_k=wins / len(tasks),
            sampled_pass_at_1=sum(
                r["first_success_attempt"] == 1 for r in records.values()
            )
            / len(tasks),
            pass_curve={
                str(j): sum(
                    r["first_success_attempt"] is not None
                    and r["first_success_attempt"] <= j
                    for r in records.values()
                )
                / len(tasks)
                for j in range(1, k + 1)
            },
            wilson95=wilson(wins, len(tasks)),
            attempts_executed=len(attempts),
            mean_first_reward=sum(e["reward"] for e in attempts if e["attempt"] == 1)
            / len(tasks),
            format_rate=sum(t["format_valid"] for t in turns) / max(1, len(turns)),
            parser_errors=sum(t["parse_error"] for t in turns),
            context_limit_failures=sum(
                e["termination"] == "context_limit" for e in attempts
            ),
            seconds=time.monotonic() - start,
            protocol=dict(
                cutoff=c["cutoff_length"],
                max_turns=c["max_turns"],
                max_new_tokens=c["max_new_tokens"],
                sampling=c["dev_sampling"] if mode != "greedy" else {"temperature": 0},
                adapter=request["adapter"],
                adapter_sha256=digest(
                    ROOT / request["adapter"] / "adapter_model.safetensors"
                ),
                prefix_caching=c.get("enable_prefix_caching", False),
                group_filter=False,
                seed=c["seed"],
                task_sha256=digest(ROOT / c["dev_data"]),
            ),
        )
        write_json(folder / "summary.json", result)
        return result

    def close(self):
        for b in self.bridges:
            b.close()
        self.log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    a = p.parse_args()
    output = sys.stdout
    sys.stdout = sys.stderr
    run = pathlib.Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    sampler = None
    try:
        sampler = Sampler(json.loads(pathlib.Path(a.config).read_text()), run)
        output.write("__GRPO_RPC__" + json.dumps(dict(ready=True)) + "\n")
        output.flush()
        for line in sys.stdin:
            request = json.loads(line)
            try:
                op = request.pop("op")
                if op == "close":
                    result = dict(closed=True)
                elif op == "sleep":
                    result = sampler.sleep()
                elif op == "collect":
                    result = sampler.collect(request)
                elif op == "evaluate":
                    result = sampler.evaluate(request)
                elif op == "probe":
                    result = sampler.probe(request)
                elif op == "token_probe":
                    result = sampler.token_probe(request)
                else:
                    raise ValueError(op)
                output.write("__GRPO_RPC__" + json.dumps(dict(result=result)) + "\n")
                output.flush()
                if op == "close":
                    break
            except BaseException:
                output.write(
                    "__GRPO_RPC__"
                    + json.dumps(dict(error=traceback.format_exc()))
                    + "\n"
                )
                output.flush()
                break
    finally:
        if sampler:
            sampler.close()


if __name__ == "__main__":
    main()
