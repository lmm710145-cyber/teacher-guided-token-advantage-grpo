"""Train to each requested epoch boundary, exit the trainer, then evaluate with vLLM."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.config import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--segment-target", type=int, help=argparse.SUPPRESS)
    p.add_argument("--resume", help=argparse.SUPPRESS)
    a = p.parse_args()
    c = load(a.config, a.set)
    run = (ROOT / a.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    cpath = run / "config.json"
    if cpath.exists():
        assert json.loads(cpath.read_text()) == c, "Different config in run directory"
    cpath.write_text(json.dumps(c, indent=2) + "\n")
    if a.segment_target is None:
        import fcntl

        lock = (run / "run.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        features = run / "train.jsonl"
        subprocess.run(
            [
                c["train_python"],
                str(ROOT / "scripts/prepare_data.py"),
                "tokenize",
                "--input",
                c["sft_data"],
                "--output",
                str(features),
                "--model",
                c["student_model"],
                "--benchmark",
                c["benchmark"],
            ],
            check=True,
        )
        n = sum(1 for line in features.open() if line.strip())
        steps_per_epoch = math.ceil(
            math.ceil(n / c["micro_batch"]) / c["gradient_accumulation_steps"]
        )
        targets = sorted(
            {math.ceil(e * steps_per_epoch) for e in c["evaluation_epochs"]}
        )
        assert targets[-1] == c["epochs"] * steps_per_epoch
        plan = dict(
            rows=n,
            steps_per_epoch=steps_per_epoch,
            total_steps=targets[-1],
            targets=targets,
            data_sha256=hashlib.sha256(features.read_bytes()).hexdigest(),
        )
        if (run / "plan.json").exists():
            assert json.loads((run / "plan.json").read_text()) == plan
        (run / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        previous = None
        for target in targets:
            cp = run / f"checkpoint-{target}"
            if not (cp / "complete.json").exists():
                cmd = [
                    c["train_python"],
                    str(Path(__file__).resolve()),
                    "--config",
                    str(cpath),
                    "--run",
                    str(run),
                    "--segment-target",
                    str(target),
                ]
                if previous:
                    cmd += ["--resume", str(previous)]
                subprocess.run(cmd, check=True)
            output = run / f"dev-step-{target}"
            if not (output / "summary.json").exists():
                subprocess.run(
                    [
                        c["inference_python"],
                        str(ROOT / "scripts/evaluate.py"),
                        "--config",
                        str(cpath),
                        "--adapter",
                        str(cp),
                        "--output",
                        str(output),
                    ],
                    check=True,
                )
            previous = cp
        (run / "complete.json").write_text(
            json.dumps(dict(complete=True, steps=targets[-1]))
        )
        return
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        set_seed,
    )
    from peft import LoraConfig, get_peft_model
    from training.engine import Rows, MaskedCollator
    from training.staged_sft import InstrumentedTrainer, StopAtStep

    set_seed(c["seed"])
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(c["student_model"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        c["student_model"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=c["lora_r"],
            lora_alpha=c["lora_alpha"],
            lora_dropout=c["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.enable_input_require_grads()
    rows = [json.loads(l) for l in (run / "train.jsonl").read_text().splitlines()]
    plan = json.loads((run / "plan.json").read_text())
    args = TrainingArguments(
        output_dir=str(run),
        num_train_epochs=c["epochs"],
        per_device_train_batch_size=c["micro_batch"],
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        learning_rate=c["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=c["warmup_ratio"],
        bf16=True,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        eval_strategy="no",
        logging_steps=1,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=c["seed"],
        data_seed=c["seed"],
        save_safetensors=True,
        disable_tqdm=True,
    )
    trainer = InstrumentedTrainer(
        model=model,
        args=args,
        train_dataset=Rows(rows),
        data_collator=MaskedCollator(tok),
        processing_class=tok,
        callbacks=[StopAtStep(a.segment_target)],
        metrics_path=run / "metrics.jsonl",
    )
    trainer.train(resume_from_checkpoint=a.resume)
    assert trainer.state.global_step == a.segment_target
    assert trainer.state.max_steps == plan["total_steps"]
    cp = run / f"checkpoint-{a.segment_target}"
    required = [
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ]
    assert all((cp / f).exists() for f in required)
    (cp / "complete.json").write_text(
        json.dumps(
            dict(
                step=a.segment_target,
                epoch=trainer.state.epoch,
                sha256={
                    f: hashlib.sha256((cp / f).read_bytes()).hexdigest()
                    for f in required
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
