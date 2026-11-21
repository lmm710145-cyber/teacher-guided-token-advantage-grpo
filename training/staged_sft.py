"""Logged SFT primitives for checkpoint-boundary evaluation and exact resumption."""

import json, time, math
from pathlib import Path
import torch
from transformers import Trainer, TrainerCallback
from .engine import supervised_loss


class StopAtStep(TrainerCallback):
    def __init__(self, target):
        self.target = target

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.target:
            control.should_save = True
            control.should_training_stop = True
        return control


class InstrumentedTrainer(Trainer):
    def __init__(self, *args, metrics_path=None, **kwargs):
        self.metric_path = Path(metrics_path) if metrics_path else None
        self.running = {"tokens": 0, "entropy_sum": 0.0, "correct": 0}
        self.last_log_time = time.monotonic()
        super().__init__(*args, compute_loss_func=supervised_loss, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        target_labels = inputs["labels"]
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if model.training:
            with torch.no_grad():
                labels = target_labels[:, 1:]
                b, t = labels.ne(-100).nonzero(as_tuple=True)
                for start in range(0, b.numel(), 64):
                    bi = b[start : start + 64]
                    ti = t[start : start + 64]
                    logits = outputs.logits[bi, ti].detach().float()
                    logp = logits.log_softmax(-1)
                    self.running["entropy_sum"] += (-(logp.exp() * logp).sum()).item()
                    self.running["correct"] += (
                        logits.argmax(-1).eq(labels[bi, ti]).sum().item()
                    )
                self.running["tokens"] += b.numel()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        logs = dict(logs)
        if "loss" in logs and self.running["tokens"]:
            logs.update(
                train_predictive_entropy_nats=self.running["entropy_sum"]
                / self.running["tokens"],
                train_token_accuracy=self.running["correct"] / self.running["tokens"],
                supervised_tokens=self.running["tokens"],
                optimizer_step_seconds=time.monotonic() - self.last_log_time,
            )
            if torch.cuda.is_available():
                logs.update(
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            self.running = {"tokens": 0, "entropy_sum": 0.0, "correct": 0}
            self.last_log_time = time.monotonic()
        super().log(logs, start_time)
        if self.metric_path and "loss" in logs:
            self.metric_path.parent.mkdir(parents=True, exist_ok=True)
            with self.metric_path.open("a") as f:
                f.write(
                    json.dumps(
                        dict(logs, step=self.state.global_step, epoch=self.state.epoch)
                    )
                    + "\n"
                )
