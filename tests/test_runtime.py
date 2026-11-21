import json
import tempfile
import unittest
from pathlib import Path

import torch
from training.config import load, ROOT
from training.grpo_core import selected_logprobs
from training.probe_validation import tie_candidates
from training.protocol import format_valid, render


class RuntimeTests(unittest.TestCase):
    def test_sft_boundary_resume_preserves_schedule(self):
        from types import SimpleNamespace
        from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from training.engine import Rows, MaskedCollator
        from training.staged_sft import InstrumentedTrainer, StopAtStep

        def make_model():
            torch.manual_seed(42)
            base = Qwen3ForCausalLM(
                Qwen3Config(
                    vocab_size=16,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=4,
                    max_position_embeddings=16,
                    attention_dropout=0.0,
                )
            )
            return get_peft_model(
                base,
                LoraConfig(
                    r=2, lora_alpha=4, task_type="CAUSAL_LM", target_modules=["q_proj"]
                ),
            )

        rows = Rows([dict(input_ids=[1, 2, 3], labels=[-100, 2, 3]) for _ in range(5)])
        with tempfile.TemporaryDirectory() as temp:

            def trainer(target):
                args = TrainingArguments(
                    output_dir=temp,
                    num_train_epochs=3,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=2,
                    learning_rate=1e-3,
                    lr_scheduler_type="cosine",
                    warmup_ratio=0.05,
                    use_cpu=True,
                    optim="adamw_torch",
                    save_strategy="no",
                    eval_strategy="no",
                    report_to=[],
                    remove_unused_columns=False,
                    disable_tqdm=True,
                    dataloader_pin_memory=False,
                )
                return InstrumentedTrainer(
                    model=make_model(),
                    args=args,
                    train_dataset=rows,
                    data_collator=MaskedCollator(SimpleNamespace(pad_token_id=0)),
                    callbacks=[StopAtStep(target)],
                )

            first = trainer(2)
            first.train()
            self.assertEqual(first.state.max_steps, 6)
            cp = Path(temp) / "checkpoint-2"
            for file in (
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
            ):
                self.assertTrue((cp / file).exists(), file)
            continued = trainer(6)
            continued.train(resume_from_checkpoint=str(cp))
            self.assertEqual(continued.state.global_step, 6)
            self.assertAlmostEqual(continued.state.epoch, 3.0)
            self.assertEqual(continued.lr_scheduler.last_epoch, 6)

    def test_all_configurations(self):
        files = [
            p
            for p in (ROOT / "configs").glob("*.json")
            if p.name not in ("data_manifest.json", "input_protocol.json")
        ]
        self.assertEqual(len(files), 14)
        for p in files:
            c = load(p)
            self.assertTrue(Path(c["student_model"]).is_absolute())
            self.assertFalse(c["enable_prefix_caching"])
        with self.assertRaises(ValueError):
            load(ROOT / "configs/webshop_tokenadv.json", ["enable_prefix_caching=true"])
        self.assertEqual(
            load(
                ROOT / "configs/webshop_tokenadv.json",
                ["token_lambda=1", "teacher_top_k=16"],
            )["teacher_top_k"],
            16,
        )

    def test_opd_gradient_matches_reverse_kl(self):
        x = torch.tensor([0.1, 0.2, -0.2, 0.4], dtype=torch.double, requires_grad=True)
        q = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)
        p = x.softmax(0)
        lp = x.log_softmax(0)
        exact = (p * (lp - q.log())).sum()
        surrogate = -(p.detach() * (q.log() - lp.detach()).detach() * lp).sum()
        torch.testing.assert_close(
            torch.autograd.grad(exact, x, retain_graph=True)[0],
            torch.autograd.grad(surrogate, x)[0],
        )

    def test_chunked_head_matches_dense_loss_and_gradient(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model

        # Include the real pad token ID; tiny hidden state keeps the CPU test small.
        torch.manual_seed(7)
        base = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=151646,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )
        )
        model = get_peft_model(
            base,
            LoraConfig(
                r=2, lora_alpha=4, task_type="CAUSAL_LM", target_modules=["q_proj"]
            ),
        )
        model.eval()
        rows = [
            dict(input_ids=[1, 2, 3, 4], target_positions=[2, 3]),
            dict(input_ids=[4, 5, 6], target_positions=[2]),
        ]
        lp, _, owners = selected_logprobs(model, rows, chunk=1)
        dense = []
        for r in rows:
            logits = (
                model(torch.tensor([r["input_ids"]])).logits[0].float().log_softmax(-1)
            )
            dense.extend(
                logits[t - 1, r["input_ids"][t]] for t in r["target_positions"]
            )
        dense = torch.stack(dense)
        torch.testing.assert_close(lp, dense, atol=1e-5, rtol=1e-5)
        params = [p for p in model.parameters() if p.requires_grad]
        g = torch.autograd.grad(lp.sum(), params, retain_graph=True)
        expected = torch.autograd.grad(dense.sum(), params)
        for a, b in zip(g, expected):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        self.assertEqual(owners.tolist(), [0, 0, 1])

    def test_near_tie_does_not_accept_large_errors(self):
        a = dict(ids=[1], top_logprobs=[{"1": -0.7, "2": -0.71}])
        b = dict(ids=[2], top_logprobs=[{"1": -0.72, "2": -0.69}])
        self.assertEqual(tie_candidates(a, b, 0), (1, 2))
        with self.assertRaises(AssertionError):
            tie_candidates(a, dict(ids=[2], top_logprobs=[{"1": -0.7, "2": -2.0}]), 0)

    def test_prompt_and_format(self):
        self.assertTrue(format_valid("alfworld", "Action: look"))
        self.assertFalse(format_valid("webshop", "Action: click[Buy Now]"))
        self.assertTrue(
            format_valid("webshop", "Thought: Buy it.\nAction: click[Buy Now]")
        )
        self.assertFalse(
            format_valid("webshop", "Thought: x\nAction: click[a]\nAction: click[b]")
        )
        self.assertIn(
            "<think>\n\n</think>", render([dict(role="assistant", content="x")])
        )

    def test_collection_zero_variance_policy(self):
        from scripts.sampler import Sampler

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "adapter_model.safetensors").write_bytes(b"fixture")
            for method in ("opd", "grpo", "hybrid", "tokenadv"):
                sampler = Sampler.__new__(Sampler)
                sampler.c = load(ROOT / f"configs/alfworld_{method}.json")
                sampler.c.update(
                    batch_size=4,
                    max_concurrent_trajectories=4,
                    max_candidate_groups_per_update=8,
                )
                sampler.train = [
                    dict(item_id=str(i), environment_index=i) for i in range(10)
                ]
                sampler.run = root
                sampler.activate = lambda *a: None

                def episodes(jobs, step):
                    return [
                        dict(
                            item_id=t["item_id"],
                            attempt=a,
                            initial_observation="same",
                            reward=0.0,
                            trace=[{}],
                            generated_tokens=1,
                        )
                        for t, a in jobs
                    ]

                sampler.episodes = episodes
                request = dict(
                    adapter=str(root),
                    version=1,
                    step=1,
                    cursor=0,
                    output=str(root / f"{method}.json"),
                )
                if method == "grpo":
                    with self.assertRaises(RuntimeError):
                        sampler.collect(request)
                else:
                    sampler.collect(request)
                    batch = json.loads(Path(request["output"]).read_text())
                    self.assertEqual(len(batch["episodes"]), 4)
                    self.assertTrue(all(e["advantage"] == 0 for e in batch["episodes"]))


if __name__ == "__main__":
    unittest.main()
