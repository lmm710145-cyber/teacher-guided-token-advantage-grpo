import gc, hashlib, json, os, random, queue, threading, subprocess, time
from pathlib import Path
import torch
from training.grpo_core import (
    write_json,
    digest,
    pack_episode,
    feature_segments,
    selected_logprobs,
    collate,
)
from training.objectives import (
    teacher_support,
    student_baseline,
    combine_advantages,
    clipped_objective,
)

ROOT = Path(__file__).resolve().parents[1]


def release():
    gc.collect()
    torch.cuda.empty_cache()


def load_model(c, teacher=False):
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, PeftConfig

    base = ROOT / c["teacher_model" if teacher else "student_model"]
    adapter = ROOT / c["teacher_checkpoint" if teacher else "parent_checkpoint"]
    pc = PeftConfig.from_pretrained(adapter)
    pc.lora_dropout = 0.0
    assert pc.r == c["lora_r"] and pc.lora_alpha == c["lora_alpha"]
    m = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        ),
        adapter,
        config=pc,
        is_trainable=not teacher,
    )
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    m.config.use_cache = False
    if teacher:
        m.requires_grad_(False)
        m.eval()
    else:
        for p in m.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        m.enable_input_require_grads()
        m.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        m.train()
    return m


def features(es):
    result = [
        f
        for e in es
        if e["trace"]
        for f in feature_segments([pack_episode(e)])
        if f["target_positions"]
    ]
    assert sum(len(f["target_positions"]) for f in result) == sum(
        e["generated_tokens"] for e in es
    )
    for f in result:
        assert len(f["input_ids"]) <= 4096 and len(f["target_positions"]) == len(
            f["behavior_logprobs"]
        )
    return result


def groups(fs, micro, budget=16384):
    order = sorted(range(len(fs)), key=lambda i: len(fs[i]["input_ids"]))
    group = []
    for i in order:
        if group and (
            len(group) >= micro or len(fs[i]["input_ids"]) * (len(group) + 1) > budget
        ):
            yield group
            group = []
        group.append(i)
    if group:
        yield group


@torch.no_grad()
def score(model, fs, micro=4):
    start = time.time()
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    out = [None] * len(fs)
    for ix in groups(fs, micro):
        lp, _, owners = selected_logprobs(model, [fs[i] for i in ix], chunk=256)
        for j, i in enumerate(ix):
            out[i] = lp[owners == j].float().cpu()
        del lp, owners
    stats = dict(
        seconds=time.time() - start,
        micro=micro,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
    )
    assert all(torch.isfinite(p).all() for p in out)
    model.cpu()
    release()
    return out, stats


def opt_to(opt, device):
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and k != "step":
                state[k] = v.to(device)


def optimizer(model, c):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=c["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )


def load_reference(c):
    m = load_model(c)
    m.requires_grad_(False)
    m.gradient_checkpointing_disable()
    m.eval()
    return m


@torch.no_grad()
def predictors(model, rows):
    x, attention, mask = collate(rows, next(model.parameters()).device)
    with torch.autocast(
        device_type=x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda
    ):
        hidden = (
            model.get_base_model()
            .model(
                input_ids=x, attention_mask=attention, use_cache=False, return_dict=True
            )
            .last_hidden_state
        )
    selected = hidden[:, :-1][mask[:, 1:]]
    targets = x[:, 1:][mask[:, 1:]]
    owners = torch.arange(len(rows), device=x.device)[:, None].expand_as(mask[:, 1:])[
        mask[:, 1:]
    ]
    return selected, targets, owners


@torch.no_grad()
def project(model, hidden):
    # Match selected_logprobs: BF16 projection followed by full-vocabulary FP32 normalization.
    with torch.autocast(
        device_type=hidden.device.type, dtype=torch.bfloat16, enabled=hidden.is_cuda
    ):
        return torch.nn.functional.linear(
            hidden, model.get_output_embeddings().weight
        ).float()


@torch.no_grad()
def score_topk(model, fs, c, micro=2):
    """Teacher at exact sampled prefixes. Cache O(generated_tokens*K), never T*V."""
    start = time.time()
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    out = [None] * len(fs)
    chunk = c["head_chunk_size"]
    try:
        for ix in groups(fs, micro, c["padded_token_budget"]):
            h, y, owners = predictors(model, [fs[i] for i in ix])
            parts = []
            for off in range(0, len(y), chunk):
                logits = project(model, h[off : off + chunk])
                ids, p, r, hit = teacher_support(
                    logits, y[off : off + chunk], c["teacher_top_k"]
                )
                parts.append((ids.int().cpu(), p.cpu(), r.cpu(), hit.cpu()))
                del logits
            ids, p, r, hit = [torch.cat([part[j] for part in parts]) for j in range(4)]
            for j, i in enumerate(ix):
                take = owners.cpu() == j
                out[i] = dict(
                    ids=ids[take], probabilities=p[take], reward=r[take], hit=hit[take]
                )
            del h, y, owners, parts, ids, p, r, hit
        stats = dict(
            seconds=time.time() - start,
            micro=micro,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        assert all(torch.isfinite(v["probabilities"]).all() for v in out)
        return out, stats
    finally:
        model.cpu()
        release()


@torch.no_grad()
def freeze_advantages(model, fs, support, c, micro=2):
    """Snapshot student expectation BEFORE any optimizer update; no gradient through b."""
    start = time.time()
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    errors = []
    try:
        for ix in groups(fs, micro, c["padded_token_budget"]):
            h, y, owners = predictors(model, [fs[i] for i in ix])
            ids = torch.cat([support[i]["ids"] for i in ix])
            p = torch.cat([support[i]["probabilities"] for i in ix])
            baselines = []
            lps = []
            for off in range(0, len(y), c["head_chunk_size"]):
                end = off + c["head_chunk_size"]
                logits = project(model, h[off:end])
                baselines.append(
                    student_baseline(
                        logits, ids[off:end].cuda(), p[off:end].cuda()
                    ).cpu()
                )
                lps.append(
                    (
                        logits.gather(-1, y[off:end, None]).squeeze(-1)
                        - logits.logsumexp(-1)
                    ).cpu()
                )
                del logits
            baseline = torch.cat(baselines)
            lp = torch.cat(lps)
            for j, i in enumerate(ix):
                take = owners.cpu() == j
                f = fs[i]
                r = support[i]["reward"]
                b = baseline[take]
                a = torch.full_like(r, f["advantage"])
                combined, local = combine_advantages(a, r, b, c["token_lambda"])
                assert (
                    torch.isfinite(combined).all()
                    and ((b >= 0) & (b <= 1.000001)).all()
                )
                f.update(
                    token_reward=r,
                    token_baseline=b,
                    token_advantage=local,
                    combined_advantage=combined,
                    teacher_topk_mass=support[i]["probabilities"].sum(-1),
                    teacher_topk_hit=support[i]["hit"],
                    snapshot_logprobs=lp[take],
                )
                errors.append((lp[take] - torch.tensor(f["behavior_logprobs"])).abs())
            del h, y, owners, ids, p, baselines, lps, baseline, lp
        err = torch.cat(errors)
        assert float(err.mean()) < 0.1 and float(err.quantile(0.99)) < 1.0, (
            "Behavior policy mismatch",
            float(err.mean()),
            float(err.quantile(0.99)),
        )
        return dict(
            seconds=time.time() - start,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            behavior_logprob_abs_error_mean=float(err.mean()),
            behavior_logprob_abs_error_p99=float(err.quantile(0.99)),
            **advantage_stats(fs, c),
        )
    finally:
        model.cpu()
        release()


def advantage_stats(fs, c):
    cat = lambda key: torch.cat([f[key].float() for f in fs])
    r, b, a, total, mass, hit = [
        cat(k)
        for k in (
            "token_reward",
            "token_baseline",
            "token_advantage",
            "combined_advantage",
            "teacher_topk_mass",
            "teacher_topk_hit",
        )
    ]
    g = torch.cat([torch.full_like(f["token_advantage"], f["advantage"]) for f in fs])
    nonzero = g != 0
    rms = lambda x: float(x.square().mean().sqrt())
    return dict(
        teacher_top_k=c["teacher_top_k"],
        token_lambda=c["token_lambda"],
        reward_mean=float(r.mean()),
        baseline_mean=float(b.mean()),
        local_mean=float(a.mean()),
        local_rms=rms(a),
        local_min=float(a.min()),
        local_max=float(a.max()),
        trajectory_rms=rms(g),
        weighted_local_rms=c["token_lambda"] * rms(a),
        teacher_topk_mass_mean=float(mass.mean()),
        teacher_topk_hit_rate=float(hit.mean()),
        zero_trajectory_advantage_token_fraction=float((~nonzero).float().mean()),
        negative_trajectory_positive_local_fraction=float(
            ((g < 0) & (a > 0)).float().mean()
        ),
        nonzero_trajectory_sign_flip_rate=float(
            (total[nonzero] * g[nonzero] < 0).float().mean()
        )
        if nonzero.any()
        else None,
    )


def tokenadv_update(
    model, opt, fs, tlps, c, step, micro, check_behavior=True, diagnostics=False
):
    start = time.time()
    model.cuda().train()
    opt_to(opt, "cuda")
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    lr = c["learning_rate"] * min(1, step / max(1, c["warmup_steps"]))
    for group in opt.param_groups:
        group["lr"] = lr
    n = len({(f["item_id"], f["attempt"]) for f in fs})
    nt = sum(len(f["target_positions"]) for f in fs)
    assert n > 0 and nt > 0
    totals = dict(loss=0.0, policy_loss=0.0, kl=0.0, clip_fraction=0.0)
    errors = []
    mb = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().flatten()[:16].clone() for p in trainable]
    components = (
        [[torch.zeros_like(p) for p in trainable] for _ in range(2)]
        if diagnostics
        else None
    )
    for ix in groups(fs, micro, c["padded_token_budget"]):
        rows = [fs[i] for i in ix]
        lp, _, _ = selected_logprobs(model, rows, chunk=c["head_chunk_size"])
        ref = torch.cat([tlps[i] for i in ix]).cuda()
        old = torch.tensor(
            [v for f in rows for v in f["behavior_logprobs"]], device="cuda"
        )
        adv = torch.cat([f["combined_advantage"] for f in rows]).cuda()
        weights = torch.cat(
            [
                torch.full(
                    (len(f["target_positions"]),),
                    1 / (n * f["trajectory_tokens"]),
                    device="cuda",
                )
                for f in rows
            ]
        )
        assert len(lp) == len(ref) == len(old) == len(adv)
        loss, policy, kl, ratio = clipped_objective(
            lp, old, ref, adv, weights, c["clip_epsilon"], c["kl_beta"]
        )
        assert torch.isfinite(loss)
        if diagnostics:
            # These UNCLIPPED component gradients diagnose scale/conflict. The actual
            # optimizer uses clipping on the combined advantage, which is not separable.
            ga = torch.cat(
                [torch.full_like(f["token_advantage"], f["advantage"]) for f in rows]
            ).cuda()
            la = torch.cat([f["token_advantage"] for f in rows]).cuda()
            for bucket, component in zip(
                components,
                [-(ratio * ga * weights).sum(), -(ratio * la * weights).sum()],
            ):
                grads = torch.autograd.grad(
                    component, trainable, retain_graph=True, allow_unused=True
                )
                for accum, g in zip(bucket, grads):
                    if g is not None:
                        accum.add_(g.detach())
                del grads
        loss.backward()
        totals["loss"] += float(loss.detach())
        totals["policy_loss"] += float(policy.detach())
        totals["kl"] += float(kl.detach())
        totals["clip_fraction"] += float(
            ((ratio.detach() - 1).abs().gt(c["clip_epsilon"]) * weights).sum()
        )
        errors.append((lp.detach() - old).abs().cpu())
        mb.append(dict(rows=len(ix), length=max(len(f["input_ids"]) for f in rows)))
    err = torch.cat(errors)
    if check_behavior:
        assert float(err.mean()) < 0.1 and float(err.quantile(0.99)) < 1.0, (
            "behavior alignment failed",
            float(err.mean()),
            float(err.quantile(0.99)),
        )
    diagnostic = {}
    if diagnostics:
        rg, tg = components
        rn = torch.sqrt(sum(g.float().square().sum() for g in rg))
        tn = torch.sqrt(sum(g.float().square().sum() for g in tg))
        dot = sum((a.float() * b.float()).sum() for a, b in zip(rg, tg))
        diagnostic = dict(
            unclipped_grpo_grad_norm=float(rn),
            unclipped_local_grad_norm=float(tn),
            weighted_local_to_grpo_grad_norm=float(c["token_lambda"] * tn / rn)
            if rn > 0
            else None,
            unclipped_gradient_cosine=float(dot / (rn * tn))
            if rn > 0 and tn > 0
            else None,
        )
        assert torch.isfinite(rn) and torch.isfinite(tn)
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    assert torch.isfinite(norm)
    opt.step()
    torch.cuda.synchronize()
    changed = any(
        not torch.equal(b, p.detach().flatten()[:16]) for b, p in zip(before, trainable)
    )
    assert changed
    stats = dict(
        step=step,
        lr=lr,
        **totals,
        **advantage_stats(fs, c),
        **diagnostic,
        grad_norm_before_clip=float(norm),
        parameters_changed=changed,
        trajectories=n,
        tokens=nt,
        segments=len(fs),
        seconds=time.time() - start,
        micro=micro,
        micro_batches=mb,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        behavior_logprob_abs_error_mean=float(err.mean()),
        behavior_logprob_abs_error_p99=float(err.quantile(0.99)),
        behavior_alignment_checked=check_behavior,
    )
    opt.zero_grad(set_to_none=True)
    model.cpu()
    opt_to(opt, "cpu")
    release()
    return stats


def save(model, opt, path, state):
    path = Path(path)
    assert not path.exists()
    tmp = path.with_name(path.name + ".incomplete")
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    torch.save(
        dict(
            optimizer=opt.state_dict(),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),
        ),
        tmp / "training_state.pt",
    )
    write_json(tmp / "state.json", state)
    write_json(
        tmp / "complete.json",
        dict(complete=True, adapter_sha256=digest(tmp / "adapter_model.safetensors")),
    )
    tmp.rename(path)


def audit(es):
    from scripts.sampler import stagnant

    turns = [t for e in es for t in e["trace"]]
    invalid = lambda t: any(
        s in t["observation"].lower()
        for s in [
            "nothing happens",
            "invalid action",
            "no known action",
            "invalid choice",
        ]
    )
    norm = lambda t: " ".join(str(t.get("action", t.get("output", ""))).lower().split())
    repeated_invalid = sum(
        invalid(a) and norm(a) == norm(b)
        for e in es
        for a, b in zip(e["trace"], e["trace"][1:])
    )
    return dict(
        repeated_action_after_invalid_feedback=repeated_invalid,
        tasks=len(es),
        successes=sum(e["success"] for e in es),
        mean_reward=sum(e["reward"] for e in es) / max(1, len(es)),
        mean_turns=sum(e["turns"] for e in es) / max(1, len(es)),
        tokens=sum(e["generated_tokens"] for e in es),
        format_rate=sum(t["format_valid"] for t in turns) / max(1, len(turns)),
        parser_errors=sum(t.get("parse_error", False) for t in turns),
        stagnant_episodes=sum(
            any(stagnant(e["trace"][:i]) for i in range(1, len(e["trace"]) + 1))
            for e in es
        ),
        invalid_feedback_turns=sum(
            any(
                s in t["observation"].lower()
                for s in [
                    "nothing happens",
                    "invalid action",
                    "no known action",
                    "invalid choice",
                ]
            )
            for t in turns
        ),
        turn_limit=sum(e["termination"] == "turn_limit" for e in es),
        context_limit=sum(e["termination"] == "context_limit" for e in es),
    )


def grpo_update(model, opt, fs, tlps, c, step, micro, check_behavior=True):
    start = time.time()
    model.cuda().train()
    opt_to(opt, "cuda")
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    lr = c["learning_rate"] * min(1, step / max(1, c["warmup_steps"]))
    for g in opt.param_groups:
        g["lr"] = lr
    n = len({(f["item_id"], f["attempt"]) for f in fs})
    assert n > 0
    total = 0.0
    kl_total = 0.0
    clipped = 0.0
    errors = []
    mb = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().flatten()[:16].clone() for p in trainable]
    for ix in groups(fs, micro, c["padded_token_budget"]):
        rows = [fs[i] for i in ix]
        lp, _, _ = selected_logprobs(model, rows, chunk=c["head_chunk_size"])
        ref = torch.cat([tlps[i] for i in ix]).cuda()
        old = torch.tensor(
            [v for f in rows for v in f["behavior_logprobs"]], device="cuda"
        )
        adv = torch.cat(
            [
                torch.full((len(f["target_positions"]),), f["advantage"], device="cuda")
                for f in rows
            ]
        )
        weights = torch.cat(
            [
                torch.full(
                    (len(f["target_positions"]),),
                    1 / (n * f["trajectory_tokens"]),
                    device="cuda",
                )
                for f in rows
            ]
        )
        assert len(lp) == len(ref) == len(old) == len(adv)
        ratio = (lp - old).exp()
        delta = ref - lp
        kl = delta.exp() - delta - 1
        obj = torch.minimum(
            ratio * adv, ratio.clamp(1 - c["clip_epsilon"], 1 + c["clip_epsilon"]) * adv
        )
        loss = ((-obj + c["kl_beta"] * kl) * weights).sum()
        assert torch.isfinite(loss)
        loss.backward()
        total += float(loss.detach())
        kl_total += float((kl.detach() * weights).sum())
        clipped += float(
            ((ratio.detach() - 1).abs().gt(c["clip_epsilon"]) * weights).sum()
        )
        errors.append((lp.detach() - old).abs().cpu())
        mb.append(dict(rows=len(ix), length=max(len(f["input_ids"]) for f in rows)))
        del lp, ref, old, adv, weights, ratio, delta, kl, obj, loss
    err = torch.cat(errors)
    if check_behavior:
        assert float(err.mean()) < 0.1 and float(err.quantile(0.99)) < 1.0, (
            "behavior alignment failed",
            float(err.mean()),
            float(err.quantile(0.99)),
        )
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    assert torch.isfinite(norm)
    opt.step()
    torch.cuda.synchronize()
    changed = any(
        not torch.equal(b, p.detach().flatten()[:16]) for b, p in zip(before, trainable)
    )
    assert changed
    stats = dict(
        step=step,
        lr=lr,
        loss=total,
        kl=kl_total,
        clip_fraction=clipped,
        grad_norm_before_clip=float(norm),
        parameters_changed=changed,
        trajectories=n,
        tokens=sum(len(f["target_positions"]) for f in fs),
        segments=len(fs),
        seconds=time.time() - start,
        micro=micro,
        micro_batches=mb,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        behavior_logprob_abs_error_mean=float(err.mean()),
        behavior_logprob_abs_error_p99=float(err.quantile(0.99)),
        behavior_logprob_abs_error_max=float(err.max()),
        behavior_alignment_checked=check_behavior,
    )
    opt.zero_grad(set_to_none=True)
    model.cpu()
    opt_to(opt, "cpu")
    del before, trainable
    release()
    return stats


def opd_update(model, opt, fs, tlps, c, step, micro, check_behavior=True):
    start = time.time()
    model.cuda().train()
    opt_to(opt, "cuda")
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    lr = c["learning_rate"] * min(1, step / max(1, c["warmup_steps"]))
    for g in opt.param_groups:
        g["lr"] = lr
    denom = sum(len(f["target_positions"]) for f in fs)
    total = 0.0
    errors = []
    advs = []
    mb = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().flatten()[:16].clone() for p in trainable]
    for ix in groups(fs, micro, c["padded_token_budget"]):
        rows = [fs[i] for i in ix]
        lp, _, _ = selected_logprobs(model, rows, chunk=c["head_chunk_size"])
        t = torch.cat([tlps[i] for i in ix]).cuda()
        behavior = torch.tensor(
            [v for f in rows for v in f["behavior_logprobs"]], device="cuda"
        )
        assert len(lp) == len(t) == len(behavior)
        adv = (t - lp.detach()).detach()
        loss = -(adv * lp).sum() / denom
        assert torch.isfinite(loss)
        loss.backward()
        total += float(loss.detach())
        errors.append((lp.detach() - behavior).abs().cpu())
        advs.append(adv.cpu())
        mb.append(dict(rows=len(ix), length=max(len(f["input_ids"]) for f in rows)))
        del lp, t, behavior, adv, loss
    err = torch.cat(errors)
    if check_behavior:
        assert float(err.mean()) < 0.1 and float(err.quantile(0.99)) < 1.0, (
            "behavior alignment failed",
            float(err.mean()),
            float(err.quantile(0.99)),
        )
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    assert torch.isfinite(norm)
    opt.step()
    torch.cuda.synchronize()
    changed = any(
        not torch.equal(b, p.detach().flatten()[:16]) for b, p in zip(before, trainable)
    )
    assert changed
    stats = dict(
        behavior_alignment_checked=check_behavior,
        step=step,
        lr=lr,
        loss=total,
        grad_norm_before_clip=float(norm),
        parameters_changed=changed,
        tokens=denom,
        segments=len(fs),
        seconds=time.time() - start,
        micro=micro,
        micro_batches=mb,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        behavior_logprob_abs_error_mean=float(err.mean()),
        behavior_logprob_abs_error_p99=float(err.quantile(0.99)),
        behavior_logprob_abs_error_max=float(err.max()),
        sampled_reverse_kl=float(-torch.cat(advs).mean()),
    )
    opt.zero_grad(set_to_none=True)
    model.cpu()
    opt_to(opt, "cpu")
    del before, trainable
    release()
    return stats


def hybrid_update(
    model, opt, fs, tlps, c, step, micro, check_behavior=True, diagnostics=False
):
    start = time.time()
    model.cuda().train()
    opt_to(opt, "cuda")
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    lr = c["learning_rate"] * min(1, step / max(1, c["warmup_steps"]))
    for g in opt.param_groups:
        g["lr"] = lr
    n = len({(f["item_id"], f["attempt"]) for f in fs})
    nt = sum(len(f["target_positions"]) for f in fs)
    assert n > 0 and nt > 0
    total = 0.0
    kl_total = 0.0
    rl_total = 0.0
    opd_total = 0.0
    clipped = 0.0
    errors = []
    mb = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().flatten()[:16].clone() for p in trainable]
    component_grads = (
        [[torch.zeros_like(p) for p in trainable] for _ in range(2)]
        if diagnostics
        else None
    )
    for ix in groups(fs, micro, c["padded_token_budget"]):
        rows = [fs[i] for i in ix]
        lp, _, _ = selected_logprobs(model, rows, chunk=c["head_chunk_size"])
        ref = torch.cat([tlps[i] for i in ix]).cuda()
        teacher = torch.cat(
            [torch.as_tensor(fs[i]["teacher_logprobs"]) for i in ix]
        ).cuda()
        old = torch.tensor(
            [v for f in rows for v in f["behavior_logprobs"]], device="cuda"
        )
        adv = torch.cat(
            [
                torch.full((len(f["target_positions"]),), f["advantage"], device="cuda")
                for f in rows
            ]
        )
        weights = torch.cat(
            [
                torch.full(
                    (len(f["target_positions"]),),
                    1 / (n * f["trajectory_tokens"]),
                    device="cuda",
                )
                for f in rows
            ]
        )
        assert len(lp) == len(ref) == len(old) == len(adv) == len(teacher)
        ratio = (lp - old).exp()
        delta = ref - lp
        kl = delta.exp() - delta - 1
        obj = torch.minimum(
            ratio * adv, ratio.clamp(1 - c["clip_epsilon"], 1 + c["clip_epsilon"]) * adv
        )
        rl = -(obj * weights).sum()
        ref_loss = (kl * weights).sum()
        token_adv = (teacher - lp.detach()).detach()
        opd = -(token_adv * lp).sum() / nt
        loss = rl + c["opd_weight"] * opd + c["kl_beta"] * ref_loss
        assert torch.isfinite(loss)
        if diagnostics:
            for bucket, component in zip(component_grads, [rl, opd]):
                grads = torch.autograd.grad(
                    component, trainable, retain_graph=True, allow_unused=True
                )
                for accum, g in zip(bucket, grads):
                    if g is not None:
                        accum.add_(g.detach())
                del grads
        loss.backward()
        total += float(loss.detach())
        rl_total += float(rl.detach())
        opd_total += float(opd.detach())
        kl_total += float(ref_loss.detach())
        clipped += float(
            ((ratio.detach() - 1).abs().gt(c["clip_epsilon"]) * weights).sum()
        )
        errors.append((lp.detach() - old).abs().cpu())
        mb.append(dict(rows=len(ix), length=max(len(f["input_ids"]) for f in rows)))
        del (
            lp,
            ref,
            teacher,
            old,
            adv,
            weights,
            ratio,
            delta,
            kl,
            obj,
            token_adv,
            rl,
            opd,
            ref_loss,
            loss,
        )
    err = torch.cat(errors)
    if check_behavior:
        assert float(err.mean()) < 0.1 and float(err.quantile(0.99)) < 1.0, (
            "behavior alignment failed",
            float(err.mean()),
            float(err.quantile(0.99)),
        )
    diagnostic = {}
    if diagnostics:
        rg, og = component_grads
        rn = torch.sqrt(sum((g.float() ** 2).sum() for g in rg))
        on = torch.sqrt(sum((g.float() ** 2).sum() for g in og))
        dot = sum((a.float() * b.float()).sum() for a, b in zip(rg, og))
        diagnostic = dict(
            grpo_grad_norm=float(rn),
            opd_grad_norm=float(on),
            weighted_opd_grad_norm=float(c["opd_weight"] * on),
            gradient_cosine=float(dot / (rn * on)) if rn > 0 and on > 0 else None,
            weighted_opd_to_grpo_norm=float(c["opd_weight"] * on / rn)
            if rn > 0
            else None,
        )
        assert torch.isfinite(rn) and torch.isfinite(on)
        del rg, og, rn, on, dot
    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    assert torch.isfinite(norm)
    opt.step()
    torch.cuda.synchronize()
    changed = any(
        not torch.equal(b, p.detach().flatten()[:16]) for b, p in zip(before, trainable)
    )
    assert changed
    stats = dict(
        step=step,
        lr=lr,
        loss=total,
        grpo_policy_loss=rl_total,
        opd_loss=opd_total,
        opd_weight=c["opd_weight"],
        kl=kl_total,
        clip_fraction=clipped,
        grad_norm_before_clip=float(norm),
        parameters_changed=changed,
        trajectories=n,
        tokens=nt,
        segments=len(fs),
        seconds=time.time() - start,
        micro=micro,
        micro_batches=mb,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        behavior_logprob_abs_error_mean=float(err.mean()),
        behavior_logprob_abs_error_p99=float(err.quantile(0.99)),
        behavior_logprob_abs_error_max=float(err.max()),
        behavior_alignment_checked=check_behavior,
        **diagnostic,
    )
    opt.zero_grad(set_to_none=True)
    model.cpu()
    opt_to(opt, "cpu")
    del before, trainable, component_grads
    release()
    return stats
