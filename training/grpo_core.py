"""Synchronous trajectory GRPO. No vocabulary projection at observation positions."""

import hashlib
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_advantages(rewards, epsilon=1e-6):
    assert len(rewards) == 4 and all(math.isfinite(r) for r in rewards)
    if max(rewards) == min(rewards):
        return None
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + epsilon) for r in rewards]


def collate(features, device="cuda"):
    length = max(len(f["input_ids"]) for f in features)
    # Pad at most to a multiple of eight, respecting the configured cutoff.
    x = torch.full((len(features), length), 151643, dtype=torch.long, device=device)
    attention = torch.zeros_like(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i, f in enumerate(features):
        n = len(f["input_ids"])
        x[i, :n] = torch.tensor(f["input_ids"], device=device)
        attention[i, :n] = 1
        mask[i, f["target_positions"]] = True
    assert not mask[:, 0].any()
    return x, attention, mask


def selected_logprobs(model, features, chunk=32, entropy=False):
    """Run transformer on full causal context; LM head only on action-output predictors.

    Chunked head is activation-checkpointed, so full-vocabulary tensors are not
    retained across chunks/backward. Only selected-token scalar logprobs persist.
    """
    x, attention, mask = collate(features, next(model.parameters()).device)
    with torch.autocast(
        device_type=x.device.type, dtype=torch.bfloat16, enabled=x.device.type == "cuda"
    ):
        hidden = (
            model.get_base_model()
            .model(
                input_ids=x, attention_mask=attention, use_cache=False, return_dict=True
            )
            .last_hidden_state
        )
        prediction_mask = mask[:, 1:]
        selected = hidden[:, :-1][prediction_mask]
        targets = x[:, 1:][prediction_mask]
        owners = torch.arange(len(features), device=x.device)[:, None].expand_as(
            prediction_mask
        )[prediction_mask]

        def project(h, y):
            logits = F.linear(h, model.get_output_embeddings().weight).float()
            logp = logits.log_softmax(-1)
            chosen = logp.gather(-1, y[:, None]).squeeze(-1)
            # Entropy is a metric, not an additional gradient term.
            ent = (
                -(logp.detach().exp() * logp.detach()).sum(-1)
                if entropy
                else chosen.detach() * 0
            )
            return chosen, ent

        logps = []
        entropies = []
        for start in range(0, len(targets), chunk):
            args = (selected[start : start + chunk], targets[start : start + chunk])
            if torch.is_grad_enabled():
                lp, en = checkpoint(project, *args, use_reentrant=False)
            else:
                lp, en = project(*args)
            logps.append(lp)
            entropies.append(en)
    return torch.cat(logps), torch.cat(entropies), owners


def pack_episode(episode):
    """Pack compatible turns; retain non-roundtripping turns as exact-input segments.

    Decoding and retokenizing model output is not bijective. The environment's
    next prompt is still built by the shared serializer. For an incompatible
    turn, evaluate its originally sampled prefix/output separately, with the
    same trajectory advantage and total generated-token denominator.
    """
    turns = episode["trace"]
    assert turns, "No generated tokens in episode"
    last = turns[-1]
    ids = last["prompt_ids"] + last["output_ids"]
    positions = []
    behavior = []
    action_mask = []
    extras = []
    for t in turns:
        start = len(t["prompt_ids"])
        end = start + len(t["output_ids"])
        if ids[:start] != t["prompt_ids"] or ids[start:end] != t["output_ids"]:
            extras.append(
                dict(
                    input_ids=t["prompt_ids"] + t["output_ids"],
                    target_positions=list(range(start, end)),
                    behavior_logprobs=t["behavior_logprobs"],
                    action_token_mask=t.get(
                        "action_token_mask", [False] * len(t["output_ids"])
                    ),
                    advantage=episode.get("advantage", 0),
                    item_id=episode["item_id"],
                    attempt=episode["attempt"],
                )
            )
            continue
        positions.extend(range(start, end))
        behavior.extend(t["behavior_logprobs"])
        action_mask.extend(t.get("action_token_mask", [False] * len(t["output_ids"])))
    assert len(set(positions)) == len(positions) and len(positions) == len(behavior)
    total = sum(len(t["output_ids"]) for t in turns)
    assert len(positions) + sum(len(f["target_positions"]) for f in extras) == total
    return dict(
        input_ids=ids,
        target_positions=positions,
        behavior_logprobs=behavior,
        action_token_mask=action_mask,
        advantage=episode.get("advantage", 0),
        item_id=episode["item_id"],
        attempt=episode["attempt"],
        extra_segments=extras,
        trajectory_tokens=total,
    )


def feature_segments(features):
    rows = []
    for f in features:
        pieces = [f] + f.get("extra_segments", [])
        total = sum(len(p["target_positions"]) for p in pieces)
        for p in pieces:
            rows.append({**p, "trajectory_tokens": total})
    return rows


def grpo_loss(
    new,
    old,
    reference,
    advantages,
    owners,
    counts,
    batch_trajectories,
    beta=0.01,
    clip=0.2,
):
    ratio = (new - old).exp()
    clipped = ratio.clamp(1 - clip, 1 + clip)
    objective = torch.minimum(ratio * advantages, clipped * advantages)
    delta = reference - new
    kl = delta.exp() - delta - 1
    # Original GRPO: mean over generated tokens per trajectory, then trajectories.
    weights = 1 / counts[owners] / batch_trajectories
    loss = ((-objective + beta * kl) * weights).sum()
    return loss, dict(
        kl=float((kl.detach() * weights).sum()),
        clip_fraction=float((((ratio.detach() - 1).abs() > 0.2) * weights).sum()),
    )
