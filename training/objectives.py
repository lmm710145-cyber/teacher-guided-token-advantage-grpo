"""Teacher probability rewards; deliberately NOT log-ratio OPD.

All distributions have temperature one. Top-K is a support mask, not a
renormalization. The baseline and advantages are fixed before optimizer.step.
"""

import torch


@torch.no_grad()
def teacher_support(logits, sampled_ids, k):
    if not 1 <= k <= logits.shape[-1]:
        raise ValueError("top_k must be within the vocabulary")
    logits = logits.float()
    values, ids = logits.topk(k, dim=-1)
    probabilities = (values - logits.logsumexp(-1, keepdim=True)).exp()
    hit = ids.eq(sampled_ids[:, None])
    reward = (probabilities * hit).sum(-1)
    return ids, probabilities, reward, hit.any(-1)


@torch.no_grad()
def student_baseline(logits, top_ids, teacher_probabilities):
    logits = logits.float()
    student_probabilities = (
        logits.gather(-1, top_ids.long()) - logits.logsumexp(-1, keepdim=True)
    ).exp()
    return (student_probabilities * teacher_probabilities).sum(-1)


def combine_advantages(trajectory_advantage, reward, baseline, token_lambda):
    if token_lambda < 0:
        raise ValueError("token_lambda must be nonnegative")
    local = (reward - baseline).detach()
    combined = trajectory_advantage.detach() + token_lambda * local
    return combined.detach(), local


def clipped_objective(
    new_logp, old_logp, reference_logp, advantage, weights, epsilon=0.2, beta=0.01
):
    """Per-trajectory token mean, then mean over trajectories, including zeros."""
    ratio = (new_logp - old_logp.detach()).exp()
    advantage = advantage.detach()
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    )
    policy = -(surrogate * weights).sum()
    delta = reference_logp.detach() - new_logp
    kl = ((delta.exp() - delta - 1) * weights).sum()
    return policy + beta * kl, policy, kl, ratio
