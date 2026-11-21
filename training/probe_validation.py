"""Compare probabilities on exact prefixes; permit only verified near-tie argmax changes."""

import torch


def tie_candidates(a, b, index, tolerance=0.2):
    x, y = a["ids"][index], b["ids"][index]
    for out in (a, b):
        p = out["top_logprobs"][index]
        assert str(x) in p and str(y) in p, "Divergent token outside top candidates"
        assert (
            max(p.values()) - p[str(x)] <= tolerance
            and max(p.values()) - p[str(y)] <= tolerance
        ), "Not a near-tie"
    return x, y


def parity(model, probe, score):
    prefix = probe["prompt_ids"]
    a, b = probe["outputs"]
    rows = []
    for out in (a, b):
        rows.append(
            dict(
                input_ids=prefix + out["ids"],
                target_positions=list(
                    range(len(prefix), len(prefix) + len(out["ids"]))
                ),
            )
        )
    lps, _ = score(model, rows, 1)
    errors = [(p - torch.tensor(out["logprobs"])).abs() for p, out in zip(lps, (a, b))]
    # Keep the existing HF/vLLM accuracy gates on BOTH complete branches.
    for error in errors:
        assert float(error.mean()) < 0.1 and float(error.max()) < 1.0, (
            "HF/vLLM alignment failed"
        )
    divergence = next(
        (i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None
    )
    if divergence is None:
        assert a["ids"] == b["ids"]
    common = len(a["ids"]) if divergence is None else divergence
    repeat = (
        torch.tensor(a["logprobs"][:common]) - torch.tensor(b["logprobs"][:common])
    ).abs()
    assert not len(repeat) or float(repeat.mean()) < 0.05, (
        "Common-prefix probability mismatch"
    )
    margin = None
    if divergence is not None:
        x, y = tie_candidates(a, b, divergence)
        shared = prefix + a["ids"][:divergence]
        candidates = [
            dict(input_ids=shared + [t], target_positions=[len(shared)]) for t in (x, y)
        ]
        p, _ = score(model, candidates, 1)
        margin = float((p[0] - p[1]).abs().item())
        assert margin <= 0.2, ("HF does not confirm a near-tie", margin)
    return dict(
        mean=max(float(e.mean()) for e in errors),
        max=max(float(e.max()) for e in errors),
        cold_warm_mean=float(repeat.mean()) if len(repeat) else 0.0,
        cold_warm_max=float(repeat.max()) if len(repeat) else 0.0,
        same_tokens=a["ids"] == b["ids"],
        first_divergence=divergence,
        hf_candidate_logprob_margin=margin,
        near_tie_verified=divergence is not None,
        both_branches_hf_verified=True,
    )


if __name__ == "__main__":
    a = dict(ids=[1], top_logprobs=[{"1": -0.70, "2": -0.71}])
    b = dict(ids=[2], top_logprobs=[{"1": -0.72, "2": -0.69}])
    assert tie_candidates(a, b, 0) == (1, 2)
    for bad in ({"1": -0.7, "2": -2.0}, {"1": -0.7}):
        try:
            tie_candidates(a, {**b, "top_logprobs": [bad]}, 0)
        except AssertionError:
            pass
        else:
            raise AssertionError("Large mismatch accepted")
    print("Near-tie gate tests passed; missing and distant candidates rejected.")
