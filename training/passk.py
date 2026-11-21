"""Pure task-level accounting for serial, success-stopped pass@k attempts."""

import hashlib, math


def sample_seed(benchmark, item_id, attempt, turn, base_seed=42):
    key = f"{base_seed}|{benchmark}|{item_id}|{attempt}|{turn}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**31 - 1)


def attempt_summary(record):
    return {
        k: record[k]
        for k in [
            "attempt",
            "success",
            "final_score",
            "final_reward",
            "termination",
            "turns",
            "generated_tokens",
        ]
    }


def task_complete(attempts, k):
    assert [a["attempt"] for a in attempts] == list(range(1, len(attempts) + 1))
    assert 0 <= len(attempts) <= k
    assert not any(a["success"] for a in attempts[:-1]), (
        "Attempt performed after an earlier success"
    )
    return bool(attempts and (attempts[-1]["success"] or len(attempts) == k))


def task_summary(item_id, attempts, k):
    assert task_complete(attempts, k)
    first_success = next((a["attempt"] for a in attempts if a["success"]), None)
    return dict(
        item_id=item_id,
        success=first_success is not None,
        first_success_attempt=first_success,
        attempts_used=len(attempts),
        attempts=attempts,
    )


def wilson(wins, n):
    z = 1.959963984540054
    den = 1 + z * z / n
    center = (wins / n + z * z / (2 * n)) / den
    half = z * math.sqrt(wins / n * (1 - wins / n) / n + z * z / (4 * n * n)) / den
    return [max(0, center - half), min(1, center + half)]


def aggregate_tasks(records, k):
    assert records
    assert len({r["item_id"] for r in records}) == len(records)
    for r in records:
        assert (
            task_summary(r["item_id"], r["attempts"], k)["first_success_attempt"]
            == r["first_success_attempt"]
        )
    n = len(records)
    wins = sum(r["success"] for r in records)
    curve = {
        str(i): sum(
            r["first_success_attempt"] is not None and r["first_success_attempt"] <= i
            for r in records
        )
        / n
        for i in range(1, k + 1)
    }
    attempts = sum(r["attempts_used"] for r in records)
    return dict(
        tasks=n,
        successes=wins,
        pass_at_k=wins / n,
        pass_at_k_wilson95=wilson(wins, n),
        pass_curve=curve,
        sampled_pass_at_1=curve["1"],
        attempts_executed=attempts,
        attempts_saved_by_early_stop=n * k - attempts,
        mean_attempts=attempts / n,
        first_attempt_mean_reward=sum(r["attempts"][0]["final_reward"] for r in records)
        / n,
        best_observed_reward_early_stopped_mean=sum(
            max(a["final_reward"] for a in r["attempts"]) for r in records
        )
        / n,
    )
