"""Shared generation entrypoint for dev rollouts, GRPO/OPD collectors and test.

Existing clients provide observations; no dynamic action lists are added here.
RL optimizers should call build_policy_input, rather than native chat templates.
"""

from .protocol import initial_messages, normalize_messages, prompt_ids


def build_policy_input(tokenizer, messages, phase):
    ids = prompt_ids(tokenizer, messages, phase=phase)
    return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class RolloutSession:
    def __init__(self, env, client, tokenizer, phase="online"):
        assert phase in {"dev", "online", "test"}
        self.env, self.client, self.tokenizer, self.phase = (
            env,
            client,
            tokenizer,
            phase,
        )
        self.messages = []

    def reset(self, task_index):
        self.client.reset(task_index)
        self.messages = initial_messages(self.env, self.client.observe())
        return self.model_input()

    def model_input(self):
        return build_policy_input(self.tokenizer, self.messages, self.phase)

    def step(self, output):
        self.messages.append(dict(role="assistant", content=output, loss=True))
        result = self.client.step(output)
        self.messages.append(dict(role="user", content=result.state, loss=False))
        return result
