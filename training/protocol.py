"""One observation-preserving serializer for SFT, dev, online learning and test."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "qwen3_agent_nonthinking_v1"
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END = "<|im_end|>"
PHASES = {"sft", "dev", "online", "test"}


def normalize_messages(messages):
    result = []
    for m in messages:
        role = m.get("role") or {"human": "user", "gpt": "assistant"}.get(m.get("from"))
        assert role in {"system", "user", "assistant"}, role
        content = m["content"] if "content" in m else m["value"]
        assert isinstance(content, str)
        result.append(dict(role=role, content=content, loss=bool(m.get("loss", False))))
    return result


def render_message(message):
    m = normalize_messages([message])[0]
    prefix = (
        ASSISTANT_PREFIX if m["role"] == "assistant" else f"<|im_start|>{m['role']}\n"
    )
    return prefix + m["content"] + END + "\n"


def render(messages, add_generation_prompt=False):
    text = "".join(render_message(m) for m in messages)
    if add_generation_prompt:
        assert messages and normalize_messages(messages)[-1]["role"] == "user"
        text += ASSISTANT_PREFIX
    return text


def prompt_ids(tokenizer, messages, phase="online"):
    assert phase in PHASES
    # Phase never changes content or formatting.
    return tokenizer.encode(render(messages, True), add_special_tokens=False)


def tokenize_trajectory(tokenizer, row, max_length=4096):
    messages = normalize_messages(row["messages"])
    text = render(messages)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    labels = [-100] * len(ids)
    cursor = 0
    spans = []
    for i, m in enumerate(messages):
        rendered = render_message(m)
        if m["loss"]:
            assert m["role"] == "assistant"
            start = cursor + len(ASSISTANT_PREFIX)
            end = (
                cursor + len(rendered) - 1
            )  # Include im_end, exclude separator newline.
            spans.append((i, start, end))
        cursor += len(rendered)
    for i, start, end in spans:
        selected = []
        for j, (a, b) in enumerate(offsets):
            if b > start and a < end:
                assert a >= start and b <= end, (
                    "target token crosses boundary",
                    row["item_id"],
                    i,
                )
                labels[j] = ids[j]
                selected.append(j)
        assert selected and ids[selected[-1]] == tokenizer.convert_tokens_to_ids(END)
        # Exact prefix equality across every supervised turn, not only a sample.
        expected = prompt_ids(tokenizer, messages[:i], phase="sft")
        assert expected == ids[: selected[0]], (row["item_id"], i, "prefix mismatch")
    assert len(ids) <= max_length, (row["item_id"], len(ids), max_length)
    assert any(x != -100 for x in labels)
    return dict(
        input_ids=ids,
        attention_mask=[1] * len(ids),
        labels=labels,
        item_id=row["item_id"],
        length=len(ids),
    )


def initial_messages(env, observation):
    frozen = json.loads((ROOT / "configs/input_protocol.json").read_text())[
        "benchmarks"
    ][env]
    return [
        dict(role="user", content=frozen["prompt"], loss=False),
        dict(role="assistant", content=frozen["acknowledgement"], loss=False),
        dict(role="user", content=observation, loss=False),
    ]


def format_valid(env, output):
    if len(re.findall(r"(?m)^\s*Action:", output)) != 1:
        return False
    if len(re.findall(r"(?m)^\s*Thought:", output)) > 1:
        return False
    pattern = (
        r"\s*(?:Thought:\s*\S[\s\S]*?\s*)?Action:\s*(\S[^\r\n]*?)\s*"
        if env == "alfworld"
        else r"\s*Thought:\s*\S[\s\S]*?\s*Action:\s*(\S[^\r\n]*?)\s*"
    )
    match = re.fullmatch(pattern, output)
    if not match:
        return False
    if env == "webshop":
        action = re.fullmatch(r"(search|click)\[([^\r\n]+)\]", match[1])
        return bool(action and action[2].strip())
    return True


def protocol_hash():
    return hashlib.sha256(
        (ROOT / "configs/input_protocol.json").read_bytes()
    ).hexdigest()
