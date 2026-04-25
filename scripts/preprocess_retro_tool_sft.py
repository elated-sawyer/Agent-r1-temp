#!/usr/bin/env python3
"""Convert pickled retrosynthesis rollouts to LLaMA-Factory ShareGPT JSONL.

Primary input: ``data/chunk_sft.pkl`` (or ``data/chunk_sft_tagged_dedup.pkl``).
These are pickles with envelope ``{"sft_records": [...]}``; each record has
``{"id", "conversations", "meta", ...}``.

Legacy inputs (``.jsonl`` / ``.json`` / ``.txt``) are still accepted so hand-
crafted debug samples keep working.

Example:
    python scripts/preprocess_retro_tool_sft.py \\
        --input data/chunk_sft.pkl \\
        --output data/retro_tool_sft.jsonl \\
        --filter-mode drop-failed-pairs \\
        --success-only \\
        --skip-invalid
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:  # Chunk pickles upstream sometimes nest torch tensors; import is harmless.
    import torch  # noqa: F401
except ImportError:
    pass


SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "single_step_retro", "description": "Perform single step retrosynthesis for a molecule and return several possible reactions to synthesize it. Note that the reactions may be incorrect and each reactant is marked as 'available' or 'unavailable'. The unavailable molecules have to be synthesized further.", "parameters": {"type": "object", "properties": {"molecule": {"type": "string", "description": "The ID for the molecule to be synthesized. For example, 0-0 ."}}, "required": ["molecule"]}}
{"name": "select_reaction", "description": "Given several reactions to synthesize a molecule, use this tool to select one from them.", "parameters": {"type": "object", "properties": {"reaction": {"type": "string", "description": "The ID for the selected reaction. For example, 0-0-0 ."}}, "required": ["reaction"]}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


VALID_TOOL_NAMES = ("single_step_retro", "select_reaction")
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"(' + "|".join(VALID_TOOL_NAMES) + r')"')
INVALID_TOOL_MSG_PREFIX = "Invalid tool call format"


# ---------- IO helpers ----------

def _parse_obj(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def load_raw_examples(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "sft_records" not in payload:
            raise ValueError(f"{path}: expected dict with 'sft_records'")
        records = payload["sft_records"]
        if not isinstance(records, list):
            raise TypeError(f"{path}: sft_records must be a list")
        return records

    if suffix == ".jsonl":
        examples: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(_parse_obj(line))
        return examples

    text = path.read_text(encoding="utf-8").strip()

    if suffix == ".json":
        obj = _parse_obj(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        raise ValueError(f"{path}: unsupported JSON root type")

    # .txt / unknown: try whole-file parse first, then split on blank lines.
    try:
        obj = _parse_obj(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass

    chunks = re.split(r"\n\s*\n", text)
    return [_parse_obj(c) for c in chunks if c.strip()]


# ---------- Normalization helpers ----------

def remove_special_tokens(text: str) -> str:
    return (
        text.replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .strip()
    )


def fix_assistant_content(text: str) -> str:
    """Strip ChatML markers and ensure ``<think>...</think>`` wraps the tool_call.

    Returns the normalized string, which may be empty if the raw turn contained
    only special tokens (a failed rollout step). The caller decides whether to
    drop the turn or substitute a placeholder.
    """
    text = remove_special_tokens(text)
    if not text:
        return ""
    if "<think>" in text and "</think>" not in text and "<tool_call>" in text:
        text = text.replace("<tool_call>", "</think>\n<tool_call>", 1)
    if "<tool_call>" in text and "<think>" not in text:
        text = "<think>\n\n</think>\n" + text
    return text.strip()


def is_assistant_failed(assistant_value: str, tool_value: str) -> bool:
    """A pair is a failed rollout step if the assistant is empty or the tool
    response is the format-error message."""
    if not assistant_value.strip():
        return True
    if tool_value.startswith(INVALID_TOOL_MSG_PREFIX):
        return True
    return False


# ---------- Per-record conversion ----------

def convert_example(
    raw: Dict[str, Any],
    filter_mode: str,
    success_only: bool,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if "conversations" not in raw:
        raise ValueError("missing 'conversations' field")

    meta = raw.get("meta", {})
    if success_only and meta.get("success") is not True:
        raise ValueError("skip: non-success trajectory")

    stats = {"empty_assistant": 0, "invalid_tool": 0}

    convs = raw["conversations"]
    user_msg: str | None = None
    pairs: List[Tuple[str, str]] = []

    i = 0
    while i < len(convs):
        role = convs[i].get("role")
        content = convs[i].get("content", "")

        if role == "system":
            i += 1
            continue
        if role == "user":
            user_msg = remove_special_tokens(content)
            i += 1
            continue
        if role == "assistant":
            a_raw = content
            t_raw = ""
            if i + 1 < len(convs) and convs[i + 1].get("role") == "tool":
                t_raw = convs[i + 1].get("content", "")
                i += 2
            else:
                i += 1
            pairs.append((a_raw, t_raw))
            continue
        if role == "tool":
            # Orphan tool message (no preceding assistant). Not observed in the
            # current corpus but we skip defensively.
            i += 1
            continue
        raise ValueError(f"unknown role: {role}")

    if user_msg is None or not user_msg:
        raise ValueError("no user message found")

    kept_pairs: List[Tuple[str, str]] = []
    for a_raw, t_raw in pairs:
        a_norm = fix_assistant_content(a_raw)
        t_norm = remove_special_tokens(t_raw)

        failed = is_assistant_failed(a_norm, t_norm)
        if failed:
            if not a_norm.strip():
                stats["empty_assistant"] += 1
            else:
                stats["invalid_tool"] += 1

            if filter_mode == "drop-failed-pairs":
                continue
            if filter_mode == "drop-conversation":
                raise ValueError("skip: conversation contains failed pair")
            # keep-all: ensure the empty-assistant case still has a parseable value.
            if not a_norm.strip():
                a_norm = "<think>\n\n</think>"

        kept_pairs.append((a_norm, t_norm))

    conversations: List[Dict[str, str]] = [{"from": "human", "value": user_msg}]
    for a_norm, t_norm in kept_pairs:
        conversations.append({"from": "gpt", "value": a_norm})
        conversations.append({"from": "observation", "value": t_norm})

    while conversations and conversations[-1]["from"] == "observation":
        conversations.pop()

    out = {
        "system": SYSTEM_PROMPT,
        "conversations": conversations,
        "id": raw.get("id"),
        "meta": meta,
    }
    return out, stats


def validate_example(example: Dict[str, Any], index: int) -> None:
    conv = example.get("conversations", [])
    if not conv:
        raise ValueError(f"example {index}: empty conversations")
    if conv[0]["from"] != "human":
        raise ValueError(f"example {index}: does not start with human")
    if conv[-1]["from"] != "gpt":
        raise ValueError(f"example {index}: does not end with gpt")

    valid_roles = {"human", "gpt", "observation"}
    for k, m in enumerate(conv):
        role = m.get("from")
        value = m.get("value")
        if role not in valid_roles:
            raise ValueError(f"example {index}, turn {k}: invalid from={role}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"example {index}, turn {k}: empty value")
        if "<|im_start|>" in value or "<|im_end|>" in value:
            raise ValueError(f"example {index}, turn {k}: ChatML token leaked")
        if role == "gpt":
            if "<tool_call>" in value and "</tool_call>" not in value:
                raise ValueError(f"example {index}, turn {k}: unclosed tool_call")
            if "<think>" in value and "</think>" not in value:
                raise ValueError(f"example {index}, turn {k}: unclosed think")


# ---------- Main ----------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Path to raw pickle (or legacy JSONL/JSON/TXT)")
    parser.add_argument("--output", required=True, help="Path to processed JSONL output")
    parser.add_argument("--success-only", action="store_true", help="Keep only meta.success == True examples")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid examples instead of raising")
    parser.add_argument(
        "--filter-mode",
        choices=("keep-all", "drop-failed-pairs", "drop-conversation"),
        default="drop-failed-pairs",
        help="How to handle (assistant, tool) pairs where the assistant is empty or the tool reports 'Invalid tool call format'.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="If set, skip records whose processed conversations length exceeds this.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_examples = load_raw_examples(input_path)

    n_total = len(raw_examples)
    n_written = 0
    n_skipped = 0
    n_too_long = 0
    empty_dropped = 0
    invalid_dropped = 0
    pre_filter_pair_counts: List[int] = []
    post_filter_turn_counts: List[int] = []
    tool_name_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}

    with output_path.open("w", encoding="utf-8") as fout:
        for i, raw in enumerate(raw_examples):
            # Track pre-filter pair counts for stats (assistants / 2 is a fine proxy).
            if isinstance(raw, dict) and isinstance(raw.get("conversations"), list):
                n_assistant = sum(1 for m in raw["conversations"] if m.get("role") == "assistant")
                pre_filter_pair_counts.append(n_assistant)
                for tag in raw.get("_merge_source_tags", []) or []:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            try:
                example, stats = convert_example(raw, args.filter_mode, args.success_only)
                validate_example(example, i)
            except Exception as e:  # noqa: BLE001
                if args.skip_invalid:
                    n_skipped += 1
                    if n_skipped <= 5:
                        print(f"[skip] example {i}: {e}")
                    continue
                raise

            if args.max_turns is not None and len(example["conversations"]) > args.max_turns:
                n_too_long += 1
                n_skipped += 1
                continue

            empty_dropped += stats["empty_assistant"]
            invalid_dropped += stats["invalid_tool"]
            post_filter_turn_counts.append(len(example["conversations"]))

            for m in example["conversations"]:
                if m["from"] == "gpt":
                    match = TOOL_NAME_RE.search(m["value"])
                    if match:
                        name = match.group(1)
                        tool_name_counts[name] = tool_name_counts.get(name, 0) + 1

            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_written += 1

    def summ(xs: List[int]) -> str:
        if not xs:
            return "0 / 0.00 / 0"
        return f"{min(xs)} / {statistics.mean(xs):.2f} / {max(xs)}"

    print()
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Filter mode: {args.filter_mode}")
    print(f"Success-only: {args.success_only}")
    print(f"Max turns cap: {args.max_turns}")
    print("-" * 60)
    print(f"Total raw records:       {n_total}")
    print(f"Written records:         {n_written}")
    print(f"Skipped records:         {n_skipped} (of which too-long: {n_too_long})")
    print(f"Empty-assistant pairs:   {empty_dropped}")
    print(f"Invalid-tool-fmt pairs:  {invalid_dropped}")
    print(f"Pre-filter assistant turns per record (min/mean/max):  {summ(pre_filter_pair_counts)}")
    print(f"Post-filter conv length (min/mean/max):                {summ(post_filter_turn_counts)}")
    print(f"Tool names in output:    {tool_name_counts}")
    if tag_counts:
        print(f"Source-tag distribution: {tag_counts}")
    print("=" * 60)


if __name__ == "__main__":
    main()
