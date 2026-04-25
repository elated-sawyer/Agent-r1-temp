# Task: SFT an LLM with Multi-turn Tool-calling Data Using LLaMA-Factory

We need to fine-tune an LLM using LLaMA-Factory on multi-turn tool-calling trajectories from a retrosynthesis agent task.

The task is completed in two stages:

1. **Preprocess the raw pickled rollout data** → ShareGPT JSONL
2. **Launch SFT training with the processed data**

The raw data is **pickled rollout records** from the `Agent-r1-temp` retrosynthesis environment — not JSONL, not JSON text, and not rendered ChatML strings. Each record has structured `conversations` with explicit `role` fields, plus `id` and `meta`.

---

# Step 1: Preprocess the Raw Data

## 1.1 Input Files and On-disk Format

Two pickle files are available under `Agent-r1-temp/data/`:

| Path | Records | Notes |
| ---- | ------- | ----- |
| `data/chunk_sft.pkl` | 11,436 | Plain merge of all shards. Each record has `id`, `conversations`, `meta`. |
| `data/chunk_sft_tagged_dedup.pkl` | 11,436 | Same records, but deduped-by-content and with merge provenance fields (`_merge_sources`, `_merge_source_dirs`, `_merge_source_tags`). Tag distribution: 9,676 `noloop-True` + 1,760 `noloop-False`. 448 ids appear twice with *different* content (the dedupe was by content, not id, so both variants are retained). |

**Pickle envelope.** Top-level payload is a `dict`:

```python
{
  "sft_records": [record, record, ...],
  "merge_info": {...}        # only present in chunk_sft_tagged_dedup.pkl
}
```

**Per-record shape.** Each entry in `sft_records` is:

```python
{
  "id": "train_h4_10__q000000__r0__turns8",
  "conversations": [
    {"role": "system", "content": ""},
    {"role": "user", "content": "You are a professional organic chemist ..."},
    {"role": "assistant", "content": "\n\n<tool_call>\n{...}\n</tool_call><|im_end|>"},
    {"role": "tool", "content": "Reaction state 0-0: Possible reactions ..."},
    ...
  ],
  "meta": {
    "abs_query_idx": 0,
    "api_model": "A1-preview",
    "data_source": "reaction_pathway_search",
    "force_noloop": True,
    "model_path": "/mnt/shared-storage-user/wangzifu/cache/model/Qwen3.5-0.8B",
    "n_turns": 8,
    "rollout_n": 8,
    "rollout_rank": 0,
    "rollout_temp": 0.7,
    "success": True
  },
  # Only in chunk_sft_tagged_dedup.pkl:
  "_merge_sources": [{"source_dir": "...", "source_tag": "noloop-True", "chunk_file": "chunk_00000.pkl"}],
  "_merge_source_dirs": ["..."],
  "_merge_source_tags": ["noloop-True"]
}
```

**Loading notes.**

- Use `pickle.load`. **Do not** try `json.loads` / `ast.literal_eval` — these files are true pickles (~110 MB each).
- Defensively `import torch` before `pickle.load`, because the upstream chunk pickles the rollouts were merged from contained tensors; this envelope has already been re-pickled but the habit is harmless and matches the existing `scripts/merge_chunk_sft*.py`.
- If the user ever supplies a JSONL/JSON/TXT file (e.g., a hand-crafted debug sample), the script should still accept it as a legacy path.

## 1.2 Empirical Observations on the Data

These were verified by inspecting `data/chunk_sft.pkl`. The preprocessing must handle all of them:

1. **Pre-filtered by success.** All 11,436 records have `meta.success == True`. So `--success-only` is a no-op on this corpus but should remain available for future inputs.
2. **First message is always empty `system`, last message is always `tool`.** 11,436/11,436 in both respects.
3. **Roles alternate perfectly** as `system → user → (assistant → tool) × N`. There are exactly 143,467 assistant turns and 143,467 tool turns across the corpus.
4. **Assistant messages use `<|im_end|>` as trailing token.** `<|im_start|>` does not appear.
5. **Only 9 of 143,467 assistant turns already contain `<think>`.** Essentially every assistant turn must be normalized by prepending an empty `<think>\n\n</think>\n` block.
6. **~21% of tool turns are format-error responses.** 30,439 tool messages start with `"Invalid tool call format. Please use the format:"` — these are environment errors produced when the model emitted a malformed tool call the preceding turn.
7. **Malformed assistant tool-calls come in several shapes:**
   - `<function=single_step_retro>\n<parameter=molecule>...</parameter>\n</function>` (pseudo-XML)
   - `<name>select_reaction</name><parameter:reaction>...</parameter>`
   - `{"function=select_reaction", "arguments": {...}}`
   - Plain `<|im_end|>` with no tool-call content (4,712 assistant turns become the empty string after stripping special tokens).
8. **Terminal assistant is always valid.** The assistant turn immediately before the final trailing tool is always a valid `single_step_retro` or `select_reaction` tool call (11,436/11,436). Dropping the trailing tool therefore yields a conversation that always ends cleanly on a `gpt` turn.
9. **Length distribution.** min 6 / mean 27.1 / **max 202** messages per conversation. Many trajectories will exceed an 8k token window.
10. **Valid tool names in this corpus:** `single_step_retro` and `select_reaction` (matches the fixed system prompt schema).

## 1.3 Target Format: LLaMA-Factory ShareGPT

The processed output is JSONL, one example per line:

```json
{
  "system": "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n...",
  "conversations": [
    {"from": "human", "value": "You are a professional organic chemist..."},
    {"from": "gpt",   "value": "<think>\n\n</think>\n<tool_call>\n{\"name\": \"single_step_retro\", \"arguments\": {\"molecule\": \"0-0\"}}\n</tool_call>"},
    {"from": "observation", "value": "Reaction state 0-0: Possible reactions ..."},
    {"from": "gpt",   "value": "<think>\n\n</think>\n<tool_call>\n{\"name\": \"select_reaction\", \"arguments\": {\"reaction\": \"0-0-1\"}}\n</tool_call>"}
  ],
  "id": "train_h4_10__q000544__r0__turns14",
  "meta": {"success": true, "n_turns": 14, ...}
}
```

Role mapping:

| Raw role    | ShareGPT role      |
| ----------- | ------------------ |
| `system`    | dropped — replaced by fixed top-level `system` |
| `user`      | `human`            |
| `assistant` | `gpt`              |
| `tool`      | `observation`      |

## 1.4 Fixed System Prompt

Use this fixed system prompt for every processed example. Do **not** wrap it with `<|im_start|>system` / `<|im_end|>` — LLaMA-Factory's template handler adds those.

```python
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
```

## 1.5 Design Choice: Handling `<think>`

Raw assistant messages typically look like:

```text
<tool_call>
{"name": "single_step_retro", "arguments": {"molecule": "0-0"}}
</tool_call><|im_end|>
```

The user prompt asks the model to produce:

```text
<think>
...
</think>
<tool_call>
...
</tool_call>
```

Normalization rules applied to each assistant message:

1. Strip `<|im_start|>` and `<|im_end|>`, then `strip()` whitespace.
2. If the message contains `<think>` but no `</think>`, insert `</think>\n` before the first `<tool_call>`.
3. If the message contains `<tool_call>` but no `<think>`, prepend `<think>\n\n</think>\n`.
4. If the message is **empty** after step 1 (the model emitted only `<|im_end|>`), treat it as a failed turn — see §1.6.

## 1.6 Design Choice: Failed Tool Calls

About 21% of (assistant, tool) pairs are model-format failures (empty assistant, or assistant with a wrong tool-call schema such as `<function=...>`). Including them as SFT targets would teach the model the bad format. The script must support a filter policy, selectable by `--filter-mode`:

- **`keep-all`** — keep every turn (teaches recovery; teaches bad formats too). Empty assistants still need a placeholder like `<think>\n\n</think>` so ShareGPT parsing does not break.
- **`drop-failed-pairs`** *(recommended, default)* — drop any `(assistant_i, tool_i)` pair where *either*:
   - the assistant message is empty after stripping special tokens, **or**
   - the tool message starts with `"Invalid tool call format"`.
   Rebuild the `conversations` list from the remaining pairs. Verified on the corpus: every one of the 11,436 records retains ≥2 valid pairs after this filter (min 2 / mean 9.9 / max 92 pairs; 6,770 records lose zero pairs).
- **`drop-conversation`** — drop the entire record if it contains any failed pair. Strictest; yields the 6,770-record subset above.

Invalid tool-call detection for the assistant side (when a finer check is needed) should match either:
- empty content after strip, or
- contains `<tool_call>` *and* none of `re.search(r'"name"\s*:\s*"(single_step_retro|select_reaction)"', content)` matches.

Mode default: `drop-failed-pairs`.

## 1.7 Preprocessing Requirements

Implement `scripts/preprocess_retro_tool_sft.py` that:

1. Reads `.pkl` (primary), plus `.jsonl` / `.json` / `.txt` (legacy fallback).
2. For `.pkl`: `import torch` before `pickle.load(open(path, "rb"))`, read `payload["sft_records"]`.
3. For legacy text formats: try `json.loads` per line; fall back to `ast.literal_eval` for Python-literal dumps.
4. For each record, read `example["conversations"]` and iterate in order.
5. Replace the raw empty `system` turn with the fixed `SYSTEM_PROMPT` written at top level.
6. Convert roles:
   - `user` → `human`
   - `assistant` → `gpt`
   - `tool` → `observation`
   - `system` → ignored (fixed top-level system used instead).
7. Strip `<|im_start|>` and `<|im_end|>` from every message content.
8. Normalize assistant messages per §1.5 rules 1-3.
9. Apply `--filter-mode` (default `drop-failed-pairs`) per §1.6.
10. Drop trailing `observation` messages at the end of each conversation (already guaranteed by the corpus but keep the safety net).
11. `--success-only`: keep only records with `meta.success == True` (no-op on the current corpus).
12. `--max-turns N`: optionally drop conversations whose processed `conversations` list exceeds `N` messages (skip the record). Useful to bound context length.
13. `--skip-invalid`: on validation failure, skip the record and continue instead of raising.
14. Write the output as JSONL and print summary stats:
    - total raw records, written, skipped
    - counts for filter actions: empty-assistant pairs dropped, invalid-tool-response pairs dropped, full conversations dropped
    - pair / turn distribution before and after filtering (min / mean / max)
    - unique tool names observed after filtering
    - note the tag distribution if the pickle carries `_merge_source_tags`.
15. Validate each processed example before writing:
    - non-empty `conversations`, starts with `human`, ends with `gpt`
    - every `from` ∈ `{human, gpt, observation}`
    - every `value` is a non-empty string
    - no `<|im_start|>` / `<|im_end|>` survives
    - any `gpt` value containing `<think>` also contains `</think>`; `<tool_call>` implies `</tool_call>`.

## 1.8 Reference Preprocessing Script (to adapt)

This is a *reference*, not a spec. The implementing agent must (a) replace the old loader with the pickle loader described in §1.7, (b) add the `--filter-mode` logic from §1.6, and (c) handle empty assistant content. Keep the legacy JSON/JSONL/TXT path as a fallback.

```python
import argparse
import ast
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch  # noqa: F401  # safe for pickles that nest torch tensors


SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "single_step_retro", "description": "...", "parameters": {...}}
{"name": "select_reaction", "description": "...", "parameters": {...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""   # full text as in §1.4


VALID_TOOL_NAMES = ("single_step_retro", "select_reaction")
TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"(' + "|".join(VALID_TOOL_NAMES) + r')"')
INVALID_TOOL_MSG_PREFIX = "Invalid tool call format"


def remove_special_tokens(text: str) -> str:
    return (text.replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .strip())


def fix_assistant_content(text: str) -> str:
    text = remove_special_tokens(text)
    if not text:
        return ""  # caller decides what to do
    if "<think>" in text and "</think>" not in text and "<tool_call>" in text:
        text = text.replace("<tool_call>", "</think>\n<tool_call>", 1)
    if "<tool_call>" in text and "<think>" not in text:
        text = "<think>\n\n</think>\n" + text
    return text.strip()


def is_assistant_failed(assistant_value: str, tool_value: str) -> bool:
    if not assistant_value.strip():
        return True
    if tool_value.startswith(INVALID_TOOL_MSG_PREFIX):
        return True
    return False


def load_raw_examples(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or "sft_records" not in payload:
            raise ValueError(f"{path}: expected dict with 'sft_records'")
        return list(payload["sft_records"])

    # Legacy fallback: JSONL / JSON / TXT with Python-literal dicts.
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [_parse_obj(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        obj = _parse_obj(text)
        return obj if isinstance(obj, list) else [obj]
    # .txt / unknown
    try:
        obj = _parse_obj(text)
        return obj if isinstance(obj, list) else [obj]
    except Exception:
        chunks = re.split(r"\n\s*\n", text.strip())
        return [_parse_obj(c) for c in chunks if c.strip()]


def _parse_obj(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def convert_example(raw: Dict[str, Any], filter_mode: str, success_only: bool) -> Tuple[Dict[str, Any], Dict[str, int]]:
    if "conversations" not in raw:
        raise ValueError("Missing 'conversations' field.")
    meta = raw.get("meta", {})
    if success_only and meta.get("success") is not True:
        raise ValueError("skip: non-success trajectory")

    stats = {"empty_assistant_dropped": 0, "invalid_tool_dropped": 0}

    # 1. Scan raw conversations and split into (user, [(assistant, tool), ...])
    user_msg = None
    pairs: List[Tuple[str, str]] = []
    i = 0
    convs = raw["conversations"]
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
            a = content
            t = ""
            if i + 1 < len(convs) and convs[i + 1].get("role") == "tool":
                t = convs[i + 1].get("content", "")
                i += 2
            else:
                i += 1
            pairs.append((a, t))
            continue
        if role == "tool":  # orphan tool (shouldn't happen in this corpus)
            i += 1
            continue
        raise ValueError(f"Unknown role: {role}")

    if user_msg is None:
        raise ValueError("No user message found.")

    # 2. Apply filter.
    kept_pairs: List[Tuple[str, str]] = []
    had_failed = False
    for a_raw, t_raw in pairs:
        a_norm = fix_assistant_content(a_raw)
        t_norm = remove_special_tokens(t_raw)
        failed = is_assistant_failed(a_norm, t_norm)
        if failed:
            had_failed = True
            if not a_norm.strip():
                stats["empty_assistant_dropped"] += 1
            else:
                stats["invalid_tool_dropped"] += 1
            if filter_mode == "drop-failed-pairs":
                continue
            if filter_mode == "drop-conversation":
                raise ValueError("skip: conversation contains failed pair")
            # keep-all: fall through, but give empty assistants a valid placeholder
            if not a_norm.strip():
                a_norm = "<think>\n\n</think>"
        kept_pairs.append((a_norm, t_norm))

    # 3. Assemble ShareGPT turns.
    conversations = [{"from": "human", "value": user_msg}]
    for a_norm, t_norm in kept_pairs:
        conversations.append({"from": "gpt", "value": a_norm})
        conversations.append({"from": "observation", "value": t_norm})

    # 4. Trim trailing observation (safety net).
    while conversations and conversations[-1]["from"] == "observation":
        conversations.pop()

    return {
        "system": SYSTEM_PROMPT,
        "conversations": conversations,
        "id": raw.get("id"),
        "meta": meta,
    }, stats


def validate_example(example: Dict[str, Any], index: int) -> None:
    conv = example["conversations"]
    if not conv:
        raise ValueError(f"example {index}: empty conversations")
    if conv[0]["from"] != "human":
        raise ValueError(f"example {index}: does not start with human")
    if conv[-1]["from"] != "gpt":
        raise ValueError(f"example {index}: does not end with gpt")
    for k, m in enumerate(conv):
        if m["from"] not in {"human", "gpt", "observation"}:
            raise ValueError(f"example {index}, turn {k}: invalid from={m['from']}")
        if not isinstance(m["value"], str) or not m["value"].strip():
            raise ValueError(f"example {index}, turn {k}: empty value")
        if "<|im_start|>" in m["value"] or "<|im_end|>" in m["value"]:
            raise ValueError(f"example {index}, turn {k}: ChatML token leaked")
        if m["from"] == "gpt":
            if "<tool_call>" in m["value"] and "</tool_call>" not in m["value"]:
                raise ValueError(f"example {index}, turn {k}: unclosed tool_call")
            if "<think>" in m["value"] and "</think>" not in m["value"]:
                raise ValueError(f"example {index}, turn {k}: unclosed think")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument(
        "--filter-mode",
        choices=("keep-all", "drop-failed-pairs", "drop-conversation"),
        default="drop-failed-pairs",
    )
    parser.add_argument("--max-turns", type=int, default=None,
                        help="If set, skip records whose processed conversations length exceeds this.")
    args = parser.parse_args()

    raw_examples = load_raw_examples(Path(args.input))

    n_total = len(raw_examples)
    n_written = 0
    n_skipped = 0
    empty_dropped = 0
    invalid_dropped = 0
    turn_counts = []

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, raw in enumerate(raw_examples):
            try:
                example, stats = convert_example(raw, args.filter_mode, args.success_only)
                validate_example(example, i)
            except Exception as e:
                if args.skip_invalid:
                    n_skipped += 1
                    if i < 5:
                        print(f"[skip] {i}: {e}")
                    continue
                raise
            if args.max_turns and len(example["conversations"]) > args.max_turns:
                n_skipped += 1
                continue
            empty_dropped += stats["empty_assistant_dropped"]
            invalid_dropped += stats["invalid_tool_dropped"]
            turn_counts.append(len(example["conversations"]))
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_written += 1

    if turn_counts:
        avg = sum(turn_counts) / len(turn_counts)
        mx = max(turn_counts)
    else:
        avg = mx = 0
    print(f"raw={n_total} written={n_written} skipped={n_skipped}")
    print(f"empty-assistant pairs handled: {empty_dropped}")
    print(f"invalid-tool-format pairs handled: {invalid_dropped}")
    print(f"filter-mode={args.filter_mode}")
    print(f"turns min/mean/max = {min(turn_counts) if turn_counts else 0} / {avg:.2f} / {mx}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
```

## 1.9 Run Preprocessing

Primary command on the actual data:

```bash
python scripts/preprocess_retro_tool_sft.py \
  --input data/chunk_sft.pkl \
  --output data/retro_tool_sft.jsonl \
  --filter-mode drop-failed-pairs \
  --success-only \
  --skip-invalid
```

Dedup / tagged variant:

```bash
python scripts/preprocess_retro_tool_sft.py \
  --input data/chunk_sft_tagged_dedup.pkl \
  --output data/retro_tool_sft_dedup.jsonl \
  --filter-mode drop-failed-pairs \
  --success-only \
  --skip-invalid
```

## 1.10 Expected Processed Example (first record)

```json
{
  "system": "# Tools\n\nYou may call one or more functions...",
  "conversations": [
    {"from": "human", "value": "You are a professional organic chemist, skilled at finding pathways to synthesize novel molecules..."},
    {"from": "gpt", "value": "<think>\n\n</think>\n<tool_call>\n{\"name\": \"single_step_retro\", \"arguments\": {\"molecule\": \"0-0\"}}\n</tool_call>"},
    {"from": "observation", "value": "Reaction state 0-0: Possible reactions to synthesize molecule 0-0: ..."},
    {"from": "gpt", "value": "<think>\n\n</think>\n<tool_call>\n{\"name\": \"select_reaction\", \"arguments\": {\"reaction\": \"0-0-3\"}}\n</tool_call>"}
  ],
  "id": "train_h4_10__q000000__r0__turns8",
  "meta": {
    "abs_query_idx": 0,
    "api_model": "A1-preview",
    "data_source": "reaction_pathway_search",
    "force_noloop": true,
    "model_path": "/mnt/shared-storage-user/wangzifu/cache/model/Qwen3.5-0.8B",
    "n_turns": 8,
    "rollout_n": 8,
    "rollout_rank": 0,
    "rollout_temp": 0.7,
    "success": true
  }
}
```

With `--filter-mode drop-failed-pairs`, the retained pair count will usually be smaller than `meta.n_turns / 2` whenever the rollout had any format errors.

---

# Step 2: Launch Training with the Processed Data

## 2.1 Put Processed Data into LLaMA-Factory

```bash
LLaMA-Factory/
├── data/
│   ├── dataset_info.json
│   └── retro_tool_sft.jsonl
```

## 2.2 Update `dataset_info.json`

Add:

```json
{
  "retro_tool_sft": {
    "file_name": "retro_tool_sft.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
    }
  }
}
```

The tool schema is inside the fixed `system` prompt, so no top-level `tools` field is needed.

## 2.3 Create Training YAML

Create `examples/train_lora/qwen3_retro_tool_sft.yaml`:

```yaml
### model
model_name_or_path: /path/to/base-model   # e.g. Qwen3.5-0.8B (what rollouts came from) or a larger instruct checkpoint

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset_dir: data
dataset: retro_tool_sft
template: qwen3                # use `qwen` for Qwen2.5-era checkpoints
cutoff_len: 16384              # max conversation length is 202 turns; 8k will truncate a noticeable fraction
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/qwen3/lora/retro_tool_sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.02
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

Required adjustments before launch:

- `model_name_or_path` → real local checkpoint path.
- `template` → `qwen3` (if the installed LLaMA-Factory has it) or `qwen` (for Qwen2.5).
- `cutoff_len` → 16384 is the conservative starting point; raise to 24576/32768 if memory allows, or pair with `--max-turns` during preprocessing to cap length up-front.

## 2.4 Debug Training First

Make a 100-record subset from the **output** JSONL (not the pickle):

```bash
head -n 100 data/retro_tool_sft.jsonl > data/retro_tool_sft_debug.jsonl
```

Add to `dataset_info.json`:

```json
{
  "retro_tool_sft_debug": {
    "file_name": "retro_tool_sft_debug.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "system": "system"}
  }
}
```

Temporarily switch the YAML to:

```yaml
dataset: retro_tool_sft_debug
num_train_epochs: 1
save_steps: 20
eval_steps: 20
```

Launch:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/qwen3_retro_tool_sft.yaml
```

Proceed to full training only after confirming: dataset loads, tokenization works, loss is finite, no role-format errors, no sequence-length error.

## 2.5 Launch Full Training

Foreground:

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/qwen3_retro_tool_sft.yaml
```

Background:

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1 nohup llamafactory-cli train \
  examples/train_lora/qwen3_retro_tool_sft.yaml \
  > logs/retro_tool_sft.log 2>&1 &
```

Monitor:

```bash
tail -f logs/retro_tool_sft.log
nvidia-smi
```

## 2.6 Common Failure Cases

### Case 1: `pickle.load` fails
The chunk pickles were produced by `scripts/merge_chunk_sft*.py`, which imports `torch` before pickling. Replicate that by `import torch` before `pickle.load`. This covers pickles that nest torch tensors even if our envelope does not.

### Case 2: Conversation ends with `observation`
Expected — every raw trajectory ends with a `tool` turn. Drop trailing `observation`s; terminal assistant turns are always valid tool calls in this corpus, so the result always ends on `gpt`.

### Case 3: Empty assistant turn after stripping
A failed rollout where the model emitted only `<|im_end|>`. With `--filter-mode drop-failed-pairs` (default), drop the whole pair. With `keep-all`, replace the value with `<think>\n\n</think>` so ShareGPT parsing does not crash.

### Case 4: Assistant has no `<think>`
Prepend `<think>\n\n</think>\n`.

### Case 5: Assistant's tool-call schema is wrong (`<function=...>`, `<name>...`, `"function=..."`)
The paired tool message will start with `"Invalid tool call format"`. With the default filter these pairs are removed. Do *not* try to repair the malformed call — the correct next-turn format is ambiguous.

### Case 6: LLaMA-Factory rejects `observation` role
Older LLaMA-Factory versions route tool outputs through `human`. Fallback encoding:

```json
{"from": "human", "value": "<tool_response>\nReaction state ...\n</tool_response>"}
```

Prefer `observation` on newer versions.

### Case 7: Sequence length too long
Max conversation length in this corpus is 202 messages. Options:

- `cutoff_len: 16384` or higher (baseline recommendation).
- Use `--max-turns` during preprocessing to drop overly long records.
- Truncate very long tool observations (e.g., keep top-k reactions per `single_step_retro` response).

## 2.7 Deliverables

```
scripts/preprocess_retro_tool_sft.py
data/retro_tool_sft.jsonl                  # processed from chunk_sft.pkl
examples/train_lora/qwen3_retro_tool_sft.yaml
```

Updates:

```
data/dataset_info.json
```

Short report:

```text
Raw pickle path:
Pickle variant used (chunk_sft.pkl / chunk_sft_tagged_dedup.pkl):
Number of raw records:
Filter mode:
Empty-assistant pairs handled:
Invalid-tool-format pairs handled:
Number of processed records written:
Number of records skipped:
Success-only filtering used (yes/no):
Turns (min / mean / max) after processing:
Training model path:
Training dataset name:
Training command:
Output directory:
Log file:
```

---

## Final Instruction to the Coding Agent

Implement the preprocessing script for pickled rollout records with structured `conversations`. Input is `Agent-r1-temp/data/chunk_sft.pkl` (or the dedup/tagged variant). Load with `pickle.load`, import torch first. Convert `user → human`, `assistant → gpt`, `tool → observation`. Replace the empty raw system with the fixed §1.4 prompt. Strip `<|im_end|>`, normalize `<think>`/`</think>` on every assistant turn. Apply `--filter-mode drop-failed-pairs` by default — this drops both empty-assistant pairs and pairs whose tool response starts with `"Invalid tool call format"`. Drop trailing observations. Validate each example. Write JSONL. Run a 100-record debug training with LLaMA-Factory, then launch the full SFT job.
