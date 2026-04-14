
2026-04-13

Generate a Slurm batch script `test_chembl.sbatch` that wraps the command in @test_chembl_script.sh so it can be submitted via `sbatch`.

## Reference & Constraints

- **Slurm boilerplate**: Follow @scripts_example.sbatch for the structure — `#SBATCH` directives, environment setup, `srun` invocation pattern, and logging conventions.
- **Resource requirements**: 1 node, 8 GPUs (`--gres=gpu:8`), 128 CPUs, 256 GB memory, 48-hour time limit, partition `ai_science`.
- **Job naming**: Use `--job-name=test_chembl_ppo` and route stdout/stderr to `logs/%x-%j.out` / `logs/%x-%j.err` (match the example pattern).
- **Conda environment**: Activate the `retro_r1` conda environment. Use the same `srun` + `bash -c "source conda ... && conda activate ..."` pattern from the example.
- **Python entry point**: The actual command to run is from `test_chembl_script.sh` — keep all its Hydra overrides and parameter values **exactly as-is**. Do not modify any hyperparameters, paths, or config values.
- **Logging**: Replace the manual `> ./logs/... 2> ./logs/...` redirection in the original script with Slurm's `--output`/`--error` directives, plus `2>&1 | tee "logs/test_chembl_${SLURM_JOB_ID}.log"` for a combined runtime log (same pattern as the example).
- **Environment variables**: Include `HF_HUB_OFFLINE=1`, `RAY_TMPDIR`, and `CPATH` setup from the example. Add an info banner that prints `SLURM_JOB_ID`, node count, and GPU count.
- **Keep it minimal**: Do not add any memory/skill/GRPO-related variables from the example — those belong to a different experiment. Only carry over the Slurm infrastructure and environment setup.







2026-04-14

Add **progress monitoring and timing logs** to the `_validate` method in @agent_r1/src/agent_ray_trainer_retro_noback.py:521-685 so I can track where time is spent during validation runs.

## What to add

### 1. Overall validation timer
- Record `time.time()` at the start of `_validate` (line ~523) and log the total elapsed time right before the `return` (line ~684).
- Format: `[Validation] Completed in {elapsed:.1f}s — {total_samples} samples, {num_batches} batches`

### 2. Progress bar for the multi-turn generation loop in `run_llm_loop` (@agent_r1/llm_agent/generation_retro_noback.py:370-441)
The main loop runs up to `max_turns` iterations (LLM generate → tool execute per turn). Wrap the `for step in range(self.config.max_turns)` loop with a `tqdm` progress bar:
- Total = `self.config.max_turns`, updated each iteration.
- Show the number of still-active trajectories in the bar's postfix: `active={active_mask.sum().item()}/{batch_size}`.
- Break out early (as the existing code does) when `active_mask.sum() == 0`; tqdm should still close cleanly.
- Use `tqdm.auto` so it works in both terminal and notebook contexts.
- The bar description should be `"Validation LLM turns"` (or `"Train LLM turns"` depending on `self.is_validation`).

### 3. Progress bar for the validation reward computation phase
- In `_validate` (@agent_r1/src/agent_ray_trainer_retro_noback.py), the `for test_data in self.val_dataloader` loop (line ~557) currently has exactly 1 batch, but still wrap the key phases with a simple `tqdm` or log so the user can see the pipeline moving:
  - Before `run_llm_loop` starts: print `[Validation] Starting generation for {len(test_batch)} samples...`
  - After `run_llm_loop` finishes: print `[Validation] Generation complete. Computing rewards...`
  - After `val_reward_fn` finishes: print `[Validation] Rewards computed. mean_answer_acc={mean_answer:.4f}`

## Constraints
- Use Python `logging.getLogger(__name__)` with `logger.info(...)` instead of bare `print()`. If a logger is already available on `self`, use that.
- **Do not modify** any computation logic, tensor shapes, return values, or the reward function interface.
- Keep the existing `print(...)` statements at lines 595 and 609 as-is (or convert them to logger calls too, your choice).







2026-04-14

Add an **API-model generation path** to the retro_noback pipeline so I can evaluate an external LLM (via OpenAI-compatible API) on the same chembl benchmark, instead of the local vLLM model.

## Goal

Currently, `run_llm_loop` (in @agent_r1/llm_agent/generation_retro_noback.py:352-448) generates responses by calling the local vLLM model via `self._generate_with_gpu_padding()` (@agent_r1/llm_agent/generation_retro_noback.py:305-350). I want to add an alternative code path that calls an external API model instead, while keeping the entire agentic tool-calling loop (tool execution, environment stepping, state updates) unchanged.

## What to modify

### 1. `ToolGenerationManager` in @agent_r1/llm_agent/generation_retro_noback.py

- **Add imports** at the top of the file:
  ```python
  try:
      from openai import AsyncOpenAI
  except ImportError:
      AsyncOpenAI = None
  ```

- **Add a config flag** to `ToolGenerationConfig`: `use_api_model: bool = False` and `api_model_name: str = ""`.

- **Add an API client builder** method to `ToolGenerationManager`:
  ```python
  def _build_api_client(self):
      if AsyncOpenAI is None:
          return None
      api_key = os.environ.get("pjlab_APImodel_key")
      if not api_key:
          return None
      base_url = os.environ.get("pjlab_APImodel_url")
      if base_url:
          return AsyncOpenAI(base_url=base_url, api_key=api_key)
      return AsyncOpenAI(api_key=api_key)
  ```
  Call it in `__init__` and store the client as `self._api_client`.

- **Add an API generation method** (e.g., `_generate_with_api`) that:
  1. Decodes the `input_ids` from the active batch back to text using `self.tokenizer`.
  2. Sends each prompt to the API via `self._api_client.chat.completions.create(...)` (use `asyncio.run` or an event loop to call the async client). Use `self.config.api_model_name` as the model name.
  3. Tokenizes the API responses back into token IDs.
  4. Returns a `DataProto` with the same `responses` key structure that the rest of the loop expects (matching the shape/format from `_generate_with_gpu_padding`).

- **Branch in `run_llm_loop`** at the generation call site (line ~382): if `self.config.use_api_model and self._api_client is not None`, call `_generate_with_api(rollings_active)` instead of `_generate_with_gpu_padding(rollings_active)`.

### 2. Trainer in @agent_r1/src/agent_ray_trainer_retro_noback.py:601-605

- Pass the new config flags (`use_api_model`, `api_model_name`) through when constructing `ToolGenerationConfig` / `ToolGenerationManager` so the validation call at lines 601-605 uses the API path.

### 3. Slurm script @test_chembl.sbatch

- Add environment variables for the API credentials:
  ```bash
  export pjlab_APImodel_key="<placeholder>"
  export pjlab_APImodel_url="<placeholder>"
  ```
- Add Hydra overrides to enable API mode:
  ```
  tool.use_api_model=True
  tool.api_model_name="<model-name-placeholder>"
  ```

## Constraints

- **Do not change** the tool-calling loop logic, environment interaction, reward computation, or any other training/validation logic.
- **Keep the local vLLM path as the default** (`use_api_model=False`). The API path is only used when explicitly enabled.
- **Token format**: The API responses must be tokenized and padded to match the same tensor format that `_generate_with_gpu_padding` returns, so downstream code (`_postprocess_responses`, `_update_rolling_state`, etc.) works without changes.






2026-04-14

Make `api_max_concurrency` and `debug` configurable end-to-end: YAML config → trainer wiring → dataclass → runtime usage in `_generate_with_api`.

## Files to change (3 files)

### 1. YAML config — `agent_r1/src/config/agent_trainer.yaml` (line ~223)

Add two new keys under the `tool:` section, right after `api_model_name`:

```yaml
  api_max_concurrency: 8
  debug: False
```

### 2. Trainer wiring — `agent_r1/src/agent_ray_trainer_retro_noback.py`

There are **two** `ToolGenerationConfig(...)` construction sites (lines ~536 and ~1060). In **both**, add these two lines after the `api_model_name=...` line, following the same `.get()` pattern:

```python
            api_max_concurrency=self.config.tool.get('api_max_concurrency', 8),
            debug=self.config.tool.get('debug', False),
```

### 3. Dataclass + method — `agent_r1/llm_agent/generation_retro_noback.py`

#### 3a. `ToolGenerationConfig` dataclass (line ~30)

Add two new fields with defaults after `api_model_name`:

```python
api_max_concurrency: int = 8      # max concurrent async API requests
debug: bool = False                # when True, print verbose debug info
```

#### 3b. `_generate_with_api` method (lines 373-446)

Apply these changes **in order**:

a) **Strip pad tokens before decoding** — replace the current one-liner `batch_decode` with a loop that filters out `pad_token_id` per row before decoding:
```python
prompts = []
for ids in input_ids:
    non_pad_ids = ids[ids != self.tokenizer.pad_token_id]
    prompts.append(self.tokenizer.decode(non_pad_ids, skip_special_tokens=False))
```

b) **Add debug logging after building prompts** — if `self.config.debug`, print batch size, model name, and prompt length stats (min/max/avg), plus the first 2 full prompts.

c) **Throttle concurrency** — create `sem = asyncio.Semaphore(self.config.api_max_concurrency)` and wrap the body of `_call_api` with `async with sem:`.

d) **Add debug logging after collecting responses** — if `self.config.debug`, print response count and length stats (min/max/avg), plus the first 2 full responses.

### Reference implementation

The full target state for `_generate_with_api` is shown below. Use this as a guide — the key additions vs. the current code are the pad-stripping, the semaphore, and the two `if self.config.debug:` blocks:

```python
def _generate_with_api(self, active_batch: DataProto) -> DataProto:
    """Generate responses using an external API model instead of local vLLM."""
    input_ids = active_batch.batch['input_ids']
    batch_size = input_ids.shape[0]

    # Decode input_ids back to text prompts, stripping left-padding first
    prompts = []
    for ids in input_ids:
        non_pad_ids = ids[ids != self.tokenizer.pad_token_id]
        prompts.append(self.tokenizer.decode(non_pad_ids, skip_special_tokens=False))

    if self.config.debug:
        prompt_lens = [len(p) for p in prompts]
        print(f"[DEBUG API] Sending {batch_size} prompts to {self.config.api_model_name} | "
              f"len min={min(prompt_lens)} max={max(prompt_lens)} avg={sum(prompt_lens)/len(prompt_lens):.0f}")
        for i, p in enumerate(prompts[:2]):
            print(f"[DEBUG API] Prompt[{i}] (len={len(p)}): {p[:]}")

    sem = asyncio.Semaphore(self.config.api_max_concurrency)

    async def _call_api(prompt: str) -> str:
        """Call external API with concurrency limit."""
        async with sem:
            try:
                response = await self._api_client.chat.completions.create(
                    model=self.config.api_model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.max_response_length,
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                raise RuntimeError(f"API request failed: {type(exc).__name__}: {exc}") from None

    # ... (async runner logic stays the same) ...

    if self.config.debug:
        resp_lens = [len(r) for r in response_texts]
        print(f"[DEBUG API] Received {len(response_texts)} responses | "
              f"len min={min(resp_lens)} max={max(resp_lens)} avg={sum(resp_lens)/len(resp_lens):.0f}")
        for i, r in enumerate(response_texts[:2]):
            print(f"[DEBUG API] Response[{i}] (len={len(r)}): {r[:]}")

    # ... (tokenization + padding logic stays the same) ...
```

## Do NOT change

- The async runner logic (`_call_all`, event-loop detection, `ThreadPoolExecutor` fallback)
- The tokenization and padding logic after `response_texts`
- Any other methods or classes in this file
