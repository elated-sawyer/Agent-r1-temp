
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

Strip the codebase down to a **validation-only pipeline using the API model path**. Remove all Ray cluster dependencies and training logic. This is on a dedicated branch (`validation_API`), so aggressive deletion is fine.

## Overview

The current pipeline launches via Ray (`ray.init` → `@ray.remote` → `RayAgentTrainer.fit()`), initializes distributed GPU worker groups, runs a PPO training loop, and periodically calls `_validate()`. Now that we have the `_generate_with_api()` path, none of the Ray/GPU/training machinery is needed. The validation logic itself (`_validate`, `ToolGenerationManager`, `RewardManager`, `ToolRLDataset`, tool environments) is already Ray-independent.

## What to change

### 1. Entry point — @agent_r1/src/main_agent_retro_noback.py

- **Remove** `ray.init()` (line ~116-122), the `@ray.remote` wrapper (line ~127), and `ray.get()` calls.
- **Remove** all training-specific setup: `reward_fn` (training reward), `train_dataset` creation, critic/ref model resource pool config.
- **Keep** the Hydra `@hydra.main` config loading, tokenizer/processor initialization (lines ~140-142, these are pure HuggingFace), `val_reward_fn` (`RewardManager`), `val_env` creation, and `val_dataset`/`val_dataloader` setup.
- The new flow should be: load config → load tokenizer → build val_dataset/dataloader → build val_env → build RewardManager → run validation → print/log results. No Ray, no `fit()`.

### 2. Trainer — @agent_r1/src/agent_ray_trainer_retro_noback.py

**Remove entirely** (these are all training/Ray-only):
- Ray imports: `import ray`, `RayResourcePool`, `RayWorkerGroup`, `RayClassWithInitArgs`, `create_colocated_worker_cls`
- Classes/enums: `ResourcePoolManager`, `Role`, `AdvantageEstimator`
- Functions: `apply_kl_penalty()`, `compute_advantage()`
- Methods on `RayAgentTrainer`: `init_workers()`, `fit()`, `_balance_batch()`, `prime_norm()`, `_compute_process_rewards()`, `_create_loss_mask()`, `_save_checkpoint()`, `_load_checkpoint()`, `_save_env_var()`
- Instance variables: `self.critic_wg`, `self.ref_policy_wg`, `self.rm_wg`, `self.actor_rollout_wg`, `self.kl_ctrl`, `self.resource_pool_manager`, `self.role_worker_mapping`, `self.use_critic`, `self.use_rm`, `self.use_reference_policy`, `self.wg_dicts`

**Keep and simplify** `_validate()` (lines ~522-684):
- **Line ~602**: Replace `pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)` → skip padding entirely (pass `world_size=1`, or remove the pad/unpad calls since API batching doesn't need GPU-divisible sizes).
- Keep everything else: batch prep, `ToolGenerationManager` construction, `run_llm_loop()`, reward computation, metric aggregation.
- Keep `_maybe_log_val_generations()` and the `ValidationGenerationsLogger`.

**Rename the class** from `RayAgentTrainer` to something like `ValidationPipeline` — it's no longer a trainer.

### 3. Generation manager — @agent_r1/llm_agent/generation_retro_noback.py

- Make `actor_rollout_wg` **optional** (`None` by default) in `ToolGenerationManager.__init__`.
- **Remove** `_generate_with_gpu_padding()` entirely (it calls `self.actor_rollout_wg.generate_sequences()` which is a Ray worker call).
- In `run_llm_loop()`, remove the GPU/API branch — always call `_generate_with_api()`.

### 4. Config — @agent_r1/src/config/agent_trainer.yaml

**Remove** these sections (training-only):
- `critic:` (entire block, lines ~108-143)
- `reward_model:` (lines ~145-164)
- `algorithm:` (lines ~170-178)
- Training-specific fields under `trainer:`: `total_epochs`, `save_freq`, `test_freq`, `critic_warmup`, `resume_mode`, `resume_from_path`, `remove_previous_ckpt_in_save`, `del_local_ckpt_after_load`
- Training-specific fields under `actor_rollout_ref:`: `actor.ppo_*`, `actor.grad_clip`, `actor.clip_ratio`, `actor.entropy_coeff`, `actor.use_kl_loss`, `actor.kl_loss_*`, `actor.ppo_epochs`, `actor.fsdp_config`, `actor.optim`, `ref:` (entire block), `rollout.n` / `rollout.n_repeat`

**Keep**:
- `data:` — `val_files`, `prompt_key`, `max_prompt_length`, `max_response_length`, `max_start_length`, `max_tool_response_length`, `return_raw_chat`
- `actor_rollout_ref.model.path` — still needed for **tokenizer loading** (not model weights)
- `actor_rollout_ref.rollout.val_kwargs` — `temperature`, `do_sample`, `n`
- `tool:` — all fields (this drives the generation loop)
- `trainer:` — `project_name`, `experiment_name`, `logger`, `val_generations_to_log_to_wandb`, `nnodes`, `n_gpus_per_node` (can default to 1), `val_before_train` (can remove, validation is the only thing)
- `custom_reward_function:` — used by `RewardManager`

### 5. Sbatch — @test_chembl.sbatch

- **Reduce GPU request**: `--gres=gpu:1` (or `gpu:0` if tokenizer loads on CPU). Validation via API doesn't need local GPUs.
- **Remove** all actor/critic/ref/reward_model/algorithm Hydra overrides (lines ~52-84).
- **Keep** data paths, tool config, API model config, project/experiment naming, logger config.

## Constraints

- **Do not change** the tool-calling loop logic in `run_llm_loop()` (tool execution, environment stepping, `_postprocess_responses`, `_update_rolling_state`, etc.).
- **Do not change** the reward function interface or computation.
- **Do not change** `ToolRLDataset` or `RewardManager` — they are already Ray-independent.
- The pipeline must still produce the same `metric_dict` and `envs_var` output from validation.






2026-04-14

Add a **debug mode** to the API generation path so that every request/response is printed to stdout for inspection.

### What to change

**1. Config — `@agent_r1/src/config/agent_trainer.yaml`**

Add a `debug` flag under the `tool:` section (default `False`).

**2. Generation config dataclass — `@agent_r1/llm_agent/generation_retro_noback.py:29-46`**

Add a `debug: bool = False` field to `ToolGenerationConfig`.

**3. Where the config is constructed — `@agent_r1/src/agent_ray_trainer_retro_noback.py`**

Pass `debug=self.config.tool.get('debug', False)` when building `ToolGenerationConfig` inside `validate()`.

**4. API call method — `@agent_r1/llm_agent/generation_retro_noback.py:327-401`**

Inside `_generate_with_api()`, when `self.config.debug` is `True`:

- **Before the API calls** (after line 333 where `prompts` is decoded): print every prompt being sent. Print the sample index and a truncated preview (first 200 chars) for each prompt, e.g.:
  ```
  [DEBUG API] Sending 16 prompts to deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  [DEBUG API] Prompt[0] (len=1234): <first 200 chars>...
  ```
- **After collecting results** (after the `response_texts` list is built, ~line 377): print every response. Same format — index, length, truncated preview:
  ```
  [DEBUG API] Response[0] (len=567): <first 200 chars>...
  ```

### Constraints

- Guard all debug prints behind `if self.config.debug` so there is zero overhead when disabled.
- Do not change the control flow, return values, or tokenization logic — only add print statements.
- Keep the truncation preview short (200 chars) to avoid flooding the log on long sequences.






2026-04-17

## Goal

Make the validation-only pipeline in `ValidationPipeline.validate()` (`agent_r1/src/agent_ray_trainer_retro_noback.py:126`, entrypoint `test.sbatch`) **crash-safe and resumable**, so that a mid-run failure (OOM, API timeout, preemption, SIGKILL, etc.) does not throw away all progress, and a re-launch of `test.sbatch` continues from where it stopped instead of re-running every sample.

## Current behavior (why it is fragile)

- The DataLoader is constructed with `batch_size=len(self.val_dataset)` and `assert len(self.val_dataloader) == 1`, so the entire validation set (e.g. 190 or 1000 samples) is sent to `generation_manager.run_llm_loop(...)` as a **single giant batch** running in parallel via the API.
- Per-sample env trackers are only collected at the very end of that batch: `envs_var += [env.get_tracking_variables() for env in envs]`.
- `_save_env_var(...)` is only called **once**, from `run()`, *after* `validate()` returns (`agent_ray_trainer_retro_noback.py:307-308`).
- Consequence: if the process dies anywhere inside `run_llm_loop` or `val_reward_fn`, **nothing is written to `val_env_var.pkl`** and all samples, API calls, and tool-search cost are lost.
- There is currently no resume mechanism: re-launching `test.sbatch` re-runs every sample from scratch.

## Suggested implementation outline

> Reference: see `@workspace/verl/verl/trainer/ppo/ray_trainer.py:1130` for the canonical
> `for batch_dict in self.train_dataloader:` pattern — iterate over a `DataLoader`, call
> `DataProto.from_single_dict(batch_dict)`, `test_batch.pop(...)` to split into `gen_batch`,
> run generation, then reward — and adapt exactly that structure here. The current
> `validate()` already does all of this **once** over one giant batch; we only need to
> turn that single iteration into N chunked iterations + persistence/resume.

All changes below are confined to `ValidationPipeline` in
`agent_r1/src/agent_ray_trainer_retro_noback.py` and its config, plus two new Hydra keys
in `agent_r1/src/config/agent_trainer.yaml` / `test.sbatch`.

### A. Config additions

- `agent_r1/src/config/agent_trainer.yaml`:
  - Under `data:` add `val_batch_size: 64` (the chunk size).
  - Under `trainer:` add:
    - `val_resume: True` (if `False`, clear the shard dir at startup).
    - `val_shard_dir: null` (resolve to `${trainer.default_local_dir}/global_step_${...}/val_shards` when `null`).
- `test.sbatch`: expose `data.val_batch_size=...` and `trainer.val_resume=...` as Hydra overrides (no value change required for the default).

### B. `_create_dataloader` — switch to chunked batching

Replace the single-batch `StatefulDataLoader` with a chunked one. Keep `shuffle=False`
and `drop_last=False` so sample indices are stable across runs (critical for resume).

```python
val_batch_size = int(self.config.data.get('val_batch_size', 64))
self.val_dataloader = StatefulDataLoader(
    dataset=self.val_dataset,
    batch_size=val_batch_size,
    num_workers=8,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
)
# Drop the old `assert len(self.val_dataloader) == 1`.
self.val_total_samples = len(self.val_dataset)
self.val_num_chunks   = len(self.val_dataloader)
```

Each `batch_dict` yielded by the loader corresponds to a contiguous range of dataset
indices `[chunk_idx * val_batch_size, (chunk_idx + 1) * val_batch_size)` (clipped at the
end). Use `chunk_idx` (the `enumerate(...)` index) as the shard id.

### C. Shard directory layout (on-disk state)

Resolve once in `validate()` / `__init__`:

```
<shard_dir>/
  chunk_00000.pkl   # pickled dict (see schema below)
  chunk_00001.pkl
  ...
  DONE              # sentinel, written after consolidation succeeds
```

Each `chunk_XXXXX.pkl` stores everything needed to (a) rebuild final metrics without
re-running the API and (b) reconstruct `val_env_var.pkl`. Minimal schema:

```python
{
    "chunk_idx": int,
    "sample_indices": list[int],          # absolute dataset indices covered
    "envs_var":       list[dict],         # [env.get_tracking_variables() for env in envs]
    "reward_tensor":  torch.Tensor,       # shape (B, max_response_length) - saved via torch.save-compatible bytes
    "turns_tensor":   torch.Tensor,       # shape (B,)
    "data_sources":   list[str],
    "end_lst":        list,
    "answer_lst":     list,
    "format_lst":     list,
    "sample_inputs":  list[str],          # decoded prompts
    "sample_outputs": list[str],          # decoded responses
    "sample_scores":  list,               # == answer_lst, kept explicit for logger
}
```

Write atomically: dump to `chunk_XXXXX.pkl.tmp`, then `os.replace(tmp, final)`. Guard the
dir creation with the same `FileLock` pattern already used by `_save_env_var`.

### D. Resume logic (at the top of `validate()`)

```python
shard_dir = self._resolve_shard_dir()          # new helper
if not self.config.trainer.get('val_resume', True):
    shutil.rmtree(shard_dir, ignore_errors=True)
os.makedirs(shard_dir, exist_ok=True)

done_chunks = self._scan_done_chunks(shard_dir)  # -> set[int], loads each chunk_XXXXX.pkl lazily only when needed
total_done_samples = sum(len(c["sample_indices"]) for c in done_chunks.values())
```

- If `DONE` exists and `val_resume` is `True`: load every chunk into memory, skip the
  `for batch_dict` loop entirely, jump straight to step **F** (aggregation).
- Otherwise, in the main loop, **skip any `chunk_idx` already present** in
  `done_chunks` before doing any API work (but still feed the loader to advance it —
  `StatefulDataLoader` iteration is cheap because generation is what costs money, not
  tokenization).

### E. Chunked main loop — minimal diff from today's `validate()`

Structure (mirrors `ray_trainer.py:1130`):

```python
val_pbar = tqdm(
    total=self.val_total_samples,
    initial=total_done_samples,
    desc="Validation samples",
)
for chunk_idx, test_data in enumerate(self.val_dataloader):
    if chunk_idx in done_chunks:
        continue                                            # resume skip

    test_batch = DataProto.from_single_dict(test_data)
    test_batch = test_batch.repeat(
        repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
        interleave=True,
    )
    envs = [self.val_env.copy(test_batch.non_tensor_batch['target'][ii])
            for ii in range(len(test_batch))]

    # ... existing input_ids decode, test_gen_batch pop, meta_info setup ...
    # ... existing generation_manager.run_llm_loop(...) call ...
    # ... existing val_reward_fn(...) call ...

    chunk_payload = {
        "chunk_idx": chunk_idx,
        "sample_indices": list(range(
            chunk_idx * self.config.data.val_batch_size,
            chunk_idx * self.config.data.val_batch_size + len(test_batch),
        )),
        "envs_var":       [env.get_tracking_variables() for env in envs],
        "reward_tensor":  reward_tensor.cpu(),
        "turns_tensor":   test_batch.batch['turns'].cpu(),
        "data_sources":   list(test_batch.non_tensor_batch.get(
            'data_source', ['unknown'] * reward_tensor.shape[0])),
        "end_lst":        end_lst,
        "answer_lst":     answer_lst,
        "format_lst":     format_lst,
        "sample_inputs":  input_texts,
        "sample_outputs": output_texts,
        "sample_scores":  answer_lst,
    }
    self._write_chunk(shard_dir, chunk_idx, chunk_payload)   # atomic write + FileLock
    done_chunks[chunk_idx] = chunk_payload

    total_done_samples += len(test_batch)
    val_pbar.update(len(test_batch))
    val_pbar.set_postfix(
        chunk=f"{chunk_idx + 1}/{self.val_num_chunks}",
        acc=f"{np.mean([s for c in done_chunks.values() for s in c['answer_lst']]):.4f}",
    )

val_pbar.close()
```

### F. Final aggregation (unchanged semantics)

After the loop, stitch all chunks back into the same flat structures the current code
uses, then run the **exact same** `data_source_*` / `metric_dict` code:

```python
chunks_sorted   = [done_chunks[i] for i in sorted(done_chunks)]
reward_tensor   = torch.cat([c["reward_tensor"] for c in chunks_sorted], dim=0).sum(-1)
turns_tensor    = torch.cat([c["turns_tensor"]  for c in chunks_sorted], dim=0)
data_sources    = np.array([ds for c in chunks_sorted for ds in c["data_sources"]])
end_lst         = [x for c in chunks_sorted for x in c["end_lst"]]
answer_lst      = [x for c in chunks_sorted for x in c["answer_lst"]]
format_lst      = [x for c in chunks_sorted for x in c["format_lst"]]
envs_var        = [x for c in chunks_sorted for x in c["envs_var"]]
sample_inputs   = [x for c in chunks_sorted for x in c["sample_inputs"]]
sample_outputs  = [x for c in chunks_sorted for x in c["sample_outputs"]]
sample_scores   = [x for c in chunks_sorted for x in c["sample_scores"]]
```

Then reuse the existing `for i in range(reward_tensor.shape[0]): ...` block verbatim to
build `metric_dict`, and call `self._maybe_log_val_generations(...)` with the
concatenated inputs/outputs/scores. The metrics and wandb table must be bit-identical to
today's output for an uninterrupted run.

### G. Consolidation for downstream compatibility

After aggregation succeeds, write the single consolidated pickle exactly where
`_save_env_var` writes today (keep that method; just let `run()` call it with the
concatenated `envs_var`), then `touch <shard_dir>/DONE` inside the same `FileLock`.

```python
self._save_env_var(envs_var)
open(os.path.join(shard_dir, "DONE"), "w").close()
```

This keeps every downstream consumer of `val_env_var.pkl` working unchanged.

### H. Helper methods to add

- `_resolve_shard_dir(self) -> str`
- `_scan_done_chunks(self, shard_dir) -> dict[int, dict]` — load existing `chunk_*.pkl`
  (ignore `*.tmp`), return keyed by `chunk_idx`.
- `_write_chunk(self, shard_dir, chunk_idx, payload)` — pickle to `.tmp`, `os.replace`,
  guarded by the existing `FileLock` pattern from `_save_env_var`.

### I. What must **not** change

- `ToolGenerationManager.run_llm_loop(...)` signature / internals.
- `RewardManager` / `val_reward_fn` signature / return shape.
- `ToolRLDataset` / `collate_fn`.
- The final `metric_dict` keys and values for an uninterrupted run.
- The on-disk path or schema of `val_env_var.pkl` (it's still the single consolidated
  list of `env.get_tracking_variables()` dicts).


## Desired behavior

1. **Chunked execution.** Split the validation set into chunks of configurable size (default `64`, overridable from `test.sbatch` / Hydra, e.g. `data.val_batch_size=64`). Process chunks sequentially; within a chunk keep the existing parallel API calls (the goal is checkpointing granularity, not reducing concurrency).
2. **Incremental checkpointing.** After each completed chunk, persist the per-sample results (env tracking variables + any scores/metadata needed later) to disk *before* starting the next chunk. A crash must never lose more than one chunk's worth of work.
3. **Resumable re-launch.** On startup, detect already-completed samples from the on-disk state and skip them; only unfinished samples are sent to the API. If a previous run finished completely, re-launching should be a no-op (or just recompute aggregate metrics).
4. **Final aggregation unchanged.** The final `metric_dict` (per `data_source` reward / end / answer / format / turns means) and the wandb `validation_generations` table must be identical in shape/semantics to the current implementation when a run completes without crashes. A final consolidated `val_env_var.pkl` (the list that today is written once) should still exist at the end for backward compatibility with downstream tooling that already reads it.

## Acceptance criteria

- Running `sbatch test.sbatch` with the current default config produces the same final `val_env_var.pkl` content (same length, same per-sample fields) as before, plus a populated `val_shard_dir/`.
- Killing the job mid-run (e.g. `scancel` after a few chunks) leaves a partially populated `val_shard_dir/`. Re-submitting `test.sbatch` with the same `CHECKPOINT_DIR` and dataset resumes: the tqdm bar starts at `total_done = <already-completed count>`, only the remaining samples hit the API, and the final metrics / consolidated pickle match a fresh uninterrupted run.
- Setting `trainer.val_resume=False` forces a full rerun (shard dir is cleared first).
- No change is required to callers outside `ValidationPipeline` and `test.sbatch` (aside from optionally passing `data.val_batch_size=...`).

## Non-goals / constraints

- Do not change the reward function, tokenizer, tool env, or `ToolGenerationManager` internals.
- Do not introduce Ray, GPU workers, or training-side logic — this module is deliberately API-only.
- Keep the single-process, single-node assumption; multi-worker sharding is out of scope.
- Keep `FileLock` usage for the consolidated pickle so concurrent re-launches do not corrupt it.









2026-04-22

## Goal

Build a **self-distillation SFT data collection pipeline** on top of the existing
`ValidationPipeline` (`@agent_r1/src/agent_ray_trainer_retro_noback.py`, entrypoint
`@run_eval_rjob.sh`). We want to harvest high-quality retrosynthesis trajectories that
the **current policy model itself** produces, and dump them as a standard chat-format
SFT training file. We deliberately do **not** use trajectories from a stronger or
different model, because a distribution mismatch hurts subsequent SFT on this policy.

Work is scoped to a dedicated branch `sft_data_collection`. **This branch exists only
for SFT data collection** — it is not required to stay compatible with the pure
validation flow on `main`. Feel free to hard-wire collection behavior, repurpose the
shard layout, and simplify config: there is no need for an opt-in `sft_collect` flag,
no need to keep the old `DATASET=chembl`/`DATASET=retro` cases working, and no need to
preserve the exact contents of `val_env_var.pkl` / wandb tables as long as the SFT
outputs are correct.

The validation/inference machinery already does per-chunk rollouts, chunk-level pickle
checkpointing, and resume — reuse all of that and add the SFT-formatted output next to
the existing `val_shards/` directory.

## What to change

### 1. Increase rollouts per query

In `run_eval_rjob.sh`, override `actor_rollout_ref.rollout.val_kwargs.n` to a larger
value (expose as `ROLLOUT_N`, default `8`; allow `16`). More rollouts per prompt ⇒
higher chance that at least one trajectory succeeds and that the shortest successful
one is actually short. Also flip `val_kwargs.do_sample=True` and set a non-zero
temperature (expose `ROLLOUT_TEMP`, default `0.7`) — greedy (`T=0`, `do_sample=False`)
makes the `n` rollouts identical and defeats the point.

Keep in mind the existing repeat pattern at
`agent_r1/src/agent_ray_trainer_retro_noback.py:215`:
```python
test_batch = test_batch.repeat(repeat_times=n, interleave=True)
```
With `interleave=True`, for original query index `q` the `n` rollouts occupy positions
`[q*n, q*n+1, …, q*n+(n-1)]` in the expanded batch. **Use this grouping** when
selecting top-k successful trajectories per query (see §3).

### 2. Point `run_eval_rjob.sh` at the training set

This branch is dedicated to SFT collection, so the script's purpose changes:
`DATASET` no longer selects between validation files — just default `VAL_FILES` to
`./data/reaction_pathway_search/train_h4_10.parquet`. You can drop the `chembl` /
`retro` case branches entirely, or keep a single `train_h4_10` case as the only
accepted value. The `VAL_FILES` variable name can stay (it is just the input parquet
to the pipeline).

No need for a separate on/off flag: every run on this branch collects SFT data.

### 3. Success filtering + top-2 shortest per query

In the chunk loop (around `agent_ray_trainer_retro_noback.py:281-300` where
`chunk_payload` is assembled), after rewards are computed, build a per-query view:

- A rollout `i` in the expanded chunk belongs to query `q = i // n` (within the chunk)
  and has absolute dataset index `sample_indices[q]`.
- A trajectory is **successful** iff the existing success signal is positive —
  use `answer_lst[i]` (the same per-sample success flag already saved to the chunk
  pickle and summed into the reported `acc`). Confirm the exact success predicate by
  reading the reward function; if `answer_lst[i]` is a float, success is `> 0`.
- Trajectory **length** = number of agent turns. Use `test_batch.batch['turns'][i]`
  (already saved as `turns_tensor`). Shorter = better.
- For each query, sort its successful rollouts by `turns` ascending, break ties by
  total generated token length (`len(output_ids[i])`) ascending, and keep at most
  **top-2**. Queries with zero successful rollouts contribute nothing.

Do this grouping **inside the chunk** so nothing is recomputed during resume.

### 4. Store SFT records in the chunk pickle, not a separate file

Add a new key `sft_records: list[dict]` to `chunk_payload`. Each record is one accepted
trajectory in the final SFT schema:

```python
{
    "id": f"{dataset_tag}__q{abs_query_idx:06d}__r{rollout_rank}__turns{n_turns}",
    "conversations": [
        {"role": "system",    "content": <system_prompt_used_for_this_rollout>},
        {"role": "user",      "content": <decoded_initial_user_prompt>},
        {"role": "assistant", "content": <assistant_turn_1>},
        {"role": "tool",      "content": <tool_response_1>},     # if applicable
        {"role": "assistant", "content": <assistant_turn_2>},
        ...
    ],
    "meta": {
        "abs_query_idx": int,
        "rollout_rank":   int,    # 0 = shortest, 1 = second shortest
        "n_turns":        int,
        "success":        True,
        "data_source":    str,
        "force_noloop":   bool,
        "rollout_n":      int,
        "rollout_temp":   float,
        "model_path":     str,    # actor_rollout_ref.model.path
        "api_model":      str,    # tool.api_model_name if API eval
    },
}
```

The `conversations` list should be reconstructed from the actual multi-turn trace
captured by `ToolGenerationManager` / `run_llm_loop`, **not** from re-splitting the
decoded `output_texts` string. Read
`@agent_r1/llm_agent/generation_retro_noback.py` and locate the per-turn records
(tool call messages + tool responses) that the env/tracker already keeps — this is
what `env.get_tracking_variables()` exposes and what gets saved into `envs_var`.
If a clean message-list view isn't exposed yet, add a small helper on the env/tracker
that returns the exact `[{role, content}, ...]` sequence for one rollout.

Storing SFT records per-chunk means resume works for free: the existing
`chunk_XXXXX.pkl` + `val_resume=True` path already skips completed chunks.

### 5. Periodic consolidation into `data_sft/`

Add `trainer.sft_save_every: 50` (chunks, not samples). After every `sft_save_every`
newly-completed chunks (and always at the end), walk the shard dir, concatenate
`sft_records` from every present chunk, and write a single parquet to:

```
data_sft/
  sft__<dataset_tag>__model-<model_tag>__noloop-<force_noloop>__n<rollout_n>__T<temp>__step<chunks_done>.parquet
```

Three required columns in the parquet: `id` (string), `conversations` (list[struct]
or JSON-serialized string — pick whichever the downstream SFT trainer reads; if
unsure, JSON-serialize to be safe), and `meta` (JSON-serialized string of the meta
dict from §4 — serialize to a string so the parquet schema stays stable even if new
meta keys are added later). Rotate by **overwriting the latest snapshot** atomically
(`write to .tmp → os.replace`) under the same `FileLock` pattern used for the
consolidated pickle — we never want a half-written parquet.

Step counter `chunks_done` = number of chunks whose pickles exist in `val_shards/`,
so after a crash+resume the filename at the next flush reflects real progress.

### 6. Final file at end of run

When the validation loop finishes (existing `DONE` sentinel path), write one final
consolidated parquet named `…__stepFINAL.parquet` in `data_sft/`. Leave the intermediate
`__stepNNN.parquet` snapshots in place — they are the resume/debug breadcrumbs.

## Constraints / non-goals

- Do **not** re-run generation during resume. All SFT record construction must happen
  inside the chunk loop before the pickle is written, so resume simply reads pickles
  back.
- Do **not** write SFT data outside `data_sft/`. That directory is the single sink.
- No multi-node / multi-worker sharding of SFT collection — single process, same as
  the current pipeline.
- Do **not** touch `main_agent_retro_noback.py` wiring beyond forwarding the new
  config keys.
- No need to keep the old `DATASET=retro` / `DATASET=chembl` paths working on this
  branch, and no need to preserve bit-for-bit equivalence of `val_env_var.pkl` /
  wandb metrics with `main`. This branch is for SFT collection only.

## Acceptance criteria

- `DATASET=train_h4_10 ROLLOUT_N=8 FORCE_NOLOOP=True bash run_eval_rjob.sh` runs
  end-to-end and produces:
  - Populated `val_shards/chunk_XXXXX.pkl` with the new `sft_records` key.
  - Periodic `data_sft/sft__train_h4_10__…__stepNNN.parquet` snapshots (every 50
    chunks) plus a final `__stepFINAL.parquet`.
  - Each row has the required columns `id`, `conversations`, and `meta`;
    `conversations` is a valid alternating chat list starting with `system` →
    `user` → `assistant …`; `meta` is a JSON-serialized string of the per-record
    meta dict from §4.
  - Per query, at most 2 rows; each row corresponds to a **successful** rollout; the
    two rows for the same query are the shortest-by-turns ones.
- Killing the job mid-run and re-running with the same `CHECKPOINT_DIR` resumes:
  only missing chunks regenerate, and the final parquet matches an uninterrupted run
  (same `id` set, same `conversations` content).


