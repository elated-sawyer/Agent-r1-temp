
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
