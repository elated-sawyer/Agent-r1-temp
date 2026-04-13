
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

