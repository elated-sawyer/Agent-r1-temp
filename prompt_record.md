
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




