# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Validation-only pipeline using the API model path.
All Ray cluster dependencies and training logic have been removed.
"""

import os
from pprint import pprint

import numpy as np
from omegaconf import OmegaConf
from verl import DataProto
from .agent_rl_dataset import ToolRLDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger, Tracking
from torchdata.stateful_dataloader import StatefulDataLoader

from agent_r1.llm_agent.generation_retro_noback import ToolGenerationManager, ToolGenerationConfig
from agent_r1.tool.tool_env import ToolEnv

from tqdm.auto import tqdm
from filelock import FileLock
import tempfile
import pickle
import hashlib


class ValidationPipeline(object):
    """
    Validation-only pipeline. Runs inference via API and computes metrics.
    No Ray, no GPU workers, no training loop.
    """

    def __init__(self, config, tokenizer, processor=None, val_reward_fn=None, env: ToolEnv = None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.val_reward_fn = val_reward_fn
        self.val_env = env
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.global_steps = 0

        self._create_dataloader()

    def _create_dataloader(self):
        self.val_dataset = ToolRLDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_start_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            tool_env=self.val_env,
            use_custom_tool_format_func=self.config.data.get('use_custom_tool_format_func', False),
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _save_env_var(self, env_list):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        local_latest_checkpointed_iteration = os.path.join(local_global_step_folder,
                                                           'val_env_var.pkl')

        # Using hash value of path as lock file name to avoid long file name
        lock_filename = f"env_{hash(local_global_step_folder) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

        try:
            with FileLock(lock_path, timeout=60):
                os.makedirs(local_global_step_folder, exist_ok=True)
                with open(local_latest_checkpointed_iteration, 'wb') as f:
                    pickle.dump(env_list, f)
        except Exception as e:
            print(f"Warning: Failed to acquire lock for {local_global_step_folder}: {e}")
            os.makedirs(local_global_step_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Crash-safe / resumable validation helpers
    # ------------------------------------------------------------------

    def _val_shard_dir(self):
        """Directory holding per-sample shard pickles for the current step."""
        override = self.config.trainer.get('val_shard_dir', None)
        if override:
            return override
        return os.path.join(
            self.config.trainer.default_local_dir,
            f'global_step_{self.global_steps}',
            'val_shards',
        )

    def _compute_stable_keys(self, non_tensor_batch, total_n, copies_per_sample):
        """Return one stable filename-safe key per expanded sample position.

        Identifier preference (stable across reruns of the same dataset):
          1. ``non_tensor_batch['idx']`` if present,
          2. ``non_tensor_batch['index']`` (ToolRLDataset copies this from
             ``extra_info.index`` during preprocessing),
          3. SHA-1 of ``(data_source | target | raw_prompt_ids)`` as a fallback.
        A ``_c{copy_idx}`` suffix distinguishes the n copies created by
        ``val_kwargs.n`` so they do not collide on disk.
        """
        explicit_field = None
        for f in ('idx', 'index'):
            if f in non_tensor_batch:
                explicit_field = f
                break

        keys = []
        for i in range(total_n):
            copy_idx = i % max(copies_per_sample, 1)
            h = hashlib.sha1()
            if explicit_field is not None:
                h.update(b'explicit:')
                h.update(str(non_tensor_batch[explicit_field][i]).encode())
            else:
                for fld in ('data_source', 'target'):
                    val = non_tensor_batch.get(fld, None)
                    if val is not None:
                        h.update(str(val[i]).encode())
                    h.update(b'|')
                if 'raw_prompt_ids' in non_tensor_batch:
                    raw = non_tensor_batch['raw_prompt_ids'][i]
                    # raw_prompt_ids is a list/array of int token ids which may
                    # exceed 255, so bytes(raw) is unsafe — serialize textually.
                    try:
                        h.update(','.join(str(int(x)) for x in raw).encode())
                    except (TypeError, ValueError):
                        h.update(str(raw).encode())
            keys.append(f"{h.hexdigest()[:16]}_c{copy_idx}")
        return keys

    @staticmethod
    def _shard_path(shard_dir, key):
        return os.path.join(shard_dir, f'sample_{key}.pkl')

    def _write_shard(self, shard_dir, key, record):
        """Atomically persist a single-sample record (tmp + os.replace)."""
        os.makedirs(shard_dir, exist_ok=True)
        final = self._shard_path(shard_dir, key)
        tmp = final + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(record, f)
        os.replace(tmp, final)

    def _load_shards(self, shard_dir):
        """Return {key: record} for every ``sample_*.pkl`` in ``shard_dir``."""
        if not os.path.isdir(shard_dir):
            return {}
        out = {}
        for fname in os.listdir(shard_dir):
            if not (fname.startswith('sample_') and fname.endswith('.pkl')):
                continue
            key = fname[len('sample_'):-len('.pkl')]
            try:
                with open(os.path.join(shard_dir, fname), 'rb') as f:
                    out[key] = pickle.load(f)
            except Exception as e:
                print(f"[Resume] Ignoring unreadable shard {fname}: {e}")
        return out

    def _clear_shards(self, shard_dir):
        if not os.path.isdir(shard_dir):
            return
        for fname in os.listdir(shard_dir):
            if fname.startswith('sample_') and (fname.endswith('.pkl') or fname.endswith('.pkl.tmp')):
                try:
                    os.remove(os.path.join(shard_dir, fname))
                except OSError:
                    pass

    def validate(self):
        import time
        val_start_time = time.time()

        # Agent config preparation
        gen_config = ToolGenerationConfig(
            max_turns=self.config.tool.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_tool_response_length=self.config.data.max_tool_response_length,
            num_gpus=1,  # API-based generation, no GPU-divisible size requirement
            use_batch_tool_calls=self.config.tool.use_batch_tool_calls,
            tool_call_start=self.config.tool.tool_call_start,
            tool_call_end=self.config.tool.tool_call_end,
            tool_response_start=self.config.tool.tool_response_start,
            tool_response_end=self.config.tool.tool_response_end,
            tool_custom_response_template=self.config.tool.tool_custom_response_template,
            use_api_model=self.config.tool.get('use_api_model', False),
            api_model_name=self.config.tool.get('api_model_name', ''),
            api_max_concurrency=self.config.tool.get('api_max_concurrency', 32),
            debug=self.config.tool.get('debug', False),
        )

        generation_manager = ToolGenerationManager(
            tokenizer=self.tokenizer,
            config=gen_config,
            is_validation=True,
        )

        # --- Shard dir + resume handling ---
        shard_dir = self._val_shard_dir()
        val_resume = bool(self.config.trainer.get('val_resume', True))
        if not val_resume:
            self._clear_shards(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)

        # --- Build the full validation batch (single DataLoader iteration). ---
        assert len(self.val_dataloader) == 1, (
            "Validation dataloader must have a single batch; chunking is applied "
            "inside validate() via data.val_batch_size."
        )
        test_data = next(iter(self.val_dataloader))
        full_batch = DataProto.from_single_dict(test_data)
        n_copies = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        full_batch = full_batch.repeat(repeat_times=n_copies, interleave=True)
        total_n = len(full_batch)

        # Stable keys are computed on the full repeated batch before any pop,
        # while raw_prompt_ids is still available as a fallback hash input.
        keys = self._compute_stable_keys(full_batch.non_tensor_batch, total_n, n_copies)
        assert len(set(keys)) == len(keys), (
            "Stable key collision detected — dataset has duplicate samples with no "
            "distinguishing 'idx'/'index' field. Add a unique 'index' in preprocessing."
        )

        # --- Identify completed vs. pending samples. ---
        completed = self._load_shards(shard_dir) if val_resume else {}
        # Drop any completed shard that does not correspond to a current-dataset key
        # (e.g. the dataset changed between runs). These are not trustworthy to reuse.
        key_set = set(keys)
        stale = [k for k in completed.keys() if k not in key_set]
        for k in stale:
            completed.pop(k, None)
        pending = [i for i, k in enumerate(keys) if k not in completed]

        val_batch_size = int(self.config.data.get('val_batch_size', 64))

        print(
            f'[Validation] total={total_n}, completed={len(completed)}, '
            f'pending={len(pending)}, chunk_size={val_batch_size}, '
            f'shard_dir={shard_dir}'
        )

        # --- Chunk loop. Skipped entirely if everything is already on disk. ---
        running_acc = [rec['answer'] for rec in completed.values()]
        total_samples_done = len(completed)

        val_pbar = tqdm(
            total=total_n,
            initial=total_samples_done,
            desc='Validation samples',
            leave=True,
        )
        meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            'validate': True,
        }
        first_chunk_log = True

        try:
            for chunk_start in range(0, len(pending), val_batch_size):
                chunk_idx = chunk_start // val_batch_size
                chunk_indices = pending[chunk_start:chunk_start + val_batch_size]
                chunk_keys = [keys[i] for i in chunk_indices]
                chunk_batch = full_batch[np.array(chunk_indices, dtype=np.int64)]

                envs = [
                    self.val_env.copy(chunk_batch.non_tensor_batch['target'][j])
                    for j in range(len(chunk_batch))
                ]

                # Capture input texts before pop removes input_ids.
                input_ids = chunk_batch.batch['input_ids']
                input_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids
                ]

                if 'multi_modal_inputs' in chunk_batch.non_tensor_batch.keys():
                    chunk_gen_batch = chunk_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    chunk_gen_batch = chunk_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                chunk_gen_batch.meta_info = meta_info
                if first_chunk_log:
                    print(f'test_gen_batch meta info: {chunk_gen_batch.meta_info}')
                    first_chunk_log = False

                first_input_ids = chunk_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                chunk_size_now = len(chunk_batch)
                val_pbar.set_postfix(batch_samples=chunk_size_now, total_done=total_samples_done)

                gen_start_time = time.time()
                chunk_output_gen_batch = generation_manager.run_llm_loop(
                    chunk_gen_batch,
                    envs=envs,
                    initial_input_ids=first_input_ids,
                )
                gen_elapsed = time.time() - gen_start_time
                print(
                    f'[Validation chunk {chunk_idx}] Generation complete in '
                    f'{gen_elapsed:.1f}s. Computing rewards...'
                )

                for key_t in chunk_output_gen_batch.batch.keys():
                    chunk_output_gen_batch.batch[key_t] = chunk_output_gen_batch.batch[key_t].long()

                output_ids = chunk_output_gen_batch.batch['responses']
                output_texts = [
                    self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
                ]

                chunk_batch_merged = chunk_batch.union(chunk_output_gen_batch)

                # Reward-function errors must not silently exit — propagate so
                # Slurm records a non-zero exit, shards already flushed persist.
                reward_tensor, end_lst, answer_lst, format_lst = self.val_reward_fn(
                    chunk_batch_merged, envs
                )

                turns_tensor = chunk_batch_merged.batch['turns']
                data_sources_chunk = chunk_batch_merged.non_tensor_batch.get(
                    'data_source', np.array(['unknown'] * reward_tensor.shape[0], dtype=object)
                )
                reward_scalar = reward_tensor.sum(-1).cpu()

                # Persist one shard per sample — atomic writes, so a crash after
                # any shard completes leaves that shard intact on disk.
                for j, k in enumerate(chunk_keys):
                    record = {
                        'env_var': envs[j].get_tracking_variables(),
                        'reward': float(reward_scalar[j].item()),
                        'turns': int(turns_tensor[j].item()),
                        'end': end_lst[j],
                        'answer': answer_lst[j],
                        'format': format_lst[j],
                        'data_source': str(data_sources_chunk[j]),
                        'input_text': input_texts[j],
                        'output_text': output_texts[j],
                    }
                    self._write_shard(shard_dir, k, record)
                    running_acc.append(answer_lst[j])

                total_samples_done += chunk_size_now
                val_pbar.update(chunk_size_now)
                val_pbar.set_postfix(
                    total_done=total_samples_done,
                    acc=f"{np.mean(running_acc):.4f}" if running_acc else "n/a",
                    last_gen=f"{gen_elapsed:.1f}s",
                )
        except Exception as e:
            print(f"[Error] Validation aborted; {total_samples_done}/{total_n} samples checkpointed. Reason: {e}")
            val_pbar.close()
            raise

        val_pbar.close()

        # --- Rehydrate everything in dataset order from on-disk shards. ---
        all_shards = self._load_shards(shard_dir)
        missing = [k for k in keys if k not in all_shards]
        if missing:
            raise RuntimeError(
                f"[Validation] {len(missing)} shard(s) missing after chunk loop; "
                f"first few: {missing[:5]}"
            )

        envs_var = [all_shards[k]['env_var'] for k in keys]
        sample_inputs = [all_shards[k]['input_text'] for k in keys]
        sample_outputs = [all_shards[k]['output_text'] for k in keys]
        sample_scores = [all_shards[k]['answer'] for k in keys]
        reward_per_sample = [all_shards[k]['reward'] for k in keys]
        turns_per_sample = [all_shards[k]['turns'] for k in keys]
        end_per_sample = [all_shards[k]['end'] for k in keys]
        answer_per_sample = [all_shards[k]['answer'] for k in keys]
        format_per_sample = [all_shards[k]['format'] for k in keys]
        data_source_per_sample = [all_shards[k]['data_source'] for k in keys]

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_end = {}
        data_source_answer = {}
        data_source_format = {}
        data_source_turns = {}
        for i in range(total_n):
            data_source = data_source_per_sample[i]
            data_source_reward.setdefault(data_source, []).append(reward_per_sample[i])
            data_source_end.setdefault(data_source, []).append(end_per_sample[i])
            data_source_answer.setdefault(data_source, []).append(answer_per_sample[i])
            data_source_format.setdefault(data_source, []).append(format_per_sample[i])
            data_source_turns.setdefault(data_source, []).append(turns_per_sample[i])

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        for data_source, ends in data_source_end.items():
            metric_dict[f'val/end_score/{data_source}'] = np.mean(ends)
        for data_source, answers in data_source_answer.items():
            metric_dict[f'val/answer_score/{data_source}'] = np.mean(answers)
        for data_source, formats in data_source_format.items():
            metric_dict[f'val/format_score/{data_source}'] = np.mean(formats)
        for data_source, turns in data_source_turns.items():
            metric_dict[f'val/turns/{data_source}'] = np.mean(turns)
        print(f'[Validation] Completed in {time.time() - val_start_time:.1f}s — {total_n} samples')
        return metric_dict, envs_var

    def run(self):
        """Run the validation pipeline: set up logger, validate, log results."""
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        val_metrics, val_env_list = self.validate()
        self._save_env_var(val_env_list)
        pprint(f'Validation metrics: {val_metrics}')
        logger.log(data=val_metrics, step=self.global_steps)
