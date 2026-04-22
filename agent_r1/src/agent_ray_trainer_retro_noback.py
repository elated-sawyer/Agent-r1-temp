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
import glob
import json
import shutil

import numpy as np
import pandas as pd
import torch
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
        # SFT data collection requires the raw per-sample chat messages so we
        # can reconstruct clean (system, user, assistant, tool, ...) traces
        # without having to re-split the templated tokenized prompt. Force
        # this on regardless of what the user set in config/data.
        self.val_dataset = ToolRLDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_start_length,
            filter_prompts=True,
            return_raw_chat=True,
            truncation='error',
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            tool_env=self.val_env,
            use_custom_tool_format_func=self.config.data.get('use_custom_tool_format_func', False),
        )
        val_batch_size = int(self.config.data.get('val_batch_size', 64))
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        self.val_total_samples = len(self.val_dataset)
        self.val_num_chunks = len(self.val_dataloader)

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

    def _resolve_shard_dir(self):
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir,
            f'global_step_{self.global_steps}',
        )
        override_dir = self.config.trainer.get('val_shard_dir', None)
        if override_dir is not None:
            return override_dir
        return os.path.join(local_global_step_folder, 'val_shards')

    def _scan_done_chunks(self, shard_dir):
        chunk_payloads = {}
        pattern = os.path.join(shard_dir, 'chunk_*.pkl')
        for chunk_path in sorted(glob.glob(pattern)):
            chunk_name = os.path.basename(chunk_path)
            try:
                chunk_idx = int(chunk_name.replace('chunk_', '').replace('.pkl', ''))
            except ValueError:
                continue
            with open(chunk_path, 'rb') as f:
                chunk_payloads[chunk_idx] = pickle.load(f)
        return chunk_payloads

    def _write_chunk(self, shard_dir, chunk_idx, payload):
        lock_filename = f"val_shard_{hash(shard_dir) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)
        final_path = os.path.join(shard_dir, f'chunk_{chunk_idx:05d}.pkl')
        tmp_path = f'{final_path}.tmp'
        with FileLock(lock_path, timeout=60):
            os.makedirs(shard_dir, exist_ok=True)
            with open(tmp_path, 'wb') as f:
                pickle.dump(payload, f)
            os.replace(tmp_path, final_path)

    # ------------------------------------------------------------------
    # SFT self-distillation helpers
    # ------------------------------------------------------------------
    @property
    def _dataset_tag(self) -> str:
        """Stable short identifier for the input dataset, used in the SFT
        parquet filename. Falls back to 'dataset' if val_files is empty."""
        val_files = self.config.data.val_files
        if isinstance(val_files, (list, tuple)):
            first = val_files[0] if val_files else 'dataset'
        else:
            first = val_files or 'dataset'
        return os.path.splitext(os.path.basename(str(first)))[0] or 'dataset'

    @property
    def _model_tag(self) -> str:
        model_path = str(self.config.actor_rollout_ref.model.path or 'model')
        return os.path.basename(model_path.rstrip('/')) or 'model'

    def _sft_output_dir(self) -> str:
        """Directory where consolidated SFT parquet snapshots are written.

        Kept at repo-relative ``data_sft/`` per the spec: a single sink for
        all SFT collection runs; disambiguation lives in the filename.
        """
        out_dir = 'data_sft'
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _sft_snapshot_filename(self, step_tag: str) -> str:
        n = int(self.config.actor_rollout_ref.rollout.val_kwargs.get('n', 1))
        temp = float(self.config.actor_rollout_ref.rollout.val_kwargs.get('temperature', 0.0))
        force_noloop = bool(self.config.tool.get('force_noloop', False))
        return (
            f"sft__{self._dataset_tag}__model-{self._model_tag}__"
            f"noloop-{force_noloop}__n{n}__T{temp:.2f}__step{step_tag}.parquet"
        )

    def _raw_chat_to_system_user(self, raw_chat):
        """Extract (system_content, user_content) from a per-sample chat list.

        ``raw_chat`` is whatever ``ToolRLDataset`` stashed under the
        ``raw_prompt`` non-tensor key, which reflects whatever modifications
        (e.g. ``use_custom_tool_format_func``) the dataset applied before
        chat-template rendering — so re-applying the same chat template at
        SFT training time will reproduce the exact prompt the policy saw at
        rollout time, regardless of that flag's value.
        """
        if raw_chat is None:
            return "", ""
        if hasattr(raw_chat, 'tolist'):
            chat_list = raw_chat.tolist()
        else:
            chat_list = list(raw_chat)
        system_content = ""
        user_content = ""
        for msg in chat_list:
            role = msg.get('role') if isinstance(msg, dict) else None
            content = msg.get('content') if isinstance(msg, dict) else None
            if role == 'system' and not system_content:
                system_content = content or ""
            elif role == 'user' and not user_content:
                user_content = content or ""
        return system_content, user_content

    def _build_sft_records(self, sample_indices, envs, raw_prompts,
                           answer_lst, turns_list, response_lengths,
                           data_sources):
        """Assemble the per-chunk SFT record list.

        For each original query in the (pre-repeat) chunk, group its ``n``
        rollouts, keep the successful ones, sort by (turns asc, generated
        token length asc), and emit the top 2. Each record is a single SFT
        training example in the final chat schema.
        """
        n = int(self.config.actor_rollout_ref.rollout.val_kwargs.get('n', 1))
        records = []
        force_noloop = bool(self.config.tool.get('force_noloop', False))
        rollout_temp = float(self.config.actor_rollout_ref.rollout.val_kwargs.get('temperature', 0.0))
        model_path = str(self.config.actor_rollout_ref.model.path or '')
        api_model = str(self.config.tool.get('api_model_name', ''))

        for q, abs_query_idx in enumerate(sample_indices):
            # Collect successful rollouts for this query.
            rollouts = []
            for k in range(n):
                i = q * n + k
                if i >= len(answer_lst):
                    break
                success = float(answer_lst[i]) > 0.0
                if not success:
                    continue
                rollouts.append({
                    'i': i,
                    'turns': int(turns_list[i]),
                    'gen_len': int(response_lengths[i]),
                })
            if not rollouts:
                continue
            # Shortest-by-turns, tie-break by generated-token-length.
            rollouts.sort(key=lambda r: (r['turns'], r['gen_len']))
            selected = rollouts[:2]

            # All n rollouts share the same prompt thanks to
            # test_batch.repeat(..., interleave=True); use the first as canon.
            base_i = q * n
            raw_chat = None
            if raw_prompts is not None and base_i < len(raw_prompts):
                raw_chat = raw_prompts[base_i]
            system_content, user_content = self._raw_chat_to_system_user(raw_chat)

            for rank, r in enumerate(selected):
                env = envs[r['i']]
                conversations = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
                conversations.extend(env.get_conversation_messages())
                data_source = "unknown"
                if data_sources is not None and r['i'] < len(data_sources):
                    data_source = str(data_sources[r['i']])

                meta = {
                    "abs_query_idx": int(abs_query_idx),
                    "rollout_rank": int(rank),
                    "n_turns": int(r['turns']),
                    "success": True,
                    "data_source": data_source,
                    "force_noloop": force_noloop,
                    "rollout_n": n,
                    "rollout_temp": rollout_temp,
                    "model_path": model_path,
                    "api_model": api_model,
                }
                rec_id = (
                    f"{self._dataset_tag}__q{int(abs_query_idx):06d}"
                    f"__r{rank}__turns{int(r['turns'])}"
                )
                records.append({
                    "id": rec_id,
                    "conversations": conversations,
                    "meta": meta,
                })
        return records

    def _consolidate_sft(self, shard_dir: str, final: bool = False) -> str:
        """Read all chunk pickles in ``shard_dir``, concatenate their
        ``sft_records`` lists, and atomically write a single parquet
        snapshot into ``data_sft/``.

        Returns the written parquet path. The filename step tag is
        ``FINAL`` when ``final=True`` and the zero-padded count of
        completed chunk pickles otherwise — which, on resume, reflects
        real progress rather than per-run progress.
        """
        chunk_files = sorted(glob.glob(os.path.join(shard_dir, 'chunk_*.pkl')))
        chunks_done = len(chunk_files)

        all_records = []
        for cf in chunk_files:
            try:
                with open(cf, 'rb') as f:
                    payload = pickle.load(f)
            except Exception as exc:
                print(f'[SFT] Skipping unreadable chunk pickle {cf}: {exc}')
                continue
            all_records.extend(payload.get('sft_records', []) or [])

        rows = []
        for rec in all_records:
            rows.append({
                'id': rec['id'],
                # JSON-serialize both structured fields so the parquet schema
                # stays stable as meta evolves and so downstream SFT trainers
                # don't have to deal with variable-shape list[struct] columns.
                'conversations': json.dumps(rec['conversations'], ensure_ascii=False),
                'meta': json.dumps(rec['meta'], ensure_ascii=False),
            })
        df = pd.DataFrame(rows, columns=['id', 'conversations', 'meta'])

        out_dir = self._sft_output_dir()
        step_tag = 'FINAL' if final else f'{chunks_done:05d}'
        final_path = os.path.join(out_dir, self._sft_snapshot_filename(step_tag))
        tmp_path = f'{final_path}.tmp'

        lock_filename = f"val_shard_{hash(shard_dir) & 0xFFFFFFFF:08x}.lock"
        lock_path = os.path.join(tempfile.gettempdir(), lock_filename)
        with FileLock(lock_path, timeout=60):
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, final_path)
        print(
            f'[SFT] Wrote {len(rows)} records from {chunks_done} chunks '
            f'to {final_path}'
        )
        return final_path

    def validate(self):
        import time
        val_start_time = time.time()
        shard_dir = self._resolve_shard_dir()
        if not self.config.trainer.get('val_resume', True):
            shutil.rmtree(shard_dir, ignore_errors=True)
        os.makedirs(shard_dir, exist_ok=True)
        done_flag_path = os.path.join(shard_dir, 'DONE')
        done_chunks = self._scan_done_chunks(shard_dir)

        # Agent config preparation
        val_kwargs = self.config.actor_rollout_ref.rollout.val_kwargs
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
            do_sample=bool(val_kwargs.get('do_sample', False)),
            temperature=float(val_kwargs.get('temperature', 0.0)),
            top_p=float(val_kwargs.get('top_p', 1.0)),
            debug=self.config.tool.get('debug', False),
        )

        generation_manager = ToolGenerationManager(
            tokenizer=self.tokenizer,
            config=gen_config,
            is_validation=True,
        )

        if os.path.exists(done_flag_path) and self.config.trainer.get('val_resume', True):
            print(f'[Validation] DONE sentinel found at {done_flag_path}, loading shard cache only.')
        else:
            val_batch_size = int(self.config.data.get('val_batch_size', 64))
            total_done_samples = sum(len(chunk['sample_indices']) for chunk in done_chunks.values())
            all_done_answers = [score for chunk in done_chunks.values() for score in chunk['answer_lst']]
            sft_save_every = int(self.config.trainer.get('sft_save_every', 50))
            newly_done_chunks = 0
            val_pbar = tqdm(
                total=self.val_total_samples,
                initial=total_done_samples,
                desc="Validation samples",
                leave=True,
            )
            for chunk_idx, test_data in enumerate(self.val_dataloader):
                chunk_sample_count = len(test_data['input_ids'])
                if chunk_idx in done_chunks:
                    continue

                test_batch = DataProto.from_single_dict(test_data)

                # repeat test batch
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                               interleave=True)
                envs = [self.val_env.copy(test_batch.non_tensor_batch['target'][ii]) for ii in range(len(test_batch))]

                # Store original inputs
                input_ids = test_batch.batch['input_ids']
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

                if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    test_gen_batch = test_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    'validate': True,
                }
                if chunk_idx == 0:
                    print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

                # No padding needed for API-based generation (no GPU-divisible size requirement)
                first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                gen_start_time = time.time()
                test_output_gen_batch = generation_manager.run_llm_loop(
                    test_gen_batch,
                    envs=envs,
                    initial_input_ids=first_input_ids,
                )

                gen_elapsed = time.time() - gen_start_time
                print(f'[Validation chunk {chunk_idx}] Generation complete in {gen_elapsed:.1f}s. Computing rewards...')

                for key in test_output_gen_batch.batch.keys():
                    test_output_gen_batch.batch[key] = test_output_gen_batch.batch[key].long()

                # Store generated outputs
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                try:
                    reward_tensor, end_lst, answer_lst, format_lst = self.val_reward_fn(test_batch, envs)
                except Exception:
                    print(f"[Error] Something wrong with the reward function")
                    print(test_batch)
                    exit()

                all_done_answers.extend(answer_lst)
                val_pbar.update(chunk_sample_count)
                val_pbar.set_postfix(
                    chunk=f"{chunk_idx + 1}/{self.val_num_chunks}",
                    acc=f"{np.mean(all_done_answers):.4f}" if all_done_answers else "nan",
                    last_gen=f"{gen_elapsed:.1f}s",
                )

                sample_indices = list(range(
                    chunk_idx * val_batch_size,
                    min(chunk_idx * val_batch_size + chunk_sample_count, self.val_total_samples),
                ))
                data_sources_list = list(
                    test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
                )
                turns_tensor_cpu = test_batch.batch['turns'].cpu()
                # Non-pad generated-token count per rollout — used as the
                # tie-breaker when two successful rollouts share turn count.
                pad_id = self.tokenizer.pad_token_id
                response_lengths = [
                    int((row != pad_id).sum().item()) for row in output_ids
                ]
                raw_prompts = test_batch.non_tensor_batch.get('raw_prompt', None)
                sft_records = self._build_sft_records(
                    sample_indices=sample_indices,
                    envs=envs,
                    raw_prompts=raw_prompts,
                    answer_lst=answer_lst,
                    turns_list=[int(x) for x in turns_tensor_cpu.tolist()],
                    response_lengths=response_lengths,
                    data_sources=data_sources_list,
                )

                chunk_payload = {
                    'chunk_idx': chunk_idx,
                    'sample_indices': sample_indices,
                    'envs_var': [env.get_tracking_variables() for env in envs],
                    # Reduce over the response-length dim here so chunks with
                    # different dynamic response lengths can be concatenated
                    # safely later. Shape: [B] instead of [B, L].
                    'reward_tensor': reward_tensor.sum(-1).cpu(),
                    'turns_tensor': turns_tensor_cpu,
                    'data_sources': data_sources_list,
                    'end_lst': end_lst,
                    'answer_lst': answer_lst,
                    'format_lst': format_lst,
                    'sample_inputs': input_texts,
                    'sample_outputs': output_texts,
                    'sample_scores': answer_lst,
                    'sft_records': sft_records,
                }
                self._write_chunk(shard_dir, chunk_idx, chunk_payload)
                done_chunks[chunk_idx] = chunk_payload

                newly_done_chunks += 1
                if sft_save_every > 0 and newly_done_chunks % sft_save_every == 0:
                    try:
                        self._consolidate_sft(shard_dir, final=False)
                    except Exception as exc:
                        print(f'[SFT] Periodic consolidation failed (non-fatal): {exc}')
            val_pbar.close()

        chunks_sorted = [done_chunks[i] for i in sorted(done_chunks)]
        if len(chunks_sorted) == 0:
            raise RuntimeError("No validation chunks available for aggregation.")

        # Normalize chunk reward tensors to 1D [B]. New chunks are already 1D;
        # legacy cached chunks may still be 2D [B, L] with varying L, so reduce
        # here for backward compatibility before concatenating along dim 0.
        reward_tensor = torch.cat(
            [
                c['reward_tensor'] if c['reward_tensor'].dim() == 1 else c['reward_tensor'].sum(-1)
                for c in chunks_sorted
            ],
            dim=0,
        )
        turns_tensor = torch.cat([chunk['turns_tensor'] for chunk in chunks_sorted], dim=0)
        data_sources = np.array([source for chunk in chunks_sorted for source in chunk['data_sources']])
        end_lst = [item for chunk in chunks_sorted for item in chunk['end_lst']]
        answer_lst = [item for chunk in chunks_sorted for item in chunk['answer_lst']]
        format_lst = [item for chunk in chunks_sorted for item in chunk['format_lst']]
        envs_var = [item for chunk in chunks_sorted for item in chunk['envs_var']]
        sample_inputs = [item for chunk in chunks_sorted for item in chunk['sample_inputs']]
        sample_outputs = [item for chunk in chunks_sorted for item in chunk['sample_outputs']]
        sample_scores = [item for chunk in chunks_sorted for item in chunk['sample_scores']]

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        reward_tensor = reward_tensor.cpu()
        turns_tensor = turns_tensor.cpu()

        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_end = {}
        data_source_answer = {}
        data_source_format = {}
        data_source_turns = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            if data_source not in data_source_end:
                data_source_end[data_source] = []
            data_source_end[data_source].append(end_lst[i])
            if data_source not in data_source_answer:
                data_source_answer[data_source] = []
            data_source_answer[data_source].append(answer_lst[i])
            if data_source not in data_source_format:
                data_source_format[data_source] = []
            data_source_format[data_source].append(format_lst[i])
            if data_source not in data_source_turns:
                data_source_turns[data_source] = []
            data_source_turns[data_source].append(turns_tensor[i])

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
        total_samples = reward_tensor.shape[0]
        self._save_env_var(envs_var)
        with FileLock(os.path.join(tempfile.gettempdir(), f"val_shard_{hash(shard_dir) & 0xFFFFFFFF:08x}.lock"), timeout=60):
            with open(done_flag_path, 'w'):
                pass
        # Always emit the final consolidated SFT parquet, regardless of
        # whether periodic snapshots fired: this is the authoritative output
        # of the collection run.
        try:
            self._consolidate_sft(shard_dir, final=True)
        except Exception as exc:
            print(f'[SFT] Final consolidation failed: {exc}')
            raise
        print(f'[Validation] Completed in {time.time() - val_start_time:.1f}s — {total_samples} samples')
        return metric_dict, envs_var

    def run(self):
        """Run the validation pipeline: set up logger, validate, log results."""
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        val_metrics, _ = self.validate()
        pprint(f'Validation metrics: {val_metrics}')
        logger.log(data=val_metrics, step=self.global_steps)
