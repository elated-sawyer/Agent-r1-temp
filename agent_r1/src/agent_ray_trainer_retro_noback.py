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
import shutil


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
            batch_size=self.config.data.get('val_batch_size', 64),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def _get_val_checkpoint_paths(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        return {
            'global_step_dir': local_global_step_folder,
            'consolidated_env': os.path.join(local_global_step_folder, 'val_env_var.pkl'),
            'shard_dir': os.path.join(local_global_step_folder, 'val_shard_dir'),
        }

    def _get_sample_shard_path(self, sample_idx):
        return os.path.join(self._get_val_checkpoint_paths()['shard_dir'], f'sample_{sample_idx:08d}.pkl')

    @staticmethod
    def _slice_batch_dict(batch_dict, indices):
        sliced = {}
        for key, value in batch_dict.items():
            if torch.is_tensor(value):
                sliced[key] = value[indices]
            else:
                sliced[key] = [value[i] for i in indices]
        return sliced

    def _write_sample_shard(self, sample_idx, payload):
        shard_path = self._get_sample_shard_path(sample_idx)
        tmp_path = f"{shard_path}.tmp"
        with open(tmp_path, 'wb') as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, shard_path)

    def _load_sample_shard(self, sample_idx):
        with open(self._get_sample_shard_path(sample_idx), 'rb') as f:
            return pickle.load(f)

    def _load_completed_indices(self):
        shard_dir = self._get_val_checkpoint_paths()['shard_dir']
        if not os.path.isdir(shard_dir):
            return set()
        completed = set()
        for filename in os.listdir(shard_dir):
            if filename.startswith('sample_') and filename.endswith('.pkl'):
                try:
                    completed.add(int(filename[len('sample_'):-len('.pkl')]))
                except ValueError:
                    continue
        return completed

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
        paths = self._get_val_checkpoint_paths()
        local_global_step_folder = paths['global_step_dir']
        local_latest_checkpointed_iteration = paths['consolidated_env']

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

    def validate(self):
        import time
        val_start_time = time.time()
        reward_lst = []
        turns_lst = []
        data_source_lst = []
        end_score_lst = []
        answer_score_lst = []
        format_score_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

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

        val_resume = self.config.trainer.get('val_resume', True)
        checkpoint_paths = self._get_val_checkpoint_paths()
        shard_dir = checkpoint_paths['shard_dir']
        if not val_resume and os.path.isdir(shard_dir):
            shutil.rmtree(shard_dir)
        os.makedirs(shard_dir, exist_ok=True)
        completed_indices = self._load_completed_indices()

        envs_var = []
        total_samples_done = len(completed_indices)
        running_acc = []
        val_pbar = tqdm(self.val_dataloader, desc="Validation batches", leave=True)
        for batch_idx, test_data in enumerate(val_pbar):
            original_batch_size = len(test_data['input_ids'])
            batch_start = batch_idx * self.val_dataloader.batch_size
            batch_indices = list(range(batch_start, batch_start + original_batch_size))
            pending_indices = [idx for idx in batch_indices if idx not in completed_indices]

            if not pending_indices:
                for sample_idx in batch_indices:
                    shard = self._load_sample_shard(sample_idx)
                    envs_var.extend(shard['env_vars'])
                    reward_lst.extend(shard['reward'])
                    turns_lst.extend(shard['turns'])
                    data_source_lst.extend(shard['data_source'])
                    end_score_lst.extend(shard['end'])
                    answer_score_lst.extend(shard['answer'])
                    format_score_lst.extend(shard['format'])
                    sample_inputs.extend(shard['input_texts'])
                    sample_outputs.extend(shard['output_texts'])
                    sample_scores.extend(shard['sample_scores'])
                    running_acc.extend(shard['answer'])
                val_pbar.set_postfix(total_done=total_samples_done, acc=f"{np.mean(running_acc):.4f}" if running_acc else "nan")
                continue

            pending_local_positions = [i for i, idx in enumerate(batch_indices) if idx in pending_indices]
            pending_data = self._slice_batch_dict(test_data, pending_local_positions)
            test_batch = DataProto.from_single_dict(pending_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)
            envs = [self.val_env.copy(test_batch.non_tensor_batch['target'][ii]) for ii in range(len(test_batch))]

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

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
            if batch_idx == 0:
                print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # No padding needed for API-based generation (no GPU-divisible size requirement)
            first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
            batch_size = len(test_batch)
            val_pbar.set_postfix(batch_samples=batch_size, total_done=total_samples_done)
            gen_start_time = time.time()
            test_output_gen_batch = generation_manager.run_llm_loop(
                test_gen_batch,
                envs=envs,
                initial_input_ids=first_input_ids,
            )

            gen_elapsed = time.time() - gen_start_time
            print(f'[Validation batch {batch_idx}] Generation complete in {gen_elapsed:.1f}s. Computing rewards...')

            for key in test_output_gen_batch.batch.keys():
                test_output_gen_batch.batch[key] = test_output_gen_batch.batch[key].long()

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            try:
                reward_tensor, end_lst, answer_lst, format_lst = self.val_reward_fn(test_batch, envs)
            except:
                print(f"[Error] Something wrong with the reward function")
                print(test_batch)
                exit()

            # Store scores
            sample_scores.extend(answer_lst)
            total_samples_done += len(pending_indices)
            running_acc.extend(answer_lst)
            val_pbar.set_postfix(
                total_done=total_samples_done,
                acc=f"{np.mean(running_acc):.4f}",
                last_gen=f"{gen_elapsed:.1f}s",
            )

            reward_values = reward_tensor.sum(-1).cpu().tolist()
            turns_values = test_batch.batch['turns'].cpu().tolist()
            data_source_values = list(test_batch.non_tensor_batch.get('data_source', ['unknown'] * len(reward_values)))
            env_values = [env.get_tracking_variables() for env in envs]

            reward_lst.extend(reward_values)
            turns_lst.extend(turns_values)
            data_source_lst.extend(data_source_values)
            end_score_lst.extend(end_lst)
            answer_score_lst.extend(answer_lst)
            format_score_lst.extend(format_lst)
            envs_var.extend(env_values)

            repeat_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
            for local_idx, sample_idx in enumerate(pending_indices):
                start = local_idx * repeat_n
                end = start + repeat_n
                self._write_sample_shard(sample_idx, {
                    'env_vars': env_values[start:end],
                    'reward': reward_values[start:end],
                    'turns': turns_values[start:end],
                    'data_source': data_source_values[start:end],
                    'end': end_lst[start:end],
                    'answer': answer_lst[start:end],
                    'format': format_lst[start:end],
                    'input_texts': input_texts[start:end],
                    'output_texts': output_texts[start:end],
                    'sample_scores': answer_lst[start:end],
                })
                completed_indices.add(sample_idx)

        val_pbar.close()


        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.tensor(reward_lst, dtype=torch.float32)
        turns_tensor = torch.tensor(turns_lst, dtype=torch.float32)
        data_sources = np.array(data_source_lst, dtype=object)

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
            data_source_end[data_source].append(end_score_lst[i])
            if data_source not in data_source_answer:
                data_source_answer[data_source] = []
            data_source_answer[data_source].append(answer_score_lst[i])
            if data_source not in data_source_format:
                data_source_format[data_source] = []
            data_source_format[data_source].append(format_score_lst[i])
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

        val_metrics, val_env_list = self.validate()
        self._save_env_var(val_env_list)
        pprint(f'Validation metrics: {val_metrics}')
        logger.log(data=val_metrics, step=self.global_steps)
